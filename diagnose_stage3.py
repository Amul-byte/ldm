"""Standalone Stage-3 diagnostic script.

Runs all five diagnostic checks and saves results to --output_dir:
  noise_pred_error_by_timestep.png/.json  — denoiser error vs noise level
  generation_diversity.png/.json          — pairwise latent distance (mode collapse check)
  per_class_accuracy.png/.json            — per-activity classifier accuracy real vs gen
  latent_distribution_comparison.png/.json — OOD check: real z0 vs generated z0 stats
  conditioning_comparison_overlay.png/.json — conditional vs unconditional generation
  attention/                              — cross-attention weight maps from inspect_attention.py
  diagnosis_summary.json                  — aggregated numeric results

Usage:
  python diagnose_stage3.py \\
    --stage1_ckpt new_checkpoints/gcnn_stage1/stage1_best.pt \\
    --stage2_ckpt new_checkpoints/stage1_gcnn_mseonly/stage2_best.pt \\
    --stage3_ckpt new_checkpoints/stage3_gcnn/stage3_best.pt \\
    --skeleton_folder /path/to/skeleton \\
    --hip_folder /path/to/hip \\
    --wrist_folder /path/to/wrist \\
    --encoder_type gcn --imu_graph multiscale \\
    --output_dir outputs/stage3_diagnosis \\
    --max_batches 4 --sample_steps 20
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from diffusion_model.dataset import create_dataset, parse_subject_list, split_train_val_dataset
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.training_eval import (
    ensure_dir,
    plot_conditioning_comparison_overlay,
    plot_generation_diversity,
    plot_latent_distribution_comparison,
    plot_noise_pred_error_by_timestep,
    plot_per_class_accuracy,
    write_json,
)
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
    JOINT_LABELS,
    set_seed,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from PIL import Image, ImageDraw
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


DEFAULT_TRAIN_SUBJECTS = [
    28, 29, 30, 31, 33, 35, 38, 39, 32, 36, 37, 43, 44, 45, 46, 49, 51, 56, 57, 58, 59, 61, 62
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Stage-3 conditional latent diffusion model.")
    # Checkpoints
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/stage3_diagnosis")
    # Data
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="")
    parser.add_argument("--hip_folder", type=str, default="")
    parser.add_argument("--wrist_folder", type=str, default="")
    parser.add_argument("--gait_cache_dir", type=str, default="")
    parser.add_argument("--disable_gait_cache", action="store_true")
    # Model
    parser.add_argument("--imu_graph", type=str, default="chain")
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    # Dataset split
    parser.add_argument("--train_subjects", type=str, default=",".join(str(s) for s in DEFAULT_TRAIN_SUBJECTS))
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--stride", type=int, default=30)
    # Eval
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=4)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--n_diversity_samples", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def select_dataset(args: argparse.Namespace) -> Dataset:
    dataset = create_dataset(
        dataset_path=args.dataset_path or None,
        window=args.window,
        joints=args.joints,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
    )
    train_subjects = parse_subject_list(args.train_subjects) if args.train_subjects else None
    train_dataset, val_dataset = split_train_val_dataset(
        dataset,
        val_split=args.val_split,
        seed=args.seed,
        train_subjects=train_subjects,
        logger=None,
    )
    if args.split == "train":
        return train_dataset
    if args.split == "val":
        if val_dataset is None:
            raise ValueError("No validation split available. Use --val_split > 0 or --split all.")
        return val_dataset
    return dataset


# ── Attention inspection (inlined from inspect_attention.py) ──────────────────

@contextmanager
def _capture_multihead(module: nn.MultiheadAttention, sink: list):
    original_forward = module.forward

    def wrapped(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        out, weights = original_forward(*args, **kwargs)
        sink.append(weights)
        return out, weights

    module.forward = wrapped  # type: ignore[method-assign]
    try:
        yield
    finally:
        module.forward = original_forward  # type: ignore[method-assign]


def _enter_all(managers: list, fn):
    if not managers:
        fn()
        return
    with managers[0]:
        _enter_all(managers[1:], fn)


def _normalize_arr(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    vmax = arr.max()
    if vmax > 0:
        arr /= vmax
    return arr


def _save_gray(arr: np.ndarray, out_path: Path, upscale: int = 8) -> None:
    if not _PIL_AVAILABLE:
        return
    norm = _normalize_arr(arr)
    img = Image.fromarray(np.uint8(norm * 255.0), mode="L")
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), resample=Image.Resampling.NEAREST)
    img.save(out_path)


def _save_bar(values: np.ndarray, out_path: Path, width: int = 900, height: int = 240) -> None:
    if not _PIL_AVAILABLE:
        return
    vals = _normalize_arr(values.reshape(-1))
    n = max(len(vals), 1)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    bar_w = max(1, width // n)
    for i, v in enumerate(vals):
        x0 = i * bar_w
        x1 = min(width - 1, (i + 1) * bar_w - 1)
        y1 = height - 20
        y0 = int(y1 - v * (height - 40))
        draw.rectangle((x0, y0, x1, y1), fill=(30, 90, 200))
    img.save(out_path)


def _run_attention_inspection(stage2, stage3, loader, device: torch.device, out_dir: Path) -> None:
    """Inline of inspect_attention.py: captures cross-attention and classifier attention weights."""
    ensure_dir(out_dir)
    batch = next(iter(loader))
    x = batch["skeleton"].to(device)
    a_hip = batch["A_hip"].to(device)
    a_wrist = batch["A_wrist"].to(device)
    gait_metrics = batch["gait_metrics"].to(device)
    y = batch["label"].to(device)

    cross_sinks: list[list[torch.Tensor]] = [[] for _ in stage3.denoiser.cross_attn_blocks]
    classifier_sinks: list[list[torch.Tensor]] = [[] for _ in stage3.classifier.encoder.layers]
    managers = []
    for sink, block in zip(cross_sinks, stage3.denoiser.cross_attn_blocks):
        managers.append(_capture_multihead(block.attn, sink))
    for sink, layer in zip(classifier_sinks, stage3.classifier.encoder.layers):
        managers.append(_capture_multihead(layer.self_attn, sink))

    def _run():
        with torch.no_grad():
            h_tokens, h_global = stage2.aligner(a_hip, a_wrist)
            out = stage3(
                x=x,
                y=y,
                h_tokens=h_tokens,
                h_global=h_global,
                gait_metrics=gait_metrics,
                a_hip_stream=a_hip,
                a_wrist_stream=a_wrist,
            )
            _ = stage3.classifier(out["x_hat"])

    _enter_all(managers, _run)

    joint_labels = list(JOINT_LABELS[: x.shape[2]])
    t = x.shape[1]
    j = x.shape[2]

    for idx, sink in enumerate(cross_sinks):
        if not sink:
            continue
        w = sink[-1].detach().cpu()
        attn_qk = w[0].mean(dim=0).numpy()  # [T*J, T]
        np.save(out_dir / f"cross_block_{idx:02d}_raw.npy", attn_qk)
        _save_gray(attn_qk, out_dir / f"cross_block_{idx:02d}_raw.png", upscale=4)

        attn_tjt = attn_qk.reshape(t, j, t)
        sensor_importance = attn_tjt.mean(axis=(0, 1))
        frame_to_sensor = attn_tjt.mean(axis=1)
        joint_importance = attn_tjt.mean(axis=(0, 2))

        np.save(out_dir / f"cross_block_{idx:02d}_frame_to_sensor.npy", frame_to_sensor)
        _save_gray(frame_to_sensor, out_dir / f"cross_block_{idx:02d}_frame_to_sensor.png", upscale=8)
        _save_bar(sensor_importance, out_dir / f"cross_block_{idx:02d}_sensor_importance.png")

        if plt is not None and _PIL_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 3))
            y_pos = np.arange(len(joint_labels))
            ax.barh(y_pos, joint_importance, color="#2563eb")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(joint_labels, fontsize=7)
            ax.set_title(f"Cross-attention block {idx} joint importance")
            ax.set_xlabel("Mean attention weight")
            fig.tight_layout()
            fig.savefig(out_dir / f"cross_block_{idx:02d}_joint_importance.png", dpi=150)
            plt.close(fig)

    for idx, sink in enumerate(classifier_sinks):
        if not sink:
            continue
        w = sink[-1].detach().cpu()
        attn_tt = w[0].mean(dim=0).numpy()  # [T, T]
        np.save(out_dir / f"classifier_layer_{idx:02d}_tt.npy", attn_tt)
        _save_gray(attn_tt, out_dir / f"classifier_layer_{idx:02d}_tt.png", upscale=8)
        _save_bar(attn_tt.mean(axis=0), out_dir / f"classifier_layer_{idx:02d}_attended_by_all.png")
        _save_bar(attn_tt.mean(axis=1), out_dir / f"classifier_layer_{idx:02d}_querying_focus.png")


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    encoder_graph_op, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading Stage-1 …")
    stage1 = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        encoder_type=encoder_graph_op,
        skeleton_graph_op=skeleton_graph_op,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=False)
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad_(False)

    print("Loading Stage-2 …")
    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
        num_classes=args.num_classes,
        imu_graph_type=args.imu_graph,
    ).to(device)
    load_checkpoint(args.stage2_ckpt, stage2, strict=False)
    stage2.eval()
    for p in stage2.parameters():
        p.requires_grad_(False)

    print("Loading Stage-3 …")
    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage3_ckpt, stage3, strict=False)
    stage3.eval()
    for p in stage3.parameters():
        p.requires_grad_(False)

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("Building dataset …")
    dataset = select_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print(f"  {len(dataset)} samples, {len(loader)} batches")

    # ── Run diagnostics ────────────────────────────────────────────────────────
    summary: dict = {
        "stage1_ckpt": args.stage1_ckpt,
        "stage2_ckpt": args.stage2_ckpt,
        "stage3_ckpt": args.stage3_ckpt,
        "output_dir": str(out_dir),
        "diagnostics": {},
    }

    diagnostics = [
        ("noise_pred_by_timestep",  plot_noise_pred_error_by_timestep,
         {"num_bins": args.num_bins, "max_batches": args.max_batches}),
        ("generation_diversity",    plot_generation_diversity,
         {"k_samples": args.n_diversity_samples, "sample_steps": args.sample_steps,
          "sampler": args.sampler, "max_batches": max(1, args.max_batches // 2)}),
        ("per_class_accuracy",      plot_per_class_accuracy,
         {"sample_steps": args.sample_steps, "sampler": args.sampler,
          "max_batches": args.max_batches, "num_classes": args.num_classes}),
        ("latent_distribution",     plot_latent_distribution_comparison,
         {"sample_steps": args.sample_steps, "sampler": args.sampler, "max_batches": args.max_batches}),
        ("conditioning_comparison", plot_conditioning_comparison_overlay,
         {"sample_steps": args.sample_steps, "sampler": args.sampler, "sample_seed": args.sample_seed}),
    ]

    with torch.no_grad():
        for name, fn, kwargs in diagnostics:
            print(f"Running {name} …")
            try:
                result = fn(stage2, stage3, loader, device, out_dir, **kwargs)
                summary["diagnostics"][name] = result
                print(f"  ✓ {name}")
            except Exception as e:
                summary["diagnostics"][name] = {"error": str(e)}
                print(f"  ✗ {name}: {e}")

    print("Running attention inspection …")
    try:
        _run_attention_inspection(stage2, stage3, loader, device, out_dir / "attention")
        summary["diagnostics"]["attention_inspection"] = {"status": "ok"}
        print("  ✓ attention_inspection")
    except Exception as e:
        summary["diagnostics"]["attention_inspection"] = {"error": str(e)}
        print(f"  ✗ attention_inspection: {e}")

    write_json(out_dir / "diagnosis_summary.json", summary)
    print(f"\nDiagnosis complete. Results saved to {out_dir}")
    print(json.dumps(summary["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
