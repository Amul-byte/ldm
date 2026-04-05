"""Open the encoder->decoder bridge and visualize where structure gets lost.

This diagnostic compares three paths on the same real skeleton batch:

1. Stage-1 decoder on clean encoder latents  (before Stage-3 decoder training)
2. Stage-3 decoder on clean encoder latents  (after Stage-3 decoder training)
3. Full Stage-3 diffusion path               (optional, requires Stage-2 ckpt)

It also opens the Stage-3 decoder itself:
- per-layer activation statistics
- partial decodes after each graph/temporal block
- latent ablations (time mean / joint mean / global mean / shuffled)
- latent sensitivity to controlled Gaussian perturbations
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_model.dataset import create_dataset
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.training_eval import (
    _mpjpe,
    _root_trajectory_error,
    _velocity_error,
    render_skeleton_panels,
    sample_stage3_latents,
    write_json,
)
from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_NUM_CLASSES, DEFAULT_TIMESTEPS, DEFAULT_WINDOW


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def _infer_latent_dim(stage1_ckpt: str) -> int:
    state_dict = _load_state_dict(stage1_ckpt)
    return int(state_dict["encoder.in_proj.weight"].shape[0])


def _infer_d_shared(stage2_ckpt: str, default: int = 64) -> int:
    state_dict = _load_state_dict(stage2_ckpt)
    key = "shared_motion_layer.net.0.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    return int(default)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "abs_mean": float(tensor.abs().mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "l2_mean": float(torch.linalg.norm(tensor.reshape(tensor.shape[0], -1), dim=1).mean().item()),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _decode_with_trace(decoder: torch.nn.Module, z: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, object]], list[tuple[str, torch.Tensor]]]:
    adjacency = decoder._skel_adjacency
    edge_index = decoder._skel_edge_index
    h = z
    stats = [_tensor_stats("input_latent", h)]
    partial_decodes: list[tuple[str, torch.Tensor]] = [("input_latent", decoder.out_proj(h))]
    for idx, (g_block, t_block) in enumerate(zip(decoder.graph_blocks, decoder.temporal_blocks)):
        h = g_block(h, adjacency=adjacency, edge_index=edge_index)
        stats.append(_tensor_stats(f"after_graph_{idx}", h))
        partial_decodes.append((f"after_graph_{idx}", decoder.out_proj(h)))
        h = t_block(h)
        stats.append(_tensor_stats(f"after_temporal_{idx}", h))
        partial_decodes.append((f"after_temporal_{idx}", decoder.out_proj(h)))
    x_hat = decoder.out_proj(h)
    stats.append(_tensor_stats("final_output", x_hat))
    partial_decodes.append(("final_output", x_hat))
    return x_hat, stats, partial_decodes


def _panel_chunks(items: Sequence[tuple[str, np.ndarray]], chunk_size: int = 4) -> list[list[tuple[str, np.ndarray]]]:
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def _mean_time(z: torch.Tensor) -> torch.Tensor:
    return z.mean(dim=1, keepdim=True).expand_as(z)


def _mean_joint(z: torch.Tensor) -> torch.Tensor:
    return z.mean(dim=2, keepdim=True).expand_as(z)


def _mean_global(z: torch.Tensor) -> torch.Tensor:
    return z.mean(dim=(1, 2), keepdim=True).expand_as(z)


def _shuffle_time(z: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(z.shape[1] - 1, -1, -1, device=z.device)
    return z[:, idx]


def _shuffle_joint(z: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(z.shape[2] - 1, -1, -1, device=z.device)
    return z[:, :, idx]


def _latent_ablation_rows(decoder: torch.nn.Module, z: torch.Tensor, x: torch.Tensor) -> tuple[list[dict[str, object]], list[tuple[str, torch.Tensor]]]:
    variants = {
        "full_latent": z,
        "time_mean": _mean_time(z),
        "joint_mean": _mean_joint(z),
        "global_mean": _mean_global(z),
        "time_reversed": _shuffle_time(z),
        "joint_reversed": _shuffle_joint(z),
    }
    rows: list[dict[str, object]] = []
    outputs: list[tuple[str, torch.Tensor]] = []
    for name, z_variant in variants.items():
        x_hat = decoder(z_variant)
        outputs.append((name, x_hat))
        rows.append(
            {
                "variant": name,
                "mpjpe_m": float(_mpjpe(x_hat, x).item()),
                "root_trajectory_error_m": float(_root_trajectory_error(x_hat, x).item()),
                "velocity_error_m": float(_velocity_error(x_hat, x).item()),
                "decoder_output_std": float(x_hat.std(unbiased=False).item()),
            }
        )
    return rows, outputs


def _latent_sensitivity_rows(decoder: torch.nn.Module, z: torch.Tensor, x: torch.Tensor) -> tuple[list[dict[str, object]], list[tuple[str, torch.Tensor]]]:
    rows: list[dict[str, object]] = []
    outputs: list[tuple[str, torch.Tensor]] = []
    base = decoder(z)
    outputs.append(("clean", base))
    z_std = max(float(z.std(unbiased=False).item()), 1e-6)
    for level in (0.01, 0.05, 0.10):
        noisy = z + torch.randn_like(z) * (level * z_std)
        x_hat = decoder(noisy)
        outputs.append((f"noise_{level:.2f}", x_hat))
        rows.append(
            {
                "noise_level": level,
                "mpjpe_to_real_m": float(_mpjpe(x_hat, x).item()),
                "mpjpe_to_clean_decode_m": float(_mpjpe(x_hat, base).item()),
                "output_delta_l2": float(torch.linalg.norm((x_hat - base).reshape(x_hat.shape[0], -1), dim=1).mean().item()),
            }
        )
    return rows, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encoder/decoder bridge diagnosis")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--output_dir", type=str, default="outputs/encoder_decoder_bridge")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--latent_dim", type=int, default=0)
    parser.add_argument("--d_shared", type=int, default=0)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=45)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--imu_graph", type=str, default="multiscale", choices=["chain", "multiscale"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    encoder_graph_op, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)
    latent_dim = args.latent_dim or _infer_latent_dim(args.stage1_ckpt)
    d_shared = args.d_shared or (_infer_d_shared(args.stage2_ckpt) if args.stage2_ckpt else 64)
    print(
        "Resolved config:",
        f"latent_dim={latent_dim}",
        f"d_shared={d_shared}",
        f"encoder_graph_op={encoder_graph_op}",
        f"skeleton_graph_op={skeleton_graph_op}",
    )

    stage1_before = Stage1Model(
        latent_dim=latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        num_classes=args.num_classes,
        encoder_type=encoder_graph_op,
        skeleton_graph_op=skeleton_graph_op,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1_before, strict=False)
    stage1_before.eval()

    stage1_for_stage3 = Stage1Model(
        latent_dim=latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        num_classes=args.num_classes,
        encoder_type=encoder_graph_op,
        skeleton_graph_op=skeleton_graph_op,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1_for_stage3, strict=False)
    stage1_for_stage3.eval()

    stage3 = Stage3Model(
        encoder=stage1_for_stage3.encoder,
        decoder=stage1_for_stage3.decoder,
        denoiser=stage1_for_stage3.denoiser,
        latent_dim=latent_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        d_shared=d_shared,
    ).to(device)
    load_checkpoint(args.stage3_ckpt, stage3, strict=False)
    stage3.eval()

    stage2 = None
    if args.stage2_ckpt:
        stage2 = Stage2Model(
            encoder=stage1_for_stage3.encoder,
            latent_dim=latent_dim,
            num_joints=args.joints,
            gait_metrics_dim=args.gait_metrics_dim,
            num_classes=args.num_classes,
            imu_graph_type=args.imu_graph,
            d_shared=d_shared,
        ).to(device)
        load_checkpoint(args.stage2_ckpt, stage2, strict=False)
        stage2.eval()

    dataset = create_dataset(
        dataset_path=args.dataset_path or None,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.num_samples, shuffle=True, drop_last=False)
    batch = next(iter(loader))
    x = batch["skeleton"].to(device)
    a_hip = batch["A_hip"].to(device)
    a_wrist = batch["A_wrist"].to(device)

    with torch.no_grad():
        z0_real = stage3.encoder(x, gait_metrics=None)
        x_hat_stage1, stage1_trace, _ = _decode_with_trace(stage1_before.decoder, z0_real)
        x_hat_stage3, stage3_trace, stage3_partials = _decode_with_trace(stage3.decoder, z0_real)

        ablation_rows, ablation_outputs = _latent_ablation_rows(stage3.decoder, z0_real, x)
        sensitivity_rows, sensitivity_outputs = _latent_sensitivity_rows(stage3.decoder, z0_real, x)

        x_hat_diff = None
        z0_diff = None
        if stage2 is not None:
            h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)
            z0_diff = sample_stage3_latents(
                stage3=stage3,
                shape=z0_real.shape,
                device=device,
                h_tokens=h_tokens,
                h_global=h_global,
                a_hip_stream=a_hip,
                a_wrist_stream=a_wrist,
                gait_metrics=None,
                sample_steps=args.sample_steps,
                sampler=args.sampler,
                sample_seed=args.sample_seed,
            )
            x_hat_diff = stage3.decoder(z0_diff)

    summary = {
        "config": {
            "stage1_ckpt": args.stage1_ckpt,
            "stage2_ckpt": args.stage2_ckpt,
            "stage3_ckpt": args.stage3_ckpt,
            "latent_dim": latent_dim,
            "d_shared": d_shared,
            "sample_steps": args.sample_steps,
            "sampler": args.sampler,
            "sample_seed": args.sample_seed,
        },
        "clean_latent": _tensor_stats("z0_real", z0_real),
        "stage1_decoder_clean": {
            "mpjpe_m": float(_mpjpe(x_hat_stage1, x).item()),
            "root_trajectory_error_m": float(_root_trajectory_error(x_hat_stage1, x).item()),
            "velocity_error_m": float(_velocity_error(x_hat_stage1, x).item()),
        },
        "stage3_decoder_clean": {
            "mpjpe_m": float(_mpjpe(x_hat_stage3, x).item()),
            "root_trajectory_error_m": float(_root_trajectory_error(x_hat_stage3, x).item()),
            "velocity_error_m": float(_velocity_error(x_hat_stage3, x).item()),
        },
        "decoder_improvement_mpjpe_m": float(_mpjpe(x_hat_stage1, x).item() - _mpjpe(x_hat_stage3, x).item()),
        "stage3_decoder_trace": stage3_trace,
        "stage1_decoder_trace": stage1_trace,
        "latent_ablations": ablation_rows,
        "latent_sensitivity": sensitivity_rows,
    }
    if x_hat_diff is not None and z0_diff is not None:
        summary["stage3_diffusion_path"] = {
            "mpjpe_m": float(_mpjpe(x_hat_diff, x).item()),
            "root_trajectory_error_m": float(_root_trajectory_error(x_hat_diff, x).item()),
            "velocity_error_m": float(_velocity_error(x_hat_diff, x).item()),
            "latent_l2_mean": float(torch.linalg.norm((z0_diff - z0_real).reshape(z0_real.shape[0], -1), dim=1).mean().item()),
            "latent_stats": _tensor_stats("z0_diff", z0_diff),
        }

    write_json(out_dir / "bridge_summary.json", summary)
    _write_csv(out_dir / "stage3_decoder_trace.csv", stage3_trace)
    _write_csv(out_dir / "stage1_decoder_trace.csv", stage1_trace)
    _write_csv(out_dir / "latent_ablation_metrics.csv", ablation_rows)
    _write_csv(out_dir / "latent_sensitivity_metrics.csv", sensitivity_rows)

    sample_idx = 0
    main_sequences: list[np.ndarray] = [
        x[sample_idx].cpu().numpy(),
        x_hat_stage1[sample_idx].cpu().numpy(),
        x_hat_stage3[sample_idx].cpu().numpy(),
    ]
    main_titles = ["Real", "Stage1 decoder\n(clean latent)", "Stage3 decoder\n(clean latent)"]
    if x_hat_diff is not None:
        main_sequences.append(x_hat_diff[sample_idx].cpu().numpy())
        main_titles.append("Stage3 full path\n(diffusion->decoder)")
    render_skeleton_panels(out_dir / "bridge_comparison_sample0.png", main_sequences, main_titles)

    partial_items = [("Real", x[sample_idx].cpu().numpy())]
    for name, seq in stage3_partials:
        partial_items.append((name, seq[sample_idx].cpu().numpy()))
    for chunk_idx, chunk in enumerate(_panel_chunks(partial_items, chunk_size=4)):
        render_skeleton_panels(
            out_dir / f"stage3_decoder_progression_sample0_part{chunk_idx + 1}.png",
            [seq for _, seq in chunk],
            [name for name, _ in chunk],
        )

    ablation_items = [("Real", x[sample_idx].cpu().numpy())]
    ablation_items.extend((name, seq[sample_idx].cpu().numpy()) for name, seq in ablation_outputs)
    for chunk_idx, chunk in enumerate(_panel_chunks(ablation_items, chunk_size=4)):
        render_skeleton_panels(
            out_dir / f"latent_ablation_sample0_part{chunk_idx + 1}.png",
            [seq for _, seq in chunk],
            [name for name, _ in chunk],
        )

    sensitivity_items = [("Real", x[sample_idx].cpu().numpy())]
    sensitivity_items.extend((name, seq[sample_idx].cpu().numpy()) for name, seq in sensitivity_outputs)
    render_skeleton_panels(
        out_dir / "latent_sensitivity_sample0.png",
        [seq for _, seq in sensitivity_items[:4]],
        [name for name, _ in sensitivity_items[:4]],
    )
    if len(sensitivity_items) > 4:
        render_skeleton_panels(
            out_dir / "latent_sensitivity_sample0_part2.png",
            [seq for _, seq in sensitivity_items[4:]],
            [name for name, _ in sensitivity_items[4:]],
        )

    print(json.dumps(summary, indent=2))
    print(f"Saved bridge diagnosis to: {out_dir}")


if __name__ == "__main__":
    main()
