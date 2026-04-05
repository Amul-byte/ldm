"""Save interpretable attention views for Stage-3 cross-attention and classifier attention."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusion_model.dataset import create_dataloader
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
    JOINT_LABELS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect attention weights in Stage-3 models")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="")
    parser.add_argument("--hip_folder", type=str, default="")
    parser.add_argument("--wrist_folder", type=str, default="")
    parser.add_argument("--gait_cache_dir", type=str, default="")
    parser.add_argument("--disable_gait_cache", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--disable_sensor_norm", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/attention")
    return parser.parse_args()


def _load_state(path: str, model: torch.nn.Module, strict: bool) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"loaded: {path}")
    print(f"missing keys ({len(missing)}): {missing}")
    print(f"unexpected keys ({len(unexpected)}): {unexpected}")


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    vmax = arr.max()
    if vmax > 0:
        arr = arr / vmax
    return arr


def _save_gray(arr: np.ndarray, out_path: str, upscale: int = 8) -> None:
    norm = _normalize(arr)
    img = Image.fromarray(np.uint8(norm * 255.0), mode="L")
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), resample=Image.Resampling.NEAREST)
    img.save(out_path)


def _save_bar(values: np.ndarray, out_path: str, width: int = 900, height: int = 240) -> None:
    vals = _normalize(values.reshape(-1))
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


def _save_joint_bar(values: np.ndarray, out_path: str, labels: list[str]) -> None:
    vals = _normalize(values.reshape(-1))
    width = 1100
    row_h = 26
    height = 20 + row_h * len(vals)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for i, (label, value) in enumerate(zip(labels, vals)):
        y = 10 + i * row_h
        draw.text((10, y), label, fill=(0, 0, 0))
        x0 = 220
        x1 = int(x0 + value * 840)
        draw.rectangle((x0, y + 4, x1, y + 20), fill=(30, 90, 200))
    img.save(out_path)


def _mean_over_heads(weights: torch.Tensor, sample_idx: int = 0) -> np.ndarray:
    w = weights.detach().cpu()
    if w.ndim != 4:
        raise ValueError(f"Expected attention weights [B,H,Q,K], got {tuple(w.shape)}")
    return w[sample_idx].mean(dim=0).numpy()


@contextmanager
def capture_multihead(module: torch.nn.MultiheadAttention, sink: list[torch.Tensor]):
    original_forward = module.forward

    def wrapped_forward(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        out, weights = original_forward(*args, **kwargs)
        sink.append(weights)
        return out, weights

    module.forward = wrapped_forward  # type: ignore[method-assign]
    try:
        yield
    finally:
        module.forward = original_forward  # type: ignore[method-assign]


def _enter_all(items: list[contextmanager], fn: Callable[[], None]) -> None:
    if not items:
        fn()
        return
    with items[0]:
        _enter_all(items[1:], fn)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_graph_op, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)

    stage1 = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        encoder_type=encoder_graph_op,
        skeleton_graph_op=skeleton_graph_op,
    ).to(device)
    _load_state(args.stage1_ckpt, stage1, strict=True)

    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    _load_state(args.stage2_ckpt, stage2, strict=True)

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
    _load_state(args.stage3_ckpt, stage3, strict=False)

    stage1.eval()
    stage2.eval()
    stage3.eval()

    loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
        shuffle=False,
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
        drop_last=False,
    )

    batch = next(iter(loader))
    x = batch["skeleton"].to(device)
    y = batch["label"].to(device)
    a_hip = batch["A_hip"].to(device)
    a_wrist = batch["A_wrist"].to(device)
    gait_metrics = batch["gait_metrics"].to(device)

    os.makedirs(args.out_dir, exist_ok=True)

    cross_sinks: list[list[torch.Tensor]] = [[] for _ in stage3.denoiser.cross_attn_blocks]
    classifier_sinks: list[list[torch.Tensor]] = [[] for _ in stage3.classifier.encoder.layers]
    managers: list[contextmanager] = []
    for sink, block in zip(cross_sinks, stage3.denoiser.cross_attn_blocks):
        managers.append(capture_multihead(block.attn, sink))
    for sink, layer in zip(classifier_sinks, stage3.classifier.encoder.layers):
        managers.append(capture_multihead(layer.self_attn, sink))

    def _run() -> None:
        with torch.no_grad():
            h_tokens, h_global = stage2.aligner(a_hip, a_wrist, gait_metrics=gait_metrics)
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

    joint_labels = list(JOINT_LABELS[: args.joints])
    t = x.shape[1]
    j = x.shape[2]

    for idx, sink in enumerate(cross_sinks):
        if not sink:
            continue
        attn_qk = _mean_over_heads(sink[-1], sample_idx=0)  # [T*J, T]
        np.save(os.path.join(args.out_dir, f"cross_block_{idx:02d}_raw.npy"), attn_qk)
        _save_gray(attn_qk, os.path.join(args.out_dir, f"cross_block_{idx:02d}_raw.png"), upscale=4)

        attn_tjt = attn_qk.reshape(t, j, t)
        sensor_importance = attn_tjt.mean(axis=(0, 1))
        frame_to_sensor = attn_tjt.mean(axis=1)
        joint_importance = attn_tjt.mean(axis=(0, 2))

        np.save(os.path.join(args.out_dir, f"cross_block_{idx:02d}_frame_to_sensor.npy"), frame_to_sensor)
        _save_gray(
            frame_to_sensor,
            os.path.join(args.out_dir, f"cross_block_{idx:02d}_frame_to_sensor.png"),
            upscale=8,
        )
        _save_bar(sensor_importance, os.path.join(args.out_dir, f"cross_block_{idx:02d}_sensor_importance.png"))
        _save_joint_bar(
            joint_importance,
            os.path.join(args.out_dir, f"cross_block_{idx:02d}_joint_importance.png"),
            joint_labels,
        )
        print(
            f"saved cross block {idx}: raw={attn_qk.shape} frame_to_sensor={frame_to_sensor.shape} joints={joint_importance.shape}"
        )

    for idx, sink in enumerate(classifier_sinks):
        if not sink:
            continue
        attn_tt = _mean_over_heads(sink[-1], sample_idx=0)  # [T, T]
        np.save(os.path.join(args.out_dir, f"classifier_layer_{idx:02d}_tt.npy"), attn_tt)
        _save_gray(attn_tt, os.path.join(args.out_dir, f"classifier_layer_{idx:02d}_tt.png"), upscale=8)

        attended_by_all = attn_tt.mean(axis=0)
        querying_focus = attn_tt.mean(axis=1)
        _save_bar(attended_by_all, os.path.join(args.out_dir, f"classifier_layer_{idx:02d}_attended_by_all.png"))
        _save_bar(querying_focus, os.path.join(args.out_dir, f"classifier_layer_{idx:02d}_querying_focus.png"))
        print(f"saved classifier layer {idx}: tt={attn_tt.shape}")


if __name__ == "__main__":
    main()
