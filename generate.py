"""Conditional generation script for stage-3 latent diffusion."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusion_model.dataset import create_dataloader
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.training_eval import sample_stage3_latents
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
    get_skeleton_edges,
)


def _normalize_xy(points_xy: np.ndarray, canvas_size: int, root_index: int = 0) -> np.ndarray:
    """Normalize xy coordinates into drawable canvas coordinates.

    Anchor the sequence to the first-frame root instead of re-centering every
    frame so the rendered motion can keep its global trajectory. Use robust
    percentile bounds so a single outlier frame does not shrink the whole body.
    """
    assert points_xy.ndim == 3, "points_xy must be [T, J, 2]"
    assert 0 <= root_index < points_xy.shape[1], "root_index out of range"
    anchor = points_xy[:1, root_index : root_index + 1, :]
    centered = points_xy - anchor
    flat_xy = centered.reshape(-1, 2)
    min_xy = np.percentile(flat_xy, 2.0, axis=0, keepdims=True).reshape(1, 1, 2)
    max_xy = np.percentile(flat_xy, 98.0, axis=0, keepdims=True).reshape(1, 1, 2)
    span = float(np.maximum(max_xy - min_xy, 1e-6).max())
    normalized = (centered - (min_xy + max_xy) * 0.5) / span
    margin = 0.1 * canvas_size
    drawable = canvas_size - 2 * margin
    canvas_center = canvas_size * 0.5
    scaled = normalized * drawable + canvas_center
    return scaled


def save_skeleton_gif(sequence: torch.Tensor, out_path: str, fps: int = 12, canvas_size: int = 512) -> None:
    """Render one skeleton sequence [T, J, 3] to an animated GIF."""
    assert sequence.ndim == 3 and sequence.shape[-1] == 3, "sequence must have shape [T, J, 3]"
    points = sequence.detach().cpu().numpy().astype(np.float32)
    num_joints = points.shape[1]
    points_xy = points[..., :2]
    points_xy = _normalize_xy(points_xy, canvas_size=canvas_size)
    edges = [(i, j) for i, j in get_skeleton_edges() if i < num_joints and j < num_joints]
    frames: List[Image.Image] = []

    for frame_xy in points_xy:
        img = Image.new("RGB", (canvas_size, canvas_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Draw bones first so joints stay visible on top.
        for i, j in edges:
            xi, yi = float(frame_xy[i][0]), float(frame_xy[i][1])
            xj, yj = float(frame_xy[j][0]), float(frame_xy[j][1])
            draw.line((xi, yi, xj, yj), fill=(30, 30, 30), width=3)
        for joint_xy in frame_xy:
            x, y = float(joint_xy[0]), float(joint_xy[1])
            r = 4
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(20, 60, 210))
        frames.append(img)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    duration_ms = int(1000 / max(fps, 1))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _class_name(class_id: int, num_classes: int) -> str:
    """Return readable class name for a predicted class id."""
    if 0 <= class_id < num_classes and num_classes == 14:
        return f"A{class_id + 1:02d}"
    return f"class_{class_id}"


def _parse_target_class(raw: str, num_classes: int) -> int:
    """Parse target class from formats like 'A12', '12', or zero-based ids."""
    value = raw.strip().upper()
    if not value:
        raise ValueError("target class is empty")
    match = re.fullmatch(r"A(\d{2})", value)
    if match:
        cid = int(match.group(1)) - 1
    else:
        n = int(value)
        if 0 <= n < num_classes:
            cid = n
        elif 1 <= n <= num_classes:
            cid = n - 1
        else:
            raise ValueError(f"target class '{raw}' is out of range for num_classes={num_classes}")
    if cid < 0 or cid >= num_classes:
        raise ValueError(f"target class '{raw}' is out of range for num_classes={num_classes}")
    return cid


def _pick_target_conditioning(
    loader: torch.utils.data.DataLoader,
    target_class: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Find first dataset sample whose real label matches target_class."""
    for batch in loader:
        labels = batch["label"]
        hit = torch.nonzero(labels == target_class, as_tuple=False)
        if hit.numel() == 0:
            continue
        i = int(hit[0].item())
        skeleton = batch["skeleton"][i : i + 1].to(device)
        a_hip = batch["A_hip"][i : i + 1].to(device)
        a_wrist = batch["A_wrist"][i : i + 1].to(device)
        gait_metrics = batch["gait_metrics"][i : i + 1].to(device)
        return skeleton, a_hip, a_wrist, gait_metrics, int(labels[i].item())
    raise ValueError(
        f"No sample with real label target_class={target_class} ({_class_name(target_class, 14)}) found in provided data."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate skeletons with conditional latent diffusion")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="")
    parser.add_argument("--hip_folder", type=str, default="")
    parser.add_argument("--wrist_folder", type=str, default="")
    parser.add_argument("--gait_cache_dir", type=str, default="")
    parser.add_argument("--disable_gait_cache", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--disable_sensor_norm", action="store_true")
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--gif_dir", type=str, default="outputs/results_new")
    parser.add_argument("--gif_prefix", type=str, default="sample")
    parser.add_argument("--gif_fps", type=int, default=12)
    parser.add_argument("--gif_index", type=int, default=0, help="Legacy single-GIF index; ignored when saving full batch.")
    parser.add_argument(
        "--target_class",
        type=str,
        default="",
        help="Target class to generate (examples: A12, 12, or 0-based id).",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=64,
        help="Max sampling attempts to hit target predicted class.",
    )
    parser.add_argument("--sample_steps", type=int, default=50, help="Reverse-process sampling steps.")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"], help="Reverse-process sampler.")
    return parser.parse_args()


def main() -> None:
    """Run conditional sampling and optional classification."""
    args = parse_args()
    if args.gait_metrics_dim != DEFAULT_GAIT_METRICS_DIM:
        raise ValueError(
            f"--gait_metrics_dim must equal the fixed auto-computed gait-summary size ({DEFAULT_GAIT_METRICS_DIM})."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)

    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage2_ckpt, stage2, strict=True)

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
    load_checkpoint(args.stage3_ckpt, stage3, strict=True)

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
        shuffle=True,
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
    )

    target_class = _parse_target_class(args.target_class, args.num_classes) if args.target_class.strip() else None

    if target_class is not None:
        skeleton, a_hip_stream, a_wrist_stream, gait_metrics, cond_label = _pick_target_conditioning(
            loader, target_class, device
        )
        y_cond = torch.full((skeleton.shape[0],), target_class, device=device, dtype=torch.long)
        print(
            "conditioning sample class: "
            f"id={cond_label} name={_class_name(cond_label, args.num_classes)}"
        )
    else:
        batch = next(iter(loader))
        skeleton = batch["skeleton"].to(device)
        a_hip_stream = batch["A_hip"].to(device)
        a_wrist_stream = batch["A_wrist"].to(device)
        gait_metrics = batch["gait_metrics"].to(device)
        y_cond = batch["label"].to(device)

    def _sample_once() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sensor_tokens_local, h_local = stage2.aligner(a_hip_stream, a_wrist_stream, gait_metrics=gait_metrics)
        cond_tokens_local, cond_global_local = stage3.condition_with_labels(
            h_tokens=sensor_tokens_local,
            h_global=h_local,
            y=y_cond,
        )
        shape = (skeleton.shape[0], skeleton.shape[1], skeleton.shape[2], args.latent_dim)
        z0_local = sample_stage3_latents(
            stage3=stage3,
            shape=torch.Size(shape),
            device=device,
            h_tokens=cond_tokens_local,
            h_global=cond_global_local,
            gait_metrics=gait_metrics,
            sample_steps=args.sample_steps,
            sampler=args.sampler,
        )
        x_local = stage3.decoder(z0_local)
        logits_local = stage3.classifier(x_local)
        return x_local, logits_local, z0_local, cond_global_local

    if target_class is None:
        x_hat, logits, z0_gen, h_global = _sample_once()
        matched = True
        attempts = 1
    else:
        best_pack = None
        best_target_logit = float("-inf")
        matched = False
        attempts = 0
        for attempt in range(1, max(1, args.max_attempts) + 1):
            x_try, logits_try, z0_try, h_global_try = _sample_once()
            pred_try = torch.argmax(logits_try, dim=1)
            pred_id = int(pred_try[0].item())
            target_logit = float(logits_try[0, target_class].item())
            if target_logit > best_target_logit:
                best_target_logit = target_logit
                best_pack = (x_try, logits_try, z0_try, h_global_try, pred_id)
            attempts = attempt
            if pred_id == target_class:
                matched = True
                best_pack = (x_try, logits_try, z0_try, h_global_try, pred_id)
                break
        assert best_pack is not None
        x_hat, logits, z0_gen, h_global, _ = best_pack
        print(
            f"target class: id={target_class} name={_class_name(target_class, args.num_classes)} "
            f"| matched={matched} | attempts={attempts}"
        )

    print(f"h_global shape: {None if h_global is None else tuple(h_global.shape)}")
    print(f"z0_gen shape: {tuple(z0_gen.shape)}")
    print(f"x_hat shape: {tuple(x_hat.shape)}")

    pred = torch.argmax(logits, dim=1)
    for sample_idx in range(x_hat.shape[0]):
        pred_id = int(pred[sample_idx].item())
        pred_name = _class_name(pred_id, args.num_classes)
        print(f"predicted class for sample[{sample_idx}]: id={pred_id} name={pred_name}")

    if args.save_gif:
        run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        for sample_idx in range(x_hat.shape[0]):
            pred_id = int(pred[sample_idx].item())
            pred_name = _class_name(pred_id, args.num_classes)
            gif_path = os.path.join(
                args.gif_dir,
                f"{args.gif_prefix}_{run_tag}_idx{sample_idx:02d}_{pred_name}.gif",
            )
            save_skeleton_gif(x_hat[sample_idx], gif_path, fps=args.gif_fps)
            print(f"saved gif: {gif_path}")

    if args.classify:
        print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
