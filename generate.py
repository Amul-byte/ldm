"""Conditional generation script for stage-3 latent diffusion."""

from __future__ import annotations

import argparse
import os
import re
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusion_model.dataset import create_dataloader
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
    get_skeleton_edges,
)


def _normalize_xy(points_xy: np.ndarray, canvas_size: int) -> np.ndarray:
    """Normalize xy coordinates into drawable canvas coordinates."""
    assert points_xy.ndim == 3, "points_xy must be [T, J, 2]"
    min_xy = points_xy.min(axis=(0, 1), keepdims=True)
    max_xy = points_xy.max(axis=(0, 1), keepdims=True)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scaled = (points_xy - min_xy) / span
    margin = 0.1 * canvas_size
    scaled = scaled * (canvas_size - 2 * margin) + margin
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
        return skeleton, a_hip, a_wrist, int(labels[i].item())
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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--disable_sensor_norm", action="store_true")
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--h_none", action="store_true")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--gif_dir", type=str, default="outputs/gifs")
    parser.add_argument("--gif_prefix", type=str, default="sample")
    parser.add_argument("--gif_fps", type=int, default=12)
    parser.add_argument("--gif_index", type=int, default=0, help="Batch index to save as a single GIF.")
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
    return parser.parse_args()


def main() -> None:
    """Run conditional sampling and optional classification."""
    args = parse_args()
    if args.h_none:
        raise ValueError("Strict proposal behavior requires sensor conditioning during generation. Remove --h_none.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)

    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=args.latent_dim, num_joints=args.joints).to(device)
    load_checkpoint(args.stage2_ckpt, stage2, strict=True)

    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
    ).to(device)
    load_checkpoint(args.stage3_ckpt, stage3, strict=True)

    loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        synthetic_length=args.batch_size,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
        shuffle=True,
    )

    target_class = _parse_target_class(args.target_class, args.num_classes) if args.target_class.strip() else None

    if target_class is not None:
        skeleton, a_hip_stream, a_wrist_stream, cond_label = _pick_target_conditioning(loader, target_class, device)
        print(
            "conditioning sample class: "
            f"id={cond_label} name={_class_name(cond_label, args.num_classes)}"
        )
    else:
        batch = next(iter(loader))
        skeleton = batch["skeleton"].to(device)
        a_hip_stream = batch["A_hip"].to(device)
        a_wrist_stream = batch["A_wrist"].to(device)

    def _sample_once() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_global_local, sensor_tokens_local = stage2.aligner(a_hip_stream, a_wrist_stream)
        shape = (skeleton.shape[0], skeleton.shape[1], skeleton.shape[2], args.latent_dim)
        z0_local = stage3.diffusion.p_sample_loop(
            stage3.denoiser,
            shape=torch.Size(shape),
            device=device,
            sensor_tokens=sensor_tokens_local,
            h_global=h_global_local,
        )
        x_local = stage3.decoder(z0_local)
        logits_local = stage3.classifier(x_local)
        return x_local, logits_local, z0_local, h_global_local

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
    gif_idx = 0 if target_class is not None else int(max(0, min(args.gif_index, x_hat.shape[0] - 1)))
    pred_id = int(pred[gif_idx].item())
    pred_name = _class_name(pred_id, args.num_classes)
    print(f"predicted class for sample[{gif_idx}]: id={pred_id} name={pred_name}")

    if args.save_gif:
        gif_path = os.path.join(args.gif_dir, f"{args.gif_prefix}_{pred_name}.gif")
        save_skeleton_gif(x_hat[gif_idx], gif_path, fps=args.gif_fps)
        print(f"saved gif: {gif_path}")

    if args.classify:
        print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
