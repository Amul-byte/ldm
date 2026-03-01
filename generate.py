"""Conditional generation script for stage-3 latent diffusion."""

from __future__ import annotations

import argparse
import os
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
    points_xy = points[..., :2]
    points_xy = _normalize_xy(points_xy, canvas_size=canvas_size)
    frames: List[Image.Image] = []

    for frame_xy in points_xy:
        img = Image.new("RGB", (canvas_size, canvas_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate skeletons with conditional latent diffusion")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--h_none", action="store_true")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--gif_dir", type=str, default="outputs/gifs")
    parser.add_argument("--gif_prefix", type=str, default="sample")
    parser.add_argument("--gif_fps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    """Run conditional sampling and optional classification."""
    args = parse_args()
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
        shuffle=False,
    )

    batch = next(iter(loader))
    skeleton = batch["skeleton"].to(device)
    a_stream = batch["A"].to(device)
    omega_stream = batch["Omega"].to(device)

    if args.h_none:
        h_global = None
        sensor_tokens = None
    else:
        h_global, sensor_tokens = stage2.aligner(a_stream, omega_stream)

    shape = (skeleton.shape[0], skeleton.shape[1], skeleton.shape[2], args.latent_dim)
    z0_gen = stage3.diffusion.p_sample_loop(
        stage3.denoiser,
        shape=torch.Size(shape),
        device=device,
        sensor_tokens=sensor_tokens,
        h_global=h_global,
    )
    x_hat = stage3.decoder(z0_gen)

    print(f"h_global shape: {None if h_global is None else tuple(h_global.shape)}")
    print(f"sensor_tokens shape: {None if sensor_tokens is None else tuple(sensor_tokens.shape)}")
    print(f"z0_gen shape: {tuple(z0_gen.shape)}")
    print(f"x_hat shape: {tuple(x_hat.shape)}")

    if args.save_gif:
        for i in range(x_hat.shape[0]):
            gif_path = os.path.join(args.gif_dir, f"{args.gif_prefix}_{i:03d}.gif")
            save_skeleton_gif(x_hat[i], gif_path, fps=args.gif_fps)
            print(f"saved gif: {gif_path}")

    if args.classify:
        logits = stage3.classifier(x_hat)
        print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
