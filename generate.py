"""Deterministic IMU-to-skeleton generation for stage-3 latent diffusion."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusion_model.dataset import create_dataloader
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM, compute_gait_metrics_torch
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.training_eval import sample_stage3_latents
from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_LATENT_DIM, DEFAULT_TIMESTEPS, DEFAULT_WINDOW, get_skeleton_edges


def _project_pose_to_2d(points_xyz: np.ndarray, root_index: int = 0) -> np.ndarray:
    """Project [T, J, 3] skeletons to a stable 2D view using the two widest axes."""
    assert points_xyz.ndim == 3 and points_xyz.shape[-1] == 3, "points_xyz must be [T, J, 3]"
    assert 0 <= root_index < points_xyz.shape[1], "root_index out of range"
    centered = points_xyz - points_xyz[:, root_index : root_index + 1, :]
    flat = centered.reshape(-1, 3)
    finite = flat[np.isfinite(flat).all(axis=-1)]
    if finite.size == 0:
        raise ValueError("Sequence has no finite XYZ coordinates to render.")
    spreads = np.percentile(finite, 98.0, axis=0) - np.percentile(finite, 2.0, axis=0)
    axis_ids = np.argsort(spreads)[-2:]
    axis_ids.sort()
    return centered[..., axis_ids]


def _normalize_xy(points_xy: np.ndarray, canvas_size: int, root_index: int = 0) -> np.ndarray:
    """Normalize xy coordinates into drawable canvas coordinates."""
    assert points_xy.ndim == 3, "points_xy must be [T, J, 2]"
    assert 0 <= root_index < points_xy.shape[1], "root_index out of range"
    finite_mask = np.isfinite(points_xy).all(axis=-1)
    if not finite_mask.any():
        raise ValueError("Sequence has no finite XY coordinates to render.")
    valid_xy = points_xy[finite_mask]
    anchor = points_xy[:, root_index : root_index + 1, :]
    centered = points_xy - anchor
    flat_xy = centered[np.isfinite(centered).all(axis=-1)]
    if flat_xy.size == 0:
        flat_xy = valid_xy.reshape(-1, 2)
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
    points_xy = _normalize_xy(_project_pose_to_2d(points), canvas_size=canvas_size)
    edges = [(i, j) for i, j in get_skeleton_edges() if i < num_joints and j < num_joints]
    frames: List[Image.Image] = []
    line_width = max(3, canvas_size // 160)
    joint_radius = max(4, canvas_size // 96)

    for frame_xy in points_xy:
        img = Image.new("RGB", (canvas_size, canvas_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        for i, j in edges:
            xi, yi = float(frame_xy[i][0]), float(frame_xy[i][1])
            xj, yj = float(frame_xy[j][0]), float(frame_xy[j][1])
            draw.line((xi, yi, xj, yj), fill=(30, 30, 30), width=line_width)
        for joint_xy in frame_xy:
            x, y = float(joint_xy[0]), float(joint_xy[1])
            r = joint_radius
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(20, 60, 210))
        frames.append(img)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    duration_ms = int(1000 / max(fps, 1))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def save_skeleton_panel(sequence: torch.Tensor, out_path: str, canvas_size: int = 1024, num_frames: int = 5) -> None:
    """Save a large horizontal panel of sampled frames for visual inspection."""
    assert sequence.ndim == 3 and sequence.shape[-1] == 3, "sequence must have shape [T, J, 3]"
    points = sequence.detach().cpu().numpy().astype(np.float32)
    num_joints = points.shape[1]
    points_xy = _normalize_xy(_project_pose_to_2d(points), canvas_size=canvas_size)
    edges = [(i, j) for i, j in get_skeleton_edges() if i < num_joints and j < num_joints]
    num_frames = max(1, min(int(num_frames), points_xy.shape[0]))
    frame_ids = np.linspace(0, points_xy.shape[0] - 1, num=num_frames, dtype=int)
    margin = max(24, canvas_size // 20)
    panel_width = canvas_size * num_frames + margin * (num_frames + 1)
    panel_height = canvas_size + 2 * margin
    line_width = max(4, canvas_size // 140)
    joint_radius = max(5, canvas_size // 88)
    panel = Image.new("RGB", (panel_width, panel_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    for panel_idx, frame_id in enumerate(frame_ids.tolist()):
        x_offset = margin + panel_idx * (canvas_size + margin)
        y_offset = margin
        frame_xy = points_xy[frame_id]
        draw.rectangle((x_offset, y_offset, x_offset + canvas_size, y_offset + canvas_size), outline=(220, 220, 220), width=2)
        for i, j in edges:
            xi, yi = float(frame_xy[i][0]) + x_offset, float(frame_xy[i][1]) + y_offset
            xj, yj = float(frame_xy[j][0]) + x_offset, float(frame_xy[j][1]) + y_offset
            draw.line((xi, yi, xj, yj), fill=(30, 30, 30), width=line_width)
        for joint_xy in frame_xy:
            x, y = float(joint_xy[0]) + x_offset, float(joint_xy[1]) + y_offset
            r = joint_radius
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(20, 60, 210))
        draw.text((x_offset + 12, y_offset + 12), f"t={frame_id}", fill=(40, 40, 40))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    panel.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired-reconstruction skeletons from IMU with deterministic latent diffusion")
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
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--d_shared", type=int, default=64)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--disable_sensor_norm", action="store_true")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--save_panel", action="store_true", help="Save a large multi-frame PNG for inspection.")
    parser.add_argument("--gif_dir", type=str, default="outputs/results_new")
    parser.add_argument("--gif_prefix", type=str, default="sample")
    parser.add_argument("--gif_fps", type=int, default=12)
    parser.add_argument("--canvas_size", type=int, default=1024, help="Canvas size for saved visualizations.")
    parser.add_argument("--panel_frames", type=int, default=5, help="Number of frames to show in the saved PNG panel.")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim"])
    parser.add_argument("--sample_seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gait_metrics_dim != DEFAULT_GAIT_METRICS_DIM:
        raise ValueError(
            f"--gait_metrics_dim must equal the fixed auto-computed gait-summary size ({DEFAULT_GAIT_METRICS_DIM})."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_graph_op, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)

    stage1 = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        encoder_type=encoder_graph_op,
        skeleton_graph_op=skeleton_graph_op,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)

    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
        d_shared=args.d_shared,
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
        use_gait_conditioning=False,
        d_shared=args.d_shared,
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

    batch = next(iter(loader))
    skeleton = batch["skeleton"].to(device)
    a_hip_stream = batch["A_hip"].to(device)
    a_wrist_stream = batch["A_wrist"].to(device)
    gait_target = batch["gait_metrics"].to(device)

    h_tokens, h_global = stage2.aligner(a_hip_stream, a_wrist_stream)
    shape = torch.Size((skeleton.shape[0], skeleton.shape[1], skeleton.shape[2], args.latent_dim))
    z0_gen = sample_stage3_latents(
        stage3=stage3,
        shape=shape,
        device=device,
        h_tokens=h_tokens,
        h_global=h_global,
        a_hip_stream=a_hip_stream,
        a_wrist_stream=a_wrist_stream,
        gait_metrics=None,
        sample_steps=args.sample_steps,
        sampler=args.sampler,
        sample_seed=args.sample_seed,
    )
    x_hat = stage3.decoder(z0_gen)

    recon_error = torch.mean(torch.norm(x_hat - skeleton, dim=-1)).item()
    gait_pred = compute_gait_metrics_torch(x_hat, fps=30.0)
    gait_error = torch.mean((gait_pred - gait_target) ** 2).item()
    print(f"h_global shape: {tuple(h_global.shape)}")
    print(f"z0_gen shape: {tuple(z0_gen.shape)}")
    print(f"x_hat shape: {tuple(x_hat.shape)}")
    print(f"mean reconstruction joint error: {recon_error:.6f}")
    print(f"mean gait MSE: {gait_error:.6f}")

    if args.save_gif:
        run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        for sample_idx in range(x_hat.shape[0]):
            gif_path = os.path.join(args.gif_dir, f"{args.gif_prefix}_{run_tag}_idx{sample_idx:02d}.gif")
            save_skeleton_gif(x_hat[sample_idx], gif_path, fps=args.gif_fps, canvas_size=args.canvas_size)
            print(f"saved gif: {gif_path}")
            if args.save_panel:
                panel_path = os.path.join(args.gif_dir, f"{args.gif_prefix}_{run_tag}_idx{sample_idx:02d}.png")
                save_skeleton_panel(
                    x_hat[sample_idx],
                    panel_path,
                    canvas_size=args.canvas_size,
                    num_frames=args.panel_frames,
                )
                print(f"saved panel: {panel_path}")


if __name__ == "__main__":
    main()
