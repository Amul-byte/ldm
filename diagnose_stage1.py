"""Stage-1 diagnosis utility for reconstruction and latent-collapse checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusion_model.dataset import create_dataset, parse_subject_list, split_train_val_dataset
from diffusion_model.model import Stage1Model
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
    get_joint_labels,
    get_skeleton_edges,
    set_seed,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_TRAIN_SUBJECTS = [
    28, 29, 30, 31, 33, 35, 38, 39, 32, 36, 37, 43, 44, 45, 46, 49, 51, 56, 57, 58, 59, 61, 62
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Stage-1 reconstruction quality and latent collapse.")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="")
    parser.add_argument("--hip_folder", type=str, default="")
    parser.add_argument("--wrist_folder", type=str, default="")
    parser.add_argument("--gait_cache_dir", type=str, default="")
    parser.add_argument("--disable_gait_cache", action="store_true")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--gait_metrics_dim", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--train_subjects", type=str, default=",".join(str(item) for item in DEFAULT_TRAIN_SUBJECTS))
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--max_batches", type=int, default=4)
    parser.add_argument("--num_plot_samples", type=int, default=4)
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
            raise ValueError("Validation split is unavailable. Set --val_split > 0 or use --split all.")
        return val_dataset
    return dataset


def save_reconstruction_plot(sample_idx: int, x: torch.Tensor, x_hat: torch.Tensor, output_path: Path) -> None:
    if plt is None:
        return
    x_np = x.detach().cpu().reshape(x.shape[0], -1).numpy()
    x_hat_np = x_hat.detach().cpu().reshape(x_hat.shape[0], -1).numpy()
    err_np = abs(x_np - x_hat_np)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
    panels = [
        (x_np, f"Input sample {sample_idx}", "coolwarm"),
        (x_hat_np, f"Decoded sample {sample_idx}", "coolwarm"),
        (err_np, f"Absolute error {sample_idx}", "magma"),
    ]
    for ax, (arr, title, cmap) in zip(axes, panels):
        im = ax.imshow(arr.T, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Flattened joint-axis feature")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Stage-1 Reconstruction Diagnostic", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_joint_trajectory_plot(sample_idx: int, x: torch.Tensor, x_hat: torch.Tensor, output_path: Path) -> None:
    if plt is None:
        return
    joint_labels = list(get_joint_labels())
    tracked_joint_names = ["PELVIS", "HEAD", "WRIST_LEFT", "WRIST_RIGHT", "ANKLE_LEFT", "ANKLE_RIGHT"]
    tracked_joint_ids = [joint_labels.index(name) for name in tracked_joint_names]
    axis_labels = ["x", "y", "z"]
    colors = ["#2563eb", "#dc2626", "#059669"]

    x_np = x.detach().cpu().numpy()
    x_hat_np = x_hat.detach().cpu().numpy()

    fig, axes = plt.subplots(len(tracked_joint_ids), 1, figsize=(14, 2.4 * len(tracked_joint_ids)), sharex=True, constrained_layout=True)
    if len(tracked_joint_ids) == 1:
        axes = [axes]

    for ax, joint_idx, joint_name in zip(axes, tracked_joint_ids, tracked_joint_names):
        for axis_idx, (axis_name, color) in enumerate(zip(axis_labels, colors)):
            ax.plot(x_np[:, joint_idx, axis_idx], color=color, linestyle="-", linewidth=1.8, label=f"input {axis_name}")
            ax.plot(x_hat_np[:, joint_idx, axis_idx], color=color, linestyle="--", linewidth=1.4, label=f"decoded {axis_name}")
        ax.set_ylabel(joint_name)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"Stage-1 Joint Trajectories Sample {sample_idx}", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_skeleton_frame_plot(sample_idx: int, x: torch.Tensor, x_hat: torch.Tensor, output_path: Path) -> None:
    if plt is None:
        return
    edges = get_skeleton_edges()
    frame_ids = [0, x.shape[0] // 2, x.shape[0] - 1]
    panel_titles = ["Start", "Middle", "End"]
    x_np = x.detach().cpu().numpy()
    x_hat_np = x_hat.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row_idx, (source, row_name, color) in enumerate(((x_np, "Input", "#2563eb"), (x_hat_np, "Decoded", "#dc2626"))):
        for col_idx, (frame_idx, panel_title) in enumerate(zip(frame_ids, panel_titles)):
            ax = axes[row_idx][col_idx]
            pose = source[frame_idx]
            for start, end in edges:
                ax.plot(
                    [pose[start, 0], pose[end, 0]],
                    [pose[start, 2], pose[end, 2]],
                    color=color,
                    linewidth=1.2,
                    alpha=0.9,
                )
            ax.scatter(pose[:, 0], pose[:, 2], color=color, s=10)
            ax.set_title(f"{row_name} {panel_title} f={frame_idx}")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)

    fig.suptitle(f"Stage-1 Skeleton Frames Sample {sample_idx} (x-z projection)", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = select_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, model, strict=True)
    model.eval()

    sum_recon_mse = 0.0
    sum_recon_mae = 0.0
    sum_baseline_joint_mse = 0.0
    sum_baseline_time_mse = 0.0
    sum_baseline_all_mse = 0.0
    sum_latent_mean_abs = 0.0
    sum_latent_std = 0.0
    sum_latent_joint_std = 0.0
    sum_latent_time_std = 0.0
    n_batches = 0
    plotted = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            x = batch["skeleton"].to(device)
            gait_metrics: Optional[torch.Tensor] = None
            if args.gait_metrics_dim > 0:
                gait_metrics = batch["gait_metrics"].to(device)

            z0 = model.encoder(x, gait_metrics=gait_metrics)
            x_hat = model.decoder(z0)

            recon_mse = F.mse_loss(x_hat, x)
            recon_mae = F.l1_loss(x_hat, x)
            mean_joint = z0.mean(dim=2, keepdim=True).expand_as(z0)
            mean_time = z0.mean(dim=1, keepdim=True).expand_as(z0)
            mean_all = z0.mean(dim=(1, 2), keepdim=True).expand_as(z0)

            sum_recon_mse += float(recon_mse.item())
            sum_recon_mae += float(recon_mae.item())
            sum_baseline_joint_mse += float(F.mse_loss(mean_joint, z0).item())
            sum_baseline_time_mse += float(F.mse_loss(mean_time, z0).item())
            sum_baseline_all_mse += float(F.mse_loss(mean_all, z0).item())
            sum_latent_mean_abs += float(z0.abs().mean().item())
            sum_latent_std += float(z0.std().item())
            sum_latent_joint_std += float(z0.std(dim=2).mean().item())
            sum_latent_time_std += float(z0.std(dim=1).mean().item())
            n_batches += 1

            for sample_offset in range(x.shape[0]):
                if plotted >= args.num_plot_samples:
                    break
                save_reconstruction_plot(
                    sample_idx=plotted,
                    x=x[sample_offset],
                    x_hat=x_hat[sample_offset],
                    output_path=plots_dir / f"reconstruction_{plotted:02d}.png",
                )
                save_joint_trajectory_plot(
                    sample_idx=plotted,
                    x=x[sample_offset],
                    x_hat=x_hat[sample_offset],
                    output_path=plots_dir / f"joint_trajectories_{plotted:02d}.png",
                )
                save_skeleton_frame_plot(
                    sample_idx=plotted,
                    x=x[sample_offset],
                    x_hat=x_hat[sample_offset],
                    output_path=plots_dir / f"skeleton_frames_{plotted:02d}.png",
                )
                plotted += 1
            if plotted >= args.num_plot_samples:
                continue

    if n_batches == 0:
        raise ValueError("No batches were processed. Increase --max_batches or check dataset availability.")

    metrics = {
        "dataset_split": args.split,
        "num_batches_evaluated": n_batches,
        "num_plot_samples": plotted,
        "avg_reconstruction_mse": sum_recon_mse / n_batches,
        "avg_reconstruction_mae": sum_recon_mae / n_batches,
        "avg_latent_abs_mean": sum_latent_mean_abs / n_batches,
        "avg_latent_std_all": sum_latent_std / n_batches,
        "avg_latent_std_across_joints": sum_latent_joint_std / n_batches,
        "avg_latent_std_across_time": sum_latent_time_std / n_batches,
        "avg_latent_mse_to_joint_mean": sum_baseline_joint_mse / n_batches,
        "avg_latent_mse_to_time_mean": sum_baseline_time_mse / n_batches,
        "avg_latent_mse_to_global_mean": sum_baseline_all_mse / n_batches,
    }

    interpretation = {
        "reconstruction": (
            "Lower reconstruction MSE/MAE means the encoder-decoder pair preserves more skeleton information."
        ),
        "latent_collapse": (
            "If latent std across joints/time is very small and the mean-baseline MSE values are near zero, "
            "the latent is close to collapsed across those axes."
        ),
        "plotting": (
            "Reconstruction plots are only written when matplotlib is available in the active Python environment."
        ),
    }

    payload = {
        "checkpoint": args.stage1_ckpt,
        "output_dir": str(out_dir),
        "metrics": metrics,
        "interpretation": interpretation,
    }
    (out_dir / "diagnosis_summary.json").write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
