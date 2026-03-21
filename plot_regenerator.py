#!/usr/bin/env python3
"""Portable plot regeneration entrypoint for the LDM gait project.

This script regenerates plot families used in the presentation from:
- training history CSVs written by the codebase
- final checkpoints plus dataset folders

It is designed for handoff to another machine as long as the repo contains:
- `diffusion_model/`
- `train.py`
- checkpoints and dataset paths supplied through CLI args
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

np = None
torch = None
plt = None

create_dataloader = None
create_dataset = None
DEFAULT_GAIT_METRICS_DIM = None
GAIT_METRIC_NAMES = None
compute_gait_metrics_torch = None
Stage1Model = None
Stage2Model = None
Stage3Model = None
load_checkpoint = None
evaluate_stage1 = None
evaluate_stage2 = None
evaluate_stage3 = None
sample_stage3_latents = None
write_curve_plot = None
write_embedding_projection_plot = None
write_hist_grid = None
write_scatter = None
write_stage3_metric_trend_plot = None
DEFAULT_FPS = None
DEFAULT_JOINTS = None
DEFAULT_LATENT_DIM = None
DEFAULT_NUM_CLASSES = None
DEFAULT_TIMESTEPS = None
DEFAULT_WINDOW = None


def import_runtime_dependencies() -> None:
    global np, torch, plt
    global create_dataloader, create_dataset
    global DEFAULT_GAIT_METRICS_DIM, GAIT_METRIC_NAMES, compute_gait_metrics_torch
    global Stage1Model, Stage2Model, Stage3Model
    global load_checkpoint
    global evaluate_stage1, evaluate_stage2, evaluate_stage3
    global sample_stage3_latents
    global write_curve_plot, write_embedding_projection_plot, write_hist_grid, write_scatter, write_stage3_metric_trend_plot
    global DEFAULT_FPS, DEFAULT_JOINTS, DEFAULT_LATENT_DIM, DEFAULT_NUM_CLASSES, DEFAULT_TIMESTEPS, DEFAULT_WINDOW

    import numpy as np_mod
    import torch as torch_mod

    try:
        from matplotlib import pyplot as plt_mod
    except Exception:  # pragma: no cover
        plt_mod = None

    from diffusion_model.dataset import create_dataloader as create_dataloader_mod, create_dataset as create_dataset_mod
    from diffusion_model.gait_metrics import (
        DEFAULT_GAIT_METRICS_DIM as DEFAULT_GAIT_METRICS_DIM_mod,
        GAIT_METRIC_NAMES as GAIT_METRIC_NAMES_mod,
        compute_gait_metrics_torch as compute_gait_metrics_torch_mod,
    )
    from diffusion_model.model import Stage1Model as Stage1Model_mod, Stage2Model as Stage2Model_mod, Stage3Model as Stage3Model_mod
    from diffusion_model.model_loader import load_checkpoint as load_checkpoint_mod
    from diffusion_model.training_eval import (
        evaluate_stage1 as evaluate_stage1_mod,
        evaluate_stage2 as evaluate_stage2_mod,
        evaluate_stage3 as evaluate_stage3_mod,
        sample_stage3_latents as sample_stage3_latents_mod,
        write_curve_plot as write_curve_plot_mod,
        write_embedding_projection_plot as write_embedding_projection_plot_mod,
        write_hist_grid as write_hist_grid_mod,
        write_scatter as write_scatter_mod,
        write_stage3_metric_trend_plot as write_stage3_metric_trend_plot_mod,
    )
    from diffusion_model.util import (
        DEFAULT_FPS as DEFAULT_FPS_mod,
        DEFAULT_JOINTS as DEFAULT_JOINTS_mod,
        DEFAULT_LATENT_DIM as DEFAULT_LATENT_DIM_mod,
        DEFAULT_NUM_CLASSES as DEFAULT_NUM_CLASSES_mod,
        DEFAULT_TIMESTEPS as DEFAULT_TIMESTEPS_mod,
        DEFAULT_WINDOW as DEFAULT_WINDOW_mod,
    )

    np = np_mod
    torch = torch_mod
    plt = plt_mod
    create_dataloader = create_dataloader_mod
    create_dataset = create_dataset_mod
    DEFAULT_GAIT_METRICS_DIM = DEFAULT_GAIT_METRICS_DIM_mod
    GAIT_METRIC_NAMES = GAIT_METRIC_NAMES_mod
    compute_gait_metrics_torch = compute_gait_metrics_torch_mod
    Stage1Model = Stage1Model_mod
    Stage2Model = Stage2Model_mod
    Stage3Model = Stage3Model_mod
    load_checkpoint = load_checkpoint_mod
    evaluate_stage1 = evaluate_stage1_mod
    evaluate_stage2 = evaluate_stage2_mod
    evaluate_stage3 = evaluate_stage3_mod
    sample_stage3_latents = sample_stage3_latents_mod
    write_curve_plot = write_curve_plot_mod
    write_embedding_projection_plot = write_embedding_projection_plot_mod
    write_hist_grid = write_hist_grid_mod
    write_scatter = write_scatter_mod
    write_stage3_metric_trend_plot = write_stage3_metric_trend_plot_mod
    DEFAULT_FPS = DEFAULT_FPS_mod
    DEFAULT_JOINTS = DEFAULT_JOINTS_mod
    DEFAULT_LATENT_DIM = DEFAULT_LATENT_DIM_mod
    DEFAULT_NUM_CLASSES = DEFAULT_NUM_CLASSES_mod
    DEFAULT_TIMESTEPS = DEFAULT_TIMESTEPS_mod
    DEFAULT_WINDOW = DEFAULT_WINDOW_mod


def parse_args() -> argparse.Namespace:
    default_window = 90 if DEFAULT_WINDOW is None else DEFAULT_WINDOW
    default_joints = 32 if DEFAULT_JOINTS is None else DEFAULT_JOINTS
    default_latent_dim = 256 if DEFAULT_LATENT_DIM is None else DEFAULT_LATENT_DIM
    default_timesteps = 500 if DEFAULT_TIMESTEPS is None else DEFAULT_TIMESTEPS
    default_gait_metrics_dim = 9 if DEFAULT_GAIT_METRICS_DIM is None else DEFAULT_GAIT_METRICS_DIM
    default_num_classes = 14 if DEFAULT_NUM_CLASSES is None else DEFAULT_NUM_CLASSES
    default_fps = 30.0 if DEFAULT_FPS is None else DEFAULT_FPS
    default_device = "cpu"
    if torch is not None and torch.cuda.is_available():
        default_device = "cuda"

    parser = argparse.ArgumentParser(description="Regenerate training/evaluation/inference plots from dataset and model paths.")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="")
    parser.add_argument("--hip_folder", type=str, default="")
    parser.add_argument("--wrist_folder", type=str, default="")
    parser.add_argument("--gait_cache_dir", type=str, default="")
    parser.add_argument("--disable_gait_cache", action="store_true")
    parser.add_argument("--disable_sensor_norm", action="store_true")

    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--stage3_ckpt", type=str, default="")

    parser.add_argument("--stage1_history", type=str, default="")
    parser.add_argument("--stage2_history", type=str, default="")
    parser.add_argument("--stage3_history", type=str, default="")
    parser.add_argument("--stage3_metric_history", type=str, default="")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--window", type=int, default=default_window)
    parser.add_argument("--stride", type=int, default=45)
    parser.add_argument("--joints", type=int, default=default_joints)
    parser.add_argument("--latent_dim", type=int, default=default_latent_dim)
    parser.add_argument("--timesteps", type=int, default=default_timesteps)
    parser.add_argument("--gait_metrics_dim", type=int, default=default_gait_metrics_dim)
    parser.add_argument("--num_classes", type=int, default=default_num_classes)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fps", type=float, default=default_fps)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--max_feature_batches", type=int, default=8)
    parser.add_argument("--max_generation_batches", type=int, default=4)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_history_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_data_mode(args: argparse.Namespace) -> None:
    if args.dataset_path:
        return
    if args.skeleton_folder and args.hip_folder and args.wrist_folder:
        return
    raise ValueError(
        "Provide either --dataset_path or all of --skeleton_folder --hip_folder --wrist_folder."
    )


def build_dataset_and_loader(args: argparse.Namespace):
    _resolve_data_mode(args)
    dataset = create_dataset(
        dataset_path=args.dataset_path or None,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
    )
    loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        shuffle=False,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return dataset, loader


def load_stage1(args: argparse.Namespace, device: torch.device) -> Stage1Model:
    if not args.stage1_ckpt:
        raise ValueError("--stage1_ckpt is required for Stage 1/2/3 checkpoint-based plots.")
    model = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, model, strict=True)
    model.eval()
    return model


def load_stage2(args: argparse.Namespace, stage1: Stage1Model, device: torch.device) -> Stage2Model:
    if not args.stage2_ckpt:
        raise ValueError("--stage2_ckpt is required for Stage 2/3 checkpoint-based plots.")
    model = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage2_ckpt, model, strict=True)
    model.eval()
    return model


def load_stage3(args: argparse.Namespace, stage1: Stage1Model, device: torch.device) -> Stage3Model:
    if not args.stage3_ckpt:
        raise ValueError("--stage3_ckpt is required for Stage 3 checkpoint-based plots.")
    model = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage3_ckpt, model, strict=True)
    model.eval()
    return model


def write_history_plot(history_path: Path, out_path: Path, title: str, keys: list[tuple[str, str, str]]) -> None:
    rows = read_history_csv(history_path)
    epochs = [float(row["epoch"]) for row in rows]
    series = [(label, [float(row[key]) for row in rows], color) for key, label, color in keys if key in rows[0]]
    if not series:
        return
    write_curve_plot(out_path, title, epochs, series, "Epoch", "Loss")


def write_label_distribution(out_path: Path, counts: Counter[int], num_classes: int) -> None:
    if plt is None:
        return
    labels = [f"A{i + 1:02d}" for i in range(num_classes)]
    values = [counts.get(i, 0) for i in range(num_classes)]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="#2563eb")
    plt.title("Real Dataset Window Distribution by Activity")
    plt.xlabel("Activity label")
    plt.ylabel("Window count")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_metric_relationship_grid(
    out_path: Path,
    real_data: np.ndarray,
    generated_data: np.ndarray,
    metric_names: list[str],
    pair_indices: list[tuple[int, int]],
) -> None:
    if plt is None or not pair_indices:
        return
    cols = 2
    rows = int(np.ceil(len(pair_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5.5 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (i, j) in zip(axes, pair_indices):
        ax.scatter(real_data[:, i], real_data[:, j], s=16, alpha=0.4, color="#2563eb", label="Real")
        ax.scatter(generated_data[:, i], generated_data[:, j], s=16, alpha=0.4, color="#dc2626", label="Generated")
        ax.set_xlabel(metric_names[i])
        ax.set_ylabel(metric_names[j])
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{metric_names[j]} vs {metric_names[i]}")
    for idx in range(len(pair_indices), len(axes)):
        axes[idx].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Real vs Generated Metric Relationships")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def collect_stage1_stage2_features(
    stage1: Stage1Model,
    stage2: Stage2Model,
    loader,
    device: torch.device,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latent_features: list[np.ndarray] = []
    sensor_features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = batch["skeleton"].to(device)
            a_hip = batch["A_hip"].to(device)
            a_wrist = batch["A_wrist"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            y = batch["label"].cpu().numpy()
            z0 = stage1.encoder(x, gait_metrics=gait_metrics if getattr(stage1, "use_gait_conditioning", True) else None)
            _, h_global = stage2.aligner(a_hip, a_wrist)
            latent_features.append(z0.mean(dim=(1, 2)).cpu().numpy())
            sensor_features.append(h_global.cpu().numpy())
            labels.append(y)
    return (
        np.concatenate(latent_features, axis=0),
        np.concatenate(sensor_features, axis=0),
        np.concatenate(labels, axis=0),
    )


def collect_real_and_generated_gait(
    stage2: Stage2Model,
    stage3: Stage3Model,
    loader,
    device: torch.device,
    sample_steps: int,
    sampler: str,
    fps: float,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    real_gait_list: list[np.ndarray] = []
    gen_gait_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = batch["skeleton"].to(device)
            a_hip = batch["A_hip"].to(device)
            a_wrist = batch["A_wrist"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            cond_tokens, cond_global = stage2.aligner(a_hip, a_wrist)
            shape = (x.shape[0], x.shape[1], x.shape[2], stage3.latent_dim)
            z0_gen = sample_stage3_latents(
                stage3=stage3,
                shape=torch.Size(shape),
                device=device,
                h_tokens=cond_tokens,
                h_global=cond_global,
                gait_metrics=None,
                sample_steps=sample_steps,
                sampler=sampler,
            )
            x_hat = stage3.decoder(z0_gen)
            gen_gait = compute_gait_metrics_torch(x_hat, fps=fps).cpu().numpy()
            real_gait_list.append(gait_metrics.cpu().numpy())
            gen_gait_list.append(gen_gait)
    return np.concatenate(real_gait_list, axis=0), np.concatenate(gen_gait_list, axis=0)


def main() -> None:
    args = parse_args()
    import_runtime_dependencies()
    out_root = ensure_dir(Path(args.output_dir))
    device = torch.device(args.device)

    print(f"[plot_regenerator] output_dir={out_root}")
    print(f"[plot_regenerator] device={device}")

    dataset = None
    loader = None
    stage1 = None
    stage2 = None
    stage3 = None

    need_dataset = any(
        [
            args.stage1_ckpt,
            args.stage2_ckpt,
            args.stage3_ckpt,
        ]
    )
    if need_dataset:
        dataset, loader = build_dataset_and_loader(args)
        print(f"[plot_regenerator] dataset_windows={len(dataset)} batches={len(loader)}")

    training_dir = ensure_dir(out_root / "training")
    evaluation_dir = ensure_dir(out_root / "evaluation")
    reporting_dir = ensure_dir(out_root / "reporting")
    generation_dir = ensure_dir(out_root / "generation")

    if args.stage1_history:
        write_history_plot(
            Path(args.stage1_history),
            training_dir / "stage1_loss_curves.png",
            "Stage1 Loss Curves",
            [
                ("train_loss_diff", "train_loss_diff", "#2563eb"),
                ("val_loss_diff", "val_loss_diff", "#dc2626"),
            ],
        )
    if args.stage2_history:
        write_history_plot(
            Path(args.stage2_history),
            training_dir / "stage2_loss_curves.png",
            "Stage2 Loss Curves",
            [
                ("train_loss_align", "train_loss_align", "#2563eb"),
                ("val_loss_align", "val_loss_align", "#dc2626"),
            ],
        )
    if args.stage3_history:
        write_history_plot(
            Path(args.stage3_history),
            training_dir / "stage3_total_loss_curves.png",
            "Stage3 Total Loss Curves",
            [
                ("train_loss_total", "train_loss_total", "#111827"),
                ("val_loss_total", "val_loss_total", "#7c3aed"),
            ],
        )
        write_history_plot(
            Path(args.stage3_history),
            training_dir / "stage3_component_loss_curves.png",
            "Stage3 Component Losses",
            [
                ("train_loss_diff", "train_loss_diff", "#2563eb"),
                ("train_loss_pose", "train_loss_pose", "#dc2626"),
                ("train_loss_latent", "train_loss_latent", "#7c3aed"),
                ("train_loss_vel", "train_loss_vel", "#ea580c"),
                ("train_loss_gait", "train_loss_gait", "#059669"),
                ("train_loss_motion", "train_loss_motion", "#0f766e"),
            ],
        )
    if args.stage3_metric_history:
        rows = read_history_csv(Path(args.stage3_metric_history))
        write_stage3_metric_trend_plot(training_dir / "generated_gait_metric_trends.png", rows)

    if args.stage1_ckpt:
        stage1 = load_stage1(args, device)
        evaluate_stage1(
            stage1,
            loader,
            device,
            evaluation_dir / "stage1_eval",
            timestep_values=[0, 50, 100, 200, 300, 400, args.timesteps - 1],
        )

    if args.stage1_ckpt and args.stage2_ckpt:
        stage2 = load_stage2(args, stage1, device)
        evaluate_stage2(
            stage1,
            stage2,
            loader,
            device,
            evaluation_dir / "stage2_eval",
            epoch=1,
        )
        latent_arr, sensor_arr, label_arr = collect_stage1_stage2_features(
            stage1, stage2, loader, device, max_batches=args.max_feature_batches
        )
        ensure_dir(reporting_dir / "embeddings")
        for method in ["pca", "tsne", "umap"]:
            write_embedding_projection_plot(
                reporting_dir / "embeddings" / f"latent_{method}.png",
                f"Skeleton Latent Space {method.upper()}",
                latent_arr,
                label_arr,
                method=method,
                num_classes=args.num_classes,
            )
            write_embedding_projection_plot(
                reporting_dir / "embeddings" / f"sensor_embedding_{method}.png",
                f"Sensor Embedding {method.upper()}",
                sensor_arr,
                label_arr,
                method=method,
                num_classes=args.num_classes,
            )

    if args.stage1_ckpt and args.stage2_ckpt and args.stage3_ckpt:
        stage3 = load_stage3(args, stage1, device)
        evaluate_stage3(
            stage2,
            stage3,
            loader,
            device,
            evaluation_dir / "stage3_eval",
            sample_steps=args.sample_steps,
            fps=args.fps,
            epoch=1,
            sampler=args.sampler,
        )
        real_gait, gen_gait = collect_real_and_generated_gait(
            stage2,
            stage3,
            loader,
            device,
            sample_steps=args.sample_steps,
            sampler=args.sampler,
            fps=args.fps,
            max_batches=args.max_generation_batches,
        )
        write_hist_grid(
            reporting_dir / "real_vs_generated_gait_distributions.png",
            "Real vs Generated Gait Metric Distributions",
            real_gait,
            gen_gait,
            GAIT_METRIC_NAMES,
        )
        write_scatter(
            reporting_dir / "real_vs_generated_speed_vs_com_fore_aft.png",
            "Real vs Generated: Walking Speed vs Mean CoM Fore-Aft",
            real_gait[:, 6],
            real_gait[:, 0],
            gen_gait[:, 6],
            gen_gait[:, 0],
            "Mean Walking Speed",
            "Mean CoM Fore-Aft",
        )
        write_metric_relationship_grid(
            reporting_dir / "real_vs_generated_metric_relationships.png",
            real_gait,
            gen_gait,
            list(GAIT_METRIC_NAMES),
            pair_indices=[(6, 0), (6, 4), (6, 7), (6, 8), (0, 4), (7, 8)],
        )

    if dataset is not None:
        label_counter = Counter(int(dataset.label[i].item()) for i in range(len(dataset.label)))
        write_label_distribution(reporting_dir / "dataset_label_distribution.png", label_counter, args.num_classes)

    print("[plot_regenerator] done")
    print("[plot_regenerator] training plots:", training_dir)
    print("[plot_regenerator] evaluation plots:", evaluation_dir)
    print("[plot_regenerator] reporting plots:", reporting_dir)
    print("[plot_regenerator] generation artifacts:", generation_dir)


if __name__ == "__main__":
    main()
