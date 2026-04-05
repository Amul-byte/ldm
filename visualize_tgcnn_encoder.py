#!/usr/bin/env python3
"""Visualize one sample before and after the Stage-2 TGNN encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from matplotlib import pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:  # pragma: no cover
    _MATPLOTLIB_IMPORT_ERROR = None

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

from diffusion_model.dataset import create_dataset
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.sensor_model import IMU_FEATURE_NAMES, build_imu_features
from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_LATENT_DIM, DEFAULT_TIMESTEPS, DEFAULT_WINDOW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot before/after visuals for the Stage-2 TGNN encoder.")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="")
    parser.add_argument("--hip_folder", type=str, default="")
    parser.add_argument("--wrist_folder", type=str, default="")
    parser.add_argument("--gait_cache_dir", type=str, default="")
    parser.add_argument("--disable_gait_cache", action="store_true")
    parser.add_argument("--disable_sensor_norm", action="store_true")
    parser.add_argument("--stage1_ckpt", type=str, default="checkpoints/stage1_best.pt")
    parser.add_argument("--stage2_ckpt", type=str, default="checkpoints/stage2_best.pt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=45)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--encoder_type", type=str, default="gat", choices=["gat", "gcn"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_plotting() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to save plots for TGNN visualization "
            f"(import failed: {_MATPLOTLIB_IMPORT_ERROR})"
        )


def validate_data_mode(args: argparse.Namespace) -> None:
    if args.dataset_path:
        return
    if args.skeleton_folder and args.hip_folder and args.wrist_folder:
        return
    raise ValueError("Provide either --dataset_path or all of --skeleton_folder --hip_folder --wrist_folder.")


def load_models(args: argparse.Namespace, device: torch.device) -> tuple[Stage1Model, Stage2Model]:
    encoder_graph_op, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)
    stage1 = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        encoder_type=encoder_graph_op,
        skeleton_graph_op=skeleton_graph_op,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    stage1.eval()

    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage2_ckpt, stage2, strict=True)
    stage2.eval()
    return stage1, stage2


def create_sample(args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], int]:
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
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    sample_idx = max(0, min(int(args.sample_idx), len(dataset) - 1))
    return dataset[sample_idx], sample_idx


def save_heatmap(
    out_path: Path,
    values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    y_ticks: list[str] | None = None,
    cmap: str = "viridis",
) -> None:
    require_plotting()
    ensure_dir(out_path.parent)
    arr = np.asarray(values, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    image = ax.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_ticks is not None:
        ax.set_yticks(np.arange(len(y_ticks)))
        ax.set_yticklabels(y_ticks)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_accel_lines(out_path: Path, accel: np.ndarray, title: str) -> None:
    require_plotting()
    ensure_dir(out_path.parent)
    frames = np.arange(accel.shape[0], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for idx, axis_name in enumerate(("ax", "ay", "az")):
        ax.plot(frames, accel[:, idx], linewidth=2.0, label=axis_name)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Acceleration")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_branch_overview(
    out_path: Path,
    raw_accel: np.ndarray,
    engineered: np.ndarray,
    tokens: np.ndarray,
    branch_name: str,
) -> None:
    require_plotting()
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)

    frames = np.arange(raw_accel.shape[0], dtype=np.int32)
    for idx, axis_name in enumerate(("ax", "ay", "az")):
        axes[0].plot(frames, raw_accel[:, idx], linewidth=1.8, label=axis_name)
    axes[0].set_title(f"{branch_name} raw IMU before TGNN")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Acceleration")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    im1 = axes[1].imshow(engineered.T, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[1].set_title(f"{branch_name} engineered IMU features fed to TGNN")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Feature channel")
    axes[1].set_yticks(np.arange(len(IMU_FEATURE_NAMES)))
    axes[1].set_yticklabels(list(IMU_FEATURE_NAMES))
    fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.02)

    im2 = axes[2].imshow(tokens.T, aspect="auto", cmap="magma", interpolation="nearest")
    axes[2].set_title(f"{branch_name} TGNN output tokens after encoder")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Latent dimension")
    fig.colorbar(im2, ax=axes[2], fraction=0.025, pad=0.02)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_token_norm_lines(out_path: Path, tokens: np.ndarray, title: str) -> None:
    require_plotting()
    ensure_dir(out_path.parent)
    token_norm = np.linalg.norm(tokens, axis=1)
    frames = np.arange(tokens.shape[0], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(frames, token_norm, linewidth=2.2, color="#b91c1c")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Token L2 norm")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_time_similarity_heatmap(out_path: Path, tokens: np.ndarray, title: str) -> None:
    require_plotting()
    ensure_dir(out_path.parent)
    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    unit = tokens / norms
    similarity = unit @ unit.T
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _project_tokens(tokens: np.ndarray, method: str) -> np.ndarray | None:
    if tokens.shape[0] < 2:
        return None
    method_name = method.lower()
    if method_name == "pca":
        if PCA is None:
            return None
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(tokens)
    if method_name == "umap":
        if umap is None or tokens.shape[0] < 3:
            return None
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(10, tokens.shape[0] - 1),
            min_dist=0.15,
            random_state=42,
        )
        return reducer.fit_transform(tokens)
    raise ValueError(f"Unsupported projection method: {method}")


def save_token_projection(out_path: Path, tokens: np.ndarray, title: str, method: str) -> None:
    require_plotting()
    reduced = _project_tokens(tokens, method=method)
    if reduced is None:
        return
    ensure_dir(out_path.parent)
    frames = np.arange(tokens.shape[0], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=frames, cmap="viridis", s=36, alpha=0.9)
    ax.plot(reduced[:, 0], reduced[:, 1], color="#64748b", alpha=0.5, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Frame", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_before_after_projection(
    out_path: Path,
    before_features: np.ndarray,
    after_features: np.ndarray,
    before_title: str,
    after_title: str,
    title: str,
    method: str = "pca",
) -> None:
    require_plotting()
    before_proj = _project_tokens(before_features, method=method)
    after_proj = _project_tokens(after_features, method=method)
    if before_proj is None or after_proj is None:
        return

    ensure_dir(out_path.parent)
    frames = np.arange(before_features.shape[0], dtype=np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    scatter_before = axes[0].scatter(before_proj[:, 0], before_proj[:, 1], c=frames, cmap="viridis", s=34, alpha=0.9)
    axes[0].plot(before_proj[:, 0], before_proj[:, 1], color="#64748b", alpha=0.5, linewidth=1.1)
    axes[0].set_title(before_title)
    axes[0].set_xlabel(f"{method.upper()} 1")
    axes[0].set_ylabel(f"{method.upper()} 2")
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(after_proj[:, 0], after_proj[:, 1], c=frames, cmap="viridis", s=34, alpha=0.9)
    axes[1].plot(after_proj[:, 0], after_proj[:, 1], color="#64748b", alpha=0.5, linewidth=1.1)
    axes[1].set_title(after_title)
    axes[1].set_xlabel(f"{method.upper()} 1")
    axes[1].set_ylabel(f"{method.upper()} 2")
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(title)
    fig.colorbar(scatter_before, ax=axes, label="Frame", fraction=0.03, pad=0.03)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    unit = features / norms
    return unit @ unit.T


def save_before_after_similarity(
    out_path: Path,
    before_features: np.ndarray,
    after_features: np.ndarray,
    before_title: str,
    after_title: str,
    title: str,
) -> None:
    require_plotting()
    ensure_dir(out_path.parent)
    before_sim = _cosine_similarity_matrix(before_features)
    after_sim = _cosine_similarity_matrix(after_features)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    im0 = axes[0].imshow(before_sim, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    axes[0].set_title(before_title)
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Frame")

    axes[1].imshow(after_sim, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    axes[1].set_title(after_title)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Frame")

    fig.suptitle(title)
    fig.colorbar(im0, ax=axes, label="Cosine similarity", fraction=0.03, pad=0.03)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_before_after_norm_line(
    out_path: Path,
    before_features: np.ndarray,
    after_features: np.ndarray,
    before_label: str,
    after_label: str,
    title: str,
) -> None:
    require_plotting()
    ensure_dir(out_path.parent)
    frames = np.arange(before_features.shape[0], dtype=np.int32)
    before_norm = np.linalg.norm(before_features, axis=1)
    after_norm = np.linalg.norm(after_features, axis=1)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(frames, before_norm, linewidth=2.0, color="#2563eb", label=before_label)
    ax.plot(frames, after_norm, linewidth=2.0, color="#dc2626", label=after_label)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("L2 norm")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    validate_data_mode(args)
    require_plotting()

    device = torch.device(args.device)
    out_dir = ensure_dir(Path(args.output_dir).resolve())

    stage1, stage2 = load_models(args, device)
    sample, sample_idx = create_sample(args)

    skeleton = sample["skeleton"].unsqueeze(0).to(device)
    a_hip = sample["A_hip"].unsqueeze(0).to(device)
    a_wrist = sample["A_wrist"].unsqueeze(0).to(device)
    gait_metrics = sample["gait_metrics"].unsqueeze(0).to(device)
    label = int(sample["label"].item())

    hip_features = build_imu_features(a_hip)
    wrist_features = build_imu_features(a_wrist)
    hip_tokens = stage2.aligner.hip_encoder(hip_features)
    wrist_tokens = stage2.aligner.wrist_encoder(wrist_features)
    sensor_tokens, h_global = stage2.aligner(a_hip, a_wrist, gait_metrics=gait_metrics)
    z0 = stage1.encoder(skeleton, gait_metrics=gait_metrics)

    hip_accel_np = a_hip.squeeze(0).cpu().numpy()
    wrist_accel_np = a_wrist.squeeze(0).cpu().numpy()
    hip_features_np = hip_features.squeeze(0).cpu().numpy()
    wrist_features_np = wrist_features.squeeze(0).cpu().numpy()
    hip_tokens_np = hip_tokens.squeeze(0).cpu().numpy()
    wrist_tokens_np = wrist_tokens.squeeze(0).cpu().numpy()
    sensor_tokens_np = sensor_tokens.squeeze(0).cpu().numpy()
    h_global_np = h_global.squeeze(0).cpu().numpy()
    z0_norm_np = torch.linalg.norm(z0.squeeze(0), dim=-1).cpu().numpy().T
    sensor_token_norm_np = torch.linalg.norm(sensor_tokens.squeeze(0), dim=-1).cpu().numpy()[None, :]

    prefix = f"sample_{sample_idx:04d}_A{label + 1:02d}"

    save_accel_lines(out_dir / f"{prefix}_hip_raw_accel.png", hip_accel_np, "Hip IMU before TGNN")
    save_accel_lines(out_dir / f"{prefix}_wrist_raw_accel.png", wrist_accel_np, "Wrist IMU before TGNN")

    save_heatmap(
        out_dir / f"{prefix}_hip_input_features.png",
        hip_features_np.T,
        "Hip engineered features before TGNN encoder",
        x_label="Frame",
        y_label="Feature channel",
        y_ticks=list(IMU_FEATURE_NAMES),
        cmap="viridis",
    )
    save_heatmap(
        out_dir / f"{prefix}_wrist_input_features.png",
        wrist_features_np.T,
        "Wrist engineered features before TGNN encoder",
        x_label="Frame",
        y_label="Feature channel",
        y_ticks=list(IMU_FEATURE_NAMES),
        cmap="viridis",
    )
    save_heatmap(
        out_dir / f"{prefix}_hip_tgcnn_output.png",
        hip_tokens_np.T,
        "Hip branch after TGNN encoder",
        x_label="Frame",
        y_label="Latent dimension",
        cmap="magma",
    )
    save_heatmap(
        out_dir / f"{prefix}_wrist_tgcnn_output.png",
        wrist_tokens_np.T,
        "Wrist branch after TGNN encoder",
        x_label="Frame",
        y_label="Latent dimension",
        cmap="magma",
    )
    save_heatmap(
        out_dir / f"{prefix}_fused_sensor_tokens.png",
        sensor_tokens_np.T,
        "Fused sensor tokens after Stage-2 aligner",
        x_label="Frame",
        y_label="Latent dimension",
        cmap="magma",
    )
    save_heatmap(
        out_dir / f"{prefix}_fused_sensor_token_norm.png",
        sensor_token_norm_np,
        "Fused sensor token norm across time",
        x_label="Frame",
        y_label="Norm",
        cmap="inferno",
    )
    save_heatmap(
        out_dir / f"{prefix}_stage1_target_latent_norm.png",
        z0_norm_np,
        "Stage-1 skeleton target latent norm for the same sample",
        x_label="Frame",
        y_label="Joint",
        cmap="magma",
    )
    save_token_norm_lines(
        out_dir / f"{prefix}_hip_token_norm_line.png",
        hip_tokens_np,
        "Hip TGNN token strength across time",
    )
    save_token_norm_lines(
        out_dir / f"{prefix}_wrist_token_norm_line.png",
        wrist_tokens_np,
        "Wrist TGNN token strength across time",
    )
    save_token_norm_lines(
        out_dir / f"{prefix}_fused_token_norm_line.png",
        sensor_tokens_np,
        "Fused Stage-2 token strength across time",
    )
    save_time_similarity_heatmap(
        out_dir / f"{prefix}_hip_time_similarity.png",
        hip_tokens_np,
        "Hip TGNN token cosine similarity across time",
    )
    save_time_similarity_heatmap(
        out_dir / f"{prefix}_wrist_time_similarity.png",
        wrist_tokens_np,
        "Wrist TGNN token cosine similarity across time",
    )
    save_time_similarity_heatmap(
        out_dir / f"{prefix}_fused_time_similarity.png",
        sensor_tokens_np,
        "Fused Stage-2 token cosine similarity across time",
    )
    save_token_projection(
        out_dir / f"{prefix}_hip_token_pca.png",
        hip_tokens_np,
        "Hip TGNN token trajectory in PCA space",
        method="pca",
    )
    save_token_projection(
        out_dir / f"{prefix}_wrist_token_pca.png",
        wrist_tokens_np,
        "Wrist TGNN token trajectory in PCA space",
        method="pca",
    )
    save_token_projection(
        out_dir / f"{prefix}_fused_token_pca.png",
        sensor_tokens_np,
        "Fused Stage-2 token trajectory in PCA space",
        method="pca",
    )
    save_token_projection(
        out_dir / f"{prefix}_hip_token_umap.png",
        hip_tokens_np,
        "Hip TGNN token trajectory in UMAP space",
        method="umap",
    )
    save_token_projection(
        out_dir / f"{prefix}_wrist_token_umap.png",
        wrist_tokens_np,
        "Wrist TGNN token trajectory in UMAP space",
        method="umap",
    )
    save_token_projection(
        out_dir / f"{prefix}_fused_token_umap.png",
        sensor_tokens_np,
        "Fused Stage-2 token trajectory in UMAP space",
        method="umap",
    )
    save_before_after_projection(
        out_dir / f"{prefix}_hip_before_after_pca.png",
        before_features=hip_features_np,
        after_features=hip_tokens_np,
        before_title="Hip features before TGNN",
        after_title="Hip tokens after TGNN",
        title="Hip branch before vs after TGNN in PCA space",
        method="pca",
    )
    save_before_after_projection(
        out_dir / f"{prefix}_wrist_before_after_pca.png",
        before_features=wrist_features_np,
        after_features=wrist_tokens_np,
        before_title="Wrist features before TGNN",
        after_title="Wrist tokens after TGNN",
        title="Wrist branch before vs after TGNN in PCA space",
        method="pca",
    )
    save_before_after_projection(
        out_dir / f"{prefix}_hip_before_after_umap.png",
        before_features=hip_features_np,
        after_features=hip_tokens_np,
        before_title="Hip features before TGNN",
        after_title="Hip tokens after TGNN",
        title="Hip branch before vs after TGNN in UMAP space",
        method="umap",
    )
    save_before_after_projection(
        out_dir / f"{prefix}_wrist_before_after_umap.png",
        before_features=wrist_features_np,
        after_features=wrist_tokens_np,
        before_title="Wrist features before TGNN",
        after_title="Wrist tokens after TGNN",
        title="Wrist branch before vs after TGNN in UMAP space",
        method="umap",
    )
    save_before_after_similarity(
        out_dir / f"{prefix}_hip_before_after_similarity.png",
        before_features=hip_features_np,
        after_features=hip_tokens_np,
        before_title="Hip features before TGNN",
        after_title="Hip tokens after TGNN",
        title="Hip branch before vs after TGNN cosine similarity",
    )
    save_before_after_similarity(
        out_dir / f"{prefix}_wrist_before_after_similarity.png",
        before_features=wrist_features_np,
        after_features=wrist_tokens_np,
        before_title="Wrist features before TGNN",
        after_title="Wrist tokens after TGNN",
        title="Wrist branch before vs after TGNN cosine similarity",
    )
    save_before_after_norm_line(
        out_dir / f"{prefix}_hip_before_after_norm.png",
        before_features=hip_features_np,
        after_features=hip_tokens_np,
        before_label="Before TGNN",
        after_label="After TGNN",
        title="Hip branch feature strength before vs after TGNN",
    )
    save_before_after_norm_line(
        out_dir / f"{prefix}_wrist_before_after_norm.png",
        before_features=wrist_features_np,
        after_features=wrist_tokens_np,
        before_label="Before TGNN",
        after_label="After TGNN",
        title="Wrist branch feature strength before vs after TGNN",
    )

    save_branch_overview(
        out_dir / f"{prefix}_hip_overview.png",
        raw_accel=hip_accel_np,
        engineered=hip_features_np,
        tokens=hip_tokens_np,
        branch_name="Hip",
    )
    save_branch_overview(
        out_dir / f"{prefix}_wrist_overview.png",
        raw_accel=wrist_accel_np,
        engineered=wrist_features_np,
        tokens=wrist_tokens_np,
        branch_name="Wrist",
    )

    metadata = {
        "sample_idx": sample_idx,
        "label_index": label,
        "label_name": f"A{label + 1:02d}",
        "window": int(hip_accel_np.shape[0]),
        "hip_input_shape": list(hip_features.shape),
        "hip_token_shape": list(hip_tokens.shape),
        "wrist_input_shape": list(wrist_features.shape),
        "wrist_token_shape": list(wrist_tokens.shape),
        "fused_sensor_tokens_shape": list(sensor_tokens.shape),
        "global_embedding_shape": list(h_global.shape),
        "stage1_target_latent_shape": list(z0.shape),
        "global_embedding_l2_norm": float(np.linalg.norm(h_global_np)),
        "stage1_ckpt": str(Path(args.stage1_ckpt).resolve()),
        "stage2_ckpt": str(Path(args.stage2_ckpt).resolve()),
    }
    (out_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved TGNN encoder visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
