"""Reusable training-time evaluation and reporting helpers."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence
from diffusion_model.losses import motion_losses

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    from matplotlib import pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - optional plotting dependency
    PCA = None

try:
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover - optional plotting dependency
    TSNE = None

try:
    import umap
except Exception:  # pragma: no cover - optional plotting dependency
    umap = None

from diffusion_model.gait_metrics import GAIT_METRIC_NAMES, compute_gait_metrics_torch
from diffusion_model.losses import motion_losses
from diffusion_model.sensor_model import IMU_FEATURE_NAMES
from diffusion_model.util import DEFAULT_NUM_CLASSES, get_skeleton_edges


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sensor_name(path: str) -> str:
    return Path(path).name if path else ""


def write_curve_plot(out_path: Path, title: str, x_values: Sequence[float], series: Sequence[tuple[str, Sequence[float], str]], x_label: str, y_label: str) -> None:
    if plt is None:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(12, 7))
    for label, values, color in series:
        plt.plot(x_values, values, label=label, linewidth=2.5, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_hist_grid(out_path: Path, title: str, real_data: np.ndarray, gen_data: np.ndarray, metric_names: Sequence[str]) -> None:
    if plt is None:
        return
    cols = 3
    rows = int(np.ceil(len(metric_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx, name in enumerate(metric_names):
        ax = axes[idx // cols, idx % cols]
        ax.hist(real_data[:, idx], bins=24, alpha=0.55, color="#2563eb", label="Real", density=True)
        ax.hist(gen_data[:, idx], bins=24, alpha=0.55, color="#dc2626", label="Generated", density=True)
        ax.set_title(name)
        ax.grid(True, alpha=0.2)
    for idx in range(len(metric_names), rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_scatter(out_path: Path, title: str, x_real: np.ndarray, y_real: np.ndarray, x_gen: np.ndarray, y_gen: np.ndarray, x_label: str, y_label: str) -> None:
    if plt is None:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(9, 7))
    plt.scatter(x_real, y_real, s=18, alpha=0.45, label="Real", color="#2563eb")
    plt.scatter(x_gen, y_gen, s=18, alpha=0.45, label="Generated", color="#dc2626")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_heatmap(out_path: Path, title: str, values: np.ndarray, x_label: str, y_label: str, cmap: str = "viridis") -> None:
    if plt is None:
        return
    ensure_dir(out_path.parent)
    arr = np.asarray(values, dtype=np.float32)
    plt.figure(figsize=(11, 5.5))
    plt.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_pca_plot(out_path: Path, title: str, features: np.ndarray, labels: np.ndarray, num_classes: int = DEFAULT_NUM_CLASSES) -> None:
    if plt is None or PCA is None or features.shape[0] < 2:
        return
    ensure_dir(out_path.parent)
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab20", num_classes)
    for cid in sorted(set(labels.tolist())):
        mask = labels == cid
        plt.scatter(reduced[mask, 0], reduced[mask, 1], s=20, alpha=0.6, color=cmap(cid), label=f"A{cid + 1:02d}")
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_embedding_projection_plot(
    out_path: Path,
    title: str,
    features: np.ndarray,
    labels: np.ndarray,
    method: str,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> bool:
    if plt is None or features.shape[0] < 2:
        return False
    method_name = method.lower()
    if method_name == "pca":
        if PCA is None:
            return False
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(features)
        x_label = f"PC1 ({reducer.explained_variance_ratio_[0] * 100:.1f}% var)"
        y_label = f"PC2 ({reducer.explained_variance_ratio_[1] * 100:.1f}% var)"
    elif method_name == "tsne":
        if TSNE is None or features.shape[0] < 4:
            return False
        perplexity = min(30.0, max(2.0, float(features.shape[0] - 1) / 3.0))
        perplexity = min(perplexity, float(features.shape[0]) - 1.0)
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(features)
        x_label = "t-SNE 1"
        y_label = "t-SNE 2"
    elif method_name == "umap":
        if umap is None or features.shape[0] < 3:
            return False
        n_neighbors = min(15, features.shape[0] - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.15, random_state=42)
        reduced = reducer.fit_transform(features)
        x_label = "UMAP 1"
        y_label = "UMAP 2"
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported projection method: {method}")

    ensure_dir(out_path.parent)
    plt.figure(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab20", num_classes)
    for cid in sorted(set(labels.tolist())):
        mask = labels == cid
        plt.scatter(reduced[mask, 0], reduced[mask, 1], s=20, alpha=0.65, color=cmap(cid), label=f"A{cid + 1:02d}")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def write_similarity_heatmap(out_path: Path, title: str, matrix: np.ndarray, labels: Sequence[str]) -> None:
    if plt is None or matrix.size == 0:
        return
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_centroid_drift_plot(out_path: Path, rows: list[dict[str, object]]) -> None:
    if plt is None or not rows:
        return
    epochs = [int(float(row["epoch"])) for row in rows]
    drift_latent = [float(row["latent_centroid_drift"]) for row in rows]
    drift_sensor = [float(row["sensor_centroid_drift"]) for row in rows]
    write_curve_plot(
        out_path,
        "Stage2 Class-Centroid Drift Across Evaluation Epochs",
        epochs,
        [
            ("Skeleton latent drift", drift_latent, "#2563eb"),
            ("Sensor embedding drift", drift_sensor, "#dc2626"),
        ],
        "Epoch",
        "Mean class-centroid drift",
    )


def update_stage2_embedding_history(stage2_dir: Path, epoch: int, metrics: dict[str, float]) -> None:
    history_path = stage2_dir / "embedding_diagnostics_history.csv"
    fieldnames = [
        "epoch",
        "latent_sensor_cosine_mean",
        "latent_sensor_cosine_std",
        "latent_sensor_l2_mean",
        "latent_sensor_l2_std",
        "latent_within_class_scatter",
        "sensor_within_class_scatter",
        "latent_centroid_drift",
        "sensor_centroid_drift",
    ]
    existing: list[dict[str, object]] = []
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    kept = [row for row in existing if int(float(row["epoch"])) != epoch]
    merged = kept + [{"epoch": epoch, **metrics}]
    merged.sort(key=lambda row: int(float(row["epoch"])))
    write_csv(history_path, merged, fieldnames)
    write_curve_plot(
        stage2_dir / "embedding_alignment_trends.png",
        "Stage2 Embedding Alignment Across Evaluation Epochs",
        [int(float(row["epoch"])) for row in merged],
        [
            ("Cosine gap mean", [float(row["latent_sensor_cosine_mean"]) for row in merged], "#2563eb"),
            ("L2 gap mean", [float(row["latent_sensor_l2_mean"]) for row in merged], "#dc2626"),
            ("Latent scatter", [float(row["latent_within_class_scatter"]) for row in merged], "#059669"),
            ("Sensor scatter", [float(row["sensor_within_class_scatter"]) for row in merged], "#f97316"),
        ],
        "Epoch",
        "Diagnostic value",
    )
    write_centroid_drift_plot(stage2_dir / "embedding_centroid_drift.png", merged)


def _class_centroids(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    classes = sorted({int(cid) for cid in labels.tolist()})
    centroids = []
    for cid in classes:
        class_features = features[labels == cid]
        centroids.append(class_features.mean(axis=0))
    return np.asarray(centroids, dtype=np.float32), classes


def _mean_within_class_scatter(features: np.ndarray, labels: np.ndarray) -> float:
    scatters: list[float] = []
    for cid in sorted({int(cid) for cid in labels.tolist()}):
        class_features = features[labels == cid]
        if class_features.shape[0] < 2:
            continue
        centroid = class_features.mean(axis=0, keepdims=True)
        scatters.append(float(np.linalg.norm(class_features - centroid, axis=1).mean()))
    return float(np.mean(scatters)) if scatters else 0.0


def render_skeleton_panels(out_path: Path, sequences: Sequence[np.ndarray], titles: Sequence[str]) -> None:
    if plt is None or not sequences:
        return
    ensure_dir(out_path.parent)
    edges = [(i, j) for i, j in get_skeleton_edges() if i < sequences[0].shape[1] and j < sequences[0].shape[1]]
    n = len(sequences)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.5))
    axes = np.atleast_1d(axes)
    for ax, seq, title in zip(axes, sequences, titles):
        seq_arr = np.asarray(seq, dtype=np.float32)
        if seq_arr.ndim != 3 or seq_arr.shape[-1] < 2 or not np.isfinite(seq_arr).any():
            ax.text(0.5, 0.5, "Invalid sequence", ha="center", va="center", fontsize=11)
            ax.set_title(title)
            ax.axis("off")
            continue
        seq_xy = _normalize_xy(seq_arr[..., :2], canvas_size=512)
        frame = seq_xy[len(seq_xy) // 2]
        xs = frame[:, 0]
        ys = frame[:, 1]
        for i, j in edges:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#222", linewidth=1.6)
        ax.scatter(xs, ys, s=20, color="#2563eb")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _normalize_xy(points_xy: np.ndarray, canvas_size: int, root_index: int = 0) -> np.ndarray:
    assert points_xy.ndim == 3, "points_xy must be [T, J, 2]"
    assert 0 <= root_index < points_xy.shape[1], "root_index out of range"
    # Center each frame on its root joint so long walking translations do not
    # collapse the visible body pose into a tiny dot on the canvas.
    anchor = points_xy[:, root_index : root_index + 1, :]
    centered = points_xy - anchor
    flat_xy = centered.reshape(-1, 2)
    min_xy = np.percentile(flat_xy, 2.0, axis=0, keepdims=True).reshape(1, 1, 2)
    max_xy = np.percentile(flat_xy, 98.0, axis=0, keepdims=True).reshape(1, 1, 2)
    span = float(np.maximum(max_xy - min_xy, 1e-6).max())
    normalized = (centered - (min_xy + max_xy) * 0.5) / span
    margin = 0.1 * canvas_size
    drawable = canvas_size - 2 * margin
    canvas_center = canvas_size * 0.5
    return normalized * drawable + canvas_center


def save_skeleton_gif(sequence: np.ndarray, out_path: Path, fps: int = 12, canvas_size: int = 512) -> None:
    ensure_dir(out_path.parent)
    points = np.asarray(sequence, dtype=np.float32)
    points_xy = _normalize_xy(points[..., :2], canvas_size=canvas_size)
    edges = [(i, j) for i, j in get_skeleton_edges() if i < points.shape[1] and j < points.shape[1]]
    frames: list[Image.Image] = []
    for frame_xy in points_xy:
        img = Image.new("RGB", (canvas_size, canvas_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        for i, j in edges:
            xi, yi = float(frame_xy[i][0]), float(frame_xy[i][1])
            xj, yj = float(frame_xy[j][0]), float(frame_xy[j][1])
            draw.line((xi, yi, xj, yj), fill=(30, 30, 30), width=3)
        for joint_xy in frame_xy:
            x, y = float(joint_xy[0]), float(joint_xy[1])
            r = 4
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(20, 60, 210))
        frames.append(img)
    duration_ms = int(1000 / max(fps, 1))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def _snapshot_timesteps(total_timesteps: int) -> list[int]:
    if total_timesteps < 2:
        return [0]
    last = total_timesteps - 1
    candidates = [last, int(round(last * 0.6)), int(round(last * 0.2)), 0]
    ordered: list[int] = []
    for item in candidates:
        item = max(0, min(last, int(item)))
        if item not in ordered:
            ordered.append(item)
    return ordered


def write_history(run_dir: Path, stage_name: str, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = sorted({k for row in history for k in row.keys()})
    write_csv(run_dir / stage_name / "history.csv", history, fieldnames)
    epochs = [row["epoch"] for row in history]
    series = []
    color_map = {
        "train_loss_diff": "#2563eb",
        "val_loss_diff": "#dc2626",
        "train_loss_align": "#2563eb",
        "val_loss_align": "#dc2626",
        "train_loss_total": "#111827",
        "val_loss_total": "#7c3aed",
        "train_loss_pose": "#dc2626",
        "val_loss_pose": "#f97316",
        "train_loss_latent": "#7c3aed",
        "val_loss_latent": "#a855f7",
        "train_loss_vel": "#0891b2",
        "val_loss_vel": "#06b6d4",
        "train_loss_gait": "#059669",
        "val_loss_gait": "#10b981",
        "train_loss_motion": "#0f766e",
        "val_loss_motion": "#14b8a6",
        "train_loss_bone":  "#b45309",
        "val_loss_bone":    "#d97706",
        "train_loss_skate": "#7e22ce",
        "val_loss_skate":   "#a21caf",
    }
    _PLOT_KEYS = {"train_loss_total", "val_loss_total"}
    for key in history[0].keys():
        if key not in _PLOT_KEYS:
            continue
        series.append((key, [row.get(key, float("nan")) for row in history], color_map.get(key, "#374151")))
    write_curve_plot(run_dir / stage_name / "loss_curves.png", f"{stage_name} Loss Curves", epochs, series, "Epoch", "Loss")

    accuracy_series = []
    accuracy_color_map = {
        "train_acc_latent_cls": "#2563eb",
        "val_acc_latent_cls": "#dc2626",
        "train_acc_cls_aux": "#059669",
        "val_acc_cls_aux": "#10b981",
    }
    accuracy_keys = [
        key
        for key in fieldnames
        if key.startswith("train_acc_") or key.startswith("val_acc_")
    ]
    for key in accuracy_keys:
        accuracy_series.append(
            (key, [row.get(key, float("nan")) for row in history], accuracy_color_map.get(key, "#374151"))
        )
    if accuracy_series:
        write_curve_plot(
            run_dir / stage_name / "accuracy_curves.png",
            f"{stage_name} Accuracy Curves",
            epochs,
            accuracy_series,
            "Epoch",
            "Accuracy",
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            confusion[truth, pred] += 1

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)

    rows: list[dict[str, object]] = []
    for class_idx in range(num_classes):
        precision = _safe_divide(true_positive[class_idx], predicted[class_idx])
        recall = _safe_divide(true_positive[class_idx], support[class_idx])
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        rows.append(
            {
                "class_name": f"A{class_idx + 1:02d}",
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(support[class_idx]),
                "accuracy": recall,
            }
        )

    accuracy = _safe_divide(float(true_positive.sum()), float(confusion.sum()))
    macro_precision = float(np.mean([row["precision"] for row in rows])) if rows else 0.0
    macro_recall = float(np.mean([row["recall"] for row in rows])) if rows else 0.0
    macro_f1 = float(np.mean([row["f1_score"] for row in rows])) if rows else 0.0
    weighted_precision = _safe_divide(
        float(sum(float(row["precision"]) * int(row["support"]) for row in rows)),
        float(sum(int(row["support"]) for row in rows)),
    )
    weighted_recall = _safe_divide(
        float(sum(float(row["recall"]) * int(row["support"]) for row in rows)),
        float(sum(int(row["support"]) for row in rows)),
    )
    weighted_f1 = _safe_divide(
        float(sum(float(row["f1_score"]) * int(row["support"]) for row in rows)),
        float(sum(int(row["support"]) for row in rows)),
    )
    return {
        "accuracy": accuracy,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "support": int(support.sum()),
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1,
            "support": int(support.sum()),
        },
        "per_class": {row["class_name"]: row for row in rows},
        "rows": rows,
        "confusion_matrix": confusion.tolist(),
    }


def _write_confusion_matrix_plot(out_path: Path, title: str, confusion: np.ndarray) -> None:
    if plt is None:
        return
    ensure_dir(out_path.parent)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        np.where(row_sums == 0, 1, row_sums),
        out=np.zeros_like(confusion, dtype=np.float32),
        where=np.ones_like(confusion, dtype=bool),
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    labels = [f"A{i + 1:02d}" for i in range(confusion.shape[0])]
    ax.set_xticks(range(confusion.shape[0]))
    ax.set_yticks(range(confusion.shape[0]))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for row_idx in range(confusion.shape[0]):
        for col_idx in range(confusion.shape[1]):
            value = confusion[row_idx, col_idx]
            ax.text(col_idx, row_idx, str(int(value)), ha="center", va="center", fontsize=7, color="#111827")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_classification_artifacts(
    out_dir: Path,
    stem: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = DEFAULT_NUM_CLASSES,
    title: str | None = None,
) -> dict[str, object]:
    ensure_dir(out_dir)
    report = _compute_classification_report(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    base_name = stem.rstrip("_")
    rows = list(report["rows"])
    summary_rows = rows + [
        {
            "class_name": "macro_avg",
            **report["macro_avg"],
            "accuracy": report["accuracy"],
        },
        {
            "class_name": "weighted_avg",
            **report["weighted_avg"],
            "accuracy": report["accuracy"],
        },
    ]
    write_json(out_dir / f"{base_name}_report.json", report)
    write_csv(
        out_dir / f"{base_name}_report.csv",
        summary_rows,
        ["class_name", "precision", "recall", "f1_score", "support", "accuracy"],
    )
    confusion_rows = []
    confusion = np.asarray(report["confusion_matrix"], dtype=np.int64)
    labels = [f"A{i + 1:02d}" for i in range(num_classes)]
    for class_idx, class_name in enumerate(labels):
        confusion_rows.append(
            {"true_class": class_name, **{pred_name: int(confusion[class_idx, pred_idx]) for pred_idx, pred_name in enumerate(labels)}}
        )
    write_csv(out_dir / f"{base_name}_confusion_matrix.csv", confusion_rows, ["true_class", *labels])
    _write_confusion_matrix_plot(
        out_dir / f"{base_name}_confusion_matrix.png",
        title or f"{base_name.replace('_', ' ').title()} Confusion Matrix",
        confusion,
    )
    return report


def _safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.square(y_true - y_pred).sum())
    ss_tot = float(np.square(y_true - float(y_true.mean())).sum())
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def write_stage2_gait_prediction_metrics(
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: Sequence[str],
) -> dict[str, object]:
    ensure_dir(out_dir)
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    abs_err = np.abs(y_pred - y_true)
    sq_err = np.square(y_pred - y_true)
    rows = []
    for idx, metric_name in enumerate(metric_names):
        rows.append(
            {
                "metric_name": str(metric_name),
                "mae": float(abs_err[:, idx].mean()),
                "rmse": float(np.sqrt(sq_err[:, idx].mean())),
                "r2": float(_safe_r2_score(y_true[:, idx], y_pred[:, idx])),
            }
        )
    result = {
        "overall_mean_mae": float(abs_err.mean()),
        "overall_mean_rmse": float(np.sqrt(sq_err.mean())),
        "metrics": rows,
    }
    write_json(out_dir / "gait_prediction_metrics.json", result)
    write_csv(out_dir / "gait_prediction_metrics.csv", rows, ["metric_name", "mae", "rmse", "r2"])
    return result


def update_stage3_eval_history(stage3_dir: Path, epoch: int, eval_acc_cls_real: float, eval_acc_cls_generated: float) -> None:
    history_path = stage3_dir / "eval_history.csv"
    fieldnames = ["epoch", "eval_acc_cls_real", "eval_acc_cls_generated"]
    existing: list[dict[str, object]] = []
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    kept = [row for row in existing if int(float(row["epoch"])) != int(epoch)]
    merged = kept + [
        {
            "epoch": int(epoch),
            "eval_acc_cls_real": float(eval_acc_cls_real),
            "eval_acc_cls_generated": float(eval_acc_cls_generated),
        }
    ]
    merged.sort(key=lambda row: int(float(row["epoch"])))
    write_csv(history_path, merged, fieldnames)
    write_stage3_eval_accuracy_plot(stage3_dir / "eval_accuracy_curves.png", merged)


def write_stage3_eval_accuracy_plot(out_path: Path, rows: list[dict[str, object]]) -> None:
    if plt is None or not rows:
        return
    ensure_dir(out_path.parent)
    epochs = [int(float(row["epoch"])) for row in rows]
    real_acc = [float(row["eval_acc_cls_real"]) for row in rows]
    generated_acc = [float(row["eval_acc_cls_generated"]) for row in rows]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, real_acc, marker="o", linewidth=2.0, color="#2563eb", label="Real classifier accuracy")
    plt.plot(epochs, generated_acc, marker="o", linewidth=2.0, color="#dc2626", label="Generated classifier accuracy")
    plt.title("Stage3 Evaluation Classifier Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def update_stage3_metric_history(stage3_dir: Path, epoch: int, rows: list[dict[str, object]]) -> None:
    history_path = stage3_dir / "generated_gait_metric_history.csv"
    fieldnames = ["epoch", "metric_name", "real_mean", "real_std", "generated_mean", "generated_std"]
    existing: list[dict[str, object]] = []
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    kept = [row for row in existing if int(float(row["epoch"])) != epoch]
    merged = kept + [{"epoch": epoch, **row} for row in rows]
    merged.sort(key=lambda row: (int(float(row["epoch"])), list(GAIT_METRIC_NAMES).index(str(row["metric_name"]))))
    write_csv(history_path, merged, fieldnames)
    write_stage3_metric_trend_plot(stage3_dir / "generated_gait_metric_trends.png", merged)


def write_stage3_metric_trend_plot(out_path: Path, rows: list[dict[str, object]]) -> None:
    if plt is None or not rows:
        return
    metric_names = list(GAIT_METRIC_NAMES)
    cols = 3
    plot_rows = int(np.ceil(len(metric_names) / cols))
    fig, axes = plt.subplots(plot_rows, cols, figsize=(15, 4.8 * plot_rows))
    axes = np.atleast_1d(axes).reshape(plot_rows, cols)
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx // cols, idx % cols]
        metric_rows = sorted(
            [row for row in rows if str(row["metric_name"]) == metric_name],
            key=lambda row: int(float(row["epoch"])),
        )
        epochs = [int(float(row["epoch"])) for row in metric_rows]
        gen_mean = [float(row["generated_mean"]) for row in metric_rows]
        gen_std = [float(row["generated_std"]) for row in metric_rows]
        real_mean = [float(row["real_mean"]) for row in metric_rows]
        real_std = [float(row["real_std"]) for row in metric_rows]
        ax.plot(epochs, gen_mean, color="#dc2626", linewidth=2.0, label="Generated mean")
        ax.plot(epochs, gen_std, color="#f97316", linewidth=2.0, label="Generated std")
        ax.plot(epochs, real_mean, color="#2563eb", linewidth=1.5, linestyle="--", label="Real mean")
        ax.plot(epochs, real_std, color="#10b981", linewidth=1.5, linestyle="--", label="Real std")
        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
    for idx in range(len(metric_names), plot_rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Generated Gait Metric Mean/Std Across Evaluation Epochs")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_run_manifest(args, device: torch.device, runtime: dict[str, object] | None = None) -> dict[str, object]:
    runtime = runtime or {}
    return {
        "device": str(device),
        "dataset": {
            "dataset_path": args.dataset_path,
            "skeleton_folder": args.skeleton_folder,
            "hip_folder": args.hip_folder,
            "wrist_folder": args.wrist_folder,
            "window": args.window,
            "stride": args.stride,
            "overlap": args.stride < args.window,
            "sensor_modality": runtime.get("sensor_modality", "accelerometer only"),
            "sensor_locations": runtime.get("sensor_locations", [_sensor_name(args.hip_folder), _sensor_name(args.wrist_folder)]),
            "imu_feature_names": list(IMU_FEATURE_NAMES),
            "gait_metric_names": list(GAIT_METRIC_NAMES),
            "skeleton_scaling": "millimeters_to_meters",
            "sensor_normalization": False,
            "fps": args.fps,
        },
        "model": {
            "encoder_graph_op": getattr(args, "encoder_graph_op_resolved", getattr(args, "encoder_type", "gat") or "gat"),
            "skeleton_graph_op": getattr(args, "skeleton_graph_op_resolved", getattr(args, "skeleton_graph_op", "gat") or "gat"),
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": runtime.get("optimizer", "Adam"),
            "scheduler": runtime.get("scheduler", "none"),
            "lambda_pose": getattr(args, "lambda_pose", None),
            "lambda_latent": getattr(args, "lambda_latent", None),
            "lambda_vel": getattr(args, "lambda_vel", None),
            "lambda_motion": getattr(args, "lambda_motion", None),
            "lambda_gait": getattr(args, "lambda_gait", None),
        },
        "diffusion": {
            "train_timesteps": args.timesteps,
            "sample_steps": getattr(args, "sample_steps", 50),
            "sample_seed": getattr(args, "sample_seed", None),
            "sampler": getattr(args, "sampler", None),
        },
        "stage": args.stage,
        "seed": args.seed,
        "conditioning_mode": "imu_only" if getattr(args, "one_to_one", False) else "legacy",
        "gait_role": "auxiliary_supervision" if getattr(args, "one_to_one", False) else "conditioning",
        "checkpoints": {
            "stage1_ckpt": args.stage1_ckpt,
            "stage2_ckpt": args.stage2_ckpt,
            "save_dir": args.save_dir,
        },
    }


def save_run_manifest(run_dir: Path, args, device: torch.device, runtime: dict[str, object] | None = None) -> None:
    write_json(run_dir / "run_manifest.json", build_run_manifest(args, device, runtime=runtime))


def _iter_eval_batches(loader, max_batches: int):
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        yield batch


@torch.no_grad()
def sample_stage3_latents(
    stage3,
    shape: torch.Size,
    device: torch.device,
    h_tokens: torch.Tensor,
    h_global: torch.Tensor,
    gait_metrics: torch.Tensor | None,
    sample_steps: int,
    sampler: str = "ddim",
    sample_seed: int | None = None,
    a_hip_stream: torch.Tensor | None = None,
    a_wrist_stream: torch.Tensor | None = None,
) -> torch.Tensor:
    generator = None
    if sample_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(sample_seed))
    h_tokens, h_global = stage3.augment_conditioning(
        h_tokens=h_tokens,
        h_global=h_global,
        a_hip_stream=a_hip_stream,
        a_wrist_stream=a_wrist_stream,
    )
    sampler_name = sampler.lower()
    if sampler_name == "ddim":
        return stage3.diffusion.p_sample_loop_ddim(
            stage3.denoiser,
            shape=shape,
            device=device,
            sample_steps=sample_steps,
            eta=0.0,
            h_tokens=h_tokens,
            h_global=h_global,
            gait_metrics=None,
            generator=generator,
        )
    if sampler_name == "ddpm":
        return stage3.diffusion.p_sample_loop(
            stage3.denoiser,
            shape=shape,
            device=device,
            h_tokens=h_tokens,
            h_global=h_global,
            gait_metrics=None,
            generator=generator,
        )
    raise ValueError(f"Unsupported sampler: {sampler}")


def _mpjpe(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x_hat - x, dim=-1).mean()


def _velocity_error(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    vel_hat = x_hat[:, 1:] - x_hat[:, :-1]
    vel = x[:, 1:] - x[:, :-1]
    return torch.linalg.norm(vel_hat - vel, dim=-1).mean()


def _root_trajectory_error(x_hat: torch.Tensor, x: torch.Tensor, root_index: int = 0) -> torch.Tensor:
    return torch.linalg.norm(x_hat[:, :, root_index, :] - x[:, :, root_index, :], dim=-1).mean()


def _pairwise_distance_rows(arr: np.ndarray) -> np.ndarray:
    diffs = arr[:, None, :] - arr[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def _write_histogram(out_path: Path, title: str, values: np.ndarray, x_label: str, color: str = "#2563eb") -> None:
    if plt is None:
        return
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(9, 6))
    plt.hist(arr, bins=24, color=color, alpha=0.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_overlay_plot(out_path: Path, title: str, series_real: np.ndarray, series_gen: np.ndarray, y_label: str) -> None:
    if plt is None:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(11, 5.5))
    plt.plot(series_real, label="Real", color="#2563eb", linewidth=2.0)
    plt.plot(series_gen, label="Generated", color="#dc2626", linewidth=2.0)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


@torch.no_grad()
def evaluate_stage1(model, loader, device: torch.device, out_dir: Path, timestep_values: Sequence[int]) -> None:
    rows: list[dict[str, object]] = []
    latent_features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_sequences: list[np.ndarray] = []
    sample_latent_maps: list[tuple[np.ndarray, str]] = []
    recon_metrics: list[dict[str, float]] = []
    for t_val in timestep_values:
        vals = []
        for batch in _iter_eval_batches(loader, max_batches=4):
            x = batch["skeleton"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            y = batch["label"].cpu().numpy()
            z0 = model.encoder(x, gait_metrics=None)
            if len(sample_sequences) < 3:
                remaining = 3 - len(sample_sequences)
                for seq, latent, label in zip(x[:remaining], z0[:remaining], y[:remaining]):
                    sample_sequences.append(seq.cpu().numpy())
                    latent_map = torch.linalg.norm(latent.float(), dim=-1).cpu().numpy()
                    sample_latent_maps.append((latent_map, f"label_A{int(label) + 1:02d}"))
                    recon = model.decoder(latent.unsqueeze(0).float())[0].cpu().numpy()
                    sample_sequences.append(recon)
            latent_features.append(z0.mean(dim=(1, 2)).cpu().numpy())
            labels.append(y)
            x_recon = model.decoder(z0.float())
            recon_metrics.append(
                {
                    "mpjpe_recon": float(_mpjpe(x_recon, x).item()),
                    "velocity_error": float(_velocity_error(x_recon, x).item()),
                    "bone_length_drift": float(motion_losses(x_recon.float())["loss_bone"].item()),
                }
            )
            t = torch.full((x.shape[0],), int(t_val), device=device, dtype=torch.long)
            noise = torch.randn_like(z0)
            zt = model.diffusion.q_sample(z0=z0, t=t, noise=noise)
            pred_noise = model.denoiser(zt, t, gait_metrics=None)
            mse = torch.mean((pred_noise - noise) ** 2, dim=(1, 2, 3)).cpu().numpy()
            vals.extend(mse.tolist())
        if vals:
            rows.append({"timestep": int(t_val), "mean_mse": float(np.mean(vals)), "std_mse": float(np.std(vals)), "count": len(vals)})
    if not rows:
        return
    ensure_dir(out_dir)
    write_csv(out_dir / "noise_prediction_error_by_timestep.csv", rows, ["timestep", "mean_mse", "std_mse", "count"])
    write_curve_plot(
        out_dir / "noise_prediction_error_by_timestep.png",
        "Noise Prediction Error by Diffusion Timestep",
        [row["timestep"] for row in rows],
        [("Mean MSE", [row["mean_mse"] for row in rows], "#2563eb")],
        "Diffusion timestep",
        "MSE(pred_noise, true_noise)",
    )
    if recon_metrics:
        write_csv(out_dir / "stage1_recon_metrics.csv", recon_metrics, ["mpjpe_recon", "velocity_error", "bone_length_drift"])
        write_json(
            out_dir / "stage1_recon_summary.json",
            {
                "mpjpe_recon_mean": float(np.mean([row["mpjpe_recon"] for row in recon_metrics])),
                "velocity_error_mean": float(np.mean([row["velocity_error"] for row in recon_metrics])),
                "bone_length_drift_mean": float(np.mean([row["bone_length_drift"] for row in recon_metrics])),
            },
        )
        _write_histogram(
            out_dir / "stage1_recon_error_hist.png",
            "Stage1 Reconstruction MPJPE Distribution",
            np.asarray([row["mpjpe_recon"] for row in recon_metrics], dtype=np.float32),
            "MPJPE",
        )
    if sample_sequences:
        render_skeleton_panels(
            out_dir / "encoder_input_skeletons.png",
            sample_sequences,
            [f"Input sample {idx}" for idx in range(len(sample_sequences))],
        )
    for idx, (latent_map, label_text) in enumerate(sample_latent_maps):
        write_heatmap(
            out_dir / f"encoder_output_latent_norm_sample_{idx}.png",
            f"Stage1 Encoder Output Norm Map ({label_text})",
            latent_map.T,
            x_label="Frame",
            y_label="Joint",
            cmap="magma",
        )
    if latent_features and labels:
        latent_arr = np.concatenate(latent_features, axis=0)
        label_arr = np.concatenate(labels, axis=0)
        write_pca_plot(
            out_dir / "encoder_output_latent_pca.png",
            "Stage1 Encoder Output PCA (mean pooled z0)",
            latent_arr,
            label_arr,
        )

    classifier_targets: list[np.ndarray] = []
    classifier_preds: list[np.ndarray] = []
    for batch in _iter_eval_batches(loader, max_batches=4):
        x = batch["skeleton"].to(device)
        y = batch["label"].cpu().numpy()
        z0 = model.encoder(x, gait_metrics=None)
        logits = model.cls_head(z0.mean(dim=(1, 2)).float())
        classifier_targets.append(y)
        classifier_preds.append(logits.argmax(dim=1).cpu().numpy())
    if classifier_targets and classifier_preds:
        write_classification_artifacts(
            out_dir,
            "latent_classifier",
            np.concatenate(classifier_targets, axis=0),
            np.concatenate(classifier_preds, axis=0),
            num_classes=model.num_classes,
            title="Stage1 Latent Classifier Confusion Matrix",
        )


@torch.no_grad()
def evaluate_stage2_reports(
    stage2,
    loader,
    device: torch.device,
    out_dir: Path,
    max_batches: int = 8,
) -> None:
    cls_targets: list[np.ndarray] = []
    cls_preds: list[np.ndarray] = []
    gait_targets: list[np.ndarray] = []
    gait_preds: list[np.ndarray] = []
    for batch in _iter_eval_batches(loader, max_batches=max_batches):
        a_hip = batch["A_hip"].to(device)
        a_wrist = batch["A_wrist"].to(device)
        gait_metrics = batch["gait_metrics"].to(device)
        y = batch["label"].to(device)
        _, h_global = stage2.aligner(a_hip, a_wrist)
        logits = stage2.cls_head(h_global.float())
        cls_targets.append(y.cpu().numpy())
        cls_preds.append(logits.argmax(dim=1).cpu().numpy())
        if stage2.gait_pred_head is not None and gait_metrics.numel() > 0:
            gait_pred = stage2.gait_pred_head(h_global.float())
            gait_targets.append(gait_metrics.cpu().numpy())
            gait_preds.append(gait_pred.cpu().numpy())

    ensure_dir(out_dir)
    if cls_targets and cls_preds:
        write_classification_artifacts(
            out_dir,
            "imu_classifier",
            np.concatenate(cls_targets, axis=0),
            np.concatenate(cls_preds, axis=0),
            num_classes=stage2.num_classes,
            title="Stage2 IMU Classifier Confusion Matrix",
        )
    if gait_targets and gait_preds:
        gait_true = np.concatenate(gait_targets, axis=0)
        gait_pred = np.concatenate(gait_preds, axis=0)
        metric_names = list(GAIT_METRIC_NAMES[: gait_true.shape[1]])
        write_stage2_gait_prediction_metrics(out_dir, gait_true, gait_pred, metric_names=metric_names)


@torch.no_grad()
def evaluate_stage2(stage1, stage2, loader, device: torch.device, out_dir: Path, epoch: int | None = None) -> None:
    latent_features: list[np.ndarray] = []
    sensor_features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    pair_cosines: list[float] = []
    pair_l2s: list[float] = []
    retrieval_hits = 0
    retrieval_total = 0
    for batch in _iter_eval_batches(loader, max_batches=8):
        x = batch["skeleton"].to(device)
        a_hip = batch["A_hip"].to(device)
        a_wrist = batch["A_wrist"].to(device)
        gait_metrics = batch["gait_metrics"].to(device)
        y = batch["label"].cpu().numpy()
        z0 = stage1.encoder(x, gait_metrics=None)
        _, h_global = stage2.aligner(a_hip, a_wrist)
        latent_batch = z0.mean(dim=(1, 2))
        sensor_batch = h_global
        latent_features.append(latent_batch.cpu().numpy())
        sensor_features.append(sensor_batch.cpu().numpy())
        labels.append(y)
        similarity = F.cosine_similarity(sensor_batch[:, None, :], latent_batch[None, :, :], dim=-1)
        retrieval_hits += int((similarity.argmax(dim=1) == torch.arange(similarity.shape[0], device=similarity.device)).sum().item())
        retrieval_total += int(similarity.shape[0])
        pair_cosines.extend(F.cosine_similarity(latent_batch, sensor_batch, dim=1).cpu().tolist())
        pair_l2s.extend(torch.linalg.norm(latent_batch - sensor_batch, dim=1).cpu().tolist())
    if not latent_features:
        return
    ensure_dir(out_dir)
    latent_arr = np.concatenate(latent_features, axis=0)
    sensor_arr = np.concatenate(sensor_features, axis=0)
    label_arr = np.concatenate(labels, axis=0)
    valid_mask = (
        np.isfinite(latent_arr).all(axis=1)
        & np.isfinite(sensor_arr).all(axis=1)
        & np.isfinite(label_arr)
    )
    invalid_count = int((~valid_mask).sum())
    latent_arr = latent_arr[valid_mask]
    sensor_arr = sensor_arr[valid_mask]
    label_arr = label_arr[valid_mask]
    pair_cosines_arr = np.asarray(pair_cosines, dtype=np.float32)
    pair_l2s_arr = np.asarray(pair_l2s, dtype=np.float32)
    pair_cosines_arr = pair_cosines_arr[np.isfinite(pair_cosines_arr)]
    pair_l2s_arr = pair_l2s_arr[np.isfinite(pair_l2s_arr)]
    if latent_arr.shape[0] == 0 or sensor_arr.shape[0] == 0:
        write_json(
            out_dir / "embedding_diagnostics.json",
            {
                "sample_count": 0,
                "invalid_feature_rows_filtered": invalid_count,
                "error": "All Stage2 latent/sensor feature rows were non-finite during evaluation.",
            },
        )
        return
    projection_methods = [
        ("pca", "PCA"),
        ("tsne", "t-SNE"),
        ("umap", "UMAP"),
    ]
    projection_status: dict[str, bool] = {}
    for method_key, method_label in projection_methods:
        projection_status[f"latent_{method_key}"] = write_embedding_projection_plot(
            out_dir / f"latent_{method_key}.png",
            f"Skeleton Latent Space {method_label}",
            latent_arr,
            label_arr,
            method=method_key,
        )
        projection_status[f"sensor_{method_key}"] = write_embedding_projection_plot(
            out_dir / f"sensor_embedding_{method_key}.png",
            f"Sensor Embedding {method_label}",
            sensor_arr,
            label_arr,
            method=method_key,
        )

    latent_tensor = torch.from_numpy(latent_arr)
    sensor_tensor = torch.from_numpy(sensor_arr)
    cosine_gap = 1.0 - F.cosine_similarity(latent_tensor, sensor_tensor, dim=1)
    l2_gap = torch.linalg.norm(latent_tensor - sensor_tensor, dim=1)
    latent_centroids, centroid_classes = _class_centroids(latent_arr, label_arr)
    sensor_centroids, _ = _class_centroids(sensor_arr, label_arr)
    centroid_labels = [f"A{cid + 1:02d}" for cid in centroid_classes]
    latent_centroid_sim = F.cosine_similarity(
        torch.from_numpy(latent_centroids)[:, None, :],
        torch.from_numpy(latent_centroids)[None, :, :],
        dim=-1,
    ).cpu().numpy()
    sensor_centroid_sim = F.cosine_similarity(
        torch.from_numpy(sensor_centroids)[:, None, :],
        torch.from_numpy(sensor_centroids)[None, :, :],
        dim=-1,
    ).cpu().numpy()
    write_similarity_heatmap(
        out_dir / "latent_class_centroid_similarity.png",
        "Skeleton Latent Class-Centroid Cosine Similarity",
        latent_centroid_sim,
        centroid_labels,
    )
    write_similarity_heatmap(
        out_dir / "sensor_class_centroid_similarity.png",
        "Sensor Embedding Class-Centroid Cosine Similarity",
        sensor_centroid_sim,
        centroid_labels,
    )

    diagnostics = {
        "sample_count": int(label_arr.shape[0]),
        "invalid_feature_rows_filtered": invalid_count,
        "invalid_pair_cosines_filtered": int(len(pair_cosines) - pair_cosines_arr.shape[0]),
        "invalid_pair_l2_filtered": int(len(pair_l2s) - pair_l2s_arr.shape[0]),
        "latent_sensor_cosine_mean": float(cosine_gap.mean().item()),
        "latent_sensor_cosine_std": float(cosine_gap.std(unbiased=False).item()),
        "latent_sensor_l2_mean": float(l2_gap.mean().item()),
        "latent_sensor_l2_std": float(l2_gap.std(unbiased=False).item()),
        "latent_within_class_scatter": _mean_within_class_scatter(latent_arr, label_arr),
        "sensor_within_class_scatter": _mean_within_class_scatter(sensor_arr, label_arr),
        "available_projections": projection_status,
        "retrieval_at_1": float(retrieval_hits / max(retrieval_total, 1)),
    }
    if epoch is not None:
        prev_dir = out_dir.parent / f"epoch_{epoch - 1:03d}"
        diagnostics["epoch"] = int(epoch)
        if prev_dir.exists():
            prev_latent_path = prev_dir / "latent_class_centroids.npz"
            prev_sensor_path = prev_dir / "sensor_class_centroids.npz"
            if prev_latent_path.exists() and prev_sensor_path.exists():
                prev_latent = np.load(prev_latent_path)
                prev_sensor = np.load(prev_sensor_path)
                prev_latent_map = {int(cid): centroid for cid, centroid in zip(prev_latent["classes"], prev_latent["centroids"])}
                prev_sensor_map = {int(cid): centroid for cid, centroid in zip(prev_sensor["classes"], prev_sensor["centroids"])}
                shared = sorted(set(prev_latent_map).intersection(centroid_classes))
                if shared:
                    diagnostics["latent_centroid_drift"] = float(
                        np.mean([np.linalg.norm(latent_centroids[centroid_classes.index(cid)] - prev_latent_map[cid]) for cid in shared])
                    )
                    diagnostics["sensor_centroid_drift"] = float(
                        np.mean([np.linalg.norm(sensor_centroids[centroid_classes.index(cid)] - prev_sensor_map[cid]) for cid in shared])
                    )
        diagnostics.setdefault("latent_centroid_drift", 0.0)
        diagnostics.setdefault("sensor_centroid_drift", 0.0)
        update_stage2_embedding_history(
            out_dir.parent,
            epoch=epoch,
            metrics={
                "latent_sensor_cosine_mean": diagnostics["latent_sensor_cosine_mean"],
                "latent_sensor_cosine_std": diagnostics["latent_sensor_cosine_std"],
                "latent_sensor_l2_mean": diagnostics["latent_sensor_l2_mean"],
                "latent_sensor_l2_std": diagnostics["latent_sensor_l2_std"],
                "latent_within_class_scatter": diagnostics["latent_within_class_scatter"],
                "sensor_within_class_scatter": diagnostics["sensor_within_class_scatter"],
                "latent_centroid_drift": diagnostics["latent_centroid_drift"],
                "sensor_centroid_drift": diagnostics["sensor_centroid_drift"],
            },
        )

    np.savez_compressed(out_dir / "latent_features.npz", features=latent_arr, labels=label_arr)
    np.savez_compressed(out_dir / "sensor_features.npz", features=sensor_arr, labels=label_arr)
    np.savez_compressed(out_dir / "latent_class_centroids.npz", centroids=latent_centroids, classes=np.asarray(centroid_classes))
    np.savez_compressed(out_dir / "sensor_class_centroids.npz", centroids=sensor_centroids, classes=np.asarray(centroid_classes))
    write_json(out_dir / "embedding_diagnostics.json", diagnostics)
    _write_histogram(out_dir / "latent_sensor_pair_cosine_hist.png", "Stage2 Paired Latent/Sensor Cosine Similarity", pair_cosines_arr, "Cosine similarity")
    _write_histogram(out_dir / "latent_sensor_pair_l2_hist.png", "Stage2 Paired Latent/Sensor L2 Distance", pair_l2s_arr, "L2 distance", color="#dc2626")
    retrieval_matrix = sensor_arr @ latent_arr.T
    write_heatmap(out_dir / "stage2_similarity_matrix.png", "Stage2 Sensor vs Latent Similarity", retrieval_matrix, "Latent sample", "Sensor sample", cmap="magma")


@torch.no_grad()
def plot_noise_pred_error_by_timestep(
    stage2,
    stage3,
    loader,
    device: torch.device,
    out_dir: Path,
    num_bins: int = 10,
    max_batches: int = 4,
) -> dict:
    """Plot MSE of noise prediction per diffusion timestep bin.

    Reveals which part of the noise schedule the denoiser struggles with.
    High error at large t → denoiser can't handle high noise.
    High error at small t → denoiser fails on fine detail.
    """
    total_timesteps = stage3.diffusion.timesteps
    edges = np.linspace(0, total_timesteps, num_bins + 1, dtype=int)
    bin_centers = [(int(edges[b]) + int(edges[b + 1])) // 2 for b in range(num_bins)]
    bin_mses: list[list[float]] = [[] for _ in range(num_bins)]

    for b in range(num_bins):
        t_center = bin_centers[b]
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = batch["skeleton"].to(device)
            a_hip = batch["A_hip"].to(device)
            a_wrist = batch["A_wrist"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            B = x.shape[0]
            z0 = stage3.encoder(x, gait_metrics=None)
            h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)
            h_tokens, h_global = stage3.augment_conditioning(
                h_tokens=h_tokens,
                h_global=h_global,
                a_hip_stream=a_hip,
                a_wrist_stream=a_wrist,
            )
            t = torch.full((B,), t_center, device=device, dtype=torch.long)
            noise = torch.randn_like(z0)
            zt = stage3.diffusion.q_sample(z0, t, noise=noise)
            pred = stage3.denoiser(zt, t, h_tokens=h_tokens, h_global=h_global, gait_metrics=None)
            mse_per_sample = (pred - noise).pow(2).mean(dim=(1, 2, 3))
            bin_mses[b].extend(mse_per_sample.cpu().tolist())

    mean_mse = [float(np.mean(v)) if v else float("nan") for v in bin_mses]
    std_mse = [float(np.std(v)) if v else float("nan") for v in bin_mses]

    ensure_dir(out_dir)
    write_curve_plot(
        out_dir / "noise_pred_error_by_timestep.png",
        "Noise Prediction MSE by Timestep Bin",
        bin_centers,
        [("Mean MSE", mean_mse, "#2563eb")],
        "Timestep bin center",
        "MSE (pred vs true noise)",
    )
    result = {"bin_centers": bin_centers, "mean_mse_per_bin": mean_mse, "std_mse_per_bin": std_mse}
    write_json(out_dir / "noise_pred_error_by_timestep.json", result)
    return result


@torch.no_grad()
def plot_generation_diversity(
    stage2,
    stage3,
    loader,
    device: torch.device,
    out_dir: Path,
    k_samples: int = 8,
    sample_steps: int = 50,
    sampler: str = "ddim",
    max_batches: int = 2,
) -> dict:
    """Plot pairwise distance histogram of k samples generated from the same IMU input.

    Low mean pairwise distance → mode collapse (model always generates the same output).
    Uses existing _pairwise_distance_rows() helper.
    """
    all_pairwise: list[float] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch["skeleton"].to(device)
        a_hip = batch["A_hip"][0:1].to(device)
        a_wrist = batch["A_wrist"][0:1].to(device)
        gait_metrics = batch["gait_metrics"][0:1].to(device)
        x_ref = x[0:1]
        shape = torch.Size((1, x_ref.shape[1], x_ref.shape[2], stage3.latent_dim))
        h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)

        latents = []
        for seed_i in range(k_samples):
            z = sample_stage3_latents(
                stage3=stage3,
                shape=shape,
                device=device,
                h_tokens=h_tokens,
                h_global=h_global,
                a_hip_stream=a_hip,
                a_wrist_stream=a_wrist,
                gait_metrics=None,
                sample_steps=sample_steps,
                sampler=sampler,
                sample_seed=seed_i,
            )
            latents.append(z.cpu().numpy().reshape(1, -1))

        stacked = np.concatenate(latents, axis=0)  # [K, T*J*D]
        dist_matrix = _pairwise_distance_rows(stacked)
        idx = np.triu_indices(k_samples, k=1)
        all_pairwise.extend(dist_matrix[idx].tolist())

    all_pairwise_arr = np.array(all_pairwise, dtype=np.float32)
    mean_dist = float(np.mean(all_pairwise_arr)) if len(all_pairwise_arr) > 0 else float("nan")
    std_dist = float(np.std(all_pairwise_arr)) if len(all_pairwise_arr) > 0 else float("nan")

    ensure_dir(out_dir)
    _write_histogram(out_dir / "generation_diversity.png", "Generation Diversity (Pairwise Latent Distance)", all_pairwise_arr, "Pairwise L2 distance")
    result = {"mean_pairwise_dist": mean_dist, "std_pairwise_dist": std_dist, "k_samples": k_samples, "n_pairs": len(all_pairwise)}
    write_json(out_dir / "generation_diversity.json", result)
    return result


@torch.no_grad()
def plot_per_class_accuracy(
    stage2,
    stage3,
    loader,
    device: torch.device,
    out_dir: Path,
    sample_steps: int = 50,
    sampler: str = "ddim",
    max_batches: int = 4,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> dict:
    """Grouped bar chart: per-class classifier accuracy on raw real vs generated skeletons.

    Classes where gen_acc << real_acc are the hardest for Stage 3 to reproduce.
    Near-chance gen_acc across all classes → mode collapse or IMU conditioning failure.
    """
    correct_real = [0] * num_classes
    correct_gen = [0] * num_classes
    total = [0] * num_classes
    all_targets: list[np.ndarray] = []
    all_real_preds: list[np.ndarray] = []
    all_generated_preds: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch["skeleton"].to(device)
        a_hip = batch["A_hip"].to(device)
        a_wrist = batch["A_wrist"].to(device)
        gait_metrics = batch["gait_metrics"].to(device)
        y = batch["label"].to(device)
        h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)
        shape = torch.Size((x.shape[0], x.shape[1], x.shape[2], stage3.latent_dim))

        z0_gen = sample_stage3_latents(
            stage3=stage3, shape=shape, device=device,
            h_tokens=h_tokens, h_global=h_global,
            a_hip_stream=a_hip, a_wrist_stream=a_wrist,
            gait_metrics=None,
            sample_steps=sample_steps, sampler=sampler,
        )
        x_hat_gen = stage3.decoder(z0_gen)

        # Real accuracy monitors classifier quality on actual skeletons.
        pred_real = stage3.classifier(x.float()).argmax(1)
        pred_gen = stage3.classifier(x_hat_gen.float()).argmax(1)
        all_targets.append(y.cpu().numpy())
        all_real_preds.append(pred_real.cpu().numpy())
        all_generated_preds.append(pred_gen.cpu().numpy())

        for c in range(num_classes):
            mask = (y == c)
            n = int(mask.sum().item())
            total[c] += n
            correct_real[c] += int((pred_real[mask] == c).sum().item())
            correct_gen[c] += int((pred_gen[mask] == c).sum().item())

    real_acc = [correct_real[c] / max(total[c], 1) for c in range(num_classes)]
    gen_acc = [correct_gen[c] / max(total[c], 1) for c in range(num_classes)]
    overall_real = sum(correct_real) / max(sum(total), 1)
    overall_gen = sum(correct_gen) / max(sum(total), 1)

    ensure_dir(out_dir)
    if plt is not None:
        x_pos = np.arange(num_classes)
        fig, ax = plt.subplots(figsize=(max(10, num_classes), 5))
        ax.bar(x_pos - 0.2, real_acc, width=0.4, color="#2563eb", label=f"Real (overall={overall_real:.3f})")
        ax.bar(x_pos + 0.2, gen_acc, width=0.4, color="#dc2626", label=f"Generated (overall={overall_gen:.3f})")
        ax.axhline(1.0 / num_classes, color="#6b7280", linestyle="--", linewidth=1, label=f"Chance (1/{num_classes})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"A{c+1:02d}" for c in range(num_classes)], rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-Class Classifier Accuracy: Real vs Generated Skeletons")
        ax.legend()
        ax.grid(True, alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "per_class_accuracy.png", dpi=180)
        plt.close(fig)

    result = {
        "overall_real_acc": overall_real,
        "overall_gen_acc": overall_gen,
        "per_class": {
            f"A{c+1:02d}": {"real_acc": real_acc[c], "gen_acc": gen_acc[c], "total_samples": total[c]}
            for c in range(num_classes)
        },
        "real_metric_interpretation": "Classifier accuracy on raw real skeletons; low values can reflect a weak Stage3 classifier when lambda_cls is small.",
        "generated_metric_interpretation": "Classifier accuracy on full reverse-diffusion generations; this is the generation-quality monitoring metric.",
    }
    write_json(out_dir / "per_class_accuracy.json", result)
    if all_targets and all_real_preds:
        y_true = np.concatenate(all_targets, axis=0)
        write_classification_artifacts(
            out_dir,
            "real_classifier",
            y_true,
            np.concatenate(all_real_preds, axis=0),
            num_classes=num_classes,
            title="Stage3 Classifier Confusion Matrix on Real Skeletons",
        )
        write_classification_artifacts(
            out_dir,
            "generated_classifier",
            y_true,
            np.concatenate(all_generated_preds, axis=0),
            num_classes=num_classes,
            title="Stage3 Classifier Confusion Matrix on Generated Skeletons",
        )
    return result


@torch.no_grad()
def plot_latent_distribution_comparison(
    stage2,
    stage3,
    loader,
    device: torch.device,
    out_dir: Path,
    sample_steps: int = 50,
    sampler: str = "ddim",
    max_batches: int = 4,
) -> dict:
    """Compare per-channel statistics of real z0 vs generated z0.

    Large gap in channel std → generated latents are OOD for the decoder,
    causing corrupted skeleton output even if denoiser MSE looks acceptable.
    """
    z0_real_list: list[np.ndarray] = []
    z0_gen_list: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch["skeleton"].to(device)
        a_hip = batch["A_hip"].to(device)
        a_wrist = batch["A_wrist"].to(device)
        gait_metrics = batch["gait_metrics"].to(device)
        h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)
        shape = torch.Size((x.shape[0], x.shape[1], x.shape[2], stage3.latent_dim))

        z0_real = stage3.encoder(x, gait_metrics=None)
        z0_gen = sample_stage3_latents(
            stage3=stage3, shape=shape, device=device,
            h_tokens=h_tokens, h_global=h_global,
            a_hip_stream=a_hip, a_wrist_stream=a_wrist,
            gait_metrics=None,
            sample_steps=sample_steps, sampler=sampler,
        )
        z0_real_list.append(z0_real.cpu().numpy())
        z0_gen_list.append(z0_gen.cpu().numpy())

    z0_real_all = np.concatenate(z0_real_list, axis=0)  # [N, T, J, D]
    z0_gen_all = np.concatenate(z0_gen_list, axis=0)

    real_ch_std = z0_real_all.std(axis=(0, 1, 2))   # [D]
    gen_ch_std = z0_gen_all.std(axis=(0, 1, 2))
    real_ch_mean = z0_real_all.mean(axis=(0, 1, 2))
    gen_ch_mean = z0_gen_all.mean(axis=(0, 1, 2))

    ensure_dir(out_dir)
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(real_ch_std, bins=40, alpha=0.65, color="#2563eb", label="Real z0")
        axes[0].hist(gen_ch_std, bins=40, alpha=0.65, color="#dc2626", label="Generated z0")
        axes[0].set_title("Per-channel Std Distribution")
        axes[0].set_xlabel("Channel std")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)
        axes[1].hist(real_ch_mean, bins=40, alpha=0.65, color="#2563eb", label="Real z0")
        axes[1].hist(gen_ch_mean, bins=40, alpha=0.65, color="#dc2626", label="Generated z0")
        axes[1].set_title("Per-channel Mean Distribution")
        axes[1].set_xlabel("Channel mean")
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)
        fig.suptitle("Latent Distribution: Real z0 vs Generated z0", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "latent_distribution_comparison.png", dpi=180)
        plt.close(fig)

    result = {
        "real_channel_std_mean": float(real_ch_std.mean()),
        "gen_channel_std_mean": float(gen_ch_std.mean()),
        "mean_channel_std_diff": float(np.abs(real_ch_std - gen_ch_std).mean()),
        "real_channel_mean_mean": float(real_ch_mean.mean()),
        "gen_channel_mean_mean": float(gen_ch_mean.mean()),
        "mean_channel_mean_diff": float(np.abs(real_ch_mean - gen_ch_mean).mean()),
    }
    write_json(out_dir / "latent_distribution.json", result)
    return result


@torch.no_grad()
def plot_conditioning_comparison_overlay(
    stage2,
    stage3,
    loader,
    device: torch.device,
    out_dir: Path,
    sample_steps: int = 50,
    sampler: str = "ddim",
    sample_seed: int | None = None,
) -> dict:
    """Visualize real vs conditional vs unconditional generation for a single sample.

    If cond_vs_uncond_l2 ≈ 0 → the model completely ignores IMU conditioning.
    Saves a 3-panel skeleton comparison and L2 distance metrics.
    """
    batch = next(_iter_eval_batches(loader, max_batches=1))
    x = batch["skeleton"][0:1].to(device)
    a_hip = batch["A_hip"][0:1].to(device)
    a_wrist = batch["A_wrist"][0:1].to(device)
    gait_metrics = batch["gait_metrics"][0:1].to(device)

    h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)
    shape = torch.Size((1, x.shape[1], x.shape[2], stage3.latent_dim))

    generator = torch.Generator(device=device).manual_seed(sample_seed) if sample_seed is not None else None

    z0_cond = sample_stage3_latents(
        stage3=stage3, shape=shape, device=device,
        h_tokens=h_tokens, h_global=h_global,
        a_hip_stream=a_hip, a_wrist_stream=a_wrist,
        gait_metrics=None,
        sample_steps=sample_steps, sampler=sampler, sample_seed=sample_seed,
    )
    z0_uncond = stage3.diffusion.p_sample_loop_ddim(
        denoiser=stage3.denoiser, shape=shape, device=device,
        sample_steps=sample_steps, eta=0.0,
        h_tokens=None, h_global=None, gait_metrics=None,
        generator=generator,
    )

    x_hat_cond = stage3.decoder(z0_cond)
    x_hat_uncond = stage3.decoder(z0_uncond)

    real_np = x[0].cpu().numpy()
    cond_np = x_hat_cond[0].cpu().numpy()
    uncond_np = x_hat_uncond[0].cpu().numpy()

    ensure_dir(out_dir)
    render_skeleton_panels(
        out_dir / "conditioning_comparison_overlay.png",
        [real_np, cond_np, uncond_np],
        ["Real", "Conditional (IMU)", "Unconditional"],
    )

    cond_vs_real = float(np.linalg.norm(cond_np - real_np))
    uncond_vs_real = float(np.linalg.norm(uncond_np - real_np))
    cond_vs_uncond = float(np.linalg.norm(cond_np - uncond_np))

    result = {
        "cond_vs_real_l2": cond_vs_real,
        "uncond_vs_real_l2": uncond_vs_real,
        "cond_vs_uncond_l2": cond_vs_uncond,
        "interpretation": "cond_vs_uncond_l2 near 0 means model ignores IMU conditioning",
    }
    write_json(out_dir / "conditioning_comparison.json", result)
    return result


@torch.no_grad()
def evaluate_stage3(
    stage2,
    stage3,
    loader,
    device: torch.device,
    out_dir: Path,
    sample_steps: int,
    fps: float,
    epoch: int | None = None,
    sampler: str = "ddim",
    sample_seed: int | None = None,
) -> None:
    real_gait_list: list[np.ndarray] = []
    gen_gait_list: list[np.ndarray] = []
    gen_sequences: list[np.ndarray] = []
    reconstruction_rows: list[dict[str, object]] = []
    sensitivity_pairs: list[dict[str, float]] = []
    sample_real: np.ndarray | None = None
    sample_gen: np.ndarray | None = None

    for batch_idx, batch in enumerate(_iter_eval_batches(loader, max_batches=4)):
        x = batch["skeleton"].to(device)
        a_hip = batch["A_hip"].to(device)
        a_wrist = batch["A_wrist"].to(device)
        gait_metrics = batch["gait_metrics"].to(device)
        h_tokens, h_global = stage2.aligner(a_hip, a_wrist)
        shape = (x.shape[0], x.shape[1], x.shape[2], stage3.latent_dim)
        z0_gen = sample_stage3_latents(
            stage3=stage3,
            shape=torch.Size(shape),
            device=device,
            h_tokens=h_tokens,
            h_global=h_global,
            a_hip_stream=a_hip,
            a_wrist_stream=a_wrist,
            gait_metrics=None,
            sample_steps=sample_steps,
            sampler=sampler,
            sample_seed=sample_seed,
        )
        x_hat = stage3.decoder(z0_gen)
        gait_gen = compute_gait_metrics_torch(x_hat, fps=fps).cpu().numpy()
        real_gait_list.append(gait_metrics.cpu().numpy())
        gen_gait_list.append(gait_gen)
        if len(gen_sequences) < 3:
            gen_sequences.extend([seq for seq in x_hat.cpu().numpy()[: 3 - len(gen_sequences)]])
        if sample_real is None:
            sample_real = x[0].cpu().numpy()
            sample_gen = x_hat[0].cpu().numpy()
        for idx in range(x.shape[0]):
            reconstruction_rows.append(
                {
                    "sample_index": len(reconstruction_rows),
                    "mpjpe": float(_mpjpe(x_hat[idx : idx + 1], x[idx : idx + 1]).item()),
                    "root_trajectory_error": float(_root_trajectory_error(x_hat[idx : idx + 1], x[idx : idx + 1]).item()),
                    "velocity_error": float(_velocity_error(x_hat[idx : idx + 1], x[idx : idx + 1]).item()),
                    "latent_reconstruction_error": float(F.mse_loss(z0_gen[idx : idx + 1], stage3.encoder(x[idx : idx + 1])).item()),
                    "gait_error": float(F.mse_loss(torch.from_numpy(gait_gen[idx : idx + 1]), gait_metrics[idx : idx + 1].cpu()).item()),
                }
            )

        if batch_idx == 0 and len(loader.dataset) >= 2:
            sample_a = loader.dataset[0]
            sample_b = loader.dataset[min(100, len(loader.dataset) - 1)]
            a_hip_fixed = sample_a["A_hip"].unsqueeze(0).to(device)
            a_wrist_fixed = sample_a["A_wrist"].unsqueeze(0).to(device)
            a_hip_other = sample_b["A_hip"].unsqueeze(0).to(device)
            a_wrist_other = sample_b["A_wrist"].unsqueeze(0).to(device)
            # Per-sample gait metrics to ensure generation is conditioned on the
            # same input that was used during training (fixes gait_metrics=None path).
            gait_a = sample_a["gait_metrics"].unsqueeze(0).to(device)
            gait_b = sample_b["gait_metrics"].unsqueeze(0).to(device)

            def _generate(
                hip_tensor: torch.Tensor,
                wrist_tensor: torch.Tensor,
                sample_gait: torch.Tensor,
            ) -> np.ndarray:
                s_tokens, s_global = stage2.aligner(hip_tensor, wrist_tensor)
                x_ref = sample_a["skeleton"].unsqueeze(0)
                z0_local = sample_stage3_latents(
                    stage3=stage3,
                    shape=torch.Size((1, x_ref.shape[1], x_ref.shape[2], stage3.latent_dim)),
                    device=device,
                    h_tokens=s_tokens,
                    h_global=s_global,
                    a_hip_stream=hip_tensor,
                    a_wrist_stream=wrist_tensor,
                    gait_metrics=None,
                    sample_steps=sample_steps,
                    sampler=sampler,
                    sample_seed=sample_seed,
                )
                return stage3.decoder(z0_local).cpu().numpy()[0]

            gen_a1 = _generate(a_hip_fixed, a_wrist_fixed, gait_a)
            gen_a2 = _generate(a_hip_fixed, a_wrist_fixed, gait_a)
            gen_b = _generate(a_hip_other, a_wrist_other, gait_b)
            sensitivity_pairs.append(
                {
                    "repeatability_error": float(np.abs(gen_a1 - gen_a2).max()),
                    "imu_distance": float(
                        torch.linalg.norm(torch.cat([a_hip_fixed.flatten(), a_wrist_fixed.flatten()]) - torch.cat([a_hip_other.flatten(), a_wrist_other.flatten()])).item()
                    ),
                    "output_distance": float(np.linalg.norm(gen_a1.reshape(-1) - gen_b.reshape(-1))),
                }
            )

    if not gen_gait_list or not reconstruction_rows:
        return
    real_gait = np.concatenate(real_gait_list, axis=0)
    gen_gait = np.concatenate(gen_gait_list, axis=0)
    rows = []
    for i, name in enumerate(GAIT_METRIC_NAMES):
        rows.append(
            {
                "metric_name": name,
                "real_mean": float(real_gait[:, i].mean()),
                "real_std": float(real_gait[:, i].std()),
                "generated_mean": float(gen_gait[:, i].mean()),
                "generated_std": float(gen_gait[:, i].std()),
            }
        )
    write_csv(
        out_dir / "stage3_reconstruction_summary.csv",
        reconstruction_rows,
        ["sample_index", "mpjpe", "root_trajectory_error", "velocity_error", "latent_reconstruction_error", "gait_error"],
    )
    write_json(
        out_dir / "stage3_reconstruction_metrics.json",
        {
            "mpjpe_mean": float(np.mean([row["mpjpe"] for row in reconstruction_rows])),
            "root_trajectory_error_mean": float(np.mean([row["root_trajectory_error"] for row in reconstruction_rows])),
            "velocity_error_mean": float(np.mean([row["velocity_error"] for row in reconstruction_rows])),
            "latent_reconstruction_error_mean": float(np.mean([row["latent_reconstruction_error"] for row in reconstruction_rows])),
            "gait_error_mean": float(np.mean([row["gait_error"] for row in reconstruction_rows])),
        },
    )
    write_csv(out_dir / "generated_gait_metric_summary.csv", rows, ["metric_name", "real_mean", "real_std", "generated_mean", "generated_std"])
    if epoch is not None:
        update_stage3_metric_history(out_dir.parent, epoch, rows)
    write_hist_grid(out_dir / "real_vs_generated_gait_distributions.png", "Real vs Generated Gait Metric Distributions", real_gait, gen_gait, GAIT_METRIC_NAMES)
    write_scatter(out_dir / "real_vs_generated_speed_vs_com_fore_aft.png", "Real vs Generated: Walking Speed vs Mean CoM Fore-Aft", real_gait[:, 6], real_gait[:, 0], gen_gait[:, 6], gen_gait[:, 0], "Mean Walking Speed", "Mean CoM Fore-Aft")
    _write_histogram(out_dir / "stage3_mpjpe_hist.png", "Stage3 MPJPE Distribution", np.asarray([row["mpjpe"] for row in reconstruction_rows], dtype=np.float32), "MPJPE")
    if sample_real is not None and sample_gen is not None:
        render_skeleton_panels(out_dir / "stage3_real_vs_generated_panels.png", [sample_real, sample_gen], ["Real", "Generated"])
        _write_overlay_plot(out_dir / "stage3_root_trajectory_overlay.png", "Root Trajectory Overlay", sample_real[:, 0, 0], sample_gen[:, 0, 0], "Root X")
        real_speed = np.linalg.norm(sample_real[1:, 0, :] - sample_real[:-1, 0, :], axis=-1)
        gen_speed = np.linalg.norm(sample_gen[1:, 0, :] - sample_gen[:-1, 0, :], axis=-1)
        _write_overlay_plot(out_dir / "stage3_velocity_overlay.png", "Root Velocity Overlay", real_speed, gen_speed, "Speed")
        joint_error = np.linalg.norm(sample_gen - sample_real, axis=-1)
        write_heatmap(out_dir / "stage3_joint_error_heatmap_sample_0.png", "Stage3 Joint Reconstruction Error", joint_error.T, "Frame", "Joint", cmap="magma")
    if sensitivity_pairs:
        write_json(
            out_dir / "stage3_repeatability.json",
            {
                "repeatability_error": float(np.mean([row["repeatability_error"] for row in sensitivity_pairs])),
                "imu_distance_mean": float(np.mean([row["imu_distance"] for row in sensitivity_pairs])),
                "output_distance_mean": float(np.mean([row["output_distance"] for row in sensitivity_pairs])),
            },
        )
        if plt is not None:
            ensure_dir(out_dir)
            plt.figure(figsize=(8, 6))
            plt.scatter(
                [row["imu_distance"] for row in sensitivity_pairs],
                [row["output_distance"] for row in sensitivity_pairs],
                color="#2563eb",
                s=50,
            )
            plt.title("IMU Distance vs Output Distance")
            plt.xlabel("IMU distance")
            plt.ylabel("Generated output distance")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / "stage3_sensitivity_scatter.png", dpi=180)
            plt.close()
    if gen_sequences:
        render_skeleton_panels(out_dir / "generated_motion_examples.png", gen_sequences, [f"sample_{i}" for i in range(len(gen_sequences))])
        for idx, seq in enumerate(gen_sequences):
            save_skeleton_gif(seq, out_dir / f"generated_motion_example_{idx}.gif", fps=max(1, int(round(fps / 2.5))))

    per_class_result: dict[str, object] | None = None
    try:
        per_class_result = plot_per_class_accuracy(
            stage2,
            stage3,
            loader,
            device,
            out_dir,
            sample_steps=sample_steps,
            sampler=sampler,
            max_batches=4,
        )
    except Exception:
        per_class_result = None
    if epoch is not None and per_class_result is not None:
        update_stage3_eval_history(
            out_dir.parent,
            epoch=epoch,
            eval_acc_cls_real=float(per_class_result["overall_real_acc"]),
            eval_acc_cls_generated=float(per_class_result["overall_gen_acc"]),
        )

    # Extended diagnostics — each runs independently; a failure in one does not abort the others.
    for _diag_fn, _diag_kwargs in [
        (plot_noise_pred_error_by_timestep,   {"num_bins": 10, "max_batches": 4}),
        (plot_generation_diversity,           {"k_samples": 8, "sample_steps": sample_steps, "sampler": sampler, "max_batches": 2}),
        (plot_latent_distribution_comparison, {"sample_steps": sample_steps, "sampler": sampler, "max_batches": 4}),
        (plot_conditioning_comparison_overlay,{"sample_steps": sample_steps, "sampler": sampler, "sample_seed": sample_seed}),
    ]:
        try:
            _diag_fn(stage2, stage3, loader, device, out_dir, **_diag_kwargs)
        except Exception:
            pass
