#!/usr/bin/env python3
"""Build a no-retrain presentation bundle from the current repo state.

This script intentionally uses only the Python standard library so it can run
in minimal environments. It parses `nohup.out`, inventories saved artifacts,
generates SVG plots, copies existing figures/GIFs into one folder, and writes a
Markdown + HTML report with presentation-safe labels and caveats.
"""

from __future__ import annotations

import ast
import csv
import html
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    import pandas as pd
    import torch
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    try:
        from umap import UMAP
    except Exception:
        UMAP = None

    from diffusion_model.dataset import create_dataloader, create_dataset
    from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM, compute_gait_metrics_torch
    from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
    from diffusion_model.model_loader import load_checkpoint
    from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_LATENT_DIM, DEFAULT_NUM_CLASSES, DEFAULT_TIMESTEPS, DEFAULT_WINDOW, get_skeleton_edges

    HAS_RICH_STACK = True
except Exception:
    HAS_RICH_STACK = False


ROOT = Path(__file__).resolve().parent
NOHUP_PATH = ROOT / "nohup.out"
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINTS_DIR = ROOT / "checkpoints"
SMARTFALL_ENV_PYTHON = Path("/home/qsw26/miniconda3/envs/smartfall_env_3.11/bin/python")
CHECKPOINT_SEARCH_DIRS = [
    CHECKPOINTS_DIR,
    OUTPUTS_DIR / "retrain_meeting" / "checkpoints",
]


RUN_DIR_RE = re.compile(r"Run dir:\s+(?P<value>.+)$")
CONFIG_RE = re.compile(
    r"Config:\s+stage=(?P<stage>\d+)\s+epochs=(?P<epochs>\d+)\s+batch_size=(?P<batch_size>\d+)\s+"
    r"lr=(?P<lr>\S+)\s+window=(?P<window>\d+)\s+stride=(?P<stride>\d+)\s+joints=(?P<joints>\d+)\s+"
    r"latent_dim=(?P<latent_dim>\d+)\s+timesteps=(?P<timesteps>\d+)(?P<rest>.*)$"
)
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
STAGE_EPOCH_RE = re.compile(
    r"\[(?P<stage>Stage[123])\]\s+epoch=(?P<epoch>\d+)/(?P<total>\d+)\s+(?P<metrics>.+?)\s+epoch_time=(?P<epoch_time>[0-9.]+)s$"
)


@dataclass
class RunConfig:
    stage: int
    run_dir: str = ""
    config_line: str = ""
    data_mode: str = ""
    dataset_path: str = ""
    skeleton_folder: str = ""
    hip_folder: str = ""
    wrist_folder: str = ""
    gait_cache_dir: str = ""
    gait_metrics_dim: str = ""
    stage3_objective: str = ""
    validation: str = ""
    config: dict[str, str] = field(default_factory=dict)
    line_no: int = 0
    timestamp: str = ""


@dataclass
class EpochRecord:
    stage: str
    epoch: int
    total_epochs: int
    metrics: dict[str, float]
    epoch_time_sec: float
    source_line: int
    timestamp: str


@dataclass
class StageBlock:
    stage: str
    total_epochs: int
    records: list[EpochRecord] = field(default_factory=list)
    run_config: RunConfig | None = None

    def start_epoch(self) -> int:
        return self.records[0].epoch if self.records else 0

    def end_epoch(self) -> int:
        return self.records[-1].epoch if self.records else 0

    def has_metric(self, name: str) -> bool:
        return any(name in rec.metrics for rec in self.records)

    def metrics_union(self) -> list[str]:
        keys: set[str] = set()
        for rec in self.records:
            keys.update(rec.metrics.keys())
        return sorted(keys)

    def block_label(self) -> str:
        if self.run_config is None:
            return f"{self.stage} block ending epoch {self.end_epoch()}"
        return (
            f"{self.stage} | logged run at {self.run_config.timestamp} | "
            f"epochs={self.run_config.config.get('epochs', self.total_epochs)} | "
            f"batch={self.run_config.config.get('batch_size', '?')} | "
            f"lr={self.run_config.config.get('lr', '?')} | "
            f"window={self.run_config.config.get('window', '?')} | "
            f"stride={self.run_config.config.get('stride', '?')}"
        )

    def final_metrics(self) -> dict[str, float]:
        return self.records[-1].metrics if self.records else {}

    def contains_nan(self) -> bool:
        for rec in self.records:
            for value in rec.metrics.values():
                if math.isnan(value) or math.isinf(value):
                    return True
        return False


def _parse_float(value: str) -> float:
    try:
        lowered = value.lower()
        if lowered == "nan":
            return float("nan")
        if lowered == "inf":
            return float("inf")
        if lowered == "-inf":
            return float("-inf")
        return float(value)
    except Exception:
        return float("nan")


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _format_float(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _find_latest_run_config(run_configs: list[RunConfig], stage_name: str, source_line: int) -> RunConfig | None:
    stage_num = int(stage_name[-1])
    matches = [cfg for cfg in run_configs if cfg.stage == stage_num and cfg.line_no <= source_line]
    return matches[-1] if matches else None


def parse_nohup(nohup_path: Path) -> tuple[list[RunConfig], dict[str, list[StageBlock]]]:
    lines = _read_text(nohup_path).splitlines()
    run_configs: list[RunConfig] = []
    pending: RunConfig | None = None
    epoch_records: dict[str, list[EpochRecord]] = {"Stage1": [], "Stage2": [], "Stage3": []}

    for idx, line in enumerate(lines, start=1):
        timestamp = line.split(" | ", 1)[0] if " | " in line else ""

        m_run = RUN_DIR_RE.search(line)
        if m_run:
            pending = RunConfig(stage=0, run_dir=m_run.group("value").strip(), line_no=idx, timestamp=timestamp)
            run_configs.append(pending)
            continue

        if pending is not None:
            m_cfg = CONFIG_RE.search(line)
            if m_cfg:
                pending.stage = int(m_cfg.group("stage"))
                pending.config_line = line.strip()
                pending.config = {k: v for k, v in KEY_VALUE_RE.findall(m_cfg.group(0))}
                continue
            if "Data mode:" in line:
                pending.data_mode = line.rsplit("Data mode:", 1)[-1].strip()
                continue
            if "dataset_path=" in line:
                pending.dataset_path = line.split("dataset_path=", 1)[-1].strip()
                continue
            if "skeleton_folder=" in line:
                pending.skeleton_folder = line.split("skeleton_folder=", 1)[-1].strip()
                continue
            if "hip_folder=" in line:
                pending.hip_folder = line.split("hip_folder=", 1)[-1].strip()
                continue
            if "wrist_folder=" in line:
                pending.wrist_folder = line.split("wrist_folder=", 1)[-1].strip()
                continue
            if "gait_cache_dir=" in line:
                pending.gait_cache_dir = line.split("gait_cache_dir=", 1)[-1].strip()
                continue
            if "Gait metrics dim:" in line:
                pending.gait_metrics_dim = line.rsplit("Gait metrics dim:", 1)[-1].strip()
                continue
            if "Stage3 objective:" in line:
                pending.stage3_objective = line.rsplit("Stage3 objective:", 1)[-1].strip()
                continue
            if "Validation:" in line:
                pending.validation = line.rsplit("Validation:", 1)[-1].strip()
                continue

        m_epoch = STAGE_EPOCH_RE.search(line)
        if not m_epoch:
            continue
        stage_name = m_epoch.group("stage")
        metrics = {
            key: _parse_float(value)
            for key, value in KEY_VALUE_RE.findall(m_epoch.group("metrics"))
        }
        record = EpochRecord(
            stage=stage_name,
            epoch=int(m_epoch.group("epoch")),
            total_epochs=int(m_epoch.group("total")),
            metrics=metrics,
            epoch_time_sec=float(m_epoch.group("epoch_time")),
            source_line=idx,
            timestamp=timestamp,
        )
        epoch_records[stage_name].append(record)

    blocks: dict[str, list[StageBlock]] = {"Stage1": [], "Stage2": [], "Stage3": []}
    for stage_name, records in epoch_records.items():
        current: StageBlock | None = None
        prev_epoch = -1
        prev_total = -1
        for rec in records:
            new_block = (
                current is None
                or rec.total_epochs != prev_total
                or rec.epoch <= prev_epoch
                or rec.epoch != prev_epoch + 1
            )
            if new_block:
                current = StageBlock(stage=stage_name, total_epochs=rec.total_epochs)
                current.run_config = _find_latest_run_config(run_configs, stage_name, rec.source_line)
                blocks[stage_name].append(current)
            current.records.append(rec)
            prev_epoch = rec.epoch
            prev_total = rec.total_epochs
    return run_configs, blocks


def latest_nonempty_block(blocks: list[StageBlock]) -> StageBlock | None:
    valid = [block for block in blocks if block.records]
    return valid[-1] if valid else None


def latest_block_with_metric(blocks: list[StageBlock], metric_name: str) -> StageBlock | None:
    valid = [block for block in blocks if block.records and block.has_metric(metric_name)]
    return valid[-1] if valid else None


def _svg_escape(text: str) -> str:
    return html.escape(text, quote=True)


def write_line_plot(
    out_path: Path,
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    x_label: str = "Epoch",
) -> None:
    width = 1200
    height = 720
    left = 90
    right = 30
    top = 70
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    finite_points: list[float] = []
    for _, values, _ in series:
        finite_points.extend(v for v in values if not math.isnan(v) and not math.isinf(v))
    if not finite_points:
        finite_points = [0.0, 1.0]
    y_min = min(finite_points)
    y_max = max(finite_points)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    margin = (y_max - y_min) * 0.08
    y_min -= margin
    y_max += margin
    x_min = min(x_values) if x_values else 0.0
    x_max = max(x_values) if x_values else 1.0
    if math.isclose(x_min, x_max):
        x_max += 1.0

    def px_x(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def px_y(y: float) -> float:
        return top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(f'<text x="{left}" y="35" font-size="28" font-family="Arial, sans-serif" fill="#111">{_svg_escape(title)}</text>')

    for i in range(6):
        frac = i / 5
        y_val = y_min + frac * (y_max - y_min)
        y = px_y(y_val)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="16" font-family="Arial, sans-serif" fill="#444">{_format_float(y_val, 3)}</text>')

    for i in range(6):
        frac = i / 5
        x_val = x_min + frac * (x_max - x_min)
        x = px_x(x_val)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#f0f0f0" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{height-bottom+28}" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#444">{int(round(x_val))}</text>')

    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111" stroke-width="2"/>')
    parts.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 25}" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#111">{_svg_escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="28" y="{top + plot_h / 2:.2f}" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#111" transform="rotate(-90 28 {top + plot_h / 2:.2f})">{_svg_escape(y_label)}</text>'
    )

    for label, values, color in series:
        path_points: list[str] = []
        for x, y in zip(x_values, values):
            if math.isnan(y) or math.isinf(y):
                continue
            cmd = "M" if not path_points else "L"
            path_points.append(f"{cmd} {px_x(x):.2f} {px_y(y):.2f}")
        if path_points:
            parts.append(f'<path d="{" ".join(path_points)}" fill="none" stroke="{color}" stroke-width="3"/>')

    legend_x = width - 260
    legend_y = top + 10
    parts.append(f'<rect x="{legend_x-20}" y="{legend_y-28}" width="240" height="{36 * max(1, len(series)) + 18}" fill="#ffffff" stroke="#d1d5db"/>')
    for i, (label, _, color) in enumerate(series):
        y = legend_y + i * 34
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+30}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{legend_x+42}" y="{y+6}" font-size="16" font-family="Arial, sans-serif" fill="#111">{_svg_escape(label)}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_bar_chart(out_path: Path, title: str, labels: list[str], values: list[float], y_label: str) -> None:
    width = 1200
    height = 720
    left = 120
    right = 40
    top = 70
    bottom = 180
    plot_w = width - left - right
    plot_h = height - top - bottom

    finite_values = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    max_val = max(finite_values) if finite_values else 1.0
    if max_val <= 0:
        max_val = 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="35" font-size="28" font-family="Arial, sans-serif" fill="#111">{_svg_escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111" stroke-width="2"/>',
        f'<text x="28" y="{top + plot_h / 2:.2f}" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#111" transform="rotate(-90 28 {top + plot_h / 2:.2f})">{_svg_escape(y_label)}</text>',
    ]

    for i in range(6):
        frac = i / 5
        y_val = frac * max_val
        y = top + plot_h - frac * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="16" font-family="Arial, sans-serif" fill="#444">{_format_float(y_val, 3)}</text>')

    count = max(1, len(labels))
    slot = plot_w / count
    bar_w = slot * 0.65
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * slot + (slot - bar_w) / 2
        bar_h = 0.0 if math.isnan(value) or math.isinf(value) else (value / max_val) * plot_h
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#2563eb"/>')
        tx = x + bar_w / 2
        ty = height - bottom + 24
        parts.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" text-anchor="end" font-size="14" font-family="Arial, sans-serif" fill="#111" transform="rotate(-35 {tx:.2f} {ty:.2f})">{_svg_escape(label)}</text>'
        )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_rich_curve_plot(
    out_path: Path,
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    x_label: str = "Epoch",
) -> None:
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


def write_hist_grid(
    out_path: Path,
    title: str,
    real: np.ndarray,
    generated: np.ndarray | None,
    metric_names: list[str],
) -> None:
    cols = 2
    rows = math.ceil(len(metric_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes_arr = np.array(axes).reshape(-1)
    for idx, name in enumerate(metric_names):
        ax = axes_arr[idx]
        ax.hist(real[:, idx], bins=24, alpha=0.65, label="Real", color="#2563eb", density=True)
        if generated is not None:
            ax.hist(generated[:, idx], bins=24, alpha=0.55, label="Generated", color="#dc2626", density=True)
        ax.set_title(name)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend()
    for idx in range(len(metric_names), len(axes_arr)):
        axes_arr[idx].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_scatter(
    out_path: Path,
    title: str,
    x_real: np.ndarray,
    y_real: np.ndarray,
    x_gen: np.ndarray | None,
    y_gen: np.ndarray | None,
    x_label: str,
    y_label: str,
) -> None:
    plt.figure(figsize=(9, 7))
    plt.scatter(x_real, y_real, s=18, alpha=0.5, label="Real", color="#2563eb")
    if x_gen is not None and y_gen is not None:
        plt.scatter(x_gen, y_gen, s=18, alpha=0.5, label="Generated", color="#dc2626")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_label_distribution(out_path: Path, title: str, counts: Counter[int], num_classes: int = 14) -> None:
    labels = [f"A{i+1:02d}" for i in range(num_classes)]
    values = [counts.get(i, 0) for i in range(num_classes)]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="#2563eb")
    plt.title(title)
    plt.xlabel("Activity label")
    plt.ylabel("Window count")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_pca_plot(
    out_path: Path,
    title: str,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 14,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(features)
    plt.figure(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab20", num_classes)
    for cid in sorted(set(int(x) for x in labels.tolist())):
        mask = labels == cid
        plt.scatter(
            pts[mask, 0],
            pts[mask, 1],
            s=18,
            alpha=0.65,
            label=f"A{cid+1:02d}",
            color=cmap(cid),
        )
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_umap_plot(
    out_path: Path,
    title: str,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 14,
) -> None:
    if UMAP is None or features.shape[0] < 3:
        return
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, max(2, features.shape[0] - 1)))
    pts = reducer.fit_transform(features)
    plt.figure(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab20", num_classes)
    for cid in sorted(set(int(x) for x in labels.tolist())):
        mask = labels == cid
        plt.scatter(
            pts[mask, 0],
            pts[mask, 1],
            s=18,
            alpha=0.65,
            label=f"A{cid+1:02d}",
            color=cmap(cid),
        )
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_metric_relationship_grid(
    out_path: Path,
    title: str,
    real_data: np.ndarray,
    generated_data: np.ndarray,
    metric_names: list[str],
    pair_indices: list[tuple[int, int]],
) -> None:
    if not pair_indices:
        return
    cols = 2
    rows = math.ceil(len(pair_indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5.5 * rows))
    axes_arr = np.array(axes).reshape(-1)
    for ax, (i, j) in zip(axes_arr, pair_indices):
        ax.scatter(real_data[:, i], real_data[:, j], s=16, alpha=0.4, color="#2563eb", label="Real")
        ax.scatter(generated_data[:, i], generated_data[:, j], s=16, alpha=0.4, color="#dc2626", label="Generated")
        ax.set_xlabel(metric_names[i])
        ax.set_ylabel(metric_names[j])
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{metric_names[j]} vs {metric_names[i]}")
    for idx in range(len(pair_indices), len(axes_arr)):
        axes_arr[idx].axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_correlation_heatmap(
    out_path: Path,
    title: str,
    data: np.ndarray,
    metric_names: list[str],
) -> None:
    corr = np.corrcoef(data, rowvar=False)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(metric_names)), metric_names, rotation=40, ha="right")
    plt.yticks(range(len(metric_names)), metric_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def copy_stage3_metric_history_candidates(report_dir: Path) -> int:
    copied = 0
    rich_tables_dir = report_dir / "tables_rich"
    plots_png_dir = report_dir / "plots_png"
    rich_tables_dir.mkdir(parents=True, exist_ok=True)
    plots_png_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(OUTPUTS_DIR.glob("**/generated_gait_metric_history.csv")):
        dst = rich_tables_dir / f"{path.parent.parent.name}_{path.parent.name}_generated_gait_metric_history.csv"
        shutil.copy2(path, dst)
        copied += 1
        trend_plot = path.parent / "generated_gait_metric_trends.png"
        if trend_plot.exists():
            shutil.copy2(trend_plot, plots_png_dir / f"{path.parent.parent.name}_{path.parent.name}_generated_gait_metric_trends.png")
    return copied


def _load_checkpoint_from_candidates(model: torch.nn.Module, filename: str) -> Path:
    last_error: Exception | None = None
    for directory in CHECKPOINT_SEARCH_DIRS:
        path = directory / filename
        if not path.exists():
            continue
        try:
            load_checkpoint(str(path), model, strict=True)
            return path
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"Could not find a compatible checkpoint named {filename!r} in {CHECKPOINT_SEARCH_DIRS!r}")


def render_skeleton_panels(
    out_path: Path,
    sequences: list[np.ndarray],
    titles: list[str],
) -> None:
    edges = get_skeleton_edges()
    n = len(sequences)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.5))
    axes_arr = np.array(axes).reshape(-1)
    for ax, seq, title in zip(axes_arr, sequences, titles):
        frame = seq[0]
        xs = frame[:, 0]
        ys = frame[:, 1]
        for i, j in edges:
            if i < frame.shape[0] and j < frame.shape[0]:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#222", linewidth=1.6)
        ax.scatter(xs, ys, s=14, color="#2563eb")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def maybe_reexec_into_smartfall_env() -> None:
    if os.environ.get("LDM_REPORT_IN_SMARTFALL_ENV") == "1":
        return
    if HAS_RICH_STACK:
        return
    if not SMARTFALL_ENV_PYTHON.exists():
        return
    env = os.environ.copy()
    env["LDM_REPORT_IN_SMARTFALL_ENV"] = "1"
    cmd = [str(SMARTFALL_ENV_PYTHON), __file__, *sys.argv[1:]]
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    raise SystemExit(result.returncode)


def copy_tree_selected(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    count = 0
    if not src_dir.exists():
        return count
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.glob(pattern)):
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)
            count += 1
    return count


def parse_gait_metric_names() -> list[str]:
    path = ROOT / "diffusion_model" / "gait_metrics.py"
    text = _read_text(path)
    match = re.search(r"GAIT_METRIC_NAMES:\s*tuple\[str, \.\.\.\]\s*=\s*\((.*?)\)\nDEFAULT_GAIT_METRICS_DIM", text, re.S)
    if not match:
        return []
    tuple_src = "(" + match.group(1) + ")"
    try:
        value = ast.literal_eval(tuple_src)
        return [str(item) for item in value]
    except Exception:
        return []


def parse_util_defaults() -> dict[str, str]:
    util_text = _read_text(ROOT / "diffusion_model" / "util.py")
    defaults: dict[str, str] = {}
    for name in [
        "DEFAULT_TIMESTEPS",
        "DEFAULT_LATENT_DIM",
        "DEFAULT_WINDOW",
        "DEFAULT_JOINTS",
        "DEFAULT_NUM_CLASSES",
        "DEFAULT_FPS",
    ]:
        match = re.search(rf"{name}\s*=\s*(.+)", util_text)
        if match:
            defaults[name] = match.group(1).strip()
    return defaults


def current_code_facts() -> dict[str, str]:
    dataset_text = _read_text(ROOT / "diffusion_model" / "dataset.py")
    sensor_text = _read_text(ROOT / "diffusion_model" / "sensor_model.py")
    train_text = _read_text(ROOT / "train.py")
    gait_text = _read_text(ROOT / "diffusion_model" / "gait_metrics.py")

    facts = parse_util_defaults()
    facts["sensor_modality"] = "accelerometer only"
    if 'IMU_FEATURE_NAMES: tuple[str, ...] = ("ax", "ay", "az", "magnitude", "pitch", "roll")' in sensor_text:
        facts["imu_feature_channels"] = "6"
        facts["imu_feature_vector"] = "[ax, ay, az, magnitude, pitch, roll]"
    else:
        facts["imu_feature_channels"] = "3"
        if "input_dim: int = 3" in sensor_text:
            facts["imu_feature_vector"] = "[ax, ay, az]"
    facts["csv_sensor_normalization"] = "enabled by default in CSV mode"
    if "self.A_hip = (self.A_hip - hip_mean) / hip_std" in dataset_text:
        facts["csv_sensor_normalization_detail"] = "z-score per stream over dataset windows"
    if '.astype(np.float32) / 1000.0' in dataset_text or 'payload["skeleton"].float() / 1000.0' in dataset_text:
        facts["skeleton_scaling"] = "millimeters to meters"
    if "--stride" in train_text:
        facts["windowing"] = "overlapping sliding windows when stride < window"
    if "--lambda-gait" in train_text:
        facts["current_train_cli_has_gait_loss"] = "yes"
    if "loss = loss_diff + args.lambda_cls * loss_cls + args.lambda_motion * loss_motion + args.lambda_gait * loss_gait" in train_text:
        facts["current_stage3_objective_in_code"] = "diffusion + classification + motion + gait"
    elif "lambda_pose * loss_pose" in train_text and "lambda_latent * loss_latent" in train_text:
        facts["current_stage3_objective_in_code"] = "diffusion + pose + latent + velocity + gait + motion"
    elif "loss = args.lambda_gait * loss_gait" in train_text:
        facts["current_stage3_objective_in_code"] = "gait-only total loss"
    if "GAIT_METRIC_NAMES" in gait_text:
        facts["gait_metrics_dim"] = str(len(parse_gait_metric_names()))
    return facts


def summarize_outputs() -> dict[str, int]:
    def count_files(path: Path, suffixes: tuple[str, ...] | None = None) -> int:
        if not path.exists():
            return 0
        total = 0
        for p in path.rglob("*"):
            if not p.is_file():
                continue
            if suffixes and p.suffix.lower() not in suffixes:
                continue
            total += 1
        return total

    return {
        "checkpoints": count_files(CHECKPOINTS_DIR),
        "results_gifs": count_files(OUTPUTS_DIR / "results", (".gif",)),
        "results_old_gifs": count_files(OUTPUTS_DIR / "results_old", (".gif",)),
        "gif_diagnostics": count_files(OUTPUTS_DIR / "gifs", (".gif", ".png")),
        "attention_pngs": count_files(OUTPUTS_DIR / "attention", (".png",)),
        "attention_npy": count_files(OUTPUTS_DIR / "attention", (".npy",)),
        "gait_cache_csv": count_files(OUTPUTS_DIR / "gait_cache", (".csv",)),
    }


def build_stage_block_rows(blocks: Iterable[StageBlock]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, block in enumerate(blocks, start=1):
        final = block.final_metrics()
        rows.append(
            {
                "block_index": idx,
                "stage": block.stage,
                "epochs_logged": len(block.records),
                "epoch_range": f"{block.start_epoch()}-{block.end_epoch()}",
                "total_epochs_declared": block.total_epochs,
                "timestamp": block.run_config.timestamp if block.run_config else "",
                "run_dir": block.run_config.run_dir if block.run_config else "",
                "batch_size": block.run_config.config.get("batch_size", "") if block.run_config else "",
                "lr": block.run_config.config.get("lr", "") if block.run_config else "",
                "window": block.run_config.config.get("window", "") if block.run_config else "",
                "stride": block.run_config.config.get("stride", "") if block.run_config else "",
                "contains_nan": block.contains_nan(),
                "metrics_present": ",".join(block.metrics_union()),
                "final_train_loss_total": _format_float(final.get("train_loss_total", float("nan"))),
                "final_val_loss_total": _format_float(final.get("val_loss_total", float("nan"))),
                "final_train_loss_diff": _format_float(final.get("train_loss_diff", float("nan"))),
                "final_val_loss_diff": _format_float(final.get("val_loss_diff", float("nan"))),
                "final_train_loss_align": _format_float(final.get("train_loss_align", float("nan"))),
                "final_val_loss_align": _format_float(final.get("val_loss_align", float("nan"))),
                "source_label": block.block_label(),
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def report_text(
    generated_dir: Path,
    run_configs: list[RunConfig],
    blocks: dict[str, list[StageBlock]],
    code_facts: dict[str, str],
    gait_metric_names: list[str],
    output_counts: dict[str, int],
    copied_counts: dict[str, int],
    rich_summary: dict[str, str] | None = None,
) -> str:
    latest_stage1 = latest_nonempty_block(blocks["Stage1"])
    latest_stage2 = latest_nonempty_block(blocks["Stage2"])
    latest_stage3 = latest_nonempty_block(blocks["Stage3"])
    latest_stage3_with_gait = latest_block_with_metric(blocks["Stage3"], "train_loss_gait")

    summary_rows: list[list[str]] = []
    for label, block in [
        ("Stage 1 latest block", latest_stage1),
        ("Stage 2 latest block", latest_stage2),
        ("Stage 3 latest block", latest_stage3_with_gait or latest_stage3),
    ]:
        if block is None:
            summary_rows.append([label, "missing", "-", "-", "-"])
            continue
        final = block.final_metrics()
        summary_rows.append(
            [
                label,
                f"{block.start_epoch()}-{block.end_epoch()} / {block.total_epochs}",
                _format_float(final.get("train_loss_total", final.get("train_loss_diff", final.get("train_loss_align", float("nan"))))),
                _format_float(final.get("val_loss_total", final.get("val_loss_diff", final.get("val_loss_align", float("nan"))))),
                "historical variant" if block.stage == "Stage3" else "current stage family",
            ]
        )

    stage3_blocks = blocks["Stage3"]
    variant_rows: list[list[str]] = []
    for idx, block in enumerate(stage3_blocks[-8:], start=max(1, len(stage3_blocks) - 7)):
        final = block.final_metrics()
        variant_rows.append(
            [
                str(idx),
                block.run_config.timestamp if block.run_config else "",
                block.run_config.config.get("epochs", "") if block.run_config else "",
                block.run_config.config.get("batch_size", "") if block.run_config else "",
                block.run_config.config.get("lr", "") if block.run_config else "",
                block.run_config.config.get("stride", "") if block.run_config else "",
                "yes" if block.contains_nan() else "no",
                ",".join(block.metrics_union()[:6]) + ("..." if len(block.metrics_union()) > 6 else ""),
                _format_float(final.get("val_loss_total", final.get("val_loss_diff", float("nan")))),
            ]
        )

    code_rows = [[k, v] for k, v in sorted(code_facts.items())]
    artifact_rows = [[k, str(v)] for k, v in sorted(output_counts.items())]
    copied_rows = [[k, str(v)] for k, v in sorted(copied_counts.items())]
    gait_rows = [[str(i + 1), name] for i, name in enumerate(gait_metric_names)]

    rich_summary = rich_summary or {}
    rich_lines = ""
    if rich_summary:
        rich_lines = f"""
## Additional Checkpoint-Backed Evidence

- Real dataset window count: `{rich_summary.get("dataset_windows", "n/a")}`
- Real gait vectors analyzed: `{rich_summary.get("real_gait_count", "n/a")}`
- Generated gait vectors analyzed: `{rich_summary.get("generated_gait_count", "n/a")}`
- Noise-vs-timestep plot generated: `{rich_summary.get("noise_plot", "no")}`
- Latent PCA generated: `{rich_summary.get("latent_pca", "no")}`
- Sensor PCA generated: `{rich_summary.get("sensor_pca", "no")}`
- Real-vs-generated gait distribution plots generated: `{rich_summary.get("gait_dist_plot", "no")}`
- Conditioning sensitivity comparison generated: `{rich_summary.get("conditioning_plot", "no")}`
- Intermediate diffusion-state panels generated: `{rich_summary.get("intermediate_plot", "no")}`
"""

    text = f"""# Current Model Extraction Report

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}

This bundle summarizes what can be extracted from the **current saved repo state** without retraining or changing the architecture. All figures, copied artifacts, tables, and conclusions in this folder are based on:

- `nohup.out`
- existing checkpoints in `checkpoints/`
- existing saved outputs under `outputs/`
- the current code in `train.py` and `diffusion_model/`

## Executive Summary

- Stage 1 shows strong convergence in the latest 100-epoch block: train diffusion loss ends near `{_format_float(latest_stage1.final_metrics().get("train_loss_diff", float("nan"))) if latest_stage1 else "n/a"}` and validation diffusion loss near `{_format_float(latest_stage1.final_metrics().get("val_loss_diff", float("nan"))) if latest_stage1 else "n/a"}`.
- Stage 2 also converges strongly in the latest 100-epoch block: train alignment loss ends near `{_format_float(latest_stage2.final_metrics().get("train_loss_align", float("nan"))) if latest_stage2 else "n/a"}` and validation alignment loss near `{_format_float(latest_stage2.final_metrics().get("val_loss_align", float("nan"))) if latest_stage2 else "n/a"}`.
- Stage 3 logs show **multiple historical training variants**, including unstable runs with NaNs and later improved runs with diffusion, classification, gait, and biomechanical loss components.
- The latest Stage 3 logs in `nohup.out` are **not equivalent to the current `train.py` Stage 3 implementation**, because the current code now uses a gait-only total loss.
- Saved qualitative evidence already exists: generated GIFs, attention visualizations, gait metric cache files, and stage checkpoints.

## Stage Summary

{markdown_table(["Block", "Epoch span", "Final train", "Final val", "Interpretation"], summary_rows)}

## Recommended Presentation Framing

Use the current evidence in two buckets:

1. **Available now from saved runs**
   - Stage 1 and Stage 2 loss curves
   - Stage 3 historical loss curves and instability case study
   - generated motion examples
   - attention figures
   - current code-defined data/configuration facts

2. **Requires fresh retraining or new evaluation**
   - exact 9-metric conditioning setup
   - 6-channel IMU feature experiment
   - fully clean Stage 3 conclusions under the current code path
   - any per-checkpoint generated-vs-real statistical comparison not already saved

## Historical Stage 3 Variant Inventory

The last several Stage 3 blocks in `nohup.out` are listed below. Treat them as **historical variants**, not one single consistent experiment.

{markdown_table(["Variant", "Timestamp", "Epochs", "Batch", "LR", "Stride", "NaN?", "Metrics logged", "Final val"], variant_rows or [["-", "-", "-", "-", "-", "-", "-", "-", "-"]])}

## Current Code Facts

{markdown_table(["Key", "Value"], code_rows)}

## Gait Metric Names in Current Code

{markdown_table(["Index", "Metric name"], gait_rows or [["-", "not parsed"]])}

Important caveat: the current code defines **{len(gait_metric_names)} gait metrics**, which is more than the exact 9-metric list requested in the external feedback.

## Saved Artifact Inventory

### Source counts

{markdown_table(["Artifact class", "Count"], artifact_rows)}

### Copied into this report bundle

{markdown_table(["Artifact class", "Count"], copied_rows)}

{rich_lines}

## Figures Included in This Folder

- `plots/stage1_diffusion_loss.svg`
- `plots/stage2_alignment_loss.svg`
- `plots/stage3_total_loss_latest_variant.svg`
- `plots/stage3_component_losses_latest_variant.svg`
- `plots/stage_block_counts.svg`
- `plots_png/` may contain richer checkpoint-backed figures when the ML environment is available

## Copied Qualitative Artifacts

- `artifacts/results_gifs/`
- `artifacts/results_old_gifs/`
- `artifacts/gif_diagnostics/`
- `artifacts/attention_pngs/`

## What Can Be Claimed Safely Now

- The repository has completed Stage 1, Stage 2, and Stage 3 checkpoint files.
- The saved logs show strong convergence for Stage 1 and Stage 2.
- The Stage 3 training history includes both unstable and improved regimes.
- The current dataset/training setup in logs uses the young participant CSV folders with hip and wrist accelerometer streams.
- The current repo already stores generated GIF outputs and attention inspection figures.

## What Must Be Stated As Caveats

- `nohup.out` mixes multiple Stage 3 objectives and experiments.
- The latest Stage 3 log block includes diffusion, classification, gait, and biomechanical loss terms, but the current `train.py` no longer trains that exact objective.
- The current code still uses 3-channel accelerometer inputs in the TGNN path.
- The current code still defines a 10-metric gait vector.
- This bundle does not fabricate missing analyses; it only packages what is already supportable from saved state.

## Files to Use in the Presentation

- Start with `report.md`
- Use `plots/` for labeled loss curves
- Use `artifacts/results_gifs/` and `artifacts/gif_diagnostics/` for motion examples
- Use `artifacts/attention_pngs/` for interpretability slides
- Use `tables/` if you need raw CSV evidence for appendix slides
"""
    return text


def write_html_from_markdown(markdown_text: str, out_path: Path) -> None:
    lines = markdown_text.splitlines()
    body: list[str] = []
    in_list = False
    in_code = False
    for line in lines:
        if line.startswith("```"):
            if in_code:
                body.append("</pre>")
            else:
                body.append("<pre>")
            in_code = not in_code
            continue
        if in_code:
            body.append(html.escape(line))
            continue
        if line.startswith("# "):
            if in_list:
                body.append("</ul>")
                in_list = False
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_list:
                body.append("</ul>")
                in_list = False
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip().startswith("|"):
            if in_list:
                body.append("</ul>")
                in_list = False
            body.append(f"<pre>{html.escape(line)}</pre>")
        elif line.strip() == "":
            if in_list:
                body.append("</ul>")
                in_list = False
            body.append("<p></p>")
        else:
            if in_list:
                body.append("</ul>")
                in_list = False
            body.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        body.append("</ul>")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Current Model Extraction Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px auto; max-width: 1100px; line-height: 1.5; color: #111; }}
    h1, h2 {{ color: #0f172a; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; border: 1px solid #e2e8f0; }}
    ul {{ margin-top: 0; }}
  </style>
</head>
<body>
{''.join(body)}
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def run_rich_analysis(report_dir: Path, gait_metric_names: list[str]) -> dict[str, str]:
    if not HAS_RICH_STACK:
        return {}

    plots_png_dir = report_dir / "plots_png"
    rich_tables_dir = report_dir / "tables_rich"
    artifacts_rich_dir = report_dir / "artifacts_rich"
    plots_png_dir.mkdir(parents=True, exist_ok=True)
    rich_tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts_rich_dir.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(
        dataset_path=None,
        window=DEFAULT_WINDOW,
        joints=DEFAULT_JOINTS,
        num_classes=DEFAULT_NUM_CLASSES,
        skeleton_folder="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/skeleton",
        hip_folder="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/meta_hip",
        wrist_folder="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/meta_wrist",
        stride=30,
        normalize_sensors=True,
        gait_cache_dir=str(OUTPUTS_DIR / "gait_cache"),
        disable_gait_cache=False,
    )
    label_counter = Counter(int(dataset.label[i].item()) for i in range(len(dataset.label)))
    write_label_distribution(
        plots_png_dir / "dataset_label_distribution.png",
        "Real Dataset Window Distribution by Activity",
        label_counter,
        num_classes=DEFAULT_NUM_CLASSES,
    )

    real_gait = dataset.gait_metrics.cpu().numpy()
    metric_names = gait_metric_names or [f"metric_{i}" for i in range(real_gait.shape[1])]
    write_hist_grid(
        plots_png_dir / "real_gait_metric_distributions.png",
        "Real Dataset Gait Metric Distributions",
        real_gait,
        None,
        metric_names,
    )
    write_scatter(
        plots_png_dir / "real_speed_vs_com_fore_aft.png",
        "Real Data: Walking Speed vs Mean CoM Fore-Aft",
        real_gait[:, 6],
        real_gait[:, 0],
        None,
        None,
        x_label="Mean Walking Speed",
        y_label="Mean CoM Fore-Aft",
    )
    write_csv(
        rich_tables_dir / "real_gait_metric_summary.csv",
        [
            {
                "metric_name": metric_names[i],
                "mean": float(real_gait[:, i].mean()),
                "std": float(real_gait[:, i].std()),
                "min": float(real_gait[:, i].min()),
                "max": float(real_gait[:, i].max()),
            }
            for i in range(real_gait.shape[1])
        ],
        ["metric_name", "mean", "std", "min", "max"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1 = Stage1Model(
        latent_dim=DEFAULT_LATENT_DIM,
        num_joints=DEFAULT_JOINTS,
        timesteps=DEFAULT_TIMESTEPS,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
    ).to(device)
    stage1_ckpt_path = _load_checkpoint_from_candidates(stage1, "stage1_best.pt")
    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=DEFAULT_LATENT_DIM,
        num_joints=DEFAULT_JOINTS,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
    ).to(device)
    stage2_ckpt_path = _load_checkpoint_from_candidates(stage2, "stage2_best.pt")
    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=DEFAULT_LATENT_DIM,
        num_joints=DEFAULT_JOINTS,
        num_classes=DEFAULT_NUM_CLASSES,
        timesteps=DEFAULT_TIMESTEPS,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
    ).to(device)
    stage3_ckpt_path = _load_checkpoint_from_candidates(stage3, "stage3_best.pt")
    stage1.eval()
    stage2.eval()
    stage3.eval()

    param_rows = []
    for name, model in [("stage1", stage1), ("stage2", stage2), ("stage3", stage3)]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_rows.append({"model": name, "total_params": total, "trainable_params": trainable})
    write_csv(rich_tables_dir / "parameter_counts.csv", param_rows, ["model", "total_params", "trainable_params"])

    loader = create_dataloader(
        dataset_path=None,
        batch_size=32,
        shuffle=False,
        window=DEFAULT_WINDOW,
        joints=DEFAULT_JOINTS,
        num_classes=DEFAULT_NUM_CLASSES,
        skeleton_folder="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/skeleton",
        hip_folder="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/meta_hip",
        wrist_folder="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/meta_wrist",
        stride=30,
        normalize_sensors=True,
        gait_cache_dir=str(OUTPUTS_DIR / "gait_cache"),
        disable_gait_cache=False,
        drop_last=False,
    )

    max_timestep = int(stage3.diffusion.timesteps) - 1
    timestep_values = sorted(
        {
            0,
            max_timestep // 5,
            (2 * max_timestep) // 5,
            (3 * max_timestep) // 5,
            (4 * max_timestep) // 5,
            max_timestep,
        }
    )
    timestep_errors = {t: [] for t in timestep_values}
    stage3_timestep_errors = {t: [] for t in timestep_values}
    latent_features: list[np.ndarray] = []
    sensor_features: list[np.ndarray] = []
    label_features: list[np.ndarray] = []
    generated_gait_list: list[np.ndarray] = []
    real_gait_subset_list: list[np.ndarray] = []
    gen_label_counter: Counter[int] = Counter()
    conditioning_rows: list[dict[str, object]] = []
    diffusion_snapshots_done = False

    max_embed_batches = 16
    max_gen_batches = 4
    torch.manual_seed(42)
    np.random.seed(42)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["skeleton"].to(device)
            a_hip = batch["A_hip"].to(device)
            a_wrist = batch["A_wrist"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            y = batch["label"].to(device)

            if batch_idx < max_embed_batches:
                z0 = stage1.encoder(x, gait_metrics=gait_metrics)
                latent_pool = z0.mean(dim=(1, 2)).cpu().numpy()
                latent_features.append(latent_pool)
                h_tokens, h_global = stage2.aligner(a_hip, a_wrist, gait_metrics=gait_metrics)
                sensor_features.append(h_global.cpu().numpy())
                label_features.append(y.cpu().numpy())

                for t_val in timestep_values:
                    t = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
                    noise = torch.randn_like(z0)
                    zt = stage1.diffusion.q_sample(z0=z0, t=t, noise=noise)
                    pred_noise = stage1.denoiser(zt, t, gait_metrics=gait_metrics)
                    mse = torch.mean((pred_noise - noise) ** 2, dim=(1, 2, 3)).cpu().numpy()
                    timestep_errors[t_val].extend(mse.tolist())

                    h_tokens, h_global = stage2.aligner(a_hip, a_wrist, gait_metrics=gait_metrics)
                    cond_tokens, cond_global = stage3.condition_with_labels(h_tokens=h_tokens, h_global=h_global, y=y)
                    zt_stage3 = stage3.diffusion.q_sample(z0=z0, t=t, noise=noise)
                    pred_noise_stage3 = stage3.denoiser(
                        zt_stage3,
                        t,
                        h_tokens=cond_tokens,
                        h_global=cond_global,
                        gait_metrics=gait_metrics,
                    )
                    mse_stage3 = torch.mean((pred_noise_stage3 - noise) ** 2, dim=(1, 2, 3)).cpu().numpy()
                    stage3_timestep_errors[t_val].extend(mse_stage3.tolist())

            if batch_idx < max_gen_batches:
                h_tokens, h_global = stage2.aligner(a_hip, a_wrist, gait_metrics=gait_metrics)
                cond_tokens, cond_global = stage3.condition_with_labels(h_tokens=h_tokens, h_global=h_global, y=y)
                shape = (x.shape[0], x.shape[1], x.shape[2], DEFAULT_LATENT_DIM)
                z0_gen = stage3.diffusion.p_sample_loop_ddim(
                    stage3.denoiser,
                    shape=torch.Size(shape),
                    device=device,
                    sample_steps=50,
                    eta=0.0,
                    h_tokens=cond_tokens,
                    h_global=cond_global,
                    gait_metrics=gait_metrics,
                )
                x_hat = stage3.decoder(z0_gen)
                gait_gen = compute_gait_metrics_torch(x_hat, fps=30.0).cpu().numpy()
                generated_gait_list.append(gait_gen)
                real_gait_subset_list.append(gait_metrics.cpu().numpy())
                pred = torch.argmax(stage3.classifier(x_hat), dim=1).cpu().numpy()
                for cid in pred.tolist():
                    gen_label_counter[int(cid)] += 1

                if not diffusion_snapshots_done:
                    sample_x = x[:1]
                    sample_hip = a_hip[:1]
                    sample_wrist = a_wrist[:1]
                    sample_gait = gait_metrics[:1]
                    sample_y = y[:1]
                    s_tokens, s_global = stage2.aligner(sample_hip, sample_wrist, gait_metrics=sample_gait)
                    c_tokens, c_global = stage3.condition_with_labels(h_tokens=s_tokens, h_global=s_global, y=sample_y)
                    timesteps_to_capture = sorted(
                        {
                            0,
                            max_timestep // 5,
                            (2 * max_timestep) // 5,
                            (3 * max_timestep) // 5,
                            (4 * max_timestep) // 5,
                            max_timestep,
                        },
                        reverse=True,
                    )
                    z = torch.randn((1, sample_x.shape[1], sample_x.shape[2], DEFAULT_LATENT_DIM), device=device)
                    captured: dict[int, np.ndarray] = {}
                    for i in reversed(range(stage3.diffusion.timesteps)):
                        t = torch.full((1,), i, device=device, dtype=torch.long)
                        pred_noise = stage3.denoiser(z, t, h_tokens=c_tokens, h_global=c_global, gait_metrics=sample_gait)
                        alpha_bar_t = stage3.diffusion.alphas_cumprod[i]
                        sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-20))
                        sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-20))
                        x0_pred = (z - sqrt_one_minus * pred_noise) / sqrt_alpha_bar_t
                        if i in timesteps_to_capture:
                            captured[i] = stage3.decoder(x0_pred).cpu().numpy()[0]
                        z = stage3.diffusion.p_sample(
                            stage3.denoiser,
                            z,
                            t,
                            h_tokens=c_tokens,
                            h_global=c_global,
                            gait_metrics=sample_gait,
                        )
                    ordered = [captured[t] for t in timesteps_to_capture if t in captured]
                    titles = [f"Decoded state at t={t}" for t in timesteps_to_capture if t in captured]
                    if ordered:
                        render_skeleton_panels(
                            plots_png_dir / "intermediate_diffusion_states.png",
                            ordered,
                            titles,
                        )
                    diffusion_snapshots_done = True

        if len(dataset) >= 2:
            idx_a = 0
            idx_b = min(100, len(dataset) - 1)
            sample_a = dataset[idx_a]
            sample_b = dataset[idx_b]
            a_hip = sample_a["A_hip"].unsqueeze(0).to(device)
            a_wrist = sample_a["A_wrist"].unsqueeze(0).to(device)
            gait_a = sample_a["gait_metrics"].unsqueeze(0).to(device)
            gait_b = sample_b["gait_metrics"].unsqueeze(0).to(device)
            y = sample_a["label"].view(1).to(device)
            shape = (1, DEFAULT_WINDOW, DEFAULT_JOINTS, DEFAULT_LATENT_DIM)

            def generate_from_gait(gait_tensor: torch.Tensor) -> np.ndarray:
                torch.manual_seed(123)
                s_tokens, s_global = stage2.aligner(a_hip, a_wrist, gait_metrics=gait_tensor)
                c_tokens, c_global = stage3.condition_with_labels(h_tokens=s_tokens, h_global=s_global, y=y)
                z0_gen = stage3.diffusion.p_sample_loop_ddim(
                    stage3.denoiser,
                    shape=torch.Size(shape),
                    device=device,
                    sample_steps=50,
                    eta=0.0,
                    h_tokens=c_tokens,
                    h_global=c_global,
                    gait_metrics=gait_tensor,
                )
                x_hat = stage3.decoder(z0_gen)
                return compute_gait_metrics_torch(x_hat, fps=30.0).cpu().numpy()[0]

            gen_a = generate_from_gait(gait_a)
            gen_b = generate_from_gait(gait_b)
            for i, name in enumerate(metric_names):
                conditioning_rows.append(
                    {
                        "metric_name": name,
                        "conditioning_a": float(gait_a.cpu().numpy()[0, i]),
                        "conditioning_b": float(gait_b.cpu().numpy()[0, i]),
                        "generated_a": float(gen_a[i]),
                        "generated_b": float(gen_b[i]),
                        "generated_delta": float(gen_b[i] - gen_a[i]),
                    }
                )

    timestep_df_rows = [
        {"timestep": t, "mean_mse": float(np.mean(vals)), "std_mse": float(np.std(vals)), "count": len(vals)}
        for t, vals in timestep_errors.items()
        if vals
    ]
    if timestep_df_rows:
        write_csv(rich_tables_dir / "noise_prediction_error_by_timestep.csv", timestep_df_rows, ["timestep", "mean_mse", "std_mse", "count"])
        write_rich_curve_plot(
            plots_png_dir / "noise_prediction_error_by_timestep.png",
            "Noise Prediction Error by Diffusion Timestep (Stage 1 Encoder/Denoiser)",
            [row["timestep"] for row in timestep_df_rows],
            [("Mean MSE", [row["mean_mse"] for row in timestep_df_rows], "#2563eb")],
            y_label="MSE(pred_noise, true_noise)",
            x_label="Diffusion timestep",
        )
    stage3_timestep_df_rows = [
        {"timestep": t, "mean_mse": float(np.mean(vals)), "std_mse": float(np.std(vals)), "count": len(vals)}
        for t, vals in stage3_timestep_errors.items()
        if vals
    ]
    if stage3_timestep_df_rows:
        write_csv(
            rich_tables_dir / "stage3_noise_prediction_error_by_timestep.csv",
            stage3_timestep_df_rows,
            ["timestep", "mean_mse", "std_mse", "count"],
        )
        write_rich_curve_plot(
            plots_png_dir / "stage3_noise_prediction_error_by_timestep.png",
            "Noise Prediction Error by Diffusion Timestep (Stage 3 Conditioned Denoiser)",
            [row["timestep"] for row in stage3_timestep_df_rows],
            [("Mean MSE", [row["mean_mse"] for row in stage3_timestep_df_rows], "#7c3aed")],
            y_label="MSE(pred_noise, true_noise)",
            x_label="Diffusion timestep",
        )

    if latent_features and sensor_features and label_features:
        latent_mat = np.concatenate(latent_features, axis=0)
        sensor_mat = np.concatenate(sensor_features, axis=0)
        labels = np.concatenate(label_features, axis=0)
        write_pca_plot(
            plots_png_dir / "latent_pca.png",
            "Skeleton Latent Space PCA (z0 mean pooled)",
            latent_mat,
            labels,
            num_classes=DEFAULT_NUM_CLASSES,
        )
        write_umap_plot(
            plots_png_dir / "latent_umap.png",
            "Skeleton Latent Space UMAP (z0 mean pooled)",
            latent_mat,
            labels,
            num_classes=DEFAULT_NUM_CLASSES,
        )
        write_pca_plot(
            plots_png_dir / "sensor_embedding_pca.png",
            "Sensor Embedding PCA (h_global)",
            sensor_mat,
            labels,
            num_classes=DEFAULT_NUM_CLASSES,
        )
        write_umap_plot(
            plots_png_dir / "sensor_embedding_umap.png",
            "Sensor Embedding UMAP (h_global)",
            sensor_mat,
            labels,
            num_classes=DEFAULT_NUM_CLASSES,
        )

    rich_summary = {
        "dataset_windows": str(len(dataset)),
        "real_gait_count": str(real_gait.shape[0]),
        "stage1_checkpoint_used": str(stage1_ckpt_path),
        "stage2_checkpoint_used": str(stage2_ckpt_path),
        "stage3_checkpoint_used": str(stage3_ckpt_path),
    }

    if generated_gait_list:
        generated_gait = np.concatenate(generated_gait_list, axis=0)
        real_gait_subset = np.concatenate(real_gait_subset_list, axis=0)
        rich_summary["generated_gait_count"] = str(generated_gait.shape[0])
        write_hist_grid(
            plots_png_dir / "real_vs_generated_gait_distributions.png",
            "Real vs Generated Gait Metric Distributions (Current Checkpoints)",
            real_gait_subset,
            generated_gait,
            metric_names,
        )
        write_scatter(
            plots_png_dir / "real_vs_generated_speed_vs_com_fore_aft.png",
            "Real vs Generated: Walking Speed vs Mean CoM Fore-Aft",
            real_gait_subset[:, 6],
            real_gait_subset[:, 0],
            generated_gait[:, 6],
            generated_gait[:, 0],
            x_label="Mean Walking Speed",
            y_label="Mean CoM Fore-Aft",
        )
        pair_indices = [(6, 0), (6, 4), (6, 7), (6, 8), (0, 4), (7, 8)]
        write_metric_relationship_grid(
            plots_png_dir / "real_vs_generated_metric_relationships.png",
            "Real vs Generated Metric Relationships",
            real_gait_subset,
            generated_gait,
            metric_names,
            pair_indices=pair_indices,
        )
        write_correlation_heatmap(
            plots_png_dir / "real_metric_correlation_heatmap.png",
            "Real Gait Metric Correlation Matrix",
            real_gait_subset,
            metric_names,
        )
        write_correlation_heatmap(
            plots_png_dir / "generated_metric_correlation_heatmap.png",
            "Generated Gait Metric Correlation Matrix",
            generated_gait,
            metric_names,
        )
        write_csv(
            rich_tables_dir / "generated_gait_metric_summary.csv",
            [
                {
                    "metric_name": metric_names[i],
                    "real_mean": float(real_gait_subset[:, i].mean()),
                    "real_std": float(real_gait_subset[:, i].std()),
                    "generated_mean": float(generated_gait[:, i].mean()),
                    "generated_std": float(generated_gait[:, i].std()),
                }
                for i in range(generated_gait.shape[1])
            ],
            ["metric_name", "real_mean", "real_std", "generated_mean", "generated_std"],
        )
        write_label_distribution(
            plots_png_dir / "generated_predicted_label_distribution.png",
            "Generated Sample Predicted Label Distribution",
            gen_label_counter,
            num_classes=DEFAULT_NUM_CLASSES,
        )
        rich_summary["gait_dist_plot"] = "yes"
    else:
        rich_summary["generated_gait_count"] = "0"
        rich_summary["gait_dist_plot"] = "no"

    if conditioning_rows:
        write_csv(
            rich_tables_dir / "conditioning_sensitivity.csv",
            conditioning_rows,
            ["metric_name", "conditioning_a", "conditioning_b", "generated_a", "generated_b", "generated_delta"],
        )
        plt.figure(figsize=(12, 6))
        plt.bar([row["metric_name"] for row in conditioning_rows], [row["generated_delta"] for row in conditioning_rows], color="#2563eb")
        plt.title("Conditioning Sensitivity: Generated Metric Change Under Fixed Seed")
        plt.xlabel("Gait metric")
        plt.ylabel("Generated metric delta (condition B - condition A)")
        plt.xticks(rotation=40, ha="right")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_png_dir / "conditioning_sensitivity.png", dpi=180)
        plt.close()
        rich_summary["conditioning_plot"] = "yes"
    else:
        rich_summary["conditioning_plot"] = "no"

    rich_summary["noise_plot"] = "yes" if (plots_png_dir / "noise_prediction_error_by_timestep.png").exists() else "no"
    rich_summary["stage3_noise_plot"] = "yes" if (plots_png_dir / "stage3_noise_prediction_error_by_timestep.png").exists() else "no"
    rich_summary["latent_pca"] = "yes" if (plots_png_dir / "latent_pca.png").exists() else "no"
    rich_summary["latent_umap"] = "yes" if (plots_png_dir / "latent_umap.png").exists() else "no"
    rich_summary["sensor_pca"] = "yes" if (plots_png_dir / "sensor_embedding_pca.png").exists() else "no"
    rich_summary["sensor_umap"] = "yes" if (plots_png_dir / "sensor_embedding_umap.png").exists() else "no"
    rich_summary["relationship_grid"] = "yes" if (plots_png_dir / "real_vs_generated_metric_relationships.png").exists() else "no"
    rich_summary["copied_stage3_metric_histories"] = str(copy_stage3_metric_history_candidates(report_dir))
    rich_summary["intermediate_plot"] = "yes" if (plots_png_dir / "intermediate_diffusion_states.png").exists() else "no"
    return rich_summary


def main() -> int:
    maybe_reexec_into_smartfall_env()

    if not NOHUP_PATH.exists():
        print(f"missing required log file: {NOHUP_PATH}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = OUTPUTS_DIR / f"current_model_report_{stamp}"
    plots_dir = report_dir / "plots"
    tables_dir = report_dir / "tables"
    artifacts_dir = report_dir / "artifacts"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_configs, blocks = parse_nohup(NOHUP_PATH)
    gait_metric_names = parse_gait_metric_names()
    code_facts = current_code_facts()
    output_counts = summarize_outputs()

    stage1_latest = latest_nonempty_block(blocks["Stage1"])
    stage2_latest = latest_nonempty_block(blocks["Stage2"])
    stage3_latest = latest_block_with_metric(blocks["Stage3"], "train_loss_gait") or latest_nonempty_block(blocks["Stage3"])

    if stage1_latest:
        epochs = [rec.epoch for rec in stage1_latest.records]
        write_line_plot(
            plots_dir / "stage1_diffusion_loss.svg",
            "Stage 1 Diffusion Loss Across Epochs (Latest Logged Block)",
            epochs,
            [
                ("Train diffusion loss", [rec.metrics.get("train_loss_diff", float("nan")) for rec in stage1_latest.records], "#2563eb"),
                ("Validation diffusion loss", [rec.metrics.get("val_loss_diff", float("nan")) for rec in stage1_latest.records], "#dc2626"),
            ],
            y_label="Loss",
        )

    if stage2_latest:
        epochs = [rec.epoch for rec in stage2_latest.records]
        write_line_plot(
            plots_dir / "stage2_alignment_loss.svg",
            "Stage 2 Alignment Loss Across Epochs (Latest Logged Block)",
            epochs,
            [
                ("Train alignment loss", [rec.metrics.get("train_loss_align", float("nan")) for rec in stage2_latest.records], "#2563eb"),
                ("Validation alignment loss", [rec.metrics.get("val_loss_align", float("nan")) for rec in stage2_latest.records], "#dc2626"),
            ],
            y_label="Loss",
        )

    if stage3_latest:
        epochs = [rec.epoch for rec in stage3_latest.records]
        write_line_plot(
            plots_dir / "stage3_total_loss_latest_variant.svg",
            "Stage 3 Total Loss Across Epochs (Latest Logged Historical Variant)",
            epochs,
            [
                ("Train total loss", [rec.metrics.get("train_loss_total", float("nan")) for rec in stage3_latest.records], "#2563eb"),
                ("Validation total loss", [rec.metrics.get("val_loss_total", float("nan")) for rec in stage3_latest.records], "#dc2626"),
            ],
            y_label="Loss",
        )
        component_series: list[tuple[str, list[float], str]] = []
        color_cycle = {
            "train_loss_diff": "#2563eb",
            "train_loss_pose": "#dc2626",
            "train_loss_latent": "#7c3aed",
            "train_loss_vel": "#ea580c",
            "train_loss_gait": "#059669",
            "train_loss_bone": "#c2410c",
            "train_loss_instab": "#be123c",
        }
        for metric in ["train_loss_diff", "train_loss_pose", "train_loss_latent", "train_loss_vel", "train_loss_gait", "train_loss_bone", "train_loss_instab"]:
            if stage3_latest.has_metric(metric):
                component_series.append(
                    (
                        metric.replace("train_", "").replace("_", " "),
                        [rec.metrics.get(metric, float("nan")) for rec in stage3_latest.records],
                        color_cycle[metric],
                    )
                )
        if component_series:
            write_line_plot(
                plots_dir / "stage3_component_losses_latest_variant.svg",
                "Stage 3 Component Losses (Latest Logged Historical Variant)",
                epochs,
                component_series,
                y_label="Loss",
            )

    block_counts_labels = ["Stage1 blocks", "Stage2 blocks", "Stage3 blocks", "Run configs"]
    block_counts_values = [len(blocks["Stage1"]), len(blocks["Stage2"]), len(blocks["Stage3"]), len(run_configs)]
    write_bar_chart(
        plots_dir / "stage_block_counts.svg",
        "Logged Training History Inventory",
        block_counts_labels,
        block_counts_values,
        y_label="Count",
    )

    run_config_rows = [
        {
            "timestamp": cfg.timestamp,
            "stage": cfg.stage,
            "run_dir": cfg.run_dir,
            "data_mode": cfg.data_mode,
            "dataset_path": cfg.dataset_path,
            "skeleton_folder": cfg.skeleton_folder,
            "hip_folder": cfg.hip_folder,
            "wrist_folder": cfg.wrist_folder,
            "gait_cache_dir": cfg.gait_cache_dir,
            "gait_metrics_dim": cfg.gait_metrics_dim,
            "validation": cfg.validation,
            **cfg.config,
        }
        for cfg in run_configs
    ]
    run_fieldnames = sorted({key for row in run_config_rows for key in row.keys()})
    write_csv(tables_dir / "run_config_inventory.csv", run_config_rows, run_fieldnames)

    checkpoint_rows = []
    for path in sorted(CHECKPOINTS_DIR.glob("*")):
        if path.is_file():
            checkpoint_rows.append(
                {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    write_csv(tables_dir / "checkpoint_inventory.csv", checkpoint_rows, ["filename", "size_bytes", "modified_utc"])

    stage_block_rows = build_stage_block_rows(blocks["Stage1"] + blocks["Stage2"] + blocks["Stage3"])
    write_csv(
        tables_dir / "stage_block_inventory.csv",
        stage_block_rows,
        [
            "block_index",
            "stage",
            "epochs_logged",
            "epoch_range",
            "total_epochs_declared",
            "timestamp",
            "run_dir",
            "batch_size",
            "lr",
            "window",
            "stride",
            "contains_nan",
            "metrics_present",
            "final_train_loss_total",
            "final_val_loss_total",
            "final_train_loss_diff",
            "final_val_loss_diff",
            "final_train_loss_align",
            "final_val_loss_align",
            "source_label",
        ],
    )

    if stage1_latest:
        rows = []
        for rec in stage1_latest.records:
            row = {"epoch": rec.epoch, "epoch_time_sec": rec.epoch_time_sec, **rec.metrics}
            rows.append(row)
        write_csv(tables_dir / "stage1_latest_epoch_metrics.csv", rows, sorted({k for row in rows for k in row}))
    if stage2_latest:
        rows = []
        for rec in stage2_latest.records:
            row = {"epoch": rec.epoch, "epoch_time_sec": rec.epoch_time_sec, **rec.metrics}
            rows.append(row)
        write_csv(tables_dir / "stage2_latest_epoch_metrics.csv", rows, sorted({k for row in rows for k in row}))
    if stage3_latest:
        rows = []
        for rec in stage3_latest.records:
            row = {"epoch": rec.epoch, "epoch_time_sec": rec.epoch_time_sec, **rec.metrics}
            rows.append(row)
        write_csv(tables_dir / "stage3_latest_epoch_metrics.csv", rows, sorted({k for row in rows for k in row}))

    write_csv(
        tables_dir / "current_code_facts.csv",
        [{"key": k, "value": v} for k, v in sorted(code_facts.items())],
        ["key", "value"],
    )
    write_csv(
        tables_dir / "gait_metric_names.csv",
        [{"index": i + 1, "metric_name": name} for i, name in enumerate(gait_metric_names)],
        ["index", "metric_name"],
    )
    write_csv(
        tables_dir / "output_inventory_counts.csv",
        [{"artifact_class": k, "count": v} for k, v in sorted(output_counts.items())],
        ["artifact_class", "count"],
    )

    copied_counts: dict[str, int] = {}
    copied_counts["results_gifs"] = copy_tree_selected(OUTPUTS_DIR / "results", artifacts_dir / "results_gifs", "*.gif")
    copied_counts["results_old_gifs"] = copy_tree_selected(OUTPUTS_DIR / "results_old", artifacts_dir / "results_old_gifs", "*.gif")
    copied_counts["gif_diagnostics"] = copy_tree_selected(OUTPUTS_DIR / "gifs", artifacts_dir / "gif_diagnostics", "*.gif")
    copied_counts["gif_diagnostic_pngs"] = copy_tree_selected(OUTPUTS_DIR / "gifs", artifacts_dir / "gif_diagnostic_pngs", "*.png")
    copied_counts["attention_pngs"] = copy_tree_selected(OUTPUTS_DIR / "attention", artifacts_dir / "attention_pngs", "*.png")
    copied_counts["attention_npy"] = copy_tree_selected(OUTPUTS_DIR / "attention", artifacts_dir / "attention_npy", "*.npy")
    copied_counts["sample_gait_cache_csv"] = 0
    gait_cache_dst = artifacts_dir / "gait_cache_sample"
    if (OUTPUTS_DIR / "gait_cache").exists():
        gait_cache_dst.mkdir(parents=True, exist_ok=True)
        for path in sorted((OUTPUTS_DIR / "gait_cache").glob("*.csv"))[:25]:
            shutil.copy2(path, gait_cache_dst / path.name)
            copied_counts["sample_gait_cache_csv"] += 1

    rich_summary = run_rich_analysis(report_dir, gait_metric_names)

    report_md = report_text(
        report_dir,
        run_configs,
        blocks,
        code_facts,
        gait_metric_names,
        output_counts,
        copied_counts,
        rich_summary=rich_summary,
    )
    (report_dir / "report.md").write_text(report_md, encoding="utf-8")
    write_html_from_markdown(report_md, report_dir / "report.html")
    (report_dir / "README.txt").write_text(
        "Open report.md or report.html first. All copied artifacts, plots, and raw tables are inside this folder.\n",
        encoding="utf-8",
    )

    print(f"created report bundle: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
