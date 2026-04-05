"""
Visualization script for the Stage 2 sensor model (IMULatentAligner).

Generates two presentation-ready plots:
  1. IMU Graph Structure  — shows the multi-scale temporal graph topology
                            (hip nodes, wrist nodes, cross-sensor + temporal edges)
  2. Node Activations (PCA) — extracts h_tokens [B, T, 256] from the encoder,
                              PCA-reduces to 2D, coloured by timestep,
                              alongside the raw accelerometer signal.

Usage:
    python visualize_sensor_model.py \
        --stage2_ckpt checkpoints/stage2_best.pt \
        --hip_folder   /path/to/meta_hip \
        --wrist_folder /path/to/meta_wrist \
        --skeleton_folder /path/to/skeleton \
        --out_dir outputs/sensor_viz \
        --num_samples 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn

# ── project imports ────────────────────────────────────────────────────────────
from diffusion_model.dataset import create_dataset, split_train_val_dataset
from diffusion_model.model import Stage2Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.sensor_model import (
    IMULatentAligner,
    build_imu_features,
    build_imu_graph_adjacency,
    _IMU_GRAPH_SCALES,
)
from diffusion_model.graph_modules import build_edge_index_from_adjacency

# ── activity label map (A01–A14 → human-readable) ─────────────────────────────
ACTIVITY_NAMES = {
    0:  "Walking",
    1:  "Walking Upstairs",
    2:  "Walking Downstairs",
    3:  "Sitting",
    4:  "Standing",
    5:  "Lying Down",
    6:  "Sit→Stand",
    7:  "Stand→Sit",
    8:  "Lie→Stand",
    9:  "Stand→Lie",
    10: "Stand→Walk",
    11: "Walk→Stand",
    12: "Fall Forward",
    13: "Fall Backward",
}


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Graph structure
# ══════════════════════════════════════════════════════════════════════════════

def plot_imu_graph_structure(window_len: int = 90, out_path: str = "imu_graph_structure.png"):
    """
    Draw the multi-scale temporal IMU graph.
    - Hip nodes   (0 .. T-1)    → blue
    - Wrist nodes (T .. 2T-1)   → orange
    - Temporal edges within each stream coloured by hop distance
    - Cross-sensor edges (hip↔wrist) → grey dashed
    """
    import networkx as nx

    T = window_len
    N = 2 * T

    G = nx.Graph()
    G.add_nodes_from(range(N))

    scale_colors = {1: "#e74c3c", 5: "#2ecc71", 15: "#9b59b6", 30: "#f39c12"}
    edge_lists   = {s: [] for s in _IMU_GRAPH_SCALES}
    cross_edges  = []

    # Temporal edges within each stream
    for scale in _IMU_GRAPH_SCALES:
        if scale >= T:
            continue
        for i in range(T - scale):
            edge_lists[scale].append((i, i + scale))             # hip
            edge_lists[scale].append((T + i, T + i + scale))     # wrist

    # Cross-sensor edges
    for t in range(T):
        cross_edges.append((t, T + t))

    # Layout: hip on top row, wrist on bottom row
    pos = {}
    for t in range(T):
        pos[t]     = (t, 1.0)   # hip
        pos[T + t] = (t, 0.0)   # wrist

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Draw cross-sensor edges (sparse: every 10th to avoid clutter)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=cross_edges[::10],
        edge_color="white", alpha=0.15,
        style="dashed", width=0.6, ax=ax,
    )

    # Draw temporal edges per scale
    for scale, elist in edge_lists.items():
        nx.draw_networkx_edges(
            G, pos,
            edgelist=elist,
            edge_color=scale_colors[scale],
            alpha=0.5, width=1.2, ax=ax,
            label=f"hop={scale}",
        )

    # Draw nodes
    hip_nodes   = list(range(T))
    wrist_nodes = list(range(T, 2 * T))
    nx.draw_networkx_nodes(G, pos, nodelist=hip_nodes,   node_color="#3498db",
                           node_size=40, ax=ax, label="Hip")
    nx.draw_networkx_nodes(G, pos, nodelist=wrist_nodes, node_color="#e67e22",
                           node_size=40, ax=ax, label="Wrist")

    # Labels only on a few nodes for readability
    label_nodes = {i: str(i) for i in range(0, T, 15)}
    label_nodes.update({T + i: str(T + i) for i in range(0, T, 15)})
    nx.draw_networkx_labels(G, pos, labels=label_nodes,
                            font_size=6, font_color="white", ax=ax)

    ax.set_title(
        f"IMU Multi-Scale Temporal Graph  (T={T} frames, {N} nodes)\n"
        f"Blue = Hip stream  |  Orange = Wrist stream  |  Coloured edges = hop distances",
        color="white", fontsize=13, pad=12,
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=scale_colors[s], linewidth=2, label=f"Temporal hop={s}")
        for s in _IMU_GRAPH_SCALES
    ] + [
        Line2D([0], [0], color="white", linewidth=1, linestyle="dashed", label="Cross-sensor"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=8, label="Hip node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markersize=8, label="Wrist node"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#2c2c54", labelcolor="white", fontsize=9)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Node activations (PCA) + raw IMU signal
# ══════════════════════════════════════════════════════════════════════════════

def _extract_activations(aligner: IMULatentAligner,
                         hip_raw: torch.Tensor,
                         wrist_raw: torch.Tensor,
                         device: torch.device):
    """
    Run the encoder and collect per-layer node activations from
    each TemporalGCNBlock inside the IMUGraphEncoder.

    Returns:
        layer_activations: list of np.ndarray [2T, 256], one per GCN block
        sensor_tokens:     np.ndarray [T, 256]  final fused tokens
    """
    hip_raw   = hip_raw.unsqueeze(0).to(device)    # [1, T, 3]
    wrist_raw = wrist_raw.unsqueeze(0).to(device)

    hip_feat   = build_imu_features(hip_raw)        # [1, T, 6]
    wrist_feat = build_imu_features(wrist_raw)

    enc = aligner.graph_encoder
    b, t, _ = hip_feat.shape

    hip_latent   = enc.hip_proj(hip_feat)           # [1, T, 256]
    wrist_latent = enc.wrist_proj(wrist_feat)
    combined     = torch.cat([hip_latent, wrist_latent], dim=1)  # [1, 2T, 256]

    if t not in enc._imu_graph_cache:
        adj = build_imu_graph_adjacency(window_len=t, device=device)
        from diffusion_model.graph_modules import build_edge_index_from_adjacency
        edge_index = build_edge_index_from_adjacency(adj)
        enc._imu_graph_cache[t] = (adj, edge_index)
    adj, edge_index = enc._imu_graph_cache[t]

    layer_activations = []
    for block in enc.blocks:
        combined = block(combined, adjacency=adj, edge_index=edge_index)
        layer_activations.append(combined[0].detach().cpu().numpy())  # [2T, 256]

    # Fuse to get sensor_tokens
    fused        = torch.cat([combined[:, :t, :], combined[:, t:, :]], dim=-1)  # [1, T, 512]
    sensor_tokens = aligner.fuse_tokens(fused)[0].detach().cpu().numpy()        # [T, 256]

    return layer_activations, sensor_tokens


def plot_node_activations(aligner: IMULatentAligner,
                          batch: dict,
                          sample_idx: int,
                          activity_label: int,
                          device: torch.device,
                          out_path: str = "node_activations.png"):
    """
    Two-panel figure:
      Left  — PCA of final-layer node activations, nodes coloured by timestep
      Right — Raw accelerometer signal (hip X/Y/Z)
    """
    from sklearn.decomposition import PCA

    hip_raw   = batch["A_hip"][sample_idx]    # [T, 3]
    wrist_raw = batch["A_wrist"][sample_idx]  # [T, 3]
    T = hip_raw.shape[0]

    with torch.no_grad():
        layer_acts, sensor_tokens = _extract_activations(aligner, hip_raw, wrist_raw, device)

    # Use the last layer activations [2T, 256]
    acts_last = layer_acts[-1]   # [2T, 256]
    pca       = PCA(n_components=2)
    acts_2d   = pca.fit_transform(acts_last)   # [2T, 2]

    hip_2d   = acts_2d[:T]       # first T nodes = hip
    wrist_2d = acts_2d[T:]       # last T nodes  = wrist
    timesteps = np.arange(T)

    activity_name = ACTIVITY_NAMES.get(activity_label, f"Activity {activity_label + 1}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    # ── Left: PCA scatter ────────────────────────────────────────────────────
    ax = axes[0]
    cmap = cm.plasma

    # Hip nodes
    sc_hip = ax.scatter(hip_2d[:, 0], hip_2d[:, 1],
                        c=timesteps, cmap=cmap, s=60,
                        edgecolors="#3498db", linewidths=1.2,
                        label="Hip nodes", zorder=3)
    # Wrist nodes
    ax.scatter(wrist_2d[:, 0], wrist_2d[:, 1],
               c=timesteps, cmap=cmap, s=60,
               edgecolors="#e67e22", linewidths=1.2,
               marker="^", label="Wrist nodes", zorder=3)

    # Draw temporal edges (hop=1 only, for clarity)
    for i in range(T - 1):
        ax.plot([hip_2d[i, 0],   hip_2d[i+1, 0]],
                [hip_2d[i, 1],   hip_2d[i+1, 1]],
                color="#3498db", alpha=0.3, linewidth=0.7)
        ax.plot([wrist_2d[i, 0], wrist_2d[i+1, 0]],
                [wrist_2d[i, 1], wrist_2d[i+1, 1]],
                color="#e67e22", alpha=0.3, linewidth=0.7)

    # Cross-sensor edges (every 10th to avoid clutter)
    for i in range(0, T, 10):
        ax.plot([hip_2d[i, 0], wrist_2d[i, 0]],
                [hip_2d[i, 1], wrist_2d[i, 1]],
                color="white", alpha=0.15, linewidth=0.6, linestyle="--")

    cbar = plt.colorbar(sc_hip, ax=ax)
    cbar.set_label("Timestep", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_title(f"Node Activations — GCN Layer 3 (PCA)\n(Label: {activity_name})",
                 color="white", fontsize=12)
    ax.set_xlabel("PCA Component 1", color="white")
    ax.set_ylabel("PCA Component 2", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    leg = ax.legend(facecolor="#2c2c54", labelcolor="white", fontsize=9)

    # ── Right: Raw accelerometer ──────────────────────────────────────────────
    ax = axes[1]
    hip_np = hip_raw.numpy()
    t_axis = np.arange(T)

    ax.plot(t_axis, hip_np[:, 0], color="#e74c3c", label="X-axis", linewidth=1.5)
    ax.plot(t_axis, hip_np[:, 1], color="#2ecc71", label="Y-axis", linewidth=1.5)
    ax.plot(t_axis, hip_np[:, 2], color="#3498db", label="Z-axis", linewidth=1.5)

    ax.set_title(f"Original Accelerometer Data (Hip)\n(Label: {activity_name})",
                 color="white", fontsize=12)
    ax.set_xlabel("Time Steps", color="white")
    ax.set_ylabel("Accelerometer Values", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    leg = ax.legend(facecolor="#2c2c54", labelcolor="white", fontsize=9)
    ax.axhline(0, color="white", alpha=0.2, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-layer PCA grid (all 3 GCN layers side by side)
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_layers(aligner: IMULatentAligner,
                   batch: dict,
                   sample_idx: int,
                   activity_label: int,
                   device: torch.device,
                   out_path: str = "all_layers.png"):
    """Show PCA projections for all 3 GCN layers side by side."""
    from sklearn.decomposition import PCA

    hip_raw   = batch["A_hip"][sample_idx]
    wrist_raw = batch["A_wrist"][sample_idx]
    T = hip_raw.shape[0]

    with torch.no_grad():
        layer_acts, _ = _extract_activations(aligner, hip_raw, wrist_raw, device)

    activity_name = ACTIVITY_NAMES.get(activity_label, f"Activity {activity_label + 1}")
    n_layers = len(layer_acts)
    timesteps = np.arange(T)

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"GCN Layer Activations (PCA) — Label: {activity_name}",
                 color="white", fontsize=14, y=1.01)

    for li, (ax, acts) in enumerate(zip(axes, layer_acts)):
        ax.set_facecolor("#1a1a2e")
        pca    = PCA(n_components=2)
        acts_2d = pca.fit_transform(acts)   # [2T, 2]
        hip_2d   = acts_2d[:T]
        wrist_2d = acts_2d[T:]

        ax.scatter(hip_2d[:, 0],   hip_2d[:, 1],   c=timesteps,
                   cmap="plasma", s=50, edgecolors="#3498db", linewidths=1, label="Hip")
        ax.scatter(wrist_2d[:, 0], wrist_2d[:, 1], c=timesteps,
                   cmap="plasma", s=50, edgecolors="#e67e22", linewidths=1,
                   marker="^", label="Wrist")

        for i in range(T - 1):
            ax.plot([hip_2d[i, 0],   hip_2d[i+1, 0]],
                    [hip_2d[i, 1],   hip_2d[i+1, 1]],
                    color="#3498db", alpha=0.25, linewidth=0.6)
            ax.plot([wrist_2d[i, 0], wrist_2d[i+1, 0]],
                    [wrist_2d[i, 1], wrist_2d[i+1, 1]],
                    color="#e67e22", alpha=0.25, linewidth=0.6)

        ax.set_title(f"GCN Layer {li + 1}", color="white", fontsize=11)
        ax.set_xlabel("PC 1", color="white")
        ax.set_ylabel("PC 2", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.legend(facecolor="#2c2c54", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Activity separation: ALL activities in ONE PCA space
# ══════════════════════════════════════════════════════════════════════════════

def plot_activity_separation(aligner: IMULatentAligner,
                             dataset,
                             device: torch.device,
                             num_samples_per_class: int = 10,
                             out_path: str = "activity_separation.png"):
    """
    Collect h_global [256] from multiple samples per activity class,
    PCA-reduce to 2D, and plot all activities on ONE chart coloured by label.
    This directly shows whether the GCN separates different activities
    in the global embedding space.
    """
    from sklearn.decomposition import PCA

    # Palette: one colour per activity
    PALETTE = [
        "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
        "#1abc9c","#e67e22","#e91e63","#00bcd4","#8bc34a",
        "#ff5722","#607d8b","#673ab7","#ffc107",
    ]

    all_embeddings = []   # list of [256] tensors
    all_labels     = []

    # Collect samples per class
    label_buckets = {}
    for i in range(len(dataset)):
        item  = dataset[i]
        label = int(item["label"])
        if label not in label_buckets:
            label_buckets[label] = []
        if len(label_buckets[label]) < num_samples_per_class:
            label_buckets[label].append(i)

    print(f"  Collecting embeddings for {len(label_buckets)} activity classes...")
    for label, indices in sorted(label_buckets.items()):
        for idx in indices:
            item      = dataset[idx]
            hip_raw   = item["A_hip"].unsqueeze(0).to(device)
            wrist_raw = item["A_wrist"].unsqueeze(0).to(device)
            with torch.no_grad():
                hip_feat   = build_imu_features(hip_raw)
                wrist_feat = build_imu_features(wrist_raw)
                hip_t, wrist_t = aligner.graph_encoder(hip_feat, wrist_feat)
                fused      = torch.cat([hip_t, wrist_t], dim=-1)
                h_global   = aligner.fuse_tokens(fused).mean(dim=1)  # [1, 256]
            all_embeddings.append(h_global[0].cpu().numpy())
            all_labels.append(label)

    X = np.stack(all_embeddings)   # [N, 256]
    y = np.array(all_labels)

    pca    = PCA(n_components=2)
    X_2d   = pca.fit_transform(X)  # [N, 2]
    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    present_labels = sorted(np.unique(y))
    for label in present_labels:
        mask = y == label
        color = PALETTE[label % len(PALETTE)]
        name  = ACTIVITY_NAMES.get(label, f"Act {label+1}")
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=color, s=90, alpha=0.85,
                   edgecolors="white", linewidths=0.4,
                   label=name, zorder=3)
        # Draw centroid marker
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.scatter(cx, cy, color=color, s=250, marker="*",
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(name, (cx, cy), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color=color, fontweight="bold")

    ax.set_title(
        "Activity Separation in Global IMU Embedding Space (h_global)\n"
        f"PCA of [B, 256] sensor embeddings — {num_samples_per_class} samples/class\n"
        f"PC1 explains {var_explained[0]:.1f}%  |  PC2 explains {var_explained[1]:.1f}%",
        color="white", fontsize=12, pad=12,
    )
    ax.set_xlabel(f"PCA Component 1 ({var_explained[0]:.1f}% variance)", color="white")
    ax.set_ylabel(f"PCA Component 2 ({var_explained[1]:.1f}% variance)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    leg = ax.legend(facecolor="#2c2c54", labelcolor="white",
                    fontsize=8, loc="upper right",
                    title="Activity", title_fontsize=9)
    leg.get_title().set_color("white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2_ckpt",      default="checkpoints/stage2_best.pt")
    p.add_argument("--stage1_ckpt",      default="checkpoints/stage1_best.pt")
    p.add_argument("--skeleton_folder",  required=True)
    p.add_argument("--hip_folder",       required=True)
    p.add_argument("--wrist_folder",     required=True)
    p.add_argument("--out_dir",          default="outputs/sensor_viz")
    p.add_argument("--num_samples",      type=int, default=4,
                   help="Number of samples to visualize (one plot each)")
    p.add_argument("--window",           type=int, default=90)
    p.add_argument("--latent_dim",       type=int, default=256)
    p.add_argument("--device",           default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Graph structure plot (no model needed) ──────────────────────────
    print("Generating IMU graph structure plot...")
    plot_imu_graph_structure(
        window_len=args.window,
        out_path=os.path.join(args.out_dir, "imu_graph_structure.png"),
    )

    # ── 2. Load Stage 2 model ──────────────────────────────────────────────
    print("Loading Stage 2 model...")
    from diffusion_model.model import Stage1Model, Stage2Model
    encoder_graph_op, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)
    stage1 = Stage1Model(
        latent_dim=args.latent_dim,
        encoder_type=encoder_graph_op,
        skeleton_graph_op=skeleton_graph_op,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=False)

    model = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
    ).to(device)
    load_checkpoint(args.stage2_ckpt, model, strict=False)
    model.eval()
    aligner = model.aligner

    # ── 3. Load a small batch of data ─────────────────────────────────────
    print("Loading dataset...")
    dataset = create_dataset(
        dataset_path=None,
        skeleton_folder=args.skeleton_folder,
        hip_folder=args.hip_folder,
        wrist_folder=args.wrist_folder,
        window=args.window,
        stride=args.window,   # non-overlapping windows for variety
    )

    # Pick one sample per unique activity (up to num_samples)
    seen_labels = set()
    chosen_indices = []
    for i in range(len(dataset)):
        item  = dataset[i]
        label = int(item["label"])
        if label not in seen_labels:
            seen_labels.add(label)
            chosen_indices.append(i)
        if len(chosen_indices) >= args.num_samples:
            break

    print(f"Visualizing {len(chosen_indices)} samples...")
    for rank, idx in enumerate(chosen_indices):
        item  = dataset[idx]
        label = int(item["label"])
        act_name = ACTIVITY_NAMES.get(label, f"act{label}")
        safe_name = act_name.replace(" ", "_").replace("→", "to").replace("/", "_")

        print(f"  Sample {rank+1}: {act_name} (label={label})")

        batch = {
            "A_hip":   item["A_hip"].unsqueeze(0),
            "A_wrist": item["A_wrist"].unsqueeze(0),
        }

        # Node activations + raw IMU
        plot_node_activations(
            aligner=aligner,
            batch=batch,
            sample_idx=0,
            activity_label=label,
            device=device,
            out_path=os.path.join(args.out_dir, f"activations_{rank+1:02d}_{safe_name}.png"),
        )

        # All 3 GCN layers
        plot_all_layers(
            aligner=aligner,
            batch=batch,
            sample_idx=0,
            activity_label=label,
            device=device,
            out_path=os.path.join(args.out_dir, f"layers_{rank+1:02d}_{safe_name}.png"),
        )

    # ── 4. Activity separation plot ────────────────────────────────────────
    print("Generating activity separation plot...")
    plot_activity_separation(
        aligner=aligner,
        dataset=dataset,
        device=device,
        num_samples_per_class=15,
        out_path=os.path.join(args.out_dir, "activity_separation.png"),
    )

    print(f"\nDone. All plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
