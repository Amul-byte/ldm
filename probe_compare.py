"""
Compare skeleton latents (Stage-1 encoder) vs IMU latents (Stage-2 GCN encoder).

Usage
-----
python probe_compare.py \
    --stage1_ckpt  checkpoints/stage1.pt \
    --stage2_ckpt  checkpoints/stage2.pt \
    --skeleton_folder  data/skeleton \
    --hip_folder       data/hip \
    --wrist_folder     data/wrist \
    --out_dir          outputs/compare

What it plots
-------------
Three-panel UMAP (PCA → UMAP on full sequences):
  1. Raw skeleton full sequence
  2. Stage-1 skeleton latents  z0  [B, T, J, D]  → flatten → PCA → UMAP
  3. Stage-2 IMU latents       h   [B, T, D]      → flatten → PCA → UMAP

This directly answers: does the GCN IMU encoder produce latents that
are more or less discriminative than the skeleton encoder?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion_model.dataset import (
    create_dataset,
    create_dataloader,
    split_train_val_dataset,
    extract_subject_ids,
)
from diffusion_model.model import Stage1Model, Stage2Model
from diffusion_model.util import DEFAULT_LATENT_DIM, DEFAULT_NUM_CLASSES, DEFAULT_WINDOW, DEFAULT_JOINTS


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--stage2_ckpt", required=True)
    # data
    p.add_argument("--data_path", default=None)
    p.add_argument("--skeleton_folder", default=None)
    p.add_argument("--hip_folder", default=None)
    p.add_argument("--wrist_folder", default=None)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--stride", type=int, default=30)
    # split
    p.add_argument("--subject_wise_split", action="store_true")
    p.add_argument("--val_frac", type=float, default=0.2)
    # model
    p.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    p.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    p.add_argument("--imu_graph_type", default="chain")
    # probe
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pca_components", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="outputs/compare")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_all(
    stage1: Stage1Model,
    stage2: Stage2Model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect full sequences from raw skeleton, skeleton latents, and IMU latents.

    Returns
    -------
    raw_seq  : [N, T*J*3]   full raw skeleton
    skel_seq : [N, T*J*D]   Stage-1 skeleton latents
    imu_seq  : [N, T*D]     Stage-2 IMU latents (h_tokens)
    labels   : [N]
    """
    raw_list, skel_list, imu_list, label_list = [], [], [], []

    for batch in loader:
        x       = batch["skeleton"].to(device)      # [B, T, J, 3]
        a_hip   = batch["A_hip"].to(device)         # [B, T, 3]
        a_wrist = batch["A_wrist"].to(device)       # [B, T, 3]
        y       = batch["label"].cpu().numpy()
        B       = x.shape[0]

        # 1. raw skeleton — full sequence flattened
        raw_list.append(x.cpu().numpy().reshape(B, -1))              # [B, T*J*3]

        # 2. skeleton latents from Stage-1 encoder
        z0 = stage1.encoder(x, gait_metrics=None)                    # [B, T, J, D]
        skel_list.append(z0.cpu().numpy().reshape(B, -1))            # [B, T*J*D]

        # 3. IMU latents from Stage-2 GCN aligner
        out = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)
        h_tokens = out[0]                                            # [B, T, D]
        imu_list.append(h_tokens.cpu().numpy().reshape(B, -1))      # [B, T*D]

        label_list.append(y)

    return (
        np.concatenate(raw_list),
        np.concatenate(skel_list),
        np.concatenate(imu_list),
        np.concatenate(label_list),
    )


# ─────────────────────────────────────────────────────────────────────────────
def pca_fit_transform(train: np.ndarray, val: np.ndarray, n: int):
    from sklearn.decomposition import PCA
    n = min(n, train.shape[0], train.shape[1])
    pca = PCA(n_components=n, random_state=42)
    tr = pca.fit_transform(train)
    va = pca.transform(val)
    var = pca.explained_variance_ratio_.sum()
    return tr, va, var


def umap_embed(feats: np.ndarray) -> np.ndarray:
    import umap
    return umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(feats)


# ─────────────────────────────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def run_probe(
    train: np.ndarray, labels_train: np.ndarray,
    val:   np.ndarray, labels_val:   np.ndarray,
    num_classes: int, epochs: int, lr: float,
    device: torch.device, name: str,
) -> float:
    probe = LinearProbe(train.shape[1], num_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    ce    = nn.CrossEntropyLoss()

    xt = torch.tensor(train, dtype=torch.float32).to(device)
    yt = torch.tensor(labels_train, dtype=torch.long).to(device)
    xv = torch.tensor(val,   dtype=torch.float32).to(device)
    yv = torch.tensor(labels_val,  dtype=torch.long).to(device)

    for ep in range(1, epochs + 1):
        probe.train()
        opt.zero_grad()
        ce(probe(xt), yt).backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        val_acc = (probe(xv).argmax(1) == yv).float().mean().item()
    print(f"  [{name}] probe val acc: {val_acc:.3f}")
    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
def scatter_ax(ax, emb, labels, num_classes, title, subtitle=""):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20", num_classes)
    for c in range(num_classes):
        idx = labels == c
        if idx.sum():
            ax.scatter(emb[idx, 0], emb[idx, 1], s=12, alpha=0.65,
                       color=cmap(c), label=f"A{c+1}")
    ax.legend(markerscale=2, fontsize=7, ncol=2, loc="best")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")


def plot_comparison(
    emb_raw, emb_skel, emb_imu,
    labels, num_classes,
    acc_skel, acc_imu,
    out_path,
):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    scatter_ax(axes[0], emb_raw,  labels, num_classes,
               "Raw skeleton", "Full sequence (PCA→UMAP)")
    scatter_ax(axes[1], emb_skel, labels, num_classes,
               "Stage-1 skeleton latents",
               f"GCN encoder output  |  probe acc={acc_skel:.3f}")
    scatter_ax(axes[2], emb_imu,  labels, num_classes,
               "Stage-2 IMU latents (GCN)",
               f"hip+wrist GCN encoder  |  probe acc={acc_imu:.3f}")

    chance = 1.0 / num_classes
    fig.suptitle(
        f"Skeleton latents vs IMU latents — which is more discriminative?\n"
        f"(chance = {chance:.3f}  |  probe on PCA-compressed full sequences)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args    = parse_args()
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load Stage-1 ─────────────────────────────────────────────────────────
    print(f"Loading Stage-1: {args.stage1_ckpt}")
    ckpt1   = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    stage1  = Stage1Model(latent_dim=args.latent_dim)
    stage1.load_state_dict(ckpt1["state_dict"], strict=False)
    stage1.eval().to(device)
    for p in stage1.parameters():
        p.requires_grad_(False)

    # ── load Stage-2 ─────────────────────────────────────────────────────────
    print(f"Loading Stage-2: {args.stage2_ckpt}")
    ckpt2  = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
        num_joints=DEFAULT_JOINTS,
        num_classes=args.num_classes,
        imu_graph_type=args.imu_graph_type,
    )
    stage2.load_state_dict(ckpt2["state_dict"], strict=False)
    stage2.eval().to(device)
    for p in stage2.parameters():
        p.requires_grad_(False)

    # ── dataset + split ───────────────────────────────────────────────────────
    print("Building dataset …")
    ds = create_dataset(
        dataset_path=args.data_path,
        window=args.window,
        joints=DEFAULT_JOINTS,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder,
        hip_folder=args.hip_folder,
        wrist_folder=args.wrist_folder,
        stride=args.stride,
        normalize_sensors=True,
    )
    print(f"  total windows: {len(ds)}")

    if args.subject_wise_split:
        subject_ids = extract_subject_ids(ds)
        all_subs    = sorted(set(subject_ids))
        n_val       = max(1, int(len(all_subs) * args.val_frac))
        train_subs  = all_subs[:-n_val]
        print(f"  subject-wise: train={train_subs}  val={all_subs[-n_val:]}")
        train_ds, val_ds = split_train_val_dataset(
            ds, val_split=args.val_frac, seed=42, train_subjects=train_subs)
    else:
        train_ds, val_ds = split_train_val_dataset(
            ds, val_split=args.val_frac, seed=42)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    def _loader(subset, shuffle):
        return create_dataloader(
            dataset_path=None, batch_size=args.batch_size,
            shuffle=shuffle, dataset=subset,
            drop_last=False, num_workers=0,
        )

    # ── collect all representations ───────────────────────────────────────────
    print("\nEncoding training set …")
    raw_tr, skel_tr, imu_tr, y_tr = collect_all(stage1, stage2, _loader(train_ds, True),  device)
    print("Encoding validation set …")
    raw_va, skel_va, imu_va, y_va = collect_all(stage1, stage2, _loader(val_ds,   False), device)

    print(f"\n  raw  : {raw_tr.shape}  →  T×J×3  per window")
    print(f"  skel : {skel_tr.shape}  →  T×J×D  per window")
    print(f"  imu  : {imu_tr.shape}   →  T×D    per window")

    # ── PCA compress (fit on train, apply to val) ─────────────────────────────
    n = args.pca_components
    print(f"\nPCA to {n} components …")
    raw_tr_p,  raw_va_p,  var_raw  = pca_fit_transform(raw_tr,  raw_va,  n)
    skel_tr_p, skel_va_p, var_skel = pca_fit_transform(skel_tr, skel_va, n)
    imu_tr_p,  imu_va_p,  var_imu  = pca_fit_transform(imu_tr,  imu_va,  n)
    print(f"  raw  variance explained: {var_raw:.1%}")
    print(f"  skel variance explained: {var_skel:.1%}")
    print(f"  imu  variance explained: {var_imu:.1%}")

    # ── linear probes ─────────────────────────────────────────────────────────
    print("\nLinear probes …")
    acc_skel = run_probe(skel_tr_p, y_tr, skel_va_p, y_va,
                         args.num_classes, args.epochs, args.lr, device, "skeleton")
    acc_imu  = run_probe(imu_tr_p,  y_tr, imu_va_p,  y_va,
                         args.num_classes, args.epochs, args.lr, device, "IMU")
    chance   = 1.0 / args.num_classes
    print(f"\n  chance = {chance:.3f}")
    print(f"  skeleton probe acc = {acc_skel:.3f}")
    print(f"  IMU      probe acc = {acc_imu:.3f}")
    winner = "skeleton" if acc_skel > acc_imu else "IMU"
    print(f"  → {winner} latents are more discriminative")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print("\nUMAP (this takes a minute) …")
    all_raw_p  = np.concatenate([raw_tr_p,  raw_va_p])
    all_skel_p = np.concatenate([skel_tr_p, skel_va_p])
    all_imu_p  = np.concatenate([imu_tr_p,  imu_va_p])
    all_y      = np.concatenate([y_tr, y_va])

    print("  [raw]  …"); emb_raw  = umap_embed(all_raw_p)
    print("  [skel] …"); emb_skel = umap_embed(all_skel_p)
    print("  [imu]  …"); emb_imu  = umap_embed(all_imu_p)

    plot_comparison(emb_raw, emb_skel, emb_imu, all_y, args.num_classes,
                    acc_skel, acc_imu, out_dir / "skeleton_vs_imu_umap.png")

    # ── save summary ──────────────────────────────────────────────────────────
    summary = {
        "chance_acc": float(chance),
        "skeleton_probe_val_acc": float(acc_skel),
        "imu_probe_val_acc": float(acc_imu),
        "pca_variance_explained": {
            "raw": float(var_raw), "skeleton": float(var_skel), "imu": float(var_imu),
        },
    }
    with open(out_dir / "comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  results → {out_dir}/comparison_results.json")


if __name__ == "__main__":
    main()
