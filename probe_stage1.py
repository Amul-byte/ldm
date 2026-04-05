"""
Linear probe: how well do Stage-1 latents separate activity classes?

Usage (CSV folders)
-------------------
python probe_stage1.py \
    --stage1_ckpt  checkpoints/stage1.pt \
    --skeleton_folder  data/skeleton \
    --hip_folder       data/hip \
    --wrist_folder     data/wrist \
    --out_dir          outputs/probe

Usage (.pt file)
----------------
python probe_stage1.py \
    --stage1_ckpt checkpoints/stage1.pt \
    --data_path   data/dataset.pt \
    --out_dir     outputs/probe

What it does
------------
1. Load Stage-1 encoder (frozen).
2. Encode every window  →  z0 [B, T, J, D]  →  mean-pool over T → [B, J*D].
3. Train a linear probe (logistic regression) on the train split, evaluate on val.
4. Report per-class accuracy + confusion matrix + UMAP scatter.

Interpreting results
--------------------
- val acc < 2× chance (2/14 ≈ 0.14)  → latents are activity-blind.
- val acc 0.3–0.6                     → weakly discriminative.
- val acc > 0.6                       → latents carry rich activity info.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from diffusion_model.dataset import create_dataset, create_dataloader, split_train_val_dataset
from diffusion_model.model import Stage1Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint
from diffusion_model.util import DEFAULT_LATENT_DIM, DEFAULT_NUM_CLASSES, DEFAULT_WINDOW, DEFAULT_JOINTS


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # checkpoint
    p.add_argument("--stage1_ckpt", required=True)
    # data — either .pt file or three CSV folders
    p.add_argument("--data_path", default=None, help=".pt dataset file")
    p.add_argument("--skeleton_folder", default=None)
    p.add_argument("--hip_folder", default=None)
    p.add_argument("--wrist_folder", default=None)
    # sliding window (CSV mode)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--stride", type=int, default=30)
    # split
    p.add_argument("--subject_wise_split", action="store_true",
                   help="Split by subject IDs instead of random windows")
    p.add_argument("--val_frac", type=float, default=0.2)
    # model / probe
    p.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    p.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="outputs/probe")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_latents(
    model: Stage1Model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode all batches, keeping the FULL temporal sequence.

    Returns
    -------
    raw_seq    : [N, T*J*3]   — full raw skeleton sequence flattened
    latent_seq : [N, T*J*D]   — full Stage-1 latent sequence flattened
    labels     : [N]
    """
    raw_list, lat_list, label_list = [], [], []
    for batch in loader:
        x  = batch["skeleton"].to(device)      # [B, T, J, 3]
        gm = batch["gait_metrics"].to(device)
        y  = batch["label"].cpu().numpy()

        # full raw sequence — keep every frame
        raw_list.append(x.cpu().numpy().reshape(x.shape[0], -1))   # [B, T*J*3]

        z0 = model.encoder(x, gait_metrics=gm)                     # [B, T, J, D]
        lat_list.append(z0.cpu().numpy().reshape(x.shape[0], -1))   # [B, T*J*D]

        label_list.append(y)

    return (
        np.concatenate(raw_list),
        np.concatenate(lat_list),
        np.concatenate(label_list),
    )


def _pca_reduce(feats: np.ndarray, n_components: int = 64) -> np.ndarray:
    """PCA compression before UMAP — needed because T*J*D is huge (~737k dims)."""
    from sklearn.decomposition import PCA
    n_components = min(n_components, feats.shape[0], feats.shape[1])
    print(f"    PCA {feats.shape[1]}→{n_components} …", end=" ", flush=True)
    pca = PCA(n_components=n_components, random_state=42)
    out = pca.fit_transform(feats)
    explained = pca.explained_variance_ratio_.sum()
    print(f"variance explained: {explained:.1%}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_probe(
    feats_train: np.ndarray, labels_train: np.ndarray,
    feats_val:   np.ndarray, labels_val:   np.ndarray,
    num_classes: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[LinearProbe, list[dict]]:
    in_dim = feats_train.shape[1]
    probe  = LinearProbe(in_dim, num_classes).to(device)
    opt    = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    ce     = nn.CrossEntropyLoss()

    xt = torch.tensor(feats_train, dtype=torch.float32).to(device)
    yt = torch.tensor(labels_train, dtype=torch.long).to(device)
    xv = torch.tensor(feats_val,   dtype=torch.float32).to(device)
    yv = torch.tensor(labels_val,  dtype=torch.long).to(device)

    history = []
    for ep in range(1, epochs + 1):
        probe.train()
        opt.zero_grad()
        ce(probe(xt), yt).backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            tr_acc = (probe(xt).argmax(1) == yt).float().mean().item()
            va_acc = (probe(xv).argmax(1) == yv).float().mean().item()
        history.append({"epoch": ep, "train_acc": tr_acc, "val_acc": va_acc})
        if ep % 10 == 0:
            print(f"  [probe] epoch {ep:3d}  train={tr_acc:.3f}  val={va_acc:.3f}")

    return probe, history


# ─────────────────────────────────────────────────────────────────────────────
def confusion_matrix_np(pred, true, n):
    cm = np.zeros((n, n), dtype=int)
    for p, t in zip(pred, true):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1
    return cm


def plot_confusion(cm, out_path):
    try:
        import matplotlib.pyplot as plt
        n = cm.shape[0]
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([f"A{i+1}" for i in range(n)], rotation=45, ha="right")
        ax.set_yticklabels([f"A{i+1}" for i in range(n)])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Linear Probe — Confusion Matrix (Stage-1 latents)")
        thresh = cm.max() / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  confusion matrix  → {out_path}")
    except ImportError:
        pass


def _umap_embed(feats):
    import umap
    return umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(feats)


def _scatter_ax(ax, emb, labels, num_classes, title):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20", num_classes)
    for c in range(num_classes):
        idx = labels == c
        if idx.sum():
            ax.scatter(emb[idx, 0], emb[idx, 1], s=10, alpha=0.6,
                       color=cmap(c), label=f"A{c+1}")
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")


def plot_umap_comparison(raw_seqs, lat_seqs, labels, out_path, num_classes):
    """Two-panel UMAP using full sequences (PCA → UMAP).

    Left  — raw skeleton full sequence (T*J*3) → PCA → UMAP
    Right — Stage-1 latents full sequence (T*J*D) → PCA → UMAP

    This preserves the complete gait motion pattern, not just a mean snapshot.
    """
    try:
        import matplotlib.pyplot as plt

        print("  Full-sequence UMAP (PCA first to handle high dims) …")
        print("  [raw]")
        raw_pca = _pca_reduce(raw_seqs, n_components=64)
        print("  [latent]")
        lat_pca = _pca_reduce(lat_seqs, n_components=64)

        print("  Running UMAP on raw PCA …")
        emb_raw = _umap_embed(raw_pca)
        print("  Running UMAP on latent PCA …")
        emb_lat = _umap_embed(lat_pca)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        _scatter_ax(axes[0], emb_raw, labels, num_classes,
                    "Raw skeleton — full sequence\n(PCA→UMAP, all T frames × J joints × 3 coords)")
        _scatter_ax(axes[1], emb_lat, labels, num_classes,
                    "Stage-1 latents — full sequence\n(PCA→UMAP, all T frames × J joints × D dims)")
        fig.suptitle(
            "Full gait sequence structure — does the encoder add discriminability?",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  UMAP comparison   → {out_path}")
    except ImportError:
        print("  (umap-learn / sklearn not available, skipping UMAP)")


def plot_acc_curve(history, out_path):
    try:
        import matplotlib.pyplot as plt
        ep = [r["epoch"] for r in history]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, [r["train_acc"] for r in history], label="train")
        ax.plot(ep, [r["val_acc"]   for r in history], label="val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("Linear Probe Accuracy")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage-1 encoder (frozen) ─────────────────────────────────────────────
    print(f"Loading Stage-1 checkpoint: {args.stage1_ckpt}")
    ckpt  = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    encoder_graph_op, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)
    print(f"  inferred graph ops: encoder={encoder_graph_op} full_skeleton={skeleton_graph_op}")
    model = Stage1Model(
        latent_dim=args.latent_dim,
        encoder_type=encoder_graph_op,
        skeleton_graph_op=skeleton_graph_op,
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    print("  encoder frozen.")

    # ── build dataset then split ─────────────────────────────────────────────
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

    # subject_wise_split: pass train_subjects as all-but-last subject
    # random split: pass train_subjects=None
    if args.subject_wise_split:
        from diffusion_model.dataset import extract_subject_ids
        subject_ids = extract_subject_ids(ds)
        if subject_ids is None:
            raise ValueError("Dataset has no subject_ids — cannot do subject-wise split.")
        all_subs = sorted(set(subject_ids))
        n_val_subs = max(1, int(len(all_subs) * args.val_frac))
        train_subjects = all_subs[:-n_val_subs]
        print(f"  subject-wise split: train={train_subjects}  val={all_subs[-n_val_subs:]}")
        train_ds, val_ds = split_train_val_dataset(ds, val_split=args.val_frac, seed=42,
                                                   train_subjects=train_subjects)
    else:
        train_ds, val_ds = split_train_val_dataset(ds, val_split=args.val_frac, seed=42)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    def _loader(subset, shuffle):
        return create_dataloader(
            dataset_path=None,
            batch_size=args.batch_size,
            shuffle=shuffle,
            dataset=subset,
            drop_last=False,
            num_workers=0,
        )

    train_loader = _loader(train_ds, shuffle=True)
    val_loader   = _loader(val_ds,   shuffle=False)

    # ── encode latents ───────────────────────────────────────────────────────
    print("Encoding training set …")
    raw_train, feats_train, labels_train = collect_latents(model, train_loader, device)
    print(f"  raw seq: {raw_train.shape}  latent seq: {feats_train.shape}  classes: {np.unique(labels_train).tolist()}")

    print("Encoding validation set …")
    raw_val, feats_val, labels_val = collect_latents(model, val_loader, device)
    print(f"  raw seq: {raw_val.shape}  latent seq: {feats_val.shape}")

    # ── PCA compress full latent sequences for probe + UMAP ─────────────────
    print("\nPCA compressing latent sequences for probe …")
    probe_train = _pca_reduce(feats_train, n_components=64)
    # fit PCA on train, apply same transform to val
    from sklearn.decomposition import PCA as _PCA
    _pca = _PCA(n_components=min(64, feats_train.shape[0], feats_train.shape[1]), random_state=42)
    probe_train = _pca.fit_transform(feats_train)
    probe_val   = _pca.transform(feats_val)

    # ── linear probe ─────────────────────────────────────────────────────────
    print(f"\nTraining linear probe ({args.epochs} epochs) …")
    probe, history = train_probe(
        probe_train, labels_train,
        probe_val,   labels_val,
        num_classes=args.num_classes,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    final_val_acc = history[-1]["val_acc"]
    chance = 1.0 / args.num_classes
    print(f"\nFinal probe val accuracy : {final_val_acc:.3f}  (chance = {chance:.3f})")
    if final_val_acc < 2 * chance:
        print("  ⚠  Latents are NOT discriminative — activity info is absent.")
    elif final_val_acc < 0.5:
        print("  ⚠  Latents are weakly discriminative.")
    else:
        print("  ✓  Latents carry meaningful activity information.")

    # ── per-class accuracy ───────────────────────────────────────────────────
    probe.eval()
    with torch.no_grad():
        preds = probe(torch.tensor(probe_val, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()

    per_class_acc = {}
    print("\nPer-class accuracy:")
    for c in range(args.num_classes):
        idx = labels_val == c
        if idx.sum() == 0:
            continue
        acc = float((preds[idx] == c).mean())
        per_class_acc[f"A{c+1:02d}"] = acc
        bar = "█" * int(acc * 30)
        print(f"  A{c+1:02d}: {acc:.3f}  {bar}")

    # ── save JSON ────────────────────────────────────────────────────────────
    results = {
        "final_val_acc": final_val_acc,
        "chance_acc": chance,
        "per_class_acc": per_class_acc,
        "history": history,
    }
    out_json = out_dir / "probe_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ── plots ────────────────────────────────────────────────────────────────
    plot_acc_curve(history, out_dir / "probe_accuracy_curve.png")

    cm = confusion_matrix_np(preds, labels_val, args.num_classes)
    plot_confusion(cm, out_dir / "probe_confusion_matrix.png")

    all_raw    = np.concatenate([raw_train,   raw_val])
    all_feats  = np.concatenate([feats_train, feats_val])
    all_labels = np.concatenate([labels_train, labels_val])
    plot_umap_comparison(all_raw, all_feats, all_labels,
                         out_dir / "latent_umap_before_after.png", args.num_classes)


if __name__ == "__main__":
    main()
