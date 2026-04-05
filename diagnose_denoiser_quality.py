"""Evaluate denoiser quality INDEPENDENTLY of the decoder.

Compares z0_real (encoder output) vs z0_diff (50-step denoiser output) at multiple
structural levels to determine if the denoiser is producing properly structured latents.

Tests:
  1. Global stats: mean, std, range (already known — similar)
  2. Per-joint structure: does each joint's latent match encoder distribution?
  3. Per-frame structure: does temporal evolution match?
  4. Per-dimension structure: which latent dimensions diverge most?
  5. Joint-joint correlation: does the denoiser preserve spatial relationships?
  6. Temporal autocorrelation: does the denoiser preserve motion smoothness?
  7. Cosine similarity: per-joint, per-frame alignment
  8. Conditioning effect: is conditioned z0 structurally different from unconditioned?
  9. Multiple runs: is the denoiser consistent or random?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffusion_model.dataset import create_dataset
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.training_eval import sample_stage3_latents
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
)

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def _infer_latent_dim(stage1_ckpt: str) -> int:
    return int(_load_state_dict(stage1_ckpt)["encoder.in_proj.weight"].shape[0])


def _infer_d_shared(stage2_ckpt: str, default: int = 64) -> int:
    sd = _load_state_dict(stage2_ckpt)
    key = "shared_motion_layer.net.0.weight"
    return int(sd[key].shape[0]) if key in sd else int(default)


def cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=dim)


def temporal_autocorrelation(z: torch.Tensor, lag: int = 1) -> float:
    """Mean correlation between frame t and frame t+lag across all joints/dims."""
    # z: [B, T, J, D]
    z1 = z[:, :-lag].reshape(-1).float()
    z2 = z[:, lag:].reshape(-1).float()
    return float(torch.corrcoef(torch.stack([z1, z2]))[0, 1].item())


def joint_correlation_matrix(z: torch.Tensor) -> np.ndarray:
    """Correlation matrix between joints based on their mean latent vectors."""
    # z: [B, T, J, D] → per-joint mean: [J, D]
    per_joint = z.float().mean(dim=(0, 1))  # [J, D]
    # Correlation between joints
    corr = torch.corrcoef(per_joint)  # [J, J]
    return corr.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Denoiser quality evaluation (decoder-independent)")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/denoiser_quality")
    parser.add_argument("--skeleton_folder", type=str, default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--num_joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="ddpm")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--encoder_type", type=str, default="gcn")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of denoiser runs to test consistency")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = _infer_latent_dim(args.stage1_ckpt)
    d_shared = _infer_d_shared(args.stage2_ckpt)
    graph_op = args.encoder_type
    print(f"Config: latent_dim={latent_dim}, d_shared={d_shared}, graph_op={graph_op}")

    # ── Load models ──
    print("Loading models...")
    stage1 = Stage1Model(
        latent_dim=latent_dim, num_joints=args.num_joints,
        num_classes=args.num_classes, timesteps=args.timesteps,
        encoder_type=graph_op, skeleton_graph_op=graph_op,
        gait_metrics_dim=args.gait_metrics_dim, use_gait_conditioning=False,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=False)
    stage1.eval()

    stage2 = Stage2Model(
        encoder=stage1.encoder, latent_dim=latent_dim, num_joints=args.num_joints,
        num_classes=args.num_classes, d_shared=d_shared,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage2_ckpt, stage2, strict=False)
    stage2.eval()

    stage3 = Stage3Model(
        encoder=stage1.encoder, decoder=stage1.decoder, denoiser=stage1.denoiser,
        latent_dim=latent_dim, num_joints=args.num_joints, num_classes=args.num_classes,
        timesteps=args.timesteps, d_shared=d_shared, gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False, shared_motion_layer=stage2.shared_motion_layer,
    ).to(device)
    load_checkpoint(args.stage3_ckpt, stage3, strict=False)
    stage3.eval()

    # ── Load data ──
    print("Loading dataset...")
    ds = create_dataset(
        dataset_path=None, skeleton_folder=args.skeleton_folder,
        hip_folder=args.hip_folder, wrist_folder=args.wrist_folder,
        window=args.window,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    batch = next(iter(loader))
    x = batch["skeleton"].to(device)
    a_hip = batch["A_hip"].to(device)
    a_wrist = batch["A_wrist"].to(device)
    labels = batch["label"]
    print(f"Batch: {tuple(x.shape)}, labels={labels.tolist()}")

    summary = {}

    with torch.no_grad():
        # ═══════════════════════════════════════════════════════════════
        # ENCODER Z0 (ground truth latent)
        # ═══════════════════════════════════════════════════════════════
        z0_real = stage3.encoder(x, gait_metrics=None)
        print(f"\n{'='*60}")
        print(f"ENCODER Z0 (ground truth)")
        print(f"  shape: {tuple(z0_real.shape)}")
        print(f"  mean={z0_real.mean():.4f}, std={z0_real.std():.4f}")
        print(f"  range=[{z0_real.min():.3f}, {z0_real.max():.3f}]")

        summary["z0_real"] = {
            "mean": float(z0_real.mean()), "std": float(z0_real.std()),
            "min": float(z0_real.min()), "max": float(z0_real.max()),
        }

        # IMU conditioning
        h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)

        # ═══════════════════════════════════════════════════════════════
        # DENOISER RUNS — conditioned
        # ═══════════════════════════════════════════════════════════════
        z0_diffs_cond = []
        for run_i in range(args.num_runs):
            z0_diff = sample_stage3_latents(
                stage3=stage3, shape=z0_real.shape, device=device,
                h_tokens=h_tokens, h_global=h_global,
                a_hip_stream=a_hip, a_wrist_stream=a_wrist,
                gait_metrics=None, sample_steps=args.sample_steps,
                sampler=args.sampler, sample_seed=run_i,
            )
            z0_diffs_cond.append(z0_diff)
            print(f"\n  Run {run_i} (conditioned): mean={z0_diff.mean():.4f}, std={z0_diff.std():.4f}")

        # DENOISER RUNS — unconditioned
        z0_diffs_uncond = []
        for run_i in range(args.num_runs):
            z0_diff = sample_stage3_latents(
                stage3=stage3, shape=z0_real.shape, device=device,
                h_tokens=torch.zeros_like(h_tokens), h_global=torch.zeros_like(h_global),
                a_hip_stream=None, a_wrist_stream=None,
                gait_metrics=None, sample_steps=args.sample_steps,
                sampler=args.sampler, sample_seed=run_i,
            )
            z0_diffs_uncond.append(z0_diff)
            print(f"  Run {run_i} (unconditioned): mean={z0_diff.mean():.4f}, std={z0_diff.std():.4f}")

        z0_cond = z0_diffs_cond[0]
        z0_uncond = z0_diffs_uncond[0]

        # ═══════════════════════════════════════════════════════════════
        # TEST 1: Global L2 distance
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("TEST 1: Global L2 distance to encoder z0")
        for i, z in enumerate(z0_diffs_cond):
            l2 = float(torch.linalg.norm(z - z0_real, dim=-1).mean())
            print(f"  Cond run {i}: L2 = {l2:.4f}")
        for i, z in enumerate(z0_diffs_uncond):
            l2 = float(torch.linalg.norm(z - z0_real, dim=-1).mean())
            print(f"  Uncond run {i}: L2 = {l2:.4f}")

        # ═══════════════════════════════════════════════════════════════
        # TEST 2: Per-joint analysis
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("TEST 2: Per-joint latent comparison")
        print(f"  {'Joint':>5} | {'real_mean':>9} {'real_std':>9} | {'diff_mean':>9} {'diff_std':>9} | {'cosine':>7} {'L2':>7}")
        print(f"  {'-'*5}-+-{'-'*9}-{'-'*9}-+-{'-'*9}-{'-'*9}-+-{'-'*7}-{'-'*7}")
        per_joint_stats = []
        for j in range(args.num_joints):
            real_j = z0_real[:, :, j, :]  # [B, T, D]
            diff_j = z0_cond[:, :, j, :]
            cos = float(cosine_sim(real_j, diff_j, dim=-1).mean())
            l2 = float(torch.linalg.norm(real_j - diff_j, dim=-1).mean())
            r_mean = float(real_j.mean())
            r_std = float(real_j.std())
            d_mean = float(diff_j.mean())
            d_std = float(diff_j.std())
            per_joint_stats.append({"joint": j, "real_mean": r_mean, "real_std": r_std,
                                     "diff_mean": d_mean, "diff_std": d_std,
                                     "cosine": cos, "l2": l2})
            print(f"  {j:5d} | {r_mean:9.4f} {r_std:9.4f} | {d_mean:9.4f} {d_std:9.4f} | {cos:7.4f} {l2:7.4f}")
        summary["per_joint"] = per_joint_stats

        # ═══════════════════════════════════════════════════════════════
        # TEST 3: Per-frame temporal analysis
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("TEST 3: Per-frame cosine similarity (real vs denoiser)")
        per_frame_cos = cosine_sim(
            z0_real.reshape(x.shape[0], x.shape[1], -1),
            z0_cond.reshape(x.shape[0], x.shape[1], -1),
            dim=-1,
        ).mean(dim=0)  # [T]
        frame_indices = [0, 15, 30, 45, 60, 75, 89]
        for fi in frame_indices:
            if fi < per_frame_cos.shape[0]:
                print(f"  Frame {fi:3d}: cosine = {per_frame_cos[fi]:.4f}")
        summary["per_frame_cosine_mean"] = float(per_frame_cos.mean())
        summary["per_frame_cosine_std"] = float(per_frame_cos.std())
        print(f"  Mean across frames: {per_frame_cos.mean():.4f} ± {per_frame_cos.std():.4f}")

        # ═══════════════════════════════════════════════════════════════
        # TEST 4: Temporal autocorrelation (smoothness)
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("TEST 4: Temporal autocorrelation (motion smoothness)")
        for lag in [1, 5, 10]:
            ac_real = temporal_autocorrelation(z0_real, lag)
            ac_cond = temporal_autocorrelation(z0_cond, lag)
            ac_uncond = temporal_autocorrelation(z0_uncond, lag)
            print(f"  Lag {lag:2d}: real={ac_real:.4f}, cond={ac_cond:.4f}, uncond={ac_uncond:.4f}")
            summary[f"autocorr_lag{lag}"] = {"real": ac_real, "cond": ac_cond, "uncond": ac_uncond}

        # ═══════════════════════════════════════════════════════════════
        # TEST 5: Per-dimension analysis (which dims diverge most)
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("TEST 5: Per-dimension divergence (top 10 worst + bottom 10 best)")
        real_per_dim = z0_real.float().mean(dim=(0, 1, 2))  # [D]
        diff_per_dim = z0_cond.float().mean(dim=(0, 1, 2))  # [D]
        real_std_dim = z0_real.float().std(dim=(0, 1, 2))
        diff_std_dim = z0_cond.float().std(dim=(0, 1, 2))
        dim_l2 = (real_per_dim - diff_per_dim).abs()
        dim_std_ratio = diff_std_dim / (real_std_dim + 1e-8)

        sorted_dims = torch.argsort(dim_l2, descending=True)
        print(f"\n  Top 10 divergent dimensions (by mean abs diff):")
        print(f"  {'Dim':>4} | {'real_mean':>9} {'diff_mean':>9} {'abs_diff':>9} | {'real_std':>8} {'diff_std':>8} {'std_ratio':>9}")
        for i in range(10):
            d = int(sorted_dims[i])
            print(f"  {d:4d} | {real_per_dim[d]:9.4f} {diff_per_dim[d]:9.4f} {dim_l2[d]:9.4f} | {real_std_dim[d]:8.4f} {diff_std_dim[d]:8.4f} {dim_std_ratio[d]:9.4f}")

        print(f"\n  Bottom 10 closest dimensions:")
        for i in range(10):
            d = int(sorted_dims[-(i + 1)])
            print(f"  {d:4d} | {real_per_dim[d]:9.4f} {diff_per_dim[d]:9.4f} {dim_l2[d]:9.4f} | {real_std_dim[d]:8.4f} {diff_std_dim[d]:8.4f} {dim_std_ratio[d]:9.4f}")

        summary["dim_mean_abs_diff_mean"] = float(dim_l2.mean())
        summary["dim_std_ratio_mean"] = float(dim_std_ratio.mean())

        # ═══════════════════════════════════════════════════════════════
        # TEST 6: Joint-joint correlation preservation
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("TEST 6: Joint-joint correlation matrix preservation")
        corr_real = joint_correlation_matrix(z0_real)
        corr_cond = joint_correlation_matrix(z0_cond)
        corr_uncond = joint_correlation_matrix(z0_uncond)
        # Compare upper triangles
        triu_idx = np.triu_indices(args.num_joints, k=1)
        corr_diff_cond = np.abs(corr_real[triu_idx] - corr_cond[triu_idx]).mean()
        corr_diff_uncond = np.abs(corr_real[triu_idx] - corr_uncond[triu_idx]).mean()
        corr_corr_cond = float(np.corrcoef(corr_real[triu_idx], corr_cond[triu_idx])[0, 1])
        corr_corr_uncond = float(np.corrcoef(corr_real[triu_idx], corr_uncond[triu_idx])[0, 1])
        print(f"  Correlation of joint-correlation matrices:")
        print(f"    Real vs Conditioned:   r={corr_corr_cond:.4f}, mean_abs_diff={corr_diff_cond:.4f}")
        print(f"    Real vs Unconditioned: r={corr_corr_uncond:.4f}, mean_abs_diff={corr_diff_uncond:.4f}")
        summary["joint_corr"] = {
            "cond_r": corr_corr_cond, "cond_diff": float(corr_diff_cond),
            "uncond_r": corr_corr_uncond, "uncond_diff": float(corr_diff_uncond),
        }

        # ═══════════════════════════════════════════════════════════════
        # TEST 7: Conditioning effect (is conditioning doing anything?)
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("TEST 7: Conditioning effect")
        cond_vs_uncond_l2 = float(torch.linalg.norm(z0_cond - z0_uncond, dim=-1).mean())
        cond_vs_uncond_cos = float(cosine_sim(
            z0_cond.reshape(x.shape[0], -1),
            z0_uncond.reshape(x.shape[0], -1),
            dim=-1,
        ).mean())
        print(f"  L2(cond, uncond) = {cond_vs_uncond_l2:.4f}")
        print(f"  Cosine(cond, uncond) = {cond_vs_uncond_cos:.4f}")
        print(f"  (If cosine ≈ 1.0 and L2 is small, conditioning has no effect)")
        summary["conditioning_effect"] = {
            "l2": cond_vs_uncond_l2, "cosine": cond_vs_uncond_cos,
        }

        # ═══════════════════════════════════════════════════════════════
        # TEST 8: Cross-run consistency
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("TEST 8: Cross-run consistency (same conditioning, different seeds)")
        if args.num_runs >= 2:
            for i in range(args.num_runs):
                for j in range(i + 1, args.num_runs):
                    l2 = float(torch.linalg.norm(z0_diffs_cond[i] - z0_diffs_cond[j], dim=-1).mean())
                    cos = float(cosine_sim(
                        z0_diffs_cond[i].reshape(x.shape[0], -1),
                        z0_diffs_cond[j].reshape(x.shape[0], -1),
                        dim=-1,
                    ).mean())
                    print(f"  Run {i} vs {j}: L2={l2:.4f}, cosine={cos:.4f}")

        # ═══════════════════════════════════════════════════════════════
        # TEST 9: Signal-to-noise ratio of denoiser output
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("TEST 9: Signal analysis")
        # If denoiser output is close to N(0,1) random noise, it hasn't learned anything
        # Compare to actual random noise with same shape
        random_z = torch.randn_like(z0_real) * z0_real.std() + z0_real.mean()
        random_l2 = float(torch.linalg.norm(random_z - z0_real, dim=-1).mean())
        cond_l2 = float(torch.linalg.norm(z0_cond - z0_real, dim=-1).mean())
        uncond_l2 = float(torch.linalg.norm(z0_uncond - z0_real, dim=-1).mean())
        print(f"  L2(random noise, z0_real)     = {random_l2:.4f}  (worst case baseline)")
        print(f"  L2(denoiser cond, z0_real)     = {cond_l2:.4f}")
        print(f"  L2(denoiser uncond, z0_real)   = {uncond_l2:.4f}")
        print(f"  Improvement over random: {((random_l2 - cond_l2) / random_l2 * 100):.1f}%")
        summary["signal_analysis"] = {
            "random_l2": random_l2, "cond_l2": cond_l2, "uncond_l2": uncond_l2,
            "improvement_pct": float((random_l2 - cond_l2) / random_l2 * 100),
        }

        # Also: is the denoiser output closer to z0_real or to other z0_reals from different samples?
        # If closer to z0_real → it learned the mapping. If same → it's generating generic latents.
        z0_real_shuffled = z0_real[torch.randperm(z0_real.shape[0])]
        cross_l2 = float(torch.linalg.norm(z0_cond - z0_real_shuffled, dim=-1).mean())
        self_l2 = float(torch.linalg.norm(z0_cond - z0_real, dim=-1).mean())
        print(f"\n  L2(denoiser, matching z0_real)    = {self_l2:.4f}")
        print(f"  L2(denoiser, wrong z0_real)        = {cross_l2:.4f}")
        if cross_l2 > self_l2:
            print(f"  → Denoiser output is closer to the MATCHING real latent (good)")
        else:
            print(f"  → Denoiser output is NOT closer to matching real — it may be generating generic latents")
        summary["specificity"] = {"self_l2": self_l2, "cross_l2": cross_l2}

    # ═══════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════
    if HAS_PLT:
        print(f"\nRendering plots to {out_dir}/...")

        # Plot 1: Per-joint cosine similarity
        fig, ax = plt.subplots(figsize=(12, 4))
        joints = [s["joint"] for s in per_joint_stats]
        cosines = [s["cosine"] for s in per_joint_stats]
        l2s = [s["l2"] for s in per_joint_stats]
        ax.bar(joints, cosines, color="#2563eb", alpha=0.7)
        ax.set_xlabel("Joint index")
        ax.set_ylabel("Cosine similarity (real vs denoiser)")
        ax.set_title("Per-joint: how well does the denoiser match the encoder?")
        ax.set_ylim(-0.2, 1.0)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "01_per_joint_cosine.png", dpi=150)
        plt.close(fig)
        print("  -> 01_per_joint_cosine.png")

        # Plot 2: Per-joint L2
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(joints, l2s, color="#dc2626", alpha=0.7)
        ax.set_xlabel("Joint index")
        ax.set_ylabel("L2 distance (real vs denoiser)")
        ax.set_title("Per-joint: latent distance from encoder output")
        fig.tight_layout()
        fig.savefig(out_dir / "02_per_joint_l2.png", dpi=150)
        plt.close(fig)
        print("  -> 02_per_joint_l2.png")

        # Plot 3: Per-frame cosine over time
        fig, ax = plt.subplots(figsize=(12, 4))
        frames = np.arange(per_frame_cos.shape[0])
        ax.plot(frames, per_frame_cos.cpu().numpy(), color="#2563eb", linewidth=1.5)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Cosine similarity")
        ax.set_title("Per-frame: temporal alignment of denoiser vs encoder")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_ylim(-0.2, 1.0)
        fig.tight_layout()
        fig.savefig(out_dir / "03_per_frame_cosine.png", dpi=150)
        plt.close(fig)
        print("  -> 03_per_frame_cosine.png")

        # Plot 4: Per-dimension mean comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 6))
        dims = np.arange(latent_dim)
        axes[0].bar(dims, real_per_dim.cpu().numpy(), alpha=0.5, color="#2563eb", label="Encoder z0")
        axes[0].bar(dims, diff_per_dim.cpu().numpy(), alpha=0.5, color="#dc2626", label="Denoiser z0")
        axes[0].set_ylabel("Mean activation")
        axes[0].set_title("Per-dimension mean: encoder vs denoiser")
        axes[0].legend()
        axes[1].bar(dims, dim_std_ratio.cpu().numpy(), alpha=0.7, color="#059669")
        axes[1].axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
        axes[1].set_ylabel("Std ratio (denoiser/encoder)")
        axes[1].set_xlabel("Latent dimension")
        axes[1].set_title("Per-dimension std ratio (1.0 = perfect match)")
        fig.tight_layout()
        fig.savefig(out_dir / "04_per_dimension_comparison.png", dpi=150)
        plt.close(fig)
        print("  -> 04_per_dimension_comparison.png")

        # Plot 5: Joint correlation matrices side by side
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        vmin, vmax = -1, 1
        axes[0].imshow(corr_real, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0].set_title("Encoder z0\njoint correlations")
        axes[1].imshow(corr_cond, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[1].set_title("Denoiser (cond)\njoint correlations")
        axes[2].imshow(corr_uncond, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[2].set_title("Denoiser (uncond)\njoint correlations")
        for ax in axes:
            ax.set_xlabel("Joint")
            ax.set_ylabel("Joint")
        fig.tight_layout()
        fig.savefig(out_dir / "05_joint_correlation_matrices.png", dpi=150)
        plt.close(fig)
        print("  -> 05_joint_correlation_matrices.png")

        # Plot 6: Autocorrelation comparison
        lags = list(range(1, 20))
        ac_real_vals = [temporal_autocorrelation(z0_real, l) for l in lags]
        ac_cond_vals = [temporal_autocorrelation(z0_cond, l) for l in lags]
        ac_uncond_vals = [temporal_autocorrelation(z0_uncond, l) for l in lags]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lags, ac_real_vals, "o-", color="#2563eb", label="Encoder z0", linewidth=2)
        ax.plot(lags, ac_cond_vals, "s-", color="#dc2626", label="Denoiser (cond)", linewidth=2)
        ax.plot(lags, ac_uncond_vals, "^-", color="#059669", label="Denoiser (uncond)", linewidth=2)
        ax.set_xlabel("Lag (frames)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Temporal smoothness: encoder vs denoiser")
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "06_temporal_autocorrelation.png", dpi=150)
        plt.close(fig)
        print("  -> 06_temporal_autocorrelation.png")

    # ── Save summary ──
    summary_path = out_dir / "denoiser_quality_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    # ── Verdict ──
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    cond_l2 = summary["signal_analysis"]["cond_l2"]
    random_l2 = summary["signal_analysis"]["random_l2"]
    improvement = summary["signal_analysis"]["improvement_pct"]
    cos_mean = summary["per_frame_cosine_mean"]
    cond_effect = summary["conditioning_effect"]["cosine"]

    if improvement < 10:
        print(f"  DENOISER IS NEARLY RANDOM — only {improvement:.1f}% better than noise")
    elif improvement < 30:
        print(f"  DENOISER IS WEAK — only {improvement:.1f}% better than noise")
    elif cos_mean < 0.3:
        print(f"  DENOISER HAS WRONG STRUCTURE — improvement={improvement:.1f}% but cosine={cos_mean:.3f}")
    else:
        print(f"  DENOISER IS REASONABLE — {improvement:.1f}% better than noise, cosine={cos_mean:.3f}")

    if cond_effect > 0.99:
        print(f"  CONDITIONING HAS NO EFFECT — cond vs uncond cosine = {cond_effect:.4f}")
    elif cond_effect > 0.95:
        print(f"  CONDITIONING HAS MINIMAL EFFECT — cond vs uncond cosine = {cond_effect:.4f}")
    else:
        print(f"  CONDITIONING IS ACTIVE — cond vs uncond cosine = {cond_effect:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
