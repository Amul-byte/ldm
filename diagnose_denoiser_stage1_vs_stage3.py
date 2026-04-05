"""Compare Stage 1 denoiser (pure, pre-Stage3) vs Stage 3 denoiser.

Answers the key question: does the Stage 1 denoiser produce structured latents
on its own? If yes, Stage 3 training broke it. If no, it was never good.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffusion_model.dataset import create_dataset
from diffusion_model.diffusion import DiffusionProcess
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.training_eval import render_skeleton_panels
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


def _infer_latent_dim(ckpt: str) -> int:
    return int(_load_state_dict(ckpt)["encoder.in_proj.weight"].shape[0])


def temporal_autocorrelation(z: torch.Tensor, lag: int = 1) -> float:
    z1 = z[:, :-lag].reshape(-1).float()
    z2 = z[:, lag:].reshape(-1).float()
    return float(torch.corrcoef(torch.stack([z1, z2]))[0, 1].item())


def joint_correlation_matrix(z: torch.Tensor) -> np.ndarray:
    per_joint = z.float().mean(dim=(0, 1))
    return torch.corrcoef(per_joint).cpu().numpy()


def cosine_sim(a, b, dim=-1):
    return torch.nn.functional.cosine_similarity(a, b, dim=dim)


def run_stage1_reverse(stage1_model, shape, device, timesteps, seed=0):
    """Run the Stage 1 denoiser in unconditional reverse mode."""
    diffusion = stage1_model.diffusion
    denoiser = stage1_model.denoiser
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    z = torch.randn(shape, device=device, generator=generator)
    for t_val in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
        z = diffusion.p_sample(
            denoiser, z, t_tensor,
            h_tokens=None, h_global=None, gait_metrics=None,
            generator=generator,
        )
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True, help="Stage 3 checkpoint (contains the modified denoiser)")
    parser.add_argument("--output_dir", type=str, default="outputs/denoiser_stage1_vs_stage3")
    parser.add_argument("--skeleton_folder", type=str, default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--num_joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--encoder_type", type=str, default="gcn")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = _infer_latent_dim(args.stage1_ckpt)
    graph_op = args.encoder_type
    print(f"Config: latent_dim={latent_dim}, graph_op={graph_op}")

    # ── Load Stage 1 model (pure, before Stage 3 training) ──
    print("\nLoading Stage 1 model (PURE — before Stage 3)...")
    stage1_pure = Stage1Model(
        latent_dim=latent_dim, num_joints=args.num_joints,
        num_classes=args.num_classes, timesteps=args.timesteps,
        encoder_type=graph_op, skeleton_graph_op=graph_op,
        gait_metrics_dim=args.gait_metrics_dim, use_gait_conditioning=False,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1_pure, strict=False)
    stage1_pure.eval()

    # ── Load Stage 3 model (denoiser modified by Stage 3 training) ──
    print("Loading Stage 1 model shell for Stage 3 weights...")
    stage3_shell = Stage1Model(
        latent_dim=latent_dim, num_joints=args.num_joints,
        num_classes=args.num_classes, timesteps=args.timesteps,
        encoder_type=graph_op, skeleton_graph_op=graph_op,
        gait_metrics_dim=args.gait_metrics_dim, use_gait_conditioning=False,
    ).to(device)
    # Load Stage 3 checkpoint into Stage1Model shell — it has encoder, denoiser, decoder
    # Stage 3 saves the full model, so we load with strict=False to skip Stage3-only keys
    load_checkpoint(args.stage3_ckpt, stage3_shell, strict=False)
    stage3_shell.eval()

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
    print(f"Batch: {tuple(x.shape)}")

    summary = {}

    with torch.no_grad():
        # Encoder z0 (ground truth latent)
        z0_real = stage1_pure.encoder(x, gait_metrics=None)
        print(f"\nEncoder z0: mean={z0_real.mean():.4f}, std={z0_real.std():.4f}")

        shape = z0_real.shape  # [B, T, J, D]

        # ── Stage 1 denoiser: unconditional reverse ──
        print(f"\nRunning Stage 1 denoiser (pure, {args.timesteps} reverse steps)...")
        z0_stage1 = run_stage1_reverse(stage1_pure, shape, device, args.timesteps, seed=args.sample_seed)
        print(f"  Stage 1 z0: mean={z0_stage1.mean():.4f}, std={z0_stage1.std():.4f}")

        # ── Stage 3 denoiser: unconditional reverse (h=None) ──
        print(f"Running Stage 3 denoiser (unconditional, {args.timesteps} reverse steps)...")
        z0_stage3 = run_stage1_reverse(stage3_shell, shape, device, args.timesteps, seed=args.sample_seed)
        print(f"  Stage 3 z0: mean={z0_stage3.mean():.4f}, std={z0_stage3.std():.4f}")

        # ═══════════════════════════════════════════════════════════════
        # COMPARE: Structure metrics
        # ═══════════════════════════════════════════════════════════════

        print(f"\n{'='*60}")
        print("TEMPORAL AUTOCORRELATION")
        for lag in [1, 5, 10]:
            ac_real = temporal_autocorrelation(z0_real, lag)
            ac_s1 = temporal_autocorrelation(z0_stage1, lag)
            ac_s3 = temporal_autocorrelation(z0_stage3, lag)
            print(f"  Lag {lag:2d}: encoder={ac_real:.4f}, stage1_denoiser={ac_s1:.4f}, stage3_denoiser={ac_s3:.4f}")
            summary[f"autocorr_lag{lag}"] = {"encoder": ac_real, "stage1": ac_s1, "stage3": ac_s3}

        print(f"\n{'='*60}")
        print("JOINT-JOINT CORRELATION")
        corr_real = joint_correlation_matrix(z0_real)
        corr_s1 = joint_correlation_matrix(z0_stage1)
        corr_s3 = joint_correlation_matrix(z0_stage3)
        triu_idx = np.triu_indices(args.num_joints, k=1)
        corr_r_s1 = float(np.corrcoef(corr_real[triu_idx], corr_s1[triu_idx])[0, 1])
        corr_r_s3 = float(np.corrcoef(corr_real[triu_idx], corr_s3[triu_idx])[0, 1])
        print(f"  Encoder vs Stage1 denoiser: r={corr_r_s1:.4f}")
        print(f"  Encoder vs Stage3 denoiser: r={corr_r_s3:.4f}")
        summary["joint_corr"] = {"stage1": corr_r_s1, "stage3": corr_r_s3}

        print(f"\n{'='*60}")
        print("PER-JOINT COSINE SIMILARITY TO ENCODER Z0")
        cos_s1_all = cosine_sim(z0_real, z0_stage1, dim=-1).mean()
        cos_s3_all = cosine_sim(z0_real, z0_stage3, dim=-1).mean()
        print(f"  Stage1 denoiser: {cos_s1_all:.4f}")
        print(f"  Stage3 denoiser: {cos_s3_all:.4f}")
        summary["cosine_to_real"] = {"stage1": float(cos_s1_all), "stage3": float(cos_s3_all)}

        # ── Decode both and compare visually ──
        print(f"\n{'='*60}")
        print("DECODING (using Stage 1 decoder — untrained but same for both)")
        # Use stage1's decoder for fair comparison (both denoiser outputs go through same decoder)
        x_hat_real = stage1_pure.decoder(z0_real)
        x_hat_s1 = stage1_pure.decoder(z0_stage1)
        x_hat_s3 = stage1_pure.decoder(z0_stage3)

        # Also use stage3's decoder
        x_hat_s3_dec = stage3_shell.decoder(z0_stage3)
        x_hat_real_s3dec = stage3_shell.decoder(z0_real)

        mpjpe_real_s1dec = float(torch.linalg.norm(x_hat_real - x, dim=-1).mean())
        mpjpe_s1_s1dec = float(torch.linalg.norm(x_hat_s1 - x, dim=-1).mean())
        mpjpe_s3_s1dec = float(torch.linalg.norm(x_hat_s3 - x, dim=-1).mean())
        mpjpe_s3_s3dec = float(torch.linalg.norm(x_hat_s3_dec - x, dim=-1).mean())
        mpjpe_real_s3dec = float(torch.linalg.norm(x_hat_real_s3dec - x, dim=-1).mean())

        print(f"  Stage1 decoder(encoder z0):          MPJPE = {mpjpe_real_s1dec:.4f}")
        print(f"  Stage1 decoder(stage1 denoiser z0):  MPJPE = {mpjpe_s1_s1dec:.4f}")
        print(f"  Stage1 decoder(stage3 denoiser z0):  MPJPE = {mpjpe_s3_s1dec:.4f}")
        print(f"  Stage3 decoder(encoder z0):          MPJPE = {mpjpe_real_s3dec:.4f}")
        print(f"  Stage3 decoder(stage3 denoiser z0):  MPJPE = {mpjpe_s3_s3dec:.4f}")
        summary["mpjpe"] = {
            "s1dec_real": mpjpe_real_s1dec, "s1dec_s1denoise": mpjpe_s1_s1dec,
            "s1dec_s3denoise": mpjpe_s3_s1dec, "s3dec_real": mpjpe_real_s3dec,
            "s3dec_s3denoise": mpjpe_s3_s3dec,
        }

        # ── Project latents through out_proj for visualization ──
        z0_real_proj = stage1_pure.decoder.out_proj(z0_real)
        z0_s1_proj = stage1_pure.decoder.out_proj(z0_stage1)
        z0_s3_proj = stage1_pure.decoder.out_proj(z0_stage3)

    # ═══════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════
    print(f"\nRendering plots to {out_dir}/...")
    idx = 0

    # Plot 1: Projected latents comparison
    render_skeleton_panels(
        out_dir / "01_latent_projected_comparison.png",
        [
            x[idx].cpu().numpy(),
            z0_real_proj[idx].cpu().numpy(),
            z0_s1_proj[idx].cpu().numpy(),
            z0_s3_proj[idx].cpu().numpy(),
        ],
        ["Real skeleton", "Encoder z0\n(projected)", "Stage1 denoiser z0\n(projected)", "Stage3 denoiser z0\n(projected)"],
    )
    print("  -> 01_latent_projected_comparison.png")

    # Plot 2: Decoded skeletons
    render_skeleton_panels(
        out_dir / "02_decoded_comparison.png",
        [
            x[idx].cpu().numpy(),
            x_hat_real[idx].cpu().numpy(),
            x_hat_s1[idx].cpu().numpy(),
            x_hat_s3[idx].cpu().numpy(),
            x_hat_real_s3dec[idx].cpu().numpy(),
            x_hat_s3_dec[idx].cpu().numpy(),
        ],
        [
            "Real",
            "S1 dec(enc z0)",
            "S1 dec(S1 denoise)",
            "S1 dec(S3 denoise)",
            "S3 dec(enc z0)",
            "S3 dec(S3 denoise)",
        ],
    )
    print("  -> 02_decoded_comparison.png")

    # Plot 3: Joint correlation matrices
    if HAS_PLT:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].imshow(corr_real, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0].set_title("Encoder z0")
        axes[1].imshow(corr_s1, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].set_title("Stage 1 denoiser")
        axes[2].imshow(corr_s3, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[2].set_title("Stage 3 denoiser")
        for ax in axes:
            ax.set_xlabel("Joint")
            ax.set_ylabel("Joint")
        fig.tight_layout()
        fig.savefig(out_dir / "03_joint_correlations.png", dpi=150)
        plt.close(fig)
        print("  -> 03_joint_correlations.png")

        # Plot 4: Temporal autocorrelation
        lags = list(range(1, 20))
        ac_real_vals = [temporal_autocorrelation(z0_real, l) for l in lags]
        ac_s1_vals = [temporal_autocorrelation(z0_stage1, l) for l in lags]
        ac_s3_vals = [temporal_autocorrelation(z0_stage3, l) for l in lags]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lags, ac_real_vals, "o-", color="#2563eb", label="Encoder z0", linewidth=2)
        ax.plot(lags, ac_s1_vals, "s-", color="#059669", label="Stage 1 denoiser", linewidth=2)
        ax.plot(lags, ac_s3_vals, "^-", color="#dc2626", label="Stage 3 denoiser", linewidth=2)
        ax.set_xlabel("Lag (frames)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Temporal smoothness: Stage 1 vs Stage 3 denoiser")
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "04_temporal_autocorrelation.png", dpi=150)
        plt.close(fig)
        print("  -> 04_temporal_autocorrelation.png")

    # Save summary
    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_dir / 'comparison_summary.json'}")

    # ── Verdict ──
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    s1_autocorr = summary["autocorr_lag1"]["stage1"]
    s3_autocorr = summary["autocorr_lag1"]["stage3"]
    real_autocorr = summary["autocorr_lag1"]["encoder"]
    if s1_autocorr > 0.7:
        print(f"  Stage 1 denoiser has temporal structure (autocorr={s1_autocorr:.3f})")
        if s3_autocorr < 0.6:
            print(f"  Stage 3 BROKE the denoiser (autocorr dropped to {s3_autocorr:.3f})")
            print(f"  → Problem is in Stage 3 training")
        else:
            print(f"  Stage 3 denoiser preserved structure (autocorr={s3_autocorr:.3f})")
    else:
        print(f"  Stage 1 denoiser ALSO lacks temporal structure (autocorr={s1_autocorr:.3f})")
        if s1_autocorr < 0.6:
            print(f"  → The denoiser never learned temporal structure")
            print(f"  → Problem is in Stage 1 training (encoder autocorr={real_autocorr:.3f})")
        else:
            print(f"  → Partial structure, needs investigation")

    print("\nDone!")


if __name__ == "__main__":
    main()
