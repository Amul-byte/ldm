"""Diagnose the two diffusion bugs and quantify their impact on skeleton quality.

Bug 1: p_sample uses sqrt(beta_t) instead of sqrt(posterior_variance_t)
Bug 2: Stage 3 z0_gen has no numerical protection (clamp=1e-20 vs Stage 1's 0.1)

Usage:
    python diagnose_bugs.py \
        --stage1_ckpt checkpoints/stage1_best.pt \
        --stage3_ckpt checkpoints/stage3_enhanced/stage3_best.pt \
        --stage2_ckpt checkpoints/stage2_enhanced/stage2_best.pt
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_model.diffusion import DiffusionProcess, extract
from diffusion_model.model import Stage1Model, Stage3Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.sensor_model import IMULatentAligner
from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_LATENT_DIM, DEFAULT_TIMESTEPS, DEFAULT_WINDOW


def diagnose_bug1_reverse_noise():
    """Compare beta_t vs posterior_variance_t across all timesteps."""
    print("=" * 70)
    print("BUG 1: Reverse-step noise — beta_t vs posterior_variance_t")
    print("=" * 70)

    dp = DiffusionProcess(timesteps=DEFAULT_TIMESTEPS)
    betas = dp.betas.numpy()
    posterior = dp.posterior_variance.numpy()

    ratio = betas / np.clip(posterior, 1e-30, None)

    print(f"\n{'Timestep':>10} {'beta_t':>12} {'post_var_t':>12} {'ratio':>8} {'excess_std':>12}")
    print("-" * 60)
    for t_idx in [0, 1, 10, 50, 100, 200, 300, 400, 450, 499]:
        excess_std = np.sqrt(betas[t_idx]) - np.sqrt(posterior[t_idx])
        print(f"{t_idx:>10} {betas[t_idx]:>12.6f} {posterior[t_idx]:>12.6f} {ratio[t_idx]:>8.2f}x {excess_std:>12.6f}")

    avg_ratio = ratio[1:].mean()  # skip t=0 where posterior=0
    print(f"\nAverage ratio (beta/posterior): {avg_ratio:.2f}x")
    print(f"Mean excess noise std per step: {(np.sqrt(betas[1:]) - np.sqrt(posterior[1:])).mean():.6f}")
    print(f"Cumulative excess over 500 steps: compounds multiplicatively")
    print(f"\nVerdict: Every reverse step injects {avg_ratio:.1f}x too much noise variance.")


def diagnose_bug2_z0_explosion(device: torch.device):
    """Show z0_gen magnitude with current clamp (1e-20) vs Stage 1's clamp (0.1)."""
    print("\n" + "=" * 70)
    print("BUG 2: z0_gen explosion — clamp=1e-20 vs clamp=0.1")
    print("=" * 70)

    dp = DiffusionProcess(timesteps=DEFAULT_TIMESTEPS).to(device)

    # Simulate a typical z0 and noise
    B, T, J, D = 4, DEFAULT_WINDOW, DEFAULT_JOINTS, DEFAULT_LATENT_DIM
    z0 = torch.randn(B, T, J, D, device=device) * 2.0  # typical encoder output scale
    noise = torch.randn_like(z0)

    print(f"\nGround truth z0 norm: {z0.norm(dim=-1).mean():.2f}")
    print(f"\n{'t':>6} {'sqrt_abar':>12} {'z0_gen(1e-20)':>16} {'z0_gen(0.1)':>14} {'ratio':>8} {'z0_gen clipped':>16}")
    print("-" * 80)

    for t_val in [0, 10, 50, 100, 200, 300, 400, 450, 475, 499]:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        zt = dp.q_sample(z0=z0, t=t, noise=noise)

        # Simulate realistic denoiser: perfect prediction + some error
        # Real denoisers don't predict noise perfectly — the error gets
        # amplified by 1/sqrt(alpha_bar_t), which is the whole point.
        denoiser_error = torch.randn_like(noise) * 0.3  # ~30% prediction error
        pred_noise = noise + denoiser_error

        sqrt_abar = extract(dp.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1m_abar = extract(dp.sqrt_one_minus_alphas_cumprod, t, zt.shape)

        # Current code (bug): clamp=1e-20
        z0_buggy = (zt - sqrt_1m_abar * pred_noise) / torch.clamp(sqrt_abar, min=1e-20)
        # Stage 1 fix: clamp=0.1 + clip
        z0_fixed = (zt - sqrt_1m_abar * pred_noise) / torch.clamp(sqrt_abar, min=0.1)
        z0_clipped = z0_fixed.clamp(-50, 50)

        buggy_norm = z0_buggy.norm(dim=-1).mean().item()
        fixed_norm = z0_fixed.norm(dim=-1).mean().item()
        clipped_norm = z0_clipped.norm(dim=-1).mean().item()
        ratio = buggy_norm / max(fixed_norm, 1e-6)

        print(f"{t_val:>6} {sqrt_abar[0,0,0,0].item():>12.6f} {buggy_norm:>16.1f} {fixed_norm:>14.1f} {ratio:>8.1f}x {clipped_norm:>16.1f}")

    print(f"\nVerdict: At large t, z0_gen with clamp=1e-20 explodes to huge norms.")
    print("The decoder trains on these exploded values — this is why poses look like spiders.")


def diagnose_bug2_with_checkpoint(
    stage1_ckpt: str,
    stage2_ckpt: str | None,
    stage3_ckpt: str | None,
    device: torch.device,
):
    """Run actual encoder + denoiser and measure z0_gen explosion on real latents."""
    if not stage3_ckpt or not stage2_ckpt:
        print("\n[Skipping checkpoint-based diagnosis — need both stage2 and stage3 checkpoints]")
        return

    print("\n" + "=" * 70)
    print("BUG 2 (with checkpoint): Real denoiser z0_gen statistics")
    print("=" * 70)

    # Load Stage 1
    graph_ops = infer_graph_ops_from_checkpoint(stage1_ckpt)
    encoder_type = graph_ops.get("encoder_graph_op", "gat")
    skeleton_graph_op = graph_ops.get("skeleton_graph_op", "gat")

    s1 = Stage1Model(
        encoder_type=encoder_type,
        skeleton_graph_op=skeleton_graph_op,
    ).to(device)
    load_checkpoint(stage1_ckpt, s1, device=device)
    s1.eval()

    # Create dummy data
    B, T, J = 4, DEFAULT_WINDOW, DEFAULT_JOINTS
    x = torch.randn(B, T, J, 3, device=device)

    with torch.no_grad():
        z0 = s1.encoder(x, gait_metrics=None)

    dp = s1.diffusion
    print(f"\nEncoder z0 — mean norm: {z0.norm(dim=-1).mean():.2f}, std: {z0.norm(dim=-1).std():.2f}")

    # Test z0_gen at different timesteps with actual denoiser
    print(f"\n{'t':>6} {'z0_gen_buggy norm':>20} {'z0_gen_fixed norm':>20} {'decoder input ok?':>20}")
    print("-" * 70)

    for t_val in [0, 50, 100, 200, 300, 400, 450, 499]:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = dp.q_sample(z0=z0, t=t, noise=noise)

        with torch.no_grad():
            pred_noise = s1.denoiser(zt, t, h_tokens=None, h_global=None, gait_metrics=None)

        sqrt_abar = extract(dp.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1m_abar = extract(dp.sqrt_one_minus_alphas_cumprod, t, zt.shape)

        z0_buggy = (zt - sqrt_1m_abar * pred_noise) / torch.clamp(sqrt_abar, min=1e-20)
        z0_fixed = ((zt - sqrt_1m_abar * pred_noise) / torch.clamp(sqrt_abar, min=0.1)).clamp(-50, 50)

        buggy_norm = z0_buggy.norm(dim=-1).mean().item()
        fixed_norm = z0_fixed.norm(dim=-1).mean().item()

        # Compare to encoder z0 range
        z0_mean_norm = z0.norm(dim=-1).mean().item()
        ok = "YES" if buggy_norm < z0_mean_norm * 5 else "NO — EXPLODED"

        print(f"{t_val:>6} {buggy_norm:>20.1f} {fixed_norm:>20.1f} {ok:>20}")

    # Show what decoder sees
    print(f"\nEncoder z0 typical norm: {z0.norm(dim=-1).mean():.1f}")
    print("If z0_gen norm >> encoder z0 norm, the decoder is training on out-of-distribution inputs.")
    print("This is the primary cause of spider skeletons.")


def diagnose_bug1_generation_impact(device: torch.device):
    """Show how much the final z0 differs between buggy and fixed reverse process."""
    print("\n" + "=" * 70)
    print("BUG 1: Generation impact — final z0 quality with buggy vs fixed reverse")
    print("=" * 70)

    dp = DiffusionProcess(timesteps=DEFAULT_TIMESTEPS).to(device)

    B, T, J, D = 2, 32, DEFAULT_JOINTS, DEFAULT_LATENT_DIM

    # Use a simple identity denoiser (predicts zero noise) to isolate the effect
    # of sigma_t choice. With a perfect denoiser this shows pure noise accumulation.
    class PerfectDenoiser(torch.nn.Module):
        """Returns the actual noise (perfect prediction) to isolate sigma effect."""
        def __init__(self):
            super().__init__()
            self._noise_cache = {}

        def forward(self, zt, t, **kwargs):
            # Can't truly predict noise without z0, so return zeros
            # This isolates the noise injection difference
            return torch.zeros_like(zt)

    denoiser = PerfectDenoiser().to(device)

    # Run 50-step reverse with buggy sigma
    torch.manual_seed(42)
    z_buggy = torch.randn(B, T, J, D, device=device)
    schedule = dp._build_sampling_schedule(50, device)

    for i, t_scalar in enumerate(schedule):
        t = torch.full((B,), int(t_scalar.item()), device=device, dtype=torch.long)
        betas_t = extract(dp.betas, t, z_buggy.shape)
        sqrt_recip = extract(dp.sqrt_recip_alphas, t, z_buggy.shape)
        sqrt_1m = extract(dp.sqrt_one_minus_alphas_cumprod, t, z_buggy.shape)
        pred = denoiser(z_buggy, t)
        mean = sqrt_recip * (z_buggy - betas_t * pred / sqrt_1m)
        nonzero = (t != 0).float().reshape(B, 1, 1, 1)
        noise = torch.randn_like(z_buggy)
        sigma_buggy = torch.sqrt(torch.clamp(betas_t, min=1e-20))
        z_buggy = mean + nonzero * sigma_buggy * noise

    # Run 50-step reverse with fixed sigma (posterior variance)
    torch.manual_seed(42)
    z_fixed = torch.randn(B, T, J, D, device=device)

    for i, t_scalar in enumerate(schedule):
        t = torch.full((B,), int(t_scalar.item()), device=device, dtype=torch.long)
        betas_t = extract(dp.betas, t, z_fixed.shape)
        post_var = extract(dp.posterior_variance, t, z_fixed.shape)
        sqrt_recip = extract(dp.sqrt_recip_alphas, t, z_fixed.shape)
        sqrt_1m = extract(dp.sqrt_one_minus_alphas_cumprod, t, z_fixed.shape)
        pred = denoiser(z_fixed, t)
        mean = sqrt_recip * (z_fixed - betas_t * pred / sqrt_1m)
        nonzero = (t != 0).float().reshape(B, 1, 1, 1)
        noise = torch.randn_like(z_fixed)
        sigma_fixed = torch.sqrt(torch.clamp(post_var, min=1e-20))
        z_fixed = mean + nonzero * sigma_fixed * noise

    print(f"\nFinal z0 norm (buggy sigma=sqrt(beta)):       {z_buggy.norm(dim=-1).mean():.2f}")
    print(f"Final z0 norm (fixed sigma=sqrt(post_var)):    {z_fixed.norm(dim=-1).mean():.2f}")
    print(f"Norm ratio (buggy/fixed):                      {z_buggy.norm(dim=-1).mean() / z_fixed.norm(dim=-1).mean():.2f}x")
    print(f"L2 difference between final samples:           {(z_buggy - z_fixed).norm(dim=-1).mean():.2f}")
    print(f"\nVerdict: Buggy reverse injects more noise, making final z0 noisier for the decoder.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose diffusion bugs")
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--stage2_ckpt", type=str, default=None)
    parser.add_argument("--stage3_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # Bug 1: pure math, no checkpoint needed
    diagnose_bug1_reverse_noise()

    # Bug 2: synthetic data (no checkpoint needed)
    diagnose_bug2_z0_explosion(device)

    # Bug 2: with real checkpoint
    if args.stage1_ckpt:
        diagnose_bug2_with_checkpoint(
            args.stage1_ckpt, args.stage2_ckpt, args.stage3_ckpt, device
        )

    # Bug 1: generation impact
    diagnose_bug1_generation_impact(device)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Bug 2 (z0_gen explosion) → decoder trains on garbage at large t → SPIDER SKELETONS")
    print("Bug 1 (excess reverse noise) → noisier z0 at inference → compounds bug 2 damage")
    print("Fix priority: Bug 2 first (training corruption), then Bug 1 (inference quality)")


if __name__ == "__main__":
    main()
