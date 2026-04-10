"""Diagnose WHY the diffusion reverse process produces z0 at wrong scale.

Traces the reverse process step-by-step to find where scale is lost.
Also tests whether the Stage 3 decoder works on correctly-scaled latents.

Usage:
    python diagnose_scale.py \
        --stage1_ckpt new_checkpoints/stage1_enhanced/stage1_best.pt \
        --stage2_ckpt new_checkpoints/stage2_enhanced/stage2_best.pt \
        --stage3_ckpt new_checkpoints/stage3_enhanced0.1/stage3_best.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from diffusion_model.dataset import create_dataset
from diffusion_model.diffusion import DiffusionProcess, extract
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.util import (
    DEFAULT_JOINTS, DEFAULT_LATENT_DIM, DEFAULT_TIMESTEPS, DEFAULT_WINDOW,
    get_skeleton_edges,
)


# ── Visualization ───────────────────────────────────────────────

def _project_to_2d(pts):
    centered = pts - pts[:, 0:1, :]
    flat = centered.reshape(-1, 3)
    finite = flat[np.isfinite(flat).all(-1)]
    if finite.size == 0:
        return centered[..., :2]
    spread = np.ptp(finite, axis=0)
    axes = np.argsort(spread)[-2:]
    axes.sort()
    return centered[..., axes]


def _draw_skeleton(pts_2d, canvas=400, title=""):
    img = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    finite = pts_2d[np.isfinite(pts_2d).all(-1)]
    if finite.size == 0:
        draw.text((10, 10), f"{title}\n[NO DATA]", fill=(255, 0, 0))
        return img
    mn, mx = finite.min(0), finite.max(0)
    span = max((mx - mn).max(), 1e-6)
    margin = 40
    usable = canvas - 2 * margin
    pts_n = (pts_2d - mn) / span * usable + margin
    for i, j in get_skeleton_edges():
        if i < len(pts_n) and j < len(pts_n):
            x1, y1 = float(pts_n[i, 0]), float(pts_n[i, 1])
            x2, y2 = float(pts_n[j, 0]), float(pts_n[j, 1])
            if all(np.isfinite([x1, y1, x2, y2])):
                draw.line((x1, y1, x2, y2), fill=(50, 50, 50), width=2)
    for jj in range(len(pts_n)):
        x, y = float(pts_n[jj, 0]), float(pts_n[jj, 1])
        if np.isfinite(x) and np.isfinite(y):
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(20, 60, 210))
    draw.text((5, 5), title, fill=(0, 0, 0))
    return img


def _save_panel(skeletons: dict, out_path: str, frame=0, canvas=350):
    imgs = []
    for name, skel in skeletons.items():
        pts = _project_to_2d(skel)
        f = min(frame, len(pts) - 1)
        imgs.append(_draw_skeleton(pts[f], canvas=canvas, title=name))
    w = canvas * len(imgs) + 10 * (len(imgs) - 1)
    panel = Image.new("RGB", (w, canvas), (255, 255, 255))
    for i, img in enumerate(imgs):
        panel.paste(img, (i * (canvas + 10), 0))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    panel.save(out_path)


# ── Diagnostics ─────────────────────────────────────────────────

def test1_denoiser_noise_prediction_quality(model, x_real, device):
    """How well does the denoiser actually predict noise at each timestep?"""
    print("\n" + "=" * 70)
    print("TEST 1: Denoiser noise prediction quality")
    print("  If noise prediction MSE is high, the denoiser hasn't learned well.")
    print("=" * 70)

    with torch.no_grad():
        z0 = model.encoder(x_real.to(device), gait_metrics=None)

    dp = model.diffusion

    print(f"\n{'t':>6} {'noise_mse':>12} {'pred_norm':>12} {'true_norm':>12} {'cosine_sim':>12} {'pred/true':>12}")
    print("-" * 68)

    for t_val in [0, 1, 5, 10, 50, 100, 200, 300, 400, 450, 499]:
        t = torch.full((x_real.shape[0],), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = dp.q_sample(z0=z0, t=t, noise=noise)

        with torch.no_grad():
            pred_noise = model.denoiser(zt, t, h_tokens=None, h_global=None, gait_metrics=None)

        mse = F.mse_loss(pred_noise, noise).item()
        pred_n = pred_noise.norm(dim=-1).mean().item()
        true_n = noise.norm(dim=-1).mean().item()
        # Cosine similarity between predicted and true noise
        cos = F.cosine_similarity(
            pred_noise.reshape(-1, pred_noise.shape[-1]),
            noise.reshape(-1, noise.shape[-1]),
            dim=-1,
        ).mean().item()
        ratio = pred_n / max(true_n, 1e-6)

        print(f"{t_val:>6} {mse:>12.4f} {pred_n:>12.2f} {true_n:>12.2f} {cos:>12.4f} {ratio:>12.3f}")

    print("\nInterpretation:")
    print("  cosine_sim > 0.5: denoiser learned the noise direction")
    print("  cosine_sim < 0.1: denoiser is essentially guessing")
    print("  pred/true ratio != 1.0: denoiser predicts wrong noise magnitude")


def test2_reverse_process_step_by_step(model, x_real, device):
    """Trace the norm of z through the reverse process step by step."""
    print("\n" + "=" * 70)
    print("TEST 2: Reverse process z norm trajectory")
    print("  Shows how z evolves from noise (t=T) to signal (t=0)")
    print("=" * 70)

    with torch.no_grad():
        z0_real = model.encoder(x_real.to(device), gait_metrics=None)

    dp = model.diffusion
    B, T, J, D = z0_real.shape

    # Forward process: show what z0_real looks like at different t
    print(f"\nz0_real norm: {z0_real.norm(dim=-1).mean():.2f}")
    print(f"\nFORWARD PROCESS (adding noise to z0_real):")
    print(f"{'t':>6} {'zt_norm':>12} {'signal_frac':>14}")
    print("-" * 36)
    for t_val in [0, 100, 200, 300, 400, 499]:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        zt = dp.q_sample(z0=z0_real, t=t)
        abar = dp.alphas_cumprod[t_val].item()
        print(f"{t_val:>6} {zt.norm(dim=-1).mean().item():>12.2f} {abar:>14.4f}")

    # REVERSE PROCESS: start from pure noise and track norm
    print(f"\nREVERSE PROCESS (denoising from noise):")
    print(f"{'step':>6} {'t':>6} {'z_norm':>12} {'expected_norm':>14} {'ratio':>8}")
    print("-" * 50)

    z = torch.randn(B, T, J, D, device=device)
    schedule = dp._build_sampling_schedule(50, device)

    step = 0
    for t_scalar in schedule:
        t_val = int(t_scalar.item())
        t = torch.full((B,), t_val, device=device, dtype=torch.long)

        z_norm = z.norm(dim=-1).mean().item()

        # Expected norm at this timestep: sqrt(alpha_bar_t) * ||z0|| + sqrt(1-alpha_bar_t) * ||noise||
        abar = dp.alphas_cumprod[t_val].item()
        expected = np.sqrt(abar) * z0_real.norm(dim=-1).mean().item() + np.sqrt(1 - abar) * np.sqrt(D)
        ratio = z_norm / max(expected, 1e-6)

        if step % 5 == 0 or step == len(schedule) - 1:
            print(f"{step:>6} {t_val:>6} {z_norm:>12.2f} {expected:>14.2f} {ratio:>8.2f}")

        # Take reverse step
        with torch.no_grad():
            z = dp.p_sample(denoiser=model.denoiser, zt=z, t=t)
        step += 1

    final_norm = z.norm(dim=-1).mean().item()
    target_norm = z0_real.norm(dim=-1).mean().item()
    print(f"\nFinal z0_gen norm: {final_norm:.2f}")
    print(f"Target z0_real norm: {target_norm:.2f}")
    print(f"Ratio: {final_norm / target_norm:.2f}")
    return z


def test3_forward_backward_consistency(model, x_real, device):
    """Add noise then denoise — does the denoiser recover z0?"""
    print("\n" + "=" * 70)
    print("TEST 3: Forward-then-backward consistency")
    print("  Noise z0_real to t, then one-step denoise back. How close is z0_est?")
    print("=" * 70)

    with torch.no_grad():
        z0_real = model.encoder(x_real.to(device), gait_metrics=None)

    dp = model.diffusion
    z0_norm = z0_real.norm(dim=-1).mean().item()

    print(f"\n{'t':>6} {'z0_est_norm':>14} {'z0_real_norm':>14} {'l2_error':>12} {'cos_sim':>10}")
    print("-" * 60)

    for t_val in [1, 10, 50, 100, 200, 300, 400, 499]:
        t = torch.full((x_real.shape[0],), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(z0_real)
        zt = dp.q_sample(z0=z0_real, t=t, noise=noise)

        with torch.no_grad():
            pred_noise = model.denoiser(zt, t, h_tokens=None, h_global=None, gait_metrics=None)

        # One-step z0 estimate
        sqrt_abar = extract(dp.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1m_abar = extract(dp.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        z0_est = (zt - sqrt_1m_abar * pred_noise) / sqrt_abar.clamp(min=0.01)

        est_norm = z0_est.norm(dim=-1).mean().item()
        l2 = (z0_est - z0_real).norm(dim=-1).mean().item()
        cos = F.cosine_similarity(
            z0_est.reshape(-1, z0_est.shape[-1]),
            z0_real.reshape(-1, z0_real.shape[-1]),
            dim=-1
        ).mean().item()

        print(f"{t_val:>6} {est_norm:>14.2f} {z0_norm:>14.2f} {l2:>12.2f} {cos:>10.4f}")


def test4_stage3_decoder_on_real_z0(stage1, stage3, x_real, device, out_dir):
    """Does the Stage 3 decoder produce good skeletons from REAL encoder z0?"""
    print("\n" + "=" * 70)
    print("TEST 4: Stage 3 decoder on real encoder z0 (bypassing diffusion)")
    print("  If this is a spider, decoder training failed.")
    print("  If this is a human, decoder is fine but diffusion gives bad z0.")
    print("=" * 70)

    with torch.no_grad():
        z0_real = stage3.encoder(x_real.to(device), gait_metrics=None)
        x_from_real_z0 = stage3.decoder(z0_real)

    skel_real = x_real[0].cpu().numpy()
    skel_decoded = x_from_real_z0[0].cpu().numpy()

    mpjpe = np.linalg.norm(skel_real - skel_decoded, axis=-1).mean()
    print(f"  z0_real norm: {z0_real.norm(dim=-1).mean():.2f}")
    print(f"  x_from_real_z0 range: [{skel_decoded.min():.3f}, {skel_decoded.max():.3f}]")
    print(f"  x_real range: [{skel_real.min():.3f}, {skel_real.max():.3f}]")
    print(f"  MPJPE (Stage3 decoder on real z0): {mpjpe:.4f}")

    bone_real = np.stack([np.linalg.norm(skel_real[:, i] - skel_real[:, j], axis=-1)
                          for i, j in get_skeleton_edges()], -1)
    bone_dec = np.stack([np.linalg.norm(skel_decoded[:, i] - skel_decoded[:, j], axis=-1)
                         for i, j in get_skeleton_edges()], -1)
    print(f"  Bone length mean: real={bone_real.mean():.4f}  decoded={bone_dec.mean():.4f}")

    _save_panel(
        {"Real": skel_real, "S3 Dec(real z0)": skel_decoded},
        os.path.join(out_dir, "test4_s3_decoder_real_z0.png"),
    )
    print(f"  -> Saved: {out_dir}/test4_s3_decoder_real_z0.png")

    if mpjpe < 0.3:
        print("  >>> DECODER IS FINE. Problem is in diffusion (wrong z0 scale).")
    else:
        print("  >>> DECODER IS BROKEN. Cannot reconstruct even from perfect z0.")
    return mpjpe


def test5_rescaled_z0_gen(stage3, stage2, x_real, a_hip, a_wrist, device, out_dir):
    """What if we rescale z0_gen to match z0_real's norm? Does it fix spiders?"""
    print("\n" + "=" * 70)
    print("TEST 5: Rescaling z0_gen to match z0_real norm")
    print("  If rescaling fixes spiders, the denoiser is correct up to scale.")
    print("=" * 70)

    with torch.no_grad():
        z0_real = stage3.encoder(x_real.to(device), gait_metrics=None)
        h_tokens, h_global = stage2.aligner(
            a_hip_stream=a_hip.to(device),
            a_wrist_stream=a_wrist.to(device),
        )

        # Generate with DDIM
        z0_gen = stage3.diffusion.p_sample_loop_ddim(
            denoiser=stage3.denoiser,
            shape=z0_real.shape,
            device=device,
            sample_steps=50,
            h_tokens=h_tokens,
            h_global=h_global,
        )

    real_norm = z0_real.norm(dim=-1, keepdim=True).mean()
    gen_norm = z0_gen.norm(dim=-1, keepdim=True).mean()
    scale_factor = real_norm / gen_norm

    print(f"  z0_real mean norm: {real_norm.item():.2f}")
    print(f"  z0_gen mean norm: {gen_norm.item():.2f}")
    print(f"  Scale factor: {scale_factor.item():.2f}")

    z0_rescaled = z0_gen * scale_factor

    with torch.no_grad():
        x_raw = stage3.decoder(z0_gen)
        x_rescaled = stage3.decoder(z0_rescaled)
        x_from_real = stage3.decoder(z0_real)

    skel_real = x_real[0].cpu().numpy()
    skel_raw = x_raw[0].cpu().numpy()
    skel_rescaled = x_rescaled[0].cpu().numpy()
    skel_from_real = x_from_real[0].cpu().numpy()

    mpjpe_raw = np.linalg.norm(skel_real - skel_raw, axis=-1).mean()
    mpjpe_rescaled = np.linalg.norm(skel_real - skel_rescaled, axis=-1).mean()
    mpjpe_from_real = np.linalg.norm(skel_real - skel_from_real, axis=-1).mean()

    print(f"  MPJPE (raw z0_gen):         {mpjpe_raw:.4f}")
    print(f"  MPJPE (rescaled z0_gen):    {mpjpe_rescaled:.4f}")
    print(f"  MPJPE (real z0):            {mpjpe_from_real:.4f}")

    # Joint angles
    for name, (a, b, c) in [("L_knee", (18,19,20)), ("R_knee", (22,23,24))]:
        for label, s in [("real", skel_real), ("raw", skel_raw), ("rescaled", skel_rescaled), ("from_real", skel_from_real)]:
            v1 = s[:, a] - s[:, b]
            v2 = s[:, c] - s[:, b]
            cos = np.sum(v1*v2, -1) / (np.linalg.norm(v1,-1)*np.linalg.norm(v2,-1)+1e-8)
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
            print(f"  {name} ({label}): {ang:.1f} deg")

    _save_panel(
        {"Real": skel_real, "Raw z0_gen": skel_raw, "Rescaled z0_gen": skel_rescaled, "From real z0": skel_from_real},
        os.path.join(out_dir, "test5_rescaled.png"),
    )
    print(f"  -> Saved: {out_dir}/test5_rescaled.png")


def test6_encoder_latent_distribution(model, x_real, device):
    """Examine the encoder's latent distribution — is it far from N(0,I)?"""
    print("\n" + "=" * 70)
    print("TEST 6: Encoder latent distribution analysis")
    print("  DDPM assumes z0 ~ some distribution. If z0 is far from N(0,I),")
    print("  the reverse process starting from N(0,I) may not reach z0's scale.")
    print("=" * 70)

    with torch.no_grad():
        z0 = model.encoder(x_real.to(device), gait_metrics=None)

    z_flat = z0.reshape(-1, z0.shape[-1])
    per_dim_mean = z_flat.mean(dim=0)
    per_dim_std = z_flat.std(dim=0)
    per_dim_norm = z_flat.norm(dim=-1)

    print(f"  z0 shape: {z0.shape}")
    print(f"  Per-sample norm: mean={per_dim_norm.mean():.2f}, std={per_dim_norm.std():.2f}")
    print(f"  Per-dim mean: mean={per_dim_mean.mean():.4f}, std={per_dim_mean.std():.4f}")
    print(f"  Per-dim std:  mean={per_dim_std.mean():.4f}, std={per_dim_std.std():.4f}")
    print(f"  Overall mean: {z_flat.mean():.4f}")
    print(f"  Overall std:  {z_flat.std():.4f}")

    # Expected norm for N(0, sigma^2 I) is sigma * sqrt(D)
    D = z0.shape[-1]
    observed_std = z_flat.std().item()
    expected_norm = observed_std * np.sqrt(D)
    actual_norm = per_dim_norm.mean().item()
    print(f"\n  If z0 ~ N(0, {observed_std:.2f}^2 I):")
    print(f"    Expected norm: {expected_norm:.2f}")
    print(f"    Actual norm:   {actual_norm:.2f}")

    # DDPM starts reverse from N(0, I), which has norm sqrt(D)
    n01_norm = np.sqrt(D)
    print(f"\n  N(0, I) norm: {n01_norm:.2f}")
    print(f"  z0_real norm: {actual_norm:.2f}")
    print(f"  z0 is {actual_norm / n01_norm:.1f}x larger than N(0,I)")

    if actual_norm / n01_norm > 3.0:
        print(f"\n  *** z0 norm is {actual_norm/n01_norm:.1f}x larger than unit normal.")
        print(f"  *** The diffusion forward process maps z0 -> N(0,I) at t=T.")
        print(f"  *** The reverse process starts from N(0,I) and must reconstruct z0.")
        print(f"  *** If the denoiser can't amplify the norm by {actual_norm/n01_norm:.1f}x, spiders result.")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--stage3_ckpt", type=str, default="")
    parser.add_argument("--skeleton_folder", default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--output_dir", default="outputs/scale_diagnosis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Infer config from checkpoint
    encoder_type, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)
    _ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    _sd = _ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt
    temporal_block_type = "attention" if any("denoiser.temporal_blocks.0.attn" in k for k in _sd) else "conv"
    latent_dim = _sd["encoder.in_proj.weight"].shape[0]
    gait_metrics_dim = DEFAULT_GAIT_METRICS_DIM
    del _ckpt, _sd

    print(f"Config: encoder={encoder_type}, skel={skeleton_graph_op}, temporal={temporal_block_type}, latent_dim={latent_dim}")

    stage1 = Stage1Model(
        latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, timesteps=DEFAULT_TIMESTEPS,
        gait_metrics_dim=gait_metrics_dim, use_gait_conditioning=False,
        encoder_type=encoder_type, skeleton_graph_op=skeleton_graph_op,
        temporal_block_type=temporal_block_type,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    stage1.eval()

    dataset = create_dataset(
        dataset_path=None,
        skeleton_folder=args.skeleton_folder,
        hip_folder=args.hip_folder,
        wrist_folder=args.wrist_folder,
        window=DEFAULT_WINDOW, joints=DEFAULT_JOINTS, stride=30,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)
    batch = next(iter(loader))
    x_real = batch["skeleton"]
    a_hip = batch["A_hip"]
    a_wrist = batch["A_wrist"]

    # Run diagnostics
    test1_denoiser_noise_prediction_quality(stage1, x_real, device)
    test2_reverse_process_step_by_step(stage1, x_real, device)
    test3_forward_backward_consistency(stage1, x_real, device)
    test6_encoder_latent_distribution(stage1, x_real, device)

    if args.stage2_ckpt and args.stage3_ckpt:
        stage2 = Stage2Model(
            encoder=stage1.encoder, latent_dim=latent_dim, num_joints=DEFAULT_JOINTS,
            gait_metrics_dim=gait_metrics_dim, d_shared=64,
        ).to(device)
        load_checkpoint(args.stage2_ckpt, stage2, strict=True)
        stage2.eval()

        stage3 = Stage3Model(
            encoder=stage1.encoder, decoder=stage1.decoder, denoiser=stage1.denoiser,
            latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, timesteps=DEFAULT_TIMESTEPS,
            gait_metrics_dim=gait_metrics_dim, use_gait_conditioning=False, d_shared=64,
            shared_motion_layer=stage2.shared_motion_layer,
        ).to(device)
        load_checkpoint(args.stage3_ckpt, stage3, strict=True)
        stage3.eval()

        test4_stage3_decoder_on_real_z0(stage1, stage3, x_real, device, out_dir)
        test5_rescaled_z0_gen(stage3, stage2, x_real, a_hip, a_wrist, device, out_dir)

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
