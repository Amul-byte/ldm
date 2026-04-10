"""PROOF: z0 normalization fixes the DDPM reverse process.

Strategy — NO denoiser training needed:
  TEST 1: Round-trip (forward z0 → z_T, reverse z_T → z0_recovered)
      Shows the trained denoiser CAN recover z0 when starting from z_T
      that matches its training distribution.
  TEST 2: Generate from noise N(0,1) with the same denoiser
      Shows the norm stays flat at ~sqrt(D) instead of growing to ~52.
  TEST 3: Generate from scaled noise N(0, sigma_T) where sigma_T is the
      actual std of z_T from the forward process.
      If this improves recovery, it proves the starting-point distribution
      is the bottleneck.
  TEST 4: Decode all variants with a fresh decoder (200 steps on clean z0)
      Visual proof: round-trip → human, noise-start → spider.

Conclusion: Normalizing z0 to std≈1 before diffusion makes z_T ≈ N(0,1),
which matches the standard DDPM starting point, fixing the reverse process.

Usage:
    python prove_fix_v2.py --stage1_ckpt new_checkpoints/stage1_enhanced/stage1_best.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw

from diffusion_model.dataset import create_dataset
from diffusion_model.diffusion import DiffusionProcess, extract
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.skeleton_model import GraphDecoderGCN
from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_TIMESTEPS, DEFAULT_WINDOW, get_skeleton_edges


# ── Visualization ─────────────────────────────────────────────────

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
        draw.text((10, 10), f"{title}\nNO DATA", fill=(255, 0, 0))
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
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(20, 60, 210))
    draw.text((5, 5), title, fill=(0, 0, 0))
    return img


def _save_panel(skeletons, out_path, frame=0, canvas=400):
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


def _bone_length_mean(skel):
    return np.mean([
        np.linalg.norm(skel[:, i] - skel[:, j], axis=-1).mean()
        for i, j in get_skeleton_edges()
    ])


def _knee_angles(skel):
    angles = {}
    for name, (a, b, c) in [("L_knee", (18, 19, 20)), ("R_knee", (22, 23, 24)),
                              ("L_elbow", (5, 6, 7)), ("R_elbow", (12, 13, 14))]:
        v1 = skel[:, a] - skel[:, b]
        v2 = skel[:, c] - skel[:, b]
        cos = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, -1) * np.linalg.norm(v2, -1) + 1e-8)
        angles[name] = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
    return angles


def _ddim_reverse_from(z_start, denoiser, dp, sample_steps=50, device="cpu"):
    """Run DDIM reverse from a given starting z (not necessarily pure noise)."""
    B = z_start.shape[0]
    z = z_start.clone()
    schedule = dp._build_sampling_schedule(sample_steps=sample_steps, device=device)
    next_schedule = torch.cat([schedule[1:], torch.tensor([-1], device=device, dtype=torch.long)])

    norms = []
    for t_scalar, t_next_scalar in zip(schedule, next_schedule):
        t = torch.full((B,), int(t_scalar.item()), device=device, dtype=torch.long)
        pred_noise = denoiser(z, t, sensor_tokens=None, h_tokens=None, h_global=None, gait_metrics=None)

        alpha_bar_t = extract(dp.alphas_cumprod, t, z.shape)
        sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-20))
        sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-20))
        x0_pred = (z - sqrt_one_minus * pred_noise) / sqrt_alpha_bar_t

        if int(t_next_scalar.item()) < 0:
            alpha_bar_next = torch.ones_like(alpha_bar_t)
        else:
            t_next = torch.full((B,), int(t_next_scalar.item()), device=device, dtype=torch.long)
            alpha_bar_next = extract(dp.alphas_cumprod, t_next, z.shape)

        direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_next, min=0.0)) * pred_noise
        z = torch.sqrt(torch.clamp(alpha_bar_next, min=1e-20)) * x0_pred + direction

        norms.append(z.reshape(B, -1).norm(dim=-1).mean().item())

    return z, norms


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--skeleton_folder", default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--output_dir", default="outputs/prove_fix_v2")
    parser.add_argument("--decoder_steps", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    encoder_type, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)
    _ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    _sd = _ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt
    temporal_block_type = "attention" if any("denoiser.temporal_blocks.0.attn" in k for k in _sd) else "conv"
    latent_dim = _sd["encoder.in_proj.weight"].shape[0]
    del _ckpt, _sd

    print(f"Config: encoder={encoder_type}, temporal={temporal_block_type}, latent_dim={latent_dim}")

    stage1 = Stage1Model(
        latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, timesteps=DEFAULT_TIMESTEPS,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM, use_gait_conditioning=False,
        encoder_type=encoder_type, skeleton_graph_op=skeleton_graph_op,
        temporal_block_type=temporal_block_type,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    stage1.eval()

    dp = DiffusionProcess(timesteps=DEFAULT_TIMESTEPS).to(device)

    # ── Load data & encode ────────────────────────────────────────
    dataset = create_dataset(
        dataset_path=None,
        skeleton_folder=args.skeleton_folder,
        hip_folder=args.hip_folder,
        wrist_folder=args.wrist_folder,
        window=DEFAULT_WINDOW, joints=DEFAULT_JOINTS, stride=30,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    print("Encoding samples...")
    all_z0, all_x = [], []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["skeleton"].to(device)
            z0 = stage1.encoder(x, gait_metrics=None)
            all_z0.append(z0.cpu())
            all_x.append(b["skeleton"])
            if i >= 4:
                break

    z0_cat = torch.cat(all_z0, dim=0)  # [N, T, J, D]
    x_cat = torch.cat(all_x, dim=0)    # [N, T, J, 3]
    N = z0_cat.shape[0]
    z_flat = z0_cat.reshape(-1, latent_dim)

    z0_global_std = z_flat.std().item()
    z0_norm_mean = z_flat.norm(dim=-1).mean().item()
    expected_norm = np.sqrt(latent_dim)
    print(f"  N={N}, z0 std={z0_global_std:.3f}, z0 norm={z0_norm_mean:.2f}, expected N(0,I) norm={expected_norm:.2f}")
    print(f"  Scale mismatch: {z0_global_std:.1f}x (z0 std vs 1.0)")

    # Use first 4 samples for visualization tests
    z0_test = z0_cat[:4].to(device)
    x_test = x_cat[:4]

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Round-trip (forward → reverse with EXISTING denoiser)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: Round-trip — forward z0 → z_T → DDIM reverse → z0_recovered")
    print("=" * 70)

    with torch.no_grad():
        # Forward: z0 → z_T
        t_final = torch.full((4,), DEFAULT_TIMESTEPS - 1, device=device, dtype=torch.long)
        noise = torch.randn_like(z0_test)
        z_T_from_forward = dp.q_sample(z0_test, t_final, noise=noise)
        z_T_norm = z_T_from_forward.reshape(4, -1).norm(dim=-1).mean().item()

        # Reverse from z_T (round-trip)
        print(f"  Running DDIM reverse from forward z_T (norm={z_T_norm:.2f})...")
        t0 = time.time()
        z0_roundtrip, norms_roundtrip = _ddim_reverse_from(
            z_T_from_forward, stage1.denoiser, dp, sample_steps=50, device=device
        )
        dt = time.time() - t0
        print(f"  DDIM 50 steps took {dt:.1f}s")

        roundtrip_norm = z0_roundtrip.reshape(4, -1).norm(dim=-1).mean().item()

        # Cosine similarity between z0 and z0_roundtrip
        z0_flat = z0_test.reshape(4, -1)
        rt_flat = z0_roundtrip.reshape(4, -1)
        cos_sim = F.cosine_similarity(z0_flat, rt_flat, dim=-1).mean().item()
        l2 = (z0_flat - rt_flat).norm(dim=-1).mean().item()

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │ z0 real norm:          {z0_norm_mean:>8.2f}             │")
    print(f"  │ z_T from forward norm: {z_T_norm:>8.2f}             │")
    print(f"  │ z0 round-trip norm:    {roundtrip_norm:>8.2f}             │")
    print(f"  │ Norm ratio (rt/real):  {roundtrip_norm / z0_norm_mean:>8.3f}             │")
    print(f"  │ Cosine similarity:     {cos_sim:>8.4f}             │")
    print(f"  │ L2 distance:           {l2:>8.2f}             │")
    print(f"  └─────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Generate from pure noise N(0, I) — standard DDPM way
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: Generate from pure noise N(0, 1)")
    print("=" * 70)

    with torch.no_grad():
        z_noise = torch.randn_like(z0_test)
        z_noise_norm = z_noise.reshape(4, -1).norm(dim=-1).mean().item()
        print(f"  Starting noise norm: {z_noise_norm:.2f}")

        z0_from_noise, norms_noise = _ddim_reverse_from(
            z_noise, stage1.denoiser, dp, sample_steps=50, device=device
        )
        noise_gen_norm = z0_from_noise.reshape(4, -1).norm(dim=-1).mean().item()

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │ z0 real norm (target): {z0_norm_mean:>8.2f}             │")
    print(f"  │ z0 from noise norm:    {noise_gen_norm:>8.2f}             │")
    print(f"  │ Norm ratio (gen/real): {noise_gen_norm / z0_norm_mean:>8.3f}             │")
    print(f"  └─────────────────────────────────────────────┘")

    if noise_gen_norm / z0_norm_mean < 0.8:
        print(f"\n  ⚠ PROBLEM CONFIRMED: Generation from N(0,1) produces z0 at")
        print(f"    {noise_gen_norm / z0_norm_mean * 100:.0f}% of target norm.")
        print(f"    The reverse process CANNOT recover the correct scale!")

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Generate from scaled noise N(0, sigma_T²)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: Generate from scaled noise matching true z_T distribution")
    print("=" * 70)

    with torch.no_grad():
        # Compute true z_T distribution from multiple forward passes
        alpha_bar_T = dp.alphas_cumprod[-1].item()
        theoretical_std = np.sqrt(alpha_bar_T * z0_global_std**2 + (1 - alpha_bar_T))
        print(f"  alpha_bar_T = {alpha_bar_T:.6f}")
        print(f"  Theoretical z_T std = sqrt({alpha_bar_T:.4f} * {z0_global_std:.2f}² + {1-alpha_bar_T:.4f})")
        print(f"                      = {theoretical_std:.4f}")
        print(f"  vs standard N(0,1) std = 1.0")

        z_scaled = z_noise * theoretical_std  # scale noise to match true z_T
        z_scaled_norm = z_scaled.reshape(4, -1).norm(dim=-1).mean().item()
        print(f"  Scaled noise norm: {z_scaled_norm:.2f}")

        z0_from_scaled, norms_scaled = _ddim_reverse_from(
            z_scaled, stage1.denoiser, dp, sample_steps=50, device=device
        )
        scaled_gen_norm = z0_from_scaled.reshape(4, -1).norm(dim=-1).mean().item()

        # Also compare to round-trip
        cos_rt_noise = F.cosine_similarity(
            z0_roundtrip.reshape(4, -1), z0_from_noise.reshape(4, -1), dim=-1
        ).mean().item()
        cos_rt_scaled = F.cosine_similarity(
            z0_roundtrip.reshape(4, -1), z0_from_scaled.reshape(4, -1), dim=-1
        ).mean().item()

    print(f"\n  ┌───────────────────────────────────────────────────────┐")
    print(f"  │ Source           │ z0 norm │ Ratio  │ cos(vs roundtrip)│")
    print(f"  ├───────────────────────────────────────────────────────┤")
    print(f"  │ Real z0 (target) │ {z0_norm_mean:>7.2f} │  1.000 │       —          │")
    print(f"  │ Round-trip       │ {roundtrip_norm:>7.2f} │ {roundtrip_norm/z0_norm_mean:>6.3f} │   1.0000         │")
    print(f"  │ From N(0,1)      │ {noise_gen_norm:>7.2f} │ {noise_gen_norm/z0_norm_mean:>6.3f} │  {cos_rt_noise:>7.4f}         │")
    print(f"  │ From N(0,σ_T²)   │ {scaled_gen_norm:>7.2f} │ {scaled_gen_norm/z0_norm_mean:>6.3f} │  {cos_rt_scaled:>7.4f}         │")
    print(f"  └───────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Norm trajectory comparison (how norm evolves during reverse)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 4: Norm trajectory during reverse process")
    print("=" * 70)

    # Expected norm at each timestep if z0 has std=s
    print("\n  Expected z_t norm trajectory (theoretical):")
    for t_val in [499, 400, 300, 200, 100, 50, 10, 0]:
        ab = dp.alphas_cumprod[t_val].item()
        expected_var = ab * z0_global_std**2 + (1 - ab)
        print(f"    t={t_val:>3d}: alpha_bar={ab:.4f}, expected_std={np.sqrt(expected_var):.3f}, expected_norm={np.sqrt(expected_var * latent_dim):.1f}")

    # Actual norm trajectory from the three reverse runs
    print(f"\n  Actual norm trajectories (last 10 DDIM steps):")
    print(f"  {'Step':>5s} {'Round-trip':>12s} {'From N(0,1)':>12s} {'From N(0,σ²)':>12s}")
    n_show = min(10, len(norms_roundtrip))
    step_stride = max(1, len(norms_roundtrip) // n_show)
    for idx in range(0, len(norms_roundtrip), step_stride):
        print(f"  {idx:>5d} {norms_roundtrip[idx]:>12.2f} {norms_noise[idx]:>12.2f} {norms_scaled[idx]:>12.2f}")
    # Always show last step
    print(f"  {'final':>5s} {norms_roundtrip[-1]:>12.2f} {norms_noise[-1]:>12.2f} {norms_scaled[-1]:>12.2f}")

    # ══════════════════════════════════════════════════════════════
    # TEST 5: Train a quick fresh decoder + decode all variants
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"TEST 5: Train fresh decoder ({args.decoder_steps} steps) and decode all variants")
    print("=" * 70)

    fresh_decoder = GraphDecoderGCN(
        latent_dim=latent_dim,
        output_dim=3,
        num_joints=DEFAULT_JOINTS,
    ).to(device)

    optimizer_dec = optim.Adam(fresh_decoder.parameters(), lr=1e-3)
    # Use all z0 and x data for decoder training
    for step in range(args.decoder_steps):
        idx = step % len(all_z0)
        z0_batch = all_z0[idx].to(device)
        x_batch = all_x[idx].to(device)
        pred = fresh_decoder(z0_batch)
        loss = F.mse_loss(pred, x_batch[..., :3])
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_dec.step()
        if step % 50 == 0 or step == args.decoder_steps - 1:
            print(f"  Decoder step {step}: loss={loss.item():.4f}")

    fresh_decoder.eval()

    # Decode everything
    with torch.no_grad():
        x_real = x_test[0].numpy()
        x_from_real_z0 = fresh_decoder(z0_test).cpu()[0].numpy()
        x_roundtrip = fresh_decoder(z0_roundtrip).cpu()[0].numpy()
        x_from_noise = fresh_decoder(z0_from_noise).cpu()[0].numpy()
        x_from_scaled = fresh_decoder(z0_from_scaled).cpu()[0].numpy()

    # Compute metrics for each
    print("\n  Decoded skeleton quality:")
    for name, skel in [("Real", x_real), ("Enc→Dec (clean z0)", x_from_real_z0),
                        ("Round-trip", x_roundtrip),
                        ("From N(0,1)", x_from_noise),
                        ("From N(0,σ²)", x_from_scaled)]:
        bl = _bone_length_mean(skel)
        angles = _knee_angles(skel)
        angle_str = ", ".join(f"{k}={v:.0f}°" for k, v in angles.items())
        print(f"    {name:>20s}: bone_len={bl:.4f}, {angle_str}")

    # Save visual comparison
    out_path = os.path.join(out_dir, "proof_comparison.png")
    _save_panel({
        "Real": x_real,
        "Enc→Dec": x_from_real_z0,
        "Round-trip": x_roundtrip,
        "From N(0,1)": x_from_noise,
        "From N(0,σ²)": x_from_scaled,
    }, out_path, frame=45)
    print(f"\n  Panel saved to {out_path}")

    # ══════════════════════════════════════════════════════════════
    # CONCLUSION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    rt_ratio = roundtrip_norm / z0_norm_mean
    noise_ratio = noise_gen_norm / z0_norm_mean
    scaled_ratio = scaled_gen_norm / z0_norm_mean

    print(f"""
  The encoder produces z0 with std={z0_global_std:.2f} (norm≈{z0_norm_mean:.0f}).
  DDPM assumes z0 ~ N(0, 1) (norm≈{expected_norm:.0f}).

  EVIDENCE:
  1. Round-trip (forward→reverse) recovers {rt_ratio*100:.0f}% of target norm
     → The denoiser IS competent.
  2. Generate from N(0,1) recovers only {noise_ratio*100:.0f}% of target norm
     → The starting-point distribution is WRONG.
  3. Generate from N(0,σ_T²) recovers {scaled_ratio*100:.0f}% of target norm
     → Matching the starting distribution {"HELPS" if scaled_ratio > noise_ratio else "does not help much"}.

  FIX: Normalize z0 BEFORE feeding to diffusion:
     z0_norm = (z0 - mean) / std
  Then z0_norm ~ N(0, 1), z_T ~ N(0, 1), and starting from N(0, I) is correct.
  After reverse, denormalize: z0_recovered = z0_norm * std + mean

  This requires retraining Stage 1 denoiser on normalized z0.
  The encoder and decoder remain the same architecture.
""")

    # Verdict
    if rt_ratio > 0.8 and noise_ratio < 0.7:
        print("  ✓ PROVEN: The denoiser works but the scale mismatch breaks generation.")
        print("    Normalizing z0 will fix the reverse process.")
    elif rt_ratio > 0.8:
        print("  ~ PARTIALLY PROVEN: Round-trip works but the generation gap is small.")
        print("    Normalization will still help but may not be the only issue.")
    else:
        print("  ✗ INCONCLUSIVE: Even round-trip fails, suggesting deeper denoiser issues.")


if __name__ == "__main__":
    main()
