"""PROOF OF CONCEPT: Does normalizing z0 fix the reverse process?

Train a fresh denoiser on normalized z0 for a few hundred steps,
then run the reverse process and check if it recovers the correct norm.
Then decode and visualize.

This is NOT a full retrain — just enough to prove the concept.

Usage:
    python prove_fix.py --stage1_ckpt new_checkpoints/stage1_enhanced/stage1_best.pt
"""

from __future__ import annotations

import argparse
import copy
import os

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
from diffusion_model.skeleton_model import GraphDecoderGCN, GraphDenoiserMaskedGCN
from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_TIMESTEPS, DEFAULT_WINDOW, get_skeleton_edges


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
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(20, 60, 210))
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
    return np.mean([np.linalg.norm(skel[:, i] - skel[:, j], axis=-1).mean()
                     for i, j in get_skeleton_edges()])


def _knee_angles(skel):
    angles = {}
    for name, (a, b, c) in [("L_knee", (18, 19, 20)), ("R_knee", (22, 23, 24)),
                              ("L_elbow", (5, 6, 7)), ("R_elbow", (12, 13, 14))]:
        v1 = skel[:, a] - skel[:, b]
        v2 = skel[:, c] - skel[:, b]
        cos = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, -1) * np.linalg.norm(v2, -1) + 1e-8)
        angles[name] = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
    return angles


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--skeleton_folder", default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--output_dir", default="outputs/prove_fix")
    parser.add_argument("--denoiser_steps", type=int, default=500)
    parser.add_argument("--decoder_steps", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load encoder
    encoder_type, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)
    _ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    _sd = _ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt
    temporal_block_type = "attention" if any("denoiser.temporal_blocks.0.attn" in k for k in _sd) else "conv"
    latent_dim = _sd["encoder.in_proj.weight"].shape[0]
    del _ckpt, _sd

    print(f"Config: encoder={encoder_type}, skel={skeleton_graph_op}, temporal={temporal_block_type}, latent_dim={latent_dim}")

    stage1 = Stage1Model(
        latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, timesteps=DEFAULT_TIMESTEPS,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM, use_gait_conditioning=False,
        encoder_type=encoder_type, skeleton_graph_op=skeleton_graph_op,
        temporal_block_type=temporal_block_type,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    stage1.eval()

    # Load data
    dataset = create_dataset(
        dataset_path=None,
        skeleton_folder=args.skeleton_folder,
        hip_folder=args.hip_folder,
        wrist_folder=args.wrist_folder,
        window=DEFAULT_WINDOW, joints=DEFAULT_JOINTS, stride=30,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    batches = []
    for i, b in enumerate(loader):
        batches.append(b)
        if i >= 9:
            break

    # ── Step 1: Compute z0 statistics ───────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Compute z0 normalization statistics")
    print("=" * 70)

    all_z0 = []
    all_x = []
    with torch.no_grad():
        for b in batches:
            x = b["skeleton"].to(device)
            z0 = stage1.encoder(x, gait_metrics=None)
            all_z0.append(z0.cpu())
            all_x.append(b["skeleton"])

    z0_cat = torch.cat(all_z0, dim=0)
    x_cat = torch.cat(all_x, dim=0)
    z_flat = z0_cat.reshape(-1, z0_cat.shape[-1])

    z0_mean = z_flat.mean(dim=0)  # [D]
    z0_std = z_flat.std(dim=0).clamp(min=1e-3)   # [D]

    print(f"  Samples: {z0_cat.shape[0]}")
    print(f"  z0 global std: {z_flat.std():.4f}")
    print(f"  z0 norm: {z_flat.norm(dim=-1).mean():.2f}")

    # Verify normalization
    z0_normed = (z0_cat - z0_mean) / z0_std
    print(f"  After normalization: std={z0_normed.reshape(-1, latent_dim).std():.4f}, norm={z0_normed.reshape(-1, latent_dim).norm(dim=-1).mean():.2f}")
    print(f"  Expected norm for N(0,I) with D={latent_dim}: {np.sqrt(latent_dim):.2f}")

    # ── Step 2: Train fresh denoiser on NORMALIZED z0 ───────────
    print("\n" + "=" * 70)
    print(f"STEP 2: Train fresh denoiser on normalized z0 ({args.denoiser_steps} steps)")
    print("=" * 70)

    # Fresh denoiser (same architecture as original)
    fresh_denoiser = GraphDenoiserMaskedGCN(
        latent_dim=latent_dim,
        num_joints=DEFAULT_JOINTS,
        temporal_block_type=temporal_block_type,
    ).to(device)

    dp = DiffusionProcess(timesteps=DEFAULT_TIMESTEPS).to(device)
    optimizer_den = optim.Adam(fresh_denoiser.parameters(), lr=2e-4)

    z0_mean_dev = z0_mean.to(device)
    z0_std_dev = z0_std.to(device)

    fresh_denoiser.train()
    for step in range(args.denoiser_steps):
        # Random batch
        idx = torch.randint(0, z0_cat.shape[0], (8,))
        z0_batch = z0_cat[idx].to(device)
        # Normalize
        z0_n = (z0_batch - z0_mean_dev) / z0_std_dev

        t = torch.randint(0, dp.timesteps, (8,), device=device)
        noise = torch.randn_like(z0_n)
        zt = dp.q_sample(z0=z0_n, t=t, noise=noise)
        pred = fresh_denoiser(zt, t, h_tokens=None, h_global=None, gait_metrics=None)
        loss = F.mse_loss(pred, noise)

        optimizer_den.zero_grad()
        loss.backward()
        optimizer_den.step()

        if step % 100 == 0 or step == args.denoiser_steps - 1:
            print(f"  step={step:>4} loss={loss.item():.6f}")

    fresh_denoiser.eval()

    # ── Step 3: Test reverse process on fresh denoiser ──────────
    print("\n" + "=" * 70)
    print("STEP 3: Reverse process with fresh denoiser on normalized z0")
    print("=" * 70)

    x_real = x_cat[:4].to(device)
    with torch.no_grad():
        z0_real = stage1.encoder(x_real, gait_metrics=None)
    z0_real_n = (z0_real - z0_mean_dev) / z0_std_dev

    print(f"  z0_real norm (raw): {z0_real.norm(dim=-1).mean():.2f}")
    print(f"  z0_real norm (normalized): {z0_real_n.norm(dim=-1).mean():.2f}")

    # DDIM reverse with fresh denoiser
    with torch.no_grad():
        z0_gen_n = dp.p_sample_loop_ddim(
            denoiser=fresh_denoiser,
            shape=z0_real_n.shape,
            device=device,
            sample_steps=50,
        )
    print(f"  z0_gen norm (normalized): {z0_gen_n.norm(dim=-1).mean():.2f}")
    print(f"  Ratio: {z0_gen_n.norm(dim=-1).mean() / z0_real_n.norm(dim=-1).mean():.2f}")

    # DDIM with more steps (skip full 500-step DDPM — too slow on CPU)
    with torch.no_grad():
        z0_gen_n_ddpm = dp.p_sample_loop_ddim(
            denoiser=fresh_denoiser,
            shape=z0_real_n.shape,
            device=device,
            sample_steps=200,
        )
    print(f"  z0_gen norm (DDIM-200, normalized): {z0_gen_n_ddpm.norm(dim=-1).mean():.2f}")
    print(f"  Ratio: {z0_gen_n_ddpm.norm(dim=-1).mean() / z0_real_n.norm(dim=-1).mean():.2f}")

    # Denormalize
    z0_gen_raw = z0_gen_n * z0_std_dev + z0_mean_dev
    z0_gen_raw_ddpm = z0_gen_n_ddpm * z0_std_dev + z0_mean_dev
    print(f"\n  After denormalization:")
    print(f"    DDIM: z0_gen norm = {z0_gen_raw.norm(dim=-1).mean():.2f}  (target: {z0_real.norm(dim=-1).mean():.2f})")
    print(f"    DDPM: z0_gen norm = {z0_gen_raw_ddpm.norm(dim=-1).mean():.2f}  (target: {z0_real.norm(dim=-1).mean():.2f})")

    # Also run original broken denoiser for comparison
    with torch.no_grad():
        z0_broken = stage1.diffusion.p_sample_loop_ddim(
            denoiser=stage1.denoiser,
            shape=z0_real.shape,
            device=device,
            sample_steps=50,
        )
    print(f"    Original broken: z0_gen norm = {z0_broken.norm(dim=-1).mean():.2f}")

    # ── Step 4: Train fresh decoder and decode ──────────────────
    print("\n" + "=" * 70)
    print(f"STEP 4: Train fresh decoder on clean z0 ({args.decoder_steps} steps)")
    print("=" * 70)

    fresh_decoder = GraphDecoderGCN(
        latent_dim=latent_dim, output_dim=3, num_joints=DEFAULT_JOINTS,
    ).to(device)
    optimizer_dec = optim.Adam(fresh_decoder.parameters(), lr=1e-3)

    # Train on clean z0_raw (not normalized) — decoder works in raw space
    for step in range(args.decoder_steps):
        idx = torch.randint(0, z0_cat.shape[0], (8,))
        z0_batch = z0_cat[idx].to(device)
        x_batch = x_cat[idx].to(device)

        fresh_decoder.train()
        x_hat = fresh_decoder(z0_batch)
        loss = F.smooth_l1_loss(x_hat, x_batch)
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_dec.step()

        if step % 50 == 0 or step == args.decoder_steps - 1:
            print(f"  step={step:>4} loss={loss.item():.6f}")

    fresh_decoder.eval()

    # ── Step 5: Decode and compare everything ───────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Decode and compare all pipelines")
    print("=" * 70)

    skel_real = x_real[0].cpu().numpy()

    with torch.no_grad():
        x_from_clean = fresh_decoder(z0_real)
        x_from_fixed_ddim = fresh_decoder(z0_gen_raw)
        x_from_fixed_ddpm = fresh_decoder(z0_gen_raw_ddpm)
        x_from_broken = fresh_decoder(z0_broken)

    results = {
        "Real": skel_real,
        "Dec(clean z0)": x_from_clean[0].cpu().numpy(),
        "FIXED (DDIM)": x_from_fixed_ddim[0].cpu().numpy(),
        "FIXED (DDIM200)": x_from_fixed_ddpm[0].cpu().numpy(),
        "BROKEN (old)": x_from_broken[0].cpu().numpy(),
    }

    print(f"\n{'Pipeline':>20} {'MPJPE':>8} {'bone_len':>10} {'L_knee':>8} {'R_knee':>8} {'L_elbow':>8}")
    print("-" * 66)
    for name, s in results.items():
        mpjpe = 0.0 if name == "Real" else np.linalg.norm(skel_real - s, axis=-1).mean()
        bl = _bone_length_mean(s)
        ka = _knee_angles(s)
        print(f"{name:>20} {mpjpe:>8.4f} {bl:>10.4f} {ka['L_knee']:>8.1f} {ka['R_knee']:>8.1f} {ka['L_elbow']:>8.1f}")

    _save_panel(results, os.path.join(out_dir, "proof_comparison.png"), canvas=350)
    print(f"\n  -> Saved: {out_dir}/proof_comparison.png")

    # Multi-frame for the fixed DDIM output
    fixed_skel = x_from_fixed_ddim[0].cpu().numpy()
    multi = {}
    for f in [0, 15, 30, 45, min(60, len(fixed_skel)-1)]:
        multi[f"FIXED t={f}"] = fixed_skel
    frames_panel_imgs = []
    for f in [0, 15, 30, 45, min(60, len(fixed_skel)-1)]:
        pts = _project_to_2d(fixed_skel)
        frames_panel_imgs.append(_draw_skeleton(pts[f], canvas=300, title=f"FIXED frame {f}"))
    w = 300 * len(frames_panel_imgs) + 10 * (len(frames_panel_imgs) - 1)
    panel = Image.new("RGB", (w, 300), (255, 255, 255))
    for i, img in enumerate(frames_panel_imgs):
        panel.paste(img, (i * 310, 0))
    panel.save(os.path.join(out_dir, "proof_fixed_multiframe.png"))
    print(f"  -> Saved: {out_dir}/proof_fixed_multiframe.png")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    fixed_norm = z0_gen_raw.norm(dim=-1).mean().item()
    broken_norm = z0_broken.norm(dim=-1).mean().item()
    target_norm = z0_real.norm(dim=-1).mean().item()

    fixed_ratio = fixed_norm / target_norm
    broken_ratio = broken_norm / target_norm

    print(f"  z0 norm recovery:")
    print(f"    Target:  {target_norm:.2f}")
    print(f"    FIXED:   {fixed_norm:.2f} ({fixed_ratio:.0%})")
    print(f"    BROKEN:  {broken_norm:.2f} ({broken_ratio:.0%})")

    if fixed_ratio > 0.7:
        print(f"\n  NORMALIZATION FIX WORKS. Reverse process recovers {fixed_ratio:.0%} of target norm.")
        print(f"  The broken pipeline only recovers {broken_ratio:.0%}.")
        print(f"\n  Next step: implement normalization in Stage 1 and retrain.")
    else:
        print(f"\n  Normalization alone is not sufficient ({fixed_ratio:.0%} recovery).")
        print(f"  May need more denoiser training steps or schedule changes.")


if __name__ == "__main__":
    main()
