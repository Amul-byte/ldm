"""Test: does normalizing z0 before diffusion fix the reverse process?

The encoder produces z0 ~ N(0, 4.67^2 I) but DDPM assumes z0 ~ N(0, I).
This test:
1. Measures the exact z0 distribution
2. Normalizes z0 to unit variance
3. Runs the denoiser reverse on normalized z0 — does it recover correct norm?
4. Trains a fresh decoder on normalized z0 — does it produce good skeletons?
5. Full pipeline: normalize -> diffuse -> reverse -> denormalize -> decode

Usage:
    python diagnose_normalization.py \
        --stage1_ckpt new_checkpoints/stage1_enhanced/stage1_best.pt
"""

from __future__ import annotations

import argparse
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
from diffusion_model.skeleton_model import GraphDecoderGCN
from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_TIMESTEPS, DEFAULT_WINDOW, get_joint_index, get_skeleton_edges


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


def _knee_angles(skel):
    angles = {}
    for name, (a, b, c) in [
        ("L_knee", (get_joint_index("HIP_LEFT"), get_joint_index("KNEE_LEFT"), get_joint_index("ANKLE_LEFT"))),
        ("R_knee", (get_joint_index("HIP_RIGHT"), get_joint_index("KNEE_RIGHT"), get_joint_index("ANKLE_RIGHT"))),
        ("L_elbow", (get_joint_index("SHOULDER_LEFT"), get_joint_index("ELBOW_LEFT"), get_joint_index("WRIST_LEFT"))),
        ("R_elbow", (get_joint_index("SHOULDER_RIGHT"), get_joint_index("ELBOW_RIGHT"), get_joint_index("WRIST_RIGHT"))),
    ]:
        v1 = skel[:, a] - skel[:, b]
        v2 = skel[:, c] - skel[:, b]
        cos = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, -1) * np.linalg.norm(v2, -1) + 1e-8)
        angles[name] = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
    return angles


def _bone_length_mean(skel):
    return np.mean([np.linalg.norm(skel[:, i] - skel[:, j], axis=-1).mean()
                     for i, j in get_skeleton_edges()])


# ── Tests ───────────────────────────────────────────────────────

def test1_measure_z0_stats(stage1, x_all, device):
    """Compute per-dimension mean and std of z0 across the dataset."""
    print("\n" + "=" * 70)
    print("TEST 1: z0 distribution statistics across dataset")
    print("=" * 70)

    all_z0 = []
    with torch.no_grad():
        for i in range(0, len(x_all), 4):
            batch = x_all[i:i+4].to(device)
            z0 = stage1.encoder(batch, gait_metrics=None)
            all_z0.append(z0.cpu())

    z0_cat = torch.cat(all_z0, dim=0)  # [N, T, J, D]
    z_flat = z0_cat.reshape(-1, z0_cat.shape[-1])  # [N*T*J, D]

    per_dim_mean = z_flat.mean(dim=0)  # [D]
    per_dim_std = z_flat.std(dim=0)    # [D]
    global_mean = z_flat.mean().item()
    global_std = z_flat.std().item()

    print(f"  Samples: {z0_cat.shape[0]}, Total vectors: {z_flat.shape[0]}")
    print(f"  Global mean: {global_mean:.4f}")
    print(f"  Global std:  {global_std:.4f}")
    print(f"  Per-dim mean: min={per_dim_mean.min():.3f} max={per_dim_mean.max():.3f} avg={per_dim_mean.mean():.3f}")
    print(f"  Per-dim std:  min={per_dim_std.min():.3f} max={per_dim_std.max():.3f} avg={per_dim_std.mean():.3f}")
    print(f"  Norm: mean={z_flat.norm(dim=-1).mean():.2f}")
    print(f"  For N(0,I): expected norm={np.sqrt(z0_cat.shape[-1]):.2f}")
    print(f"  Scale factor needed: {global_std:.2f}")

    return per_dim_mean, per_dim_std, global_std


def test2_reverse_with_normalized_z0(stage1, x_real, per_dim_mean, per_dim_std, device):
    """Normalize z0, run reverse process, check if norm is recovered."""
    print("\n" + "=" * 70)
    print("TEST 2: Reverse process with NORMALIZED z0")
    print("  Normalize z0 -> run diffusion -> check if reverse recovers unit norm")
    print("=" * 70)

    with torch.no_grad():
        z0_raw = stage1.encoder(x_real.to(device), gait_metrics=None)

    # Normalize z0 to ~N(0, I)
    z0_norm = (z0_raw - per_dim_mean.to(device)) / per_dim_std.clamp(min=1e-6).to(device)

    print(f"  z0_raw:  norm={z0_raw.norm(dim=-1).mean():.2f}  std={z0_raw.std():.4f}")
    print(f"  z0_norm: norm={z0_norm.norm(dim=-1).mean():.2f}  std={z0_norm.std():.4f}")

    dp = stage1.diffusion

    # Forward process on normalized z0
    print(f"\n  FORWARD on normalized z0:")
    for t_val in [0, 100, 300, 499]:
        t = torch.full((x_real.shape[0],), t_val, device=device, dtype=torch.long)
        zt = dp.q_sample(z0=z0_norm, t=t)
        print(f"    t={t_val}: zt_norm={zt.norm(dim=-1).mean():.2f}")

    # Reverse process on normalized z0 (DDPM full 500 steps)
    print(f"\n  REVERSE (DDPM 500 steps) from pure noise:")
    z0_gen_ddpm = dp.p_sample_loop(
        denoiser=stage1.denoiser,
        shape=z0_norm.shape,
        device=device,
    )
    print(f"    z0_gen norm: {z0_gen_ddpm.norm(dim=-1).mean():.2f}")
    print(f"    z0_norm target: {z0_norm.norm(dim=-1).mean():.2f}")
    print(f"    Ratio: {z0_gen_ddpm.norm(dim=-1).mean() / z0_norm.norm(dim=-1).mean():.2f}")

    # Reverse process DDIM 50 steps
    print(f"\n  REVERSE (DDIM 50 steps) from pure noise:")
    z0_gen_ddim = dp.p_sample_loop_ddim(
        denoiser=stage1.denoiser,
        shape=z0_norm.shape,
        device=device,
        sample_steps=50,
    )
    print(f"    z0_gen norm: {z0_gen_ddim.norm(dim=-1).mean():.2f}")
    print(f"    Ratio: {z0_gen_ddim.norm(dim=-1).mean() / z0_norm.norm(dim=-1).mean():.2f}")

    # Denormalize back to original scale
    z0_gen_denorm_ddpm = z0_gen_ddpm * per_dim_std.to(device) + per_dim_mean.to(device)
    z0_gen_denorm_ddim = z0_gen_ddim * per_dim_std.to(device) + per_dim_mean.to(device)

    print(f"\n  After denormalization:")
    print(f"    DDPM denorm norm: {z0_gen_denorm_ddpm.norm(dim=-1).mean():.2f}")
    print(f"    DDIM denorm norm: {z0_gen_denorm_ddim.norm(dim=-1).mean():.2f}")
    print(f"    z0_raw target:    {z0_raw.norm(dim=-1).mean():.2f}")

    # Cosine similarity with z0_raw (can't match exactly since generated, but should be in-distribution)
    print(f"\n  NOTE: Generated z0 won't match z0_raw (different samples), but norms should match.")

    return z0_gen_ddpm, z0_gen_ddim, z0_gen_denorm_ddpm, z0_gen_denorm_ddim


def test3_decode_normalized_pipeline(stage1, x_real, per_dim_mean, per_dim_std,
                                      z0_gen_denorm_ddpm, z0_gen_denorm_ddim, device, out_dir):
    """Train a fresh decoder and test the full normalized pipeline."""
    print("\n" + "=" * 70)
    print("TEST 3: Fresh decoder on denormalized z0_gen")
    print("  Train decoder on CLEAN z0_raw, then test on denormalized z0_gen")
    print("=" * 70)

    with torch.no_grad():
        z0_raw = stage1.encoder(x_real.to(device), gait_metrics=None)

    latent_dim = z0_raw.shape[-1]
    skel_real = x_real[0].cpu().numpy()

    # Train fresh decoder on clean z0
    fresh_dec = GraphDecoderGCN(latent_dim=latent_dim, output_dim=3, num_joints=DEFAULT_JOINTS).to(device)
    optimizer = optim.Adam(fresh_dec.parameters(), lr=1e-3)
    x_target = x_real.to(device)

    print(f"\n  Training fresh decoder on clean z0_raw for 300 steps...")
    for step in range(301):
        fresh_dec.train()
        x_hat = fresh_dec(z0_raw.detach())
        loss = F.smooth_l1_loss(x_hat, x_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            s = x_hat[0].detach().cpu().numpy()
            mpjpe = np.linalg.norm(skel_real - s, axis=-1).mean()
            print(f"    step={step} loss={loss.item():.6f} MPJPE={mpjpe:.4f}")

    fresh_dec.eval()

    # Decode different latents
    with torch.no_grad():
        x_from_raw = fresh_dec(z0_raw)                     # clean z0 (should be good)
        x_from_ddpm = fresh_dec(z0_gen_denorm_ddpm)         # denormalized DDPM gen
        x_from_ddim = fresh_dec(z0_gen_denorm_ddim)         # denormalized DDIM gen

        # Also try the RAW (un-normalized) reverse process for comparison
        z0_raw_reverse_ddim = stage1.diffusion.p_sample_loop_ddim(
            denoiser=stage1.denoiser,
            shape=z0_raw.shape,
            device=device,
            sample_steps=50,
        )
        x_from_raw_reverse = fresh_dec(z0_raw_reverse_ddim)  # raw reverse (no normalization)

    results = {
        "Real": skel_real,
        "Dec(clean z0)": x_from_raw[0].cpu().numpy(),
        "Dec(norm+DDIM+denorm)": x_from_ddim[0].cpu().numpy(),
        "Dec(norm+DDPM+denorm)": x_from_ddpm[0].cpu().numpy(),
        "Dec(raw reverse)": x_from_raw_reverse[0].cpu().numpy(),
    }

    print(f"\n  Results:")
    print(f"{'Source':>25} {'MPJPE':>10} {'bone_len':>10} {'L_knee':>10} {'R_knee':>10}")
    print("-" * 68)
    for name, s in results.items():
        if name == "Real":
            mpjpe = 0.0
        else:
            mpjpe = np.linalg.norm(skel_real - s, axis=-1).mean()
        bl = _bone_length_mean(s)
        ka = _knee_angles(s)
        print(f"{name:>25} {mpjpe:>10.4f} {bl:>10.4f} {ka['L_knee']:>10.1f} {ka['R_knee']:>10.1f}")

    _save_panel(results, os.path.join(out_dir, "test3_normalized_pipeline.png"), canvas=300)
    print(f"\n  -> Saved: {out_dir}/test3_normalized_pipeline.png")


def test4_retrain_denoiser_quick_check(stage1, x_real, per_dim_mean, per_dim_std, device):
    """Quick sanity: does the denoiser predict noise well on NORMALIZED z0?"""
    print("\n" + "=" * 70)
    print("TEST 4: Denoiser noise prediction on normalized vs raw z0")
    print("  The denoiser was trained on RAW z0 (std=4.67).")
    print("  If we feed normalized z0, prediction quality should DROP")
    print("  (confirming the denoiser learned the wrong scale).")
    print("=" * 70)

    with torch.no_grad():
        z0_raw = stage1.encoder(x_real.to(device), gait_metrics=None)

    z0_norm = (z0_raw - per_dim_mean.to(device)) / per_dim_std.clamp(min=1e-6).to(device)
    dp = stage1.diffusion

    print(f"\n{'t':>6} {'MSE(raw)':>12} {'MSE(norm)':>12} {'cos(raw)':>10} {'cos(norm)':>10}")
    print("-" * 52)

    for t_val in [10, 50, 100, 200, 400]:
        t = torch.full((x_real.shape[0],), t_val, device=device, dtype=torch.long)

        # On raw z0
        noise = torch.randn_like(z0_raw)
        zt_raw = dp.q_sample(z0=z0_raw, t=t, noise=noise)
        with torch.no_grad():
            pred_raw = stage1.denoiser(zt_raw, t, h_tokens=None, h_global=None, gait_metrics=None)
        mse_raw = F.mse_loss(pred_raw, noise).item()
        cos_raw = F.cosine_similarity(pred_raw.reshape(-1, pred_raw.shape[-1]),
                                       noise.reshape(-1, noise.shape[-1]), dim=-1).mean().item()

        # On normalized z0
        noise2 = torch.randn_like(z0_norm)
        zt_norm = dp.q_sample(z0=z0_norm, t=t, noise=noise2)
        with torch.no_grad():
            pred_norm = stage1.denoiser(zt_norm, t, h_tokens=None, h_global=None, gait_metrics=None)
        mse_norm = F.mse_loss(pred_norm, noise2).item()
        cos_norm = F.cosine_similarity(pred_norm.reshape(-1, pred_norm.shape[-1]),
                                        noise2.reshape(-1, noise2.shape[-1]), dim=-1).mean().item()

        print(f"{t_val:>6} {mse_raw:>12.4f} {mse_norm:>12.4f} {cos_raw:>10.4f} {cos_norm:>10.4f}")

    print(f"\n  If MSE(norm) >> MSE(raw): confirms denoiser was trained on wrong-scale z0.")
    print(f"  Fix: retrain Stage 1 with z0 normalization before diffusion.")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--skeleton_folder", default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--output_dir", default="outputs/normalization_diagnosis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Infer config
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

    dataset = create_dataset(
        dataset_path=None,
        skeleton_folder=args.skeleton_folder,
        hip_folder=args.hip_folder,
        wrist_folder=args.wrist_folder,
        window=DEFAULT_WINDOW, joints=DEFAULT_JOINTS, stride=30,
    )

    # Collect more data for statistics
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
    x_batches = []
    for i, batch in enumerate(loader):
        x_batches.append(batch["skeleton"])
        if i >= 9:  # ~80 samples for stats
            break
    x_all = torch.cat(x_batches, dim=0)
    x_real = x_all[:4]  # Use first 4 for visualization
    print(f"Collected {x_all.shape[0]} samples for statistics")

    # Run tests
    per_dim_mean, per_dim_std, global_std = test1_measure_z0_stats(stage1, x_all, device)
    z0_ddpm, z0_ddim, z0_denorm_ddpm, z0_denorm_ddim = test2_reverse_with_normalized_z0(
        stage1, x_real, per_dim_mean, per_dim_std, device
    )
    test3_decode_normalized_pipeline(
        stage1, x_real, per_dim_mean, per_dim_std, z0_denorm_ddpm, z0_denorm_ddim, device, out_dir
    )
    test4_retrain_denoiser_quick_check(stage1, x_real, per_dim_mean, per_dim_std, device)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"  Encoder z0 std: {global_std:.2f} (should be ~1.0 for standard DDPM)")
    print(f"  The denoiser was trained on z0 with std={global_std:.2f}.")
    print(f"  The reverse process cannot recover this scale from N(0,I) noise.")
    print(f"")
    print(f"  FIX: Add latent normalization layer between encoder and diffusion:")
    print(f"    z0_raw = encoder(x)")
    print(f"    z0_scaled = (z0_raw - mean) / std    # ~N(0, I)")
    print(f"    ... diffusion operates on z0_scaled ...")
    print(f"    z0_recovered = z0_gen * std + mean    # back to original scale")
    print(f"    x_hat = decoder(z0_recovered)")
    print(f"")
    print(f"  This requires retraining Stage 1 (denoiser) and Stage 3 (decoder).")
    print(f"  Stage 1 encoder weights can be FROZEN — only the denoiser needs retraining.")


if __name__ == "__main__":
    main()
