"""Verify: is the decoder broken because it was trained on noisy z0_gen?

Three tests:
1. What z0_gen quality does the decoder see at each t during Stage 3 training?
2. Does the Stage 3 decoder produce same bad output regardless of z0_gen quality?
3. Can a fresh decoder learn from CLEAN z0 in just a few steps? (proves the mapping is learnable)

Usage:
    python diagnose_decoder_training.py \
        --stage1_ckpt new_checkpoints/stage1_enhanced/stage1_best.pt \
        --stage3_ckpt new_checkpoints/stage3_enhanced0.1/stage3_best.pt
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
from diffusion_model.model import Stage1Model, Stage3Model, Stage2Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.skeleton_model import GraphDecoder, GraphDecoderGCN
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


def _draw_skeleton(pts_2d, canvas=350, title=""):
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


def _save_panel(skeletons, out_path, frame=0, canvas=350):
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
    edges = get_skeleton_edges()
    return np.mean([np.linalg.norm(skel[:, i] - skel[:, j], axis=-1).mean() for i, j in edges])


def _knee_angles(skel):
    angles = {}
    for name, (a, b, c) in [
        ("L_knee", (get_joint_index("HIP_LEFT"), get_joint_index("KNEE_LEFT"), get_joint_index("ANKLE_LEFT"))),
        ("R_knee", (get_joint_index("HIP_RIGHT"), get_joint_index("KNEE_RIGHT"), get_joint_index("ANKLE_RIGHT"))),
    ]:
        v1 = skel[:, a] - skel[:, b]
        v2 = skel[:, c] - skel[:, b]
        cos = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, -1) * np.linalg.norm(v2, -1) + 1e-8)
        angles[name] = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
    return angles


# ── Tests ───────────────────────────────────────────────────────

def test1_z0_gen_quality_during_training(stage1, x_real, device):
    """What does the decoder actually receive during Stage 3 training?"""
    print("\n" + "=" * 70)
    print("TEST 1: z0_gen quality at each timestep (what decoder sees in training)")
    print("=" * 70)

    with torch.no_grad():
        z0_real = stage1.encoder(x_real.to(device), gait_metrics=None)

    dp = stage1.diffusion
    z0_norm = z0_real.norm(dim=-1).mean().item()

    print(f"\n  z0_real norm: {z0_norm:.2f}")
    print(f"\n  t is sampled UNIFORMLY from [0, 499] during training.")
    print(f"  So the decoder sees each row below with equal probability.\n")
    print(f"{'t':>6} {'z0_gen norm':>14} {'L2 to z0_real':>16} {'cos_sim':>10} {'pct of training':>18}")
    print("-" * 68)

    t_ranges = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 499)]
    for t_lo, t_hi in t_ranges:
        t_mid = (t_lo + t_hi) // 2
        t = torch.full((x_real.shape[0],), t_mid, device=device, dtype=torch.long)
        noise = torch.randn_like(z0_real)
        zt = dp.q_sample(z0=z0_real, t=t, noise=noise)

        with torch.no_grad():
            pred = stage1.denoiser(zt, t, h_tokens=None, h_global=None, gait_metrics=None)

        sqrt_abar = extract(dp.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1m = extract(dp.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        z0_gen = (zt - sqrt_1m * pred) / torch.clamp(sqrt_abar, min=1e-20)

        gen_norm = z0_gen.norm(dim=-1).mean().item()
        l2 = (z0_gen - z0_real).norm(dim=-1).mean().item()
        cos = F.cosine_similarity(
            z0_gen.reshape(-1, z0_gen.shape[-1]),
            z0_real.reshape(-1, z0_real.shape[-1]), dim=-1
        ).mean().item()
        pct = (t_hi - t_lo) / 500 * 100

        print(f"{t_lo:>3}-{t_hi:<3} {gen_norm:>14.2f} {l2:>16.2f} {cos:>10.4f} {pct:>17.0f}%")

    # Weighted average L2 error
    print(f"\n  Key question: when t is large (t>300), L2 error is huge.")
    print(f"  That means ~40% of training steps feed the decoder noisy z0_gen.")
    print(f"  The decoder must learn x_hat ≈ x_real from BOTH clean and noisy z0_gen.")


def test2_stage3_decoder_at_each_t(stage1, stage3, x_real, device, out_dir):
    """Does the Stage 3 decoder output change with z0_gen quality?"""
    print("\n" + "=" * 70)
    print("TEST 2: Stage 3 decoder output at different z0_gen quality levels")
    print("  If output is SAME regardless of z0_gen quality -> decoder learned mean pose")
    print("=" * 70)

    with torch.no_grad():
        z0_real = stage3.encoder(x_real.to(device), gait_metrics=None)

    dp = stage3.diffusion
    skel_real = x_real[0].cpu().numpy()

    skeletons = {"Real": skel_real}

    print(f"\n{'Input':>20} {'z0 norm':>10} {'MPJPE':>10} {'bone_len':>10} {'L_knee':>10} {'R_knee':>10}")
    print("-" * 74)

    # From clean z0_real
    with torch.no_grad():
        x_clean = stage3.decoder(z0_real)
    s = x_clean[0].cpu().numpy()
    ka = _knee_angles(s)
    print(f"{'clean z0_real':>20} {z0_real.norm(dim=-1).mean():.2f} {np.linalg.norm(skel_real - s, axis=-1).mean():>10.4f} {_bone_length_mean(s):>10.4f} {ka['L_knee']:>10.1f} {ka['R_knee']:>10.1f}")
    skeletons["Clean z0"] = s

    for t_val in [10, 100, 300, 499]:
        t = torch.full((x_real.shape[0],), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(z0_real)
        zt = dp.q_sample(z0=z0_real, t=t, noise=noise)

        with torch.no_grad():
            pred = stage3.denoiser(zt, t, h_tokens=None, h_global=None, gait_metrics=None)

        sqrt_abar = extract(dp.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1m = extract(dp.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        z0_gen = (zt - sqrt_1m * pred) / torch.clamp(sqrt_abar, min=1e-20)

        with torch.no_grad():
            x_hat = stage3.decoder(z0_gen)

        s = x_hat[0].cpu().numpy()
        ka = _knee_angles(s)
        mpjpe = np.linalg.norm(skel_real - s, axis=-1).mean()
        print(f"{'z0_gen t=' + str(t_val):>20} {z0_gen.norm(dim=-1).mean():.2f} {mpjpe:>10.4f} {_bone_length_mean(s):>10.4f} {ka['L_knee']:>10.1f} {ka['R_knee']:>10.1f}")
        skeletons[f"t={t_val}"] = s

    # From random noise (never seen during training)
    z_random = torch.randn_like(z0_real)
    with torch.no_grad():
        x_rand = stage3.decoder(z_random)
    s = x_rand[0].cpu().numpy()
    ka = _knee_angles(s)
    print(f"{'random noise':>20} {z_random.norm(dim=-1).mean():.2f} {np.linalg.norm(skel_real - s, axis=-1).mean():>10.4f} {_bone_length_mean(s):>10.4f} {ka['L_knee']:>10.1f} {ka['R_knee']:>10.1f}")
    skeletons["Random z"] = s

    _save_panel(skeletons, os.path.join(out_dir, "test2_decoder_per_t.png"))
    print(f"\n  -> Saved: {out_dir}/test2_decoder_per_t.png")
    print(f"\n  If all outputs look the same -> decoder collapsed to mean pose.")
    print(f"  If clean z0 looks ok but noisy z0 is spider -> decoder is input-sensitive.")


def test3_fresh_decoder_on_clean_z0(stage1, x_real, device, out_dir):
    """Train a FRESH decoder on CLEAN z0_real for a few steps. Does it learn?"""
    print("\n" + "=" * 70)
    print("TEST 3: Train fresh decoder on CLEAN z0 (no diffusion noise)")
    print("  Proves whether the z0 -> skeleton mapping is learnable at all.")
    print("=" * 70)

    with torch.no_grad():
        z0_real = stage1.encoder(x_real.to(device), gait_metrics=None)

    latent_dim = z0_real.shape[-1]
    skel_real = x_real[0].cpu().numpy()

    # Fresh decoder with same architecture
    fresh_dec = GraphDecoderGCN(latent_dim=latent_dim, output_dim=3, num_joints=DEFAULT_JOINTS).to(device)
    optimizer = optim.Adam(fresh_dec.parameters(), lr=1e-3)

    print(f"\n  Training fresh decoder on {x_real.shape[0]} samples of CLEAN z0_real...")
    print(f"  z0_real norm: {z0_real.norm(dim=-1).mean():.2f}")
    print(f"  x_real range: [{x_real.min():.3f}, {x_real.max():.3f}]")

    x_target = x_real.to(device)

    print(f"\n{'step':>6} {'loss_pose':>12} {'MPJPE':>10} {'bone_len':>10} {'L_knee':>10}")
    print("-" * 52)

    for step in range(201):
        fresh_dec.train()
        x_hat = fresh_dec(z0_real.detach())
        loss = F.smooth_l1_loss(x_hat, x_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 200:
            fresh_dec.eval()
            with torch.no_grad():
                x_hat_eval = fresh_dec(z0_real)
            s = x_hat_eval[0].cpu().numpy()
            mpjpe = np.linalg.norm(skel_real - s, axis=-1).mean()
            bl = _bone_length_mean(s)
            ka = _knee_angles(s)
            print(f"{step:>6} {loss.item():>12.6f} {mpjpe:>10.4f} {bl:>10.4f} {ka['L_knee']:>10.1f}")

    # Final comparison
    fresh_dec.eval()
    with torch.no_grad():
        x_fresh = fresh_dec(z0_real)
    s_fresh = x_fresh[0].cpu().numpy()

    _save_panel(
        {"Real": skel_real, "Fresh dec (clean z0)": s_fresh},
        os.path.join(out_dir, "test3_fresh_decoder.png"),
    )
    print(f"\n  -> Saved: {out_dir}/test3_fresh_decoder.png")

    # Now train same fresh decoder but with NOISY z0 (simulating Stage 3)
    print(f"\n  Now training ANOTHER fresh decoder with NOISY z0_gen (simulating Stage 3)...")
    noisy_dec = GraphDecoderGCN(latent_dim=latent_dim, output_dim=3, num_joints=DEFAULT_JOINTS).to(device)
    optimizer2 = optim.Adam(noisy_dec.parameters(), lr=1e-3)
    dp = stage1.diffusion

    print(f"\n{'step':>6} {'loss_pose':>12} {'MPJPE':>10} {'bone_len':>10} {'L_knee':>10}")
    print("-" * 52)

    for step in range(201):
        noisy_dec.train()
        # Sample random t (same as Stage 3 training)
        t = torch.randint(0, dp.timesteps, (x_real.shape[0],), device=device)
        noise = torch.randn_like(z0_real)
        zt = dp.q_sample(z0=z0_real, t=t, noise=noise)

        with torch.no_grad():
            pred = stage1.denoiser(zt, t, h_tokens=None, h_global=None, gait_metrics=None)

        sqrt_abar = extract(dp.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1m = extract(dp.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        z0_gen = (zt - sqrt_1m * pred) / torch.clamp(sqrt_abar, min=1e-20)

        x_hat = noisy_dec(z0_gen.detach())
        loss = F.smooth_l1_loss(x_hat, x_target)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        if step % 20 == 0 or step == 200:
            noisy_dec.eval()
            with torch.no_grad():
                x_hat_eval = noisy_dec(z0_real)  # evaluate on CLEAN z0
            s = x_hat_eval[0].cpu().numpy()
            mpjpe = np.linalg.norm(skel_real - s, axis=-1).mean()
            bl = _bone_length_mean(s)
            ka = _knee_angles(s)
            print(f"{step:>6} {loss.item():>12.6f} {mpjpe:>10.4f} {bl:>10.4f} {ka['L_knee']:>10.1f}")

    noisy_dec.eval()
    with torch.no_grad():
        x_noisy = noisy_dec(z0_real)
    s_noisy = x_noisy[0].cpu().numpy()

    _save_panel(
        {"Real": skel_real, "Fresh (clean z0)": s_fresh, "Fresh (noisy z0)": s_noisy},
        os.path.join(out_dir, "test3_clean_vs_noisy_training.png"),
    )
    print(f"\n  -> Saved: {out_dir}/test3_clean_vs_noisy_training.png")
    print(f"\n  Comparison: decoder trained on clean z0 vs decoder trained on noisy z0_gen")
    print(f"  If clean learns and noisy doesn't -> NOISY Z0_GEN IS THE CAUSE")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--stage3_ckpt", type=str, default="")
    parser.add_argument("--skeleton_folder", default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--output_dir", default="outputs/decoder_diagnosis")
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)
    batch = next(iter(loader))
    x_real = batch["skeleton"]

    test1_z0_gen_quality_during_training(stage1, x_real, device)

    # Load Stage 3 for test 2
    if args.stage3_ckpt:
        stage2_for_s3 = None
        if args.stage2_ckpt:
            stage2_for_s3 = Stage2Model(
                encoder=stage1.encoder, latent_dim=latent_dim, num_joints=DEFAULT_JOINTS,
                gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM, d_shared=64,
            ).to(device)
            load_checkpoint(args.stage2_ckpt, stage2_for_s3, strict=True)
            stage2_for_s3.eval()

        stage3 = Stage3Model(
            encoder=stage1.encoder, decoder=stage1.decoder, denoiser=stage1.denoiser,
            latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, timesteps=DEFAULT_TIMESTEPS,
            gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM, use_gait_conditioning=False, d_shared=64,
            shared_motion_layer=stage2_for_s3.shared_motion_layer if stage2_for_s3 else None,
        ).to(device)
        load_checkpoint(args.stage3_ckpt, stage3, strict=True)
        stage3.eval()

        test2_stage3_decoder_at_each_t(stage1, stage3, x_real, device, out_dir)

    test3_fresh_decoder_on_clean_z0(stage1, x_real, device, out_dir)

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
