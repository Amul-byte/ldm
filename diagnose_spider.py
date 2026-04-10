"""Comprehensive spider skeleton diagnosis.

Traces through every stage of the pipeline to find exactly where
skeleton quality breaks down. Generates side-by-side visualizations.

Usage:
    python diagnose_spider.py \
        --stage1_ckpt new_checkpoints/stage1_enhanced/stage1_best.pt \
        --stage2_ckpt new_checkpoints/stage2_enhanced/stage2_best.pt \
        --stage3_ckpt new_checkpoints/stage3_enhanced0.1/stage3_best.pt \
        --skeleton_folder /home/qsw26/smartfall/gait_loss/filtered_skeleton_data \
        --hip_folder /home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone \
        --wrist_folder /home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from diffusion_model.dataset import create_dataset
from diffusion_model.diffusion import DiffusionProcess, extract
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.training_eval import sample_stage3_latents
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
    get_joint_index,
    get_skeleton_edges,
)


# ── Visualization helpers ──────────────────────────────────────────

def _project_to_2d(pts: np.ndarray) -> np.ndarray:
    """Project [T, J, 3] -> [T, J, 2] using the two axes with most spread."""
    centered = pts - pts[:, 0:1, :]
    flat = centered.reshape(-1, 3)
    finite = flat[np.isfinite(flat).all(-1)]
    if finite.size == 0:
        return centered[..., :2]
    spread = np.ptp(finite, axis=0)
    axes = np.argsort(spread)[-2:]
    axes.sort()
    return centered[..., axes]


def _draw_skeleton(pts_2d: np.ndarray, canvas: int = 400, title: str = "") -> Image.Image:
    """Draw a single frame [J, 2] on a canvas."""
    img = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    finite = pts_2d[np.isfinite(pts_2d).all(-1)]
    if finite.size == 0:
        draw.text((10, 10), f"{title}\n[NO FINITE POINTS]", fill=(255, 0, 0))
        return img

    mn = finite.min(0)
    mx = finite.max(0)
    span = max((mx - mn).max(), 1e-6)
    margin = 40
    usable = canvas - 2 * margin
    pts_norm = (pts_2d - mn) / span * usable + margin

    edges = get_skeleton_edges()
    for i, j in edges:
        if i < len(pts_norm) and j < len(pts_norm):
            x1, y1 = float(pts_norm[i, 0]), float(pts_norm[i, 1])
            x2, y2 = float(pts_norm[j, 0]), float(pts_norm[j, 1])
            if all(np.isfinite([x1, y1, x2, y2])):
                draw.line((x1, y1, x2, y2), fill=(50, 50, 50), width=2)

    for jj in range(len(pts_norm)):
        x, y = float(pts_norm[jj, 0]), float(pts_norm[jj, 1])
        if np.isfinite(x) and np.isfinite(y):
            r = 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(20, 60, 210))

    draw.text((5, 5), title, fill=(0, 0, 0))
    return img


def _make_comparison_panel(skeletons: dict[str, np.ndarray], frame_idx: int = 0, canvas: int = 400) -> Image.Image:
    """Create a horizontal panel comparing skeletons from different sources."""
    images = []
    for name, skel in skeletons.items():
        pts_2d = _project_to_2d(skel)
        f = min(frame_idx, len(pts_2d) - 1)
        images.append(_draw_skeleton(pts_2d[f], canvas=canvas, title=name))

    total_w = canvas * len(images) + 10 * (len(images) - 1)
    panel = Image.new("RGB", (total_w, canvas), (255, 255, 255))
    for i, img in enumerate(images):
        panel.paste(img, (i * (canvas + 10), 0))
    return panel


def _compute_bone_lengths(skel: np.ndarray) -> np.ndarray:
    """Compute bone lengths [T, num_edges] from skeleton [T, J, 3]."""
    edges = get_skeleton_edges()
    lengths = []
    for i, j in edges:
        if i < skel.shape[1] and j < skel.shape[1]:
            bone = np.linalg.norm(skel[:, i, :] - skel[:, j, :], axis=-1)
            lengths.append(bone)
    return np.stack(lengths, axis=-1)


def _joint_angles(skel: np.ndarray) -> dict:
    """Compute key joint angles for sanity check."""
    angles = {}
    triplets = {
        "left_knee": (get_joint_index("HIP_LEFT"), get_joint_index("KNEE_LEFT"), get_joint_index("ANKLE_LEFT")),
        "right_knee": (get_joint_index("HIP_RIGHT"), get_joint_index("KNEE_RIGHT"), get_joint_index("ANKLE_RIGHT")),
        "left_elbow": (get_joint_index("SHOULDER_LEFT"), get_joint_index("ELBOW_LEFT"), get_joint_index("WRIST_LEFT")),
        "right_elbow": (get_joint_index("SHOULDER_RIGHT"), get_joint_index("ELBOW_RIGHT"), get_joint_index("WRIST_RIGHT")),
    }
    for name, (a, b, c) in triplets.items():
        if a >= skel.shape[1] or b >= skel.shape[1] or c >= skel.shape[1]:
            continue
        v1 = skel[:, a, :] - skel[:, b, :]
        v2 = skel[:, c, :] - skel[:, b, :]
        cos = np.sum(v1 * v2, axis=-1) / (
            np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
        )
        cos = np.clip(cos, -1, 1)
        angles[name] = np.degrees(np.arccos(cos))
    return angles


# ── Main diagnostic checks ─────────────────────────────────────────

def check1_real_data_looks_ok(x_real: torch.Tensor, out_dir: str):
    """Check 1: Does the real data look like a human?"""
    print("\n" + "=" * 60)
    print("CHECK 1: Real data sanity")
    print("=" * 60)

    skel = x_real[0].cpu().numpy()
    bone_lens = _compute_bone_lengths(skel)
    angles = _joint_angles(skel)

    print(f"  Shape: {skel.shape}")
    print(f"  Coordinate range: [{skel.min():.3f}, {skel.max():.3f}]")
    print(f"  Mean bone length: {bone_lens.mean():.4f}")
    print(f"  Bone length std across time: {bone_lens.std(axis=0).mean():.6f}")
    for name, ang in angles.items():
        print(f"  {name} angle: mean={ang.mean():.1f} std={ang.std():.1f} range=[{ang.min():.1f}, {ang.max():.1f}]")

    panel = _make_comparison_panel({"Real frame 0": skel, "Real frame 30": skel, "Real frame 60": skel},
                                    frame_idx=0, canvas=400)
    # Overwrite with actual different frames
    imgs = []
    for f in [0, 30, min(60, len(skel) - 1)]:
        pts_2d = _project_to_2d(skel)
        imgs.append(_draw_skeleton(pts_2d[f], canvas=400, title=f"Real t={f}"))
    total_w = 400 * len(imgs) + 10 * (len(imgs) - 1)
    panel = Image.new("RGB", (total_w, 400), (255, 255, 255))
    for i, img in enumerate(imgs):
        panel.paste(img, (i * 410, 0))
    panel.save(os.path.join(out_dir, "check1_real_data.png"))
    print(f"  -> Saved: {out_dir}/check1_real_data.png")
    return skel


def check2_encoder_decoder_roundtrip(model: Stage1Model, x_real: torch.Tensor, device: torch.device, out_dir: str):
    """Check 2: Encoder -> Decoder roundtrip (no diffusion). If this is a spider, decoder is broken."""
    print("\n" + "=" * 60)
    print("CHECK 2: Encoder -> Decoder roundtrip (bypass diffusion)")
    print("=" * 60)

    with torch.no_grad():
        z0 = model.encoder(x_real.to(device), gait_metrics=None)
        x_hat = model.decoder(z0)

    skel_real = x_real[0].cpu().numpy()
    skel_hat = x_hat[0].cpu().numpy()
    z0_np = z0[0].cpu().numpy()

    mpjpe = np.linalg.norm(skel_real - skel_hat, axis=-1).mean()
    bone_lens_real = _compute_bone_lengths(skel_real)
    bone_lens_hat = _compute_bone_lengths(skel_hat)
    angles_real = _joint_angles(skel_real)
    angles_hat = _joint_angles(skel_hat)

    print(f"  z0 shape: {z0.shape}, norm: {z0.norm(dim=-1).mean():.2f}")
    print(f"  z0 range: [{z0.min():.3f}, {z0.max():.3f}]")
    print(f"  x_hat range: [{skel_hat.min():.3f}, {skel_hat.max():.3f}]")
    print(f"  MPJPE (encoder->decoder, no diffusion): {mpjpe:.6f}")
    print(f"  Real bone length mean: {bone_lens_real.mean():.4f}")
    print(f"  Reconstructed bone length mean: {bone_lens_hat.mean():.4f}")
    print(f"  Bone length ratio: {bone_lens_hat.mean() / max(bone_lens_real.mean(), 1e-8):.3f}")

    for name in angles_real:
        if name in angles_hat:
            r = angles_real[name].mean()
            h = angles_hat[name].mean()
            print(f"  {name}: real={r:.1f} recon={h:.1f} diff={abs(r-h):.1f}")

    skels = {"Real": skel_real, "Enc->Dec (no diff)": skel_hat}
    panel = _make_comparison_panel(skels, frame_idx=0)
    panel.save(os.path.join(out_dir, "check2_roundtrip.png"))
    print(f"  -> Saved: {out_dir}/check2_roundtrip.png")

    if mpjpe > 1.0:
        print("  *** WARNING: MPJPE > 1.0 — decoder cannot reconstruct even clean encoder latents!")
        print("  *** This means the decoder has NOT been trained or is severely broken.")
    return z0, x_hat


def check3_diffusion_then_decode(model: Stage1Model, x_real: torch.Tensor, device: torch.device, out_dir: str):
    """Check 3: Run Stage 1 diffusion reverse process, then decode."""
    print("\n" + "=" * 60)
    print("CHECK 3: Diffusion reverse -> Decoder (Stage 1 unconditional)")
    print("=" * 60)

    with torch.no_grad():
        z0_real = model.encoder(x_real.to(device), gait_metrics=None)

        # Full DDPM reverse
        z0_gen = model.diffusion.p_sample_loop(
            denoiser=model.denoiser,
            shape=z0_real.shape,
            device=device,
        )
        x_gen = model.decoder(z0_gen)

        # Also try DDIM
        z0_ddim = model.diffusion.p_sample_loop_ddim(
            denoiser=model.denoiser,
            shape=z0_real.shape,
            device=device,
            sample_steps=50,
        )
        x_ddim = model.decoder(z0_ddim)

    skel_real = x_real[0].cpu().numpy()
    skel_gen = x_gen[0].cpu().numpy()
    skel_ddim = x_ddim[0].cpu().numpy()

    print(f"  z0_real norm: {z0_real.norm(dim=-1).mean():.2f}")
    print(f"  z0_gen (DDPM) norm: {z0_gen.norm(dim=-1).mean():.2f}")
    print(f"  z0_gen (DDIM) norm: {z0_ddim.norm(dim=-1).mean():.2f}")
    print(f"  z0 L2 distance (DDPM vs real): {(z0_gen - z0_real).norm(dim=-1).mean():.2f}")
    print(f"  z0 L2 distance (DDIM vs real): {(z0_ddim - z0_real).norm(dim=-1).mean():.2f}")
    print(f"  x_gen range (DDPM): [{skel_gen.min():.3f}, {skel_gen.max():.3f}]")
    print(f"  x_gen range (DDIM): [{skel_ddim.min():.3f}, {skel_ddim.max():.3f}]")

    mpjpe_ddpm = np.linalg.norm(skel_real - skel_gen, axis=-1).mean()
    mpjpe_ddim = np.linalg.norm(skel_real - skel_ddim, axis=-1).mean()
    print(f"  MPJPE (DDPM gen): {mpjpe_ddpm:.6f}")
    print(f"  MPJPE (DDIM gen): {mpjpe_ddim:.6f}")

    bones_gen = _compute_bone_lengths(skel_gen)
    bones_ddim = _compute_bone_lengths(skel_ddim)
    print(f"  Bone length mean (DDPM): {bones_gen.mean():.4f}")
    print(f"  Bone length mean (DDIM): {bones_ddim.mean():.4f}")
    print(f"  Bone length temporal std (DDPM): {bones_gen.std(axis=0).mean():.6f}")
    print(f"  Bone length temporal std (DDIM): {bones_ddim.std(axis=0).mean():.6f}")

    skels = {"Real": skel_real, "DDPM gen": skel_gen, "DDIM gen": skel_ddim}
    panel = _make_comparison_panel(skels, frame_idx=0)
    panel.save(os.path.join(out_dir, "check3_diffusion_gen.png"))
    print(f"  -> Saved: {out_dir}/check3_diffusion_gen.png")
    return z0_gen, z0_ddim


def check4_z0_gen_explosion(model: Stage1Model, x_real: torch.Tensor, device: torch.device, out_dir: str):
    """Check 4: Measure z0_gen (one-step x0 estimate) at different timesteps — the exact bug."""
    print("\n" + "=" * 60)
    print("CHECK 4: One-step z0 estimate quality at different timesteps")
    print("    (This is what Stage 3 trains its decoder on)")
    print("=" * 60)

    with torch.no_grad():
        z0_real = model.encoder(x_real.to(device), gait_metrics=None)

    z0_norm = z0_real.norm(dim=-1).mean().item()
    dp = model.diffusion

    print(f"  Encoder z0 norm: {z0_norm:.2f}")
    print(f"\n{'t':>6} {'z0_est norm':>14} {'z0_est(fixed)':>16} {'MPJPE raw':>12} {'MPJPE fixed':>14} {'decoded ok?':>14}")
    print("-" * 80)

    frames_buggy = []
    frames_fixed = []

    for t_val in [0, 10, 50, 100, 200, 300, 400, 450, 499]:
        t = torch.full((x_real.shape[0],), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(z0_real)
        zt = dp.q_sample(z0=z0_real, t=t, noise=noise)

        with torch.no_grad():
            pred_noise = model.denoiser(zt, t, h_tokens=None, h_global=None, gait_metrics=None)

        sqrt_abar = extract(dp.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1m_abar = extract(dp.sqrt_one_minus_alphas_cumprod, t, zt.shape)

        # Buggy (current Stage 3 code)
        z0_buggy = (zt - sqrt_1m_abar * pred_noise) / torch.clamp(sqrt_abar, min=1e-20)
        # Fixed (Stage 1 pattern)
        z0_fixed = ((zt - sqrt_1m_abar * pred_noise.float()).float()
                     / torch.clamp(sqrt_abar.float(), min=0.1)).clamp(-50, 50)

        with torch.no_grad():
            x_buggy = model.decoder(z0_buggy)
            x_fixed = model.decoder(z0_fixed)

        skel_real = x_real[0].cpu().numpy()
        skel_buggy = x_buggy[0].cpu().numpy()
        skel_fixed = x_fixed[0].cpu().numpy()

        mpjpe_buggy = np.linalg.norm(skel_real - skel_buggy, axis=-1).mean()
        mpjpe_fixed = np.linalg.norm(skel_real - skel_fixed, axis=-1).mean()

        norm_buggy = z0_buggy.norm(dim=-1).mean().item()
        norm_fixed = z0_fixed.norm(dim=-1).mean().item()

        ok = "YES" if mpjpe_buggy < 0.5 else ("MARGINAL" if mpjpe_buggy < 2.0 else "SPIDER")

        print(f"{t_val:>6} {norm_buggy:>14.1f} {norm_fixed:>16.1f} {mpjpe_buggy:>12.4f} {mpjpe_fixed:>14.4f} {ok:>14}")

        if t_val in [0, 200, 400, 499]:
            frames_buggy.append((f"t={t_val} buggy", skel_buggy))
            frames_fixed.append((f"t={t_val} fixed", skel_fixed))

    # Save comparison panels
    buggy_dict = {name: s for name, s in frames_buggy}
    fixed_dict = {name: s for name, s in frames_fixed}
    all_dict = {"Real": skel_real, **buggy_dict}
    panel = _make_comparison_panel(all_dict, frame_idx=0, canvas=300)
    panel.save(os.path.join(out_dir, "check4_z0_buggy_by_t.png"))

    all_dict2 = {"Real": skel_real, **fixed_dict}
    panel2 = _make_comparison_panel(all_dict2, frame_idx=0, canvas=300)
    panel2.save(os.path.join(out_dir, "check4_z0_fixed_by_t.png"))
    print(f"  -> Saved: {out_dir}/check4_z0_buggy_by_t.png")
    print(f"  -> Saved: {out_dir}/check4_z0_fixed_by_t.png")


def check5_stage3_conditioned(
    stage3: Stage3Model, stage2: Stage2Model,
    x_real: torch.Tensor, a_hip: torch.Tensor, a_wrist: torch.Tensor,
    device: torch.device, out_dir: str
):
    """Check 5: Full Stage 3 conditioned generation."""
    print("\n" + "=" * 60)
    print("CHECK 5: Full Stage 3 conditioned generation")
    print("=" * 60)

    with torch.no_grad():
        h_tokens, h_global = stage2.aligner(
            a_hip_stream=a_hip.to(device),
            a_wrist_stream=a_wrist.to(device),
        )

        z0_real = stage3.encoder(x_real.to(device), gait_metrics=None)

        # DDIM conditioned
        z0_cond = stage3.diffusion.p_sample_loop_ddim(
            denoiser=stage3.denoiser,
            shape=z0_real.shape,
            device=device,
            sample_steps=50,
            h_tokens=h_tokens,
            h_global=h_global,
        )
        x_cond = stage3.decoder(z0_cond)

        # DDIM unconditioned
        z0_uncond = stage3.diffusion.p_sample_loop_ddim(
            denoiser=stage3.denoiser,
            shape=z0_real.shape,
            device=device,
            sample_steps=50,
        )
        x_uncond = stage3.decoder(z0_uncond)

    skel_real = x_real[0].cpu().numpy()
    skel_cond = x_cond[0].cpu().numpy()
    skel_uncond = x_uncond[0].cpu().numpy()

    print(f"  z0_cond norm: {z0_cond.norm(dim=-1).mean():.2f}")
    print(f"  z0_uncond norm: {z0_uncond.norm(dim=-1).mean():.2f}")
    print(f"  z0_real norm: {z0_real.norm(dim=-1).mean():.2f}")
    print(f"  x_cond range: [{skel_cond.min():.3f}, {skel_cond.max():.3f}]")
    print(f"  x_uncond range: [{skel_uncond.min():.3f}, {skel_uncond.max():.3f}]")

    mpjpe_cond = np.linalg.norm(skel_real - skel_cond, axis=-1).mean()
    mpjpe_uncond = np.linalg.norm(skel_real - skel_uncond, axis=-1).mean()
    print(f"  MPJPE conditioned: {mpjpe_cond:.6f}")
    print(f"  MPJPE unconditioned: {mpjpe_uncond:.6f}")

    bones_cond = _compute_bone_lengths(skel_cond)
    bones_uncond = _compute_bone_lengths(skel_uncond)
    bones_real = _compute_bone_lengths(skel_real)
    print(f"  Bone length mean: real={bones_real.mean():.4f}  cond={bones_cond.mean():.4f}  uncond={bones_uncond.mean():.4f}")

    angles_real = _joint_angles(skel_real)
    angles_cond = _joint_angles(skel_cond)
    angles_uncond = _joint_angles(skel_uncond)
    for name in angles_real:
        r = angles_real[name].mean()
        c = angles_cond.get(name, np.array([0])).mean()
        u = angles_uncond.get(name, np.array([0])).mean()
        print(f"  {name}: real={r:.1f}  cond={c:.1f}  uncond={u:.1f}")

    skels = {"Real": skel_real, "Stage3 Cond": skel_cond, "Stage3 Uncond": skel_uncond}
    panel = _make_comparison_panel(skels, frame_idx=0, canvas=400)
    panel.save(os.path.join(out_dir, "check5_stage3_gen.png"))
    print(f"  -> Saved: {out_dir}/check5_stage3_gen.png")

    # Multi-frame for conditioned
    imgs = []
    pts_2d = _project_to_2d(skel_cond)
    for f in [0, 15, 30, 45, min(60, len(skel_cond) - 1)]:
        imgs.append(_draw_skeleton(pts_2d[f], canvas=300, title=f"Stage3 t={f}"))
    total_w = 300 * len(imgs) + 10 * (len(imgs) - 1)
    panel_multi = Image.new("RGB", (total_w, 300), (255, 255, 255))
    for i, img in enumerate(imgs):
        panel_multi.paste(img, (i * 310, 0))
    panel_multi.save(os.path.join(out_dir, "check5_stage3_multiframe.png"))
    print(f"  -> Saved: {out_dir}/check5_stage3_multiframe.png")


def check6_training_loss_analysis(model: Stage3Model, x_real: torch.Tensor,
                                   h_tokens: torch.Tensor, h_global: torch.Tensor,
                                   gait_metrics: torch.Tensor,
                                   device: torch.device, out_dir: str):
    """Check 6: Analyze per-timestep training loss to see contamination."""
    print("\n" + "=" * 60)
    print("CHECK 6: Per-timestep training loss analysis")
    print("    (Shows how z0 estimation error grows with t)")
    print("=" * 60)

    with torch.no_grad():
        z0_real = model.encoder(x_real.to(device), gait_metrics=None)

    dp = model.diffusion
    print(f"\n{'t':>6} {'loss_diff':>12} {'loss_latent':>14} {'loss_pose':>12} {'z0_gen norm':>14} {'decoder out range':>20}")
    print("-" * 82)

    for t_val in [0, 10, 50, 100, 200, 300, 400, 450, 499]:
        t = torch.full((x_real.shape[0],), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(z0_real)
        zt = dp.q_sample(z0=z0_real, t=t, noise=noise)

        with torch.no_grad():
            pred_noise = model.denoiser(
                zt, t, h_tokens=h_tokens.to(device), h_global=h_global.to(device), gait_metrics=None
            )

        sqrt_abar = extract(dp.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1m_abar = extract(dp.sqrt_one_minus_alphas_cumprod, t, zt.shape)

        z0_gen = (zt - sqrt_1m_abar * pred_noise) / torch.clamp(sqrt_abar, min=1e-20)

        with torch.no_grad():
            x_hat = model.decoder(z0_gen.float())

        loss_diff = F.mse_loss(pred_noise, noise).item()
        loss_latent = F.mse_loss(z0_gen, z0_real).item()
        loss_pose = F.smooth_l1_loss(x_hat, x_real.to(device)).item()
        z0_norm = z0_gen.norm(dim=-1).mean().item()
        x_range = f"[{x_hat.min().item():.1f}, {x_hat.max().item():.1f}]"

        print(f"{t_val:>6} {loss_diff:>12.4f} {loss_latent:>14.4f} {loss_pose:>12.4f} {z0_norm:>14.1f} {x_range:>20}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive spider skeleton diagnosis")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--stage3_ckpt", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--output_dir", type=str, default="outputs/spider_diagnosis")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--d_shared", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # ── Load checkpoints ────────────────────────────────────────
    encoder_type, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)
    # Detect temporal_block_type from checkpoint keys
    _ckpt_sd = torch.load(args.stage1_ckpt, map_location="cpu")
    _sd = _ckpt_sd["state_dict"] if "state_dict" in _ckpt_sd else _ckpt_sd
    _has_temporal_attn = any("denoiser.temporal_blocks.0.attn" in k for k in _sd)
    temporal_block_type = "attention" if _has_temporal_attn else "conv"
    # Detect latent_dim from encoder.in_proj.weight shape
    _in_proj_key = "encoder.in_proj.weight"
    if _in_proj_key in _sd:
        args.latent_dim = _sd[_in_proj_key].shape[0]
    del _ckpt_sd, _sd
    print(f"Graph ops: encoder={encoder_type}, skeleton={skeleton_graph_op}, temporal={temporal_block_type}")

    stage1 = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        encoder_type=encoder_type,
        skeleton_graph_op=skeleton_graph_op,
        temporal_block_type=temporal_block_type,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    stage1.eval()

    # ── Load real data ──────────────────────────────────────────
    dataset = create_dataset(
        dataset_path=None,
        skeleton_folder=args.skeleton_folder,
        hip_folder=args.hip_folder,
        wrist_folder=args.wrist_folder,
        window=args.window,
        joints=args.joints,
        stride=30,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)
    batch = next(iter(loader))
    x_real = batch["skeleton"]
    a_hip = batch["A_hip"]
    a_wrist = batch["A_wrist"]
    gait_metrics = batch["gait_metrics"]
    print(f"Loaded real data: skeleton={x_real.shape}, hip={a_hip.shape}, wrist={a_wrist.shape}")

    # ── Run checks ──────────────────────────────────────────────
    check1_real_data_looks_ok(x_real, out_dir)
    check2_encoder_decoder_roundtrip(stage1, x_real, device, out_dir)
    check3_diffusion_then_decode(stage1, x_real, device, out_dir)
    check4_z0_gen_explosion(stage1, x_real, device, out_dir)

    # Stage 3 checks (need stage2 + stage3 checkpoints)
    if args.stage2_ckpt and args.stage3_ckpt:
        stage2 = Stage2Model(
            encoder=stage1.encoder,
            latent_dim=args.latent_dim,
            num_joints=args.joints,
            gait_metrics_dim=args.gait_metrics_dim,
            d_shared=args.d_shared,
        ).to(device)
        load_checkpoint(args.stage2_ckpt, stage2, strict=True)
        stage2.eval()

        stage3 = Stage3Model(
            encoder=stage1.encoder,
            decoder=stage1.decoder,
            denoiser=stage1.denoiser,
            latent_dim=args.latent_dim,
            num_joints=args.joints,
            timesteps=args.timesteps,
            gait_metrics_dim=args.gait_metrics_dim,
            use_gait_conditioning=False,
            d_shared=args.d_shared,
            shared_motion_layer=stage2.shared_motion_layer,
        ).to(device)
        load_checkpoint(args.stage3_ckpt, stage3, strict=True)
        stage3.eval()

        with torch.no_grad():
            h_tokens, h_global = stage2.aligner(
                a_hip_stream=a_hip.to(device),
                a_wrist_stream=a_wrist.to(device),
            )

        check5_stage3_conditioned(stage3, stage2, x_real, a_hip, a_wrist, device, out_dir)
        check6_training_loss_analysis(stage3, x_real, h_tokens, h_global, gait_metrics, device, out_dir)
    else:
        print("\n[Skipping Stage 3 checks — provide --stage2_ckpt and --stage3_ckpt]")

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print(f"All images saved to: {out_dir}/")
    print("=" * 60)
    print("\nHow to interpret:")
    print("  Check 2: If spider HERE -> decoder is broken (never trained / only Stage 1 decoder)")
    print("  Check 3: If spider HERE but Check 2 ok -> diffusion reverse is corrupting latents")
    print("  Check 4: Shows exactly how z0 estimation degrades with t")
    print("  Check 5: If spider HERE but Check 3 ok -> Stage 3 conditioning or training broke it")
    print("  Check 6: Shows per-timestep loss contamination during Stage 3 training")


if __name__ == "__main__":
    main()
