"""End-to-end pipeline trace: visualize the skeleton at EVERY stage.

Produces a sequence of skeleton plots showing:
  1. Real skeleton (ground truth)
  2. Encoder output z0 (projected to 3D via decoder out_proj)
  3. Forward-diffused zt at several timesteps (projected to 3D)
  4. Single-step denoiser z0_gen WITHOUT conditioning (projected)
  5. Single-step denoiser z0_gen WITH conditioning (projected)
  6. Full 50-step reverse diffusion z0_diff (projected to 3D)
  7. Decoder output from z0_real (autoencoder path)
  8. Decoder output from z0_diff (full pipeline)

This lets you see exactly where the skeleton breaks down.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffusion_model.dataset import create_dataset
from diffusion_model.diffusion import extract
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model  # noqa: E402
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.training_eval import render_skeleton_panels, sample_stage3_latents
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
)


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def _infer_latent_dim(stage1_ckpt: str) -> int:
    return int(_load_state_dict(stage1_ckpt)["encoder.in_proj.weight"].shape[0])


def _infer_d_shared(stage2_ckpt: str, default: int = 64) -> int:
    sd = _load_state_dict(stage2_ckpt)
    key = "shared_motion_layer.net.0.weight"
    return int(sd[key].shape[0]) if key in sd else int(default)


def project_latent_to_3d(z: torch.Tensor, decoder: torch.nn.Module) -> torch.Tensor:
    """Project a latent [B,T,J,D] to 3D [B,T,J,3] using only the decoder's output linear."""
    with torch.no_grad():
        return decoder.out_proj(z)


def project_latent_through_decoder_layers(
    z: torch.Tensor, decoder: torch.nn.Module
) -> list[tuple[str, torch.Tensor]]:
    """Run latent through decoder layer by layer, projecting each intermediate to 3D."""
    results = []
    results.append(("input_latent\n(projected)", decoder.out_proj(z).detach()))
    h = z
    adjacency = decoder._skel_adjacency
    edge_index = decoder._skel_edge_index
    for i, (g_block, t_block) in enumerate(zip(decoder.graph_blocks, decoder.temporal_blocks)):
        h = g_block(h, adjacency=adjacency, edge_index=edge_index)
        results.append((f"after_graph_{i}\n(projected)", decoder.out_proj(h).detach()))
        h = t_block(h)
        results.append((f"after_temporal_{i}\n(projected)", decoder.out_proj(h).detach()))
    final = decoder.out_proj(h)
    results.append(("final_output", final.detach()))
    return results


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline trace")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/full_pipeline_trace")
    parser.add_argument("--skeleton_folder", type=str, default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--latent_dim", type=int, default=None, help="Auto-inferred from stage1 ckpt if omitted")
    parser.add_argument("--d_shared", type=int, default=None, help="Auto-inferred from stage2 ckpt if omitted")
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--num_joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="ddpm")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--encoder_type", type=str, default="gcn")
    parser.add_argument("--sample_index", type=int, default=0, help="Which sample in the batch to plot")
    # Forward diffusion timesteps to visualize
    parser.add_argument(
        "--noise_timesteps", type=int, nargs="+", default=[10, 50, 100, 250, 400, 499],
        help="Timesteps at which to visualize forward-diffused latent",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = args.sample_index

    # Auto-infer dimensions from checkpoints
    latent_dim = args.latent_dim or _infer_latent_dim(args.stage1_ckpt)
    d_shared = args.d_shared or _infer_d_shared(args.stage2_ckpt)
    graph_op = args.encoder_type
    print(f"Config: latent_dim={latent_dim}, d_shared={d_shared}, graph_op={graph_op}")

    # ── Load models ──────────────────────────────────────────────────────
    print("Loading Stage 1...")
    stage1 = Stage1Model(
        latent_dim=latent_dim, num_joints=args.num_joints,
        num_classes=args.num_classes, timesteps=args.timesteps,
        encoder_type=graph_op, skeleton_graph_op=graph_op,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=False)
    stage1.eval()

    print("Loading Stage 2...")
    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=latent_dim, num_joints=args.num_joints,
        num_classes=args.num_classes, d_shared=d_shared,
        gait_metrics_dim=args.gait_metrics_dim,
    ).to(device)
    load_checkpoint(args.stage2_ckpt, stage2, strict=False)
    stage2.eval()

    print("Loading Stage 3...")
    stage3 = Stage3Model(
        encoder=stage1.encoder, decoder=stage1.decoder,
        denoiser=stage1.denoiser, latent_dim=latent_dim,
        num_joints=args.num_joints, num_classes=args.num_classes,
        timesteps=args.timesteps, d_shared=d_shared,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        shared_motion_layer=stage2.shared_motion_layer,
    ).to(device)
    load_checkpoint(args.stage3_ckpt, stage3, strict=False)
    stage3.eval()

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading dataset...")
    ds = create_dataset(
        dataset_path=None,
        skeleton_folder=args.skeleton_folder,
        hip_folder=args.hip_folder,
        wrist_folder=args.wrist_folder,
        window=args.window,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                          drop_last=True, num_workers=0)
    batch = next(iter(loader))
    x = batch["skeleton"].to(device)
    a_hip = batch["A_hip"].to(device)
    a_wrist = batch["A_wrist"].to(device)
    labels = batch["label"]

    print(f"Batch: skeleton {tuple(x.shape)}, label={labels[idx].item()}")
    print(f"Skeleton range: [{x.min().item():.3f}, {x.max().item():.3f}]")

    summary = {"config": vars(args), "sample_index": idx, "label": int(labels[idx].item())}

    with torch.no_grad():
        # ════════════════════════════════════════════════════════════════
        # STEP 1: Real skeleton
        # ════════════════════════════════════════════════════════════════
        x_np = x[idx].cpu().numpy()
        print(f"\n[1] Real skeleton: shape={x_np.shape}")

        # ════════════════════════════════════════════════════════════════
        # STEP 2: Encoder → z0 (clean latent)
        # ════════════════════════════════════════════════════════════════
        z0_real = stage3.encoder(x, gait_metrics=None)
        z0_proj = project_latent_to_3d(z0_real, stage3.decoder)
        z0_proj_np = z0_proj[idx].cpu().numpy()
        print(f"[2] Encoder z0: shape={tuple(z0_real.shape)}, mean={z0_real.mean():.4f}, std={z0_real.std():.4f}")
        summary["z0_real"] = {"mean": float(z0_real.mean()), "std": float(z0_real.std())}

        # ════════════════════════════════════════════════════════════════
        # STEP 3: Forward diffusion — noise z0 at various timesteps
        # ════════════════════════════════════════════════════════════════
        noise = torch.randn_like(z0_real)
        noised_projections = {}
        print(f"\n[3] Forward diffusion (adding noise):")
        for t_val in args.noise_timesteps:
            t_tensor = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
            zt = stage3.diffusion.q_sample(z0=z0_real, t=t_tensor, noise=noise)
            zt_proj = project_latent_to_3d(zt, stage3.decoder)
            noised_projections[t_val] = zt_proj[idx].cpu().numpy()
            zt_stats = {"mean": float(zt.mean()), "std": float(zt.std())}
            print(f"    t={t_val:3d}: mean={zt_stats['mean']:.4f}, std={zt_stats['std']:.4f}")
            summary[f"zt_t{t_val}"] = zt_stats

        # ════════════════════════════════════════════════════════════════
        # STEP 4: IMU conditioning — get h_tokens, h_global from Stage 2
        # ════════════════════════════════════════════════════════════════
        h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)
        print(f"\n[4] IMU conditioning: h_tokens={tuple(h_tokens.shape)}, h_global={tuple(h_global.shape)}")
        summary["h_tokens"] = {"mean": float(h_tokens.mean()), "std": float(h_tokens.std())}
        summary["h_global"] = {"mean": float(h_global.mean()), "std": float(h_global.std())}

        # Augmented conditioning (with shared motion features)
        h_tokens_aug, h_global_aug = stage3.augment_conditioning(
            h_tokens=h_tokens, h_global=h_global,
            a_hip_stream=a_hip, a_wrist_stream=a_wrist,
        )
        print(f"    After augment: h_tokens_aug mean={h_tokens_aug.mean():.4f}, std={h_tokens_aug.std():.4f}")
        summary["h_tokens_aug"] = {"mean": float(h_tokens_aug.mean()), "std": float(h_tokens_aug.std())}

        # ════════════════════════════════════════════════════════════════
        # STEP 5: Single-step denoiser estimate (what training sees)
        #         Both WITH and WITHOUT conditioning
        # ════════════════════════════════════════════════════════════════
        # Pick a mid-range t to show a representative training example
        single_step_results = {}
        for t_val in [50, 250]:
            t_tensor = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
            zt = stage3.diffusion.q_sample(z0=z0_real, t=t_tensor, noise=noise)

            # WITH conditioning
            pred_noise_cond = stage3.denoiser(
                zt, t_tensor, sensor_tokens=h_tokens_aug,
                h_tokens=h_tokens_aug, h_global=h_global_aug, gait_metrics=None,
            )
            sqrt_alpha = extract(stage3.diffusion.sqrt_alphas_cumprod, t_tensor, zt.shape)
            sqrt_one_minus = extract(stage3.diffusion.sqrt_one_minus_alphas_cumprod, t_tensor, zt.shape)
            z0_gen_cond = (zt - sqrt_one_minus * pred_noise_cond) / torch.clamp(sqrt_alpha, min=1e-20)

            # WITHOUT conditioning (zeros)
            h_zeros = torch.zeros_like(h_tokens_aug)
            h_global_zeros = torch.zeros_like(h_global_aug)
            pred_noise_uncond = stage3.denoiser(
                zt, t_tensor, sensor_tokens=h_zeros,
                h_tokens=h_zeros, h_global=h_global_zeros, gait_metrics=None,
            )
            z0_gen_uncond = (zt - sqrt_one_minus * pred_noise_uncond) / torch.clamp(sqrt_alpha, min=1e-20)

            z0_gen_cond_proj = project_latent_to_3d(z0_gen_cond, stage3.decoder)
            z0_gen_uncond_proj = project_latent_to_3d(z0_gen_uncond, stage3.decoder)

            single_step_results[t_val] = {
                "cond_proj": z0_gen_cond_proj[idx].cpu().numpy(),
                "uncond_proj": z0_gen_uncond_proj[idx].cpu().numpy(),
                "cond_l2_to_z0": float(torch.linalg.norm(z0_gen_cond - z0_real, dim=-1).mean()),
                "uncond_l2_to_z0": float(torch.linalg.norm(z0_gen_uncond - z0_real, dim=-1).mean()),
            }
            print(f"\n[5] Single-step denoise at t={t_val}:")
            print(f"    WITH conditioning:    L2 to z0 = {single_step_results[t_val]['cond_l2_to_z0']:.4f}")
            print(f"    WITHOUT conditioning: L2 to z0 = {single_step_results[t_val]['uncond_l2_to_z0']:.4f}")
            summary[f"single_step_t{t_val}"] = {
                "cond_l2": single_step_results[t_val]["cond_l2_to_z0"],
                "uncond_l2": single_step_results[t_val]["uncond_l2_to_z0"],
            }

        # ════════════════════════════════════════════════════════════════
        # STEP 6: Full reverse diffusion (50 steps) — what inference sees
        #         Both WITH and WITHOUT conditioning
        # ════════════════════════════════════════════════════════════════
        print(f"\n[6] Full reverse diffusion ({args.sample_steps} steps, {args.sampler})...")
        z0_diff_cond = sample_stage3_latents(
            stage3=stage3, shape=z0_real.shape, device=device,
            h_tokens=h_tokens, h_global=h_global,
            a_hip_stream=a_hip, a_wrist_stream=a_wrist,
            gait_metrics=None, sample_steps=args.sample_steps,
            sampler=args.sampler, sample_seed=args.sample_seed,
        )
        z0_diff_cond_proj = project_latent_to_3d(z0_diff_cond, stage3.decoder)

        # Without conditioning
        z0_diff_uncond = sample_stage3_latents(
            stage3=stage3, shape=z0_real.shape, device=device,
            h_tokens=torch.zeros_like(h_tokens), h_global=torch.zeros_like(h_global),
            a_hip_stream=None, a_wrist_stream=None,
            gait_metrics=None, sample_steps=args.sample_steps,
            sampler=args.sampler, sample_seed=args.sample_seed,
        )
        z0_diff_uncond_proj = project_latent_to_3d(z0_diff_uncond, stage3.decoder)

        cond_l2 = float(torch.linalg.norm(z0_diff_cond - z0_real, dim=-1).mean())
        uncond_l2 = float(torch.linalg.norm(z0_diff_uncond - z0_real, dim=-1).mean())
        print(f"    WITH conditioning:    L2 to z0 = {cond_l2:.4f}, mean={z0_diff_cond.mean():.4f}, std={z0_diff_cond.std():.4f}")
        print(f"    WITHOUT conditioning: L2 to z0 = {uncond_l2:.4f}, mean={z0_diff_uncond.mean():.4f}, std={z0_diff_uncond.std():.4f}")
        summary["full_reverse_cond"] = {"l2_to_z0": cond_l2, "mean": float(z0_diff_cond.mean()), "std": float(z0_diff_cond.std())}
        summary["full_reverse_uncond"] = {"l2_to_z0": uncond_l2, "mean": float(z0_diff_uncond.mean()), "std": float(z0_diff_uncond.std())}

        # ════════════════════════════════════════════════════════════════
        # STEP 7: Decoder outputs
        # ════════════════════════════════════════════════════════════════
        x_hat_autoenc = stage3.decoder(z0_real)
        x_hat_cond = stage3.decoder(z0_diff_cond)
        x_hat_uncond = stage3.decoder(z0_diff_uncond)

        mpjpe_autoenc = float(torch.linalg.norm(x_hat_autoenc - x, dim=-1).mean())
        mpjpe_cond = float(torch.linalg.norm(x_hat_cond - x, dim=-1).mean())
        mpjpe_uncond = float(torch.linalg.norm(x_hat_uncond - x, dim=-1).mean())

        print(f"\n[7] Decoder outputs (MPJPE in meters):")
        print(f"    Autoencoder (clean z0):          {mpjpe_autoenc:.4f}")
        print(f"    Full pipeline (conditioned):     {mpjpe_cond:.4f}")
        print(f"    Full pipeline (unconditioned):   {mpjpe_uncond:.4f}")
        summary["decoder_mpjpe"] = {
            "autoencoder": mpjpe_autoenc,
            "full_cond": mpjpe_cond,
            "full_uncond": mpjpe_uncond,
        }

        # ════════════════════════════════════════════════════════════════
        # STEP 8: Decoder layer-by-layer trace on BOTH clean and diffused latents
        # ════════════════════════════════════════════════════════════════
        print(f"\n[8] Decoder layer trace on clean z0 vs diffused z0...")
        trace_clean = project_latent_through_decoder_layers(z0_real, stage3.decoder)
        trace_diff = project_latent_through_decoder_layers(z0_diff_cond, stage3.decoder)

    # ════════════════════════════════════════════════════════════════════
    # RENDER PLOTS
    # ════════════════════════════════════════════════════════════════════
    print(f"\nRendering plots to {out_dir}/...")

    # ── Plot 1: The big picture — Real → Encoder → Noise → Denoise → Decode ──
    seqs_main = [
        x_np,                                           # Real
        z0_proj_np,                                     # Encoder z0 projected
        noised_projections[50],                         # zt at t=50
        noised_projections[250],                        # zt at t=250
        noised_projections[499],                        # zt at t=499 (near pure noise)
        z0_diff_cond_proj[idx].cpu().numpy(),           # After full reverse (conditioned)
        x_hat_autoenc[idx].cpu().numpy(),               # Decoder(z0_real)
        x_hat_cond[idx].cpu().numpy(),                  # Decoder(z0_diff_cond)
    ]
    titles_main = [
        "1. Real\nskeleton",
        "2. Encoder z0\n(projected)",
        "3. Noised zt\nt=50",
        "4. Noised zt\nt=250",
        "5. Noised zt\nt=499",
        "6. After 50-step\nreverse (projected)",
        "7. Decoder\n(clean z0)",
        "8. Decoder\n(diffused z0)",
    ]
    render_skeleton_panels(out_dir / "01_full_pipeline_overview.png", seqs_main, titles_main)
    print("  -> 01_full_pipeline_overview.png")

    # ── Plot 2: Forward diffusion progression ──
    seqs_noise = [x_np, z0_proj_np]
    titles_noise = ["Real", "z0 (clean)"]
    for t_val in args.noise_timesteps:
        seqs_noise.append(noised_projections[t_val])
        titles_noise.append(f"zt (t={t_val})")
    render_skeleton_panels(out_dir / "02_forward_diffusion.png", seqs_noise, titles_noise)
    print("  -> 02_forward_diffusion.png")

    # ── Plot 3: Conditioning effect — single-step denoiser ──
    seqs_cond = [x_np, z0_proj_np]
    titles_cond = ["Real", "z0 (clean)"]
    for t_val in [50, 250]:
        seqs_cond.append(single_step_results[t_val]["uncond_proj"])
        titles_cond.append(f"1-step uncond\nt={t_val}")
        seqs_cond.append(single_step_results[t_val]["cond_proj"])
        titles_cond.append(f"1-step cond\nt={t_val}")
    render_skeleton_panels(out_dir / "03_conditioning_effect_single_step.png", seqs_cond, titles_cond)
    print("  -> 03_conditioning_effect_single_step.png")

    # ── Plot 4: Conditioning effect — full reverse ──
    seqs_reverse = [
        x_np,
        z0_proj_np,
        z0_diff_uncond_proj[idx].cpu().numpy(),
        z0_diff_cond_proj[idx].cpu().numpy(),
        x_hat_autoenc[idx].cpu().numpy(),
        x_hat_uncond[idx].cpu().numpy(),
        x_hat_cond[idx].cpu().numpy(),
    ]
    titles_reverse = [
        "Real",
        "z0 (clean)\nprojected",
        "50-step reverse\nUNCOND (projected)",
        "50-step reverse\nCOND (projected)",
        "Decoder\n(clean z0)",
        "Decoder\n(uncond z0)",
        "Decoder\n(cond z0)",
    ]
    render_skeleton_panels(out_dir / "04_conditioning_effect_full_reverse.png", seqs_reverse, titles_reverse)
    print("  -> 04_conditioning_effect_full_reverse.png")

    # ── Plot 5: Decoder layer trace — clean z0 ──
    seqs_trace_clean = [x_np] + [t[1][idx].cpu().numpy() for t in trace_clean]
    titles_trace_clean = ["Real"] + [t[0] for t in trace_clean]
    # Split into parts of 4 for readability
    chunk_size = 4
    for part_i in range(0, len(seqs_trace_clean), chunk_size):
        chunk_seqs = seqs_trace_clean[part_i:part_i + chunk_size]
        chunk_titles = titles_trace_clean[part_i:part_i + chunk_size]
        part_num = part_i // chunk_size + 1
        render_skeleton_panels(
            out_dir / f"05_decoder_trace_clean_z0_part{part_num}.png",
            chunk_seqs, chunk_titles,
        )
        print(f"  -> 05_decoder_trace_clean_z0_part{part_num}.png")

    # ── Plot 6: Decoder layer trace — diffused z0 (conditioned) ──
    seqs_trace_diff = [x_np] + [t[1][idx].cpu().numpy() for t in trace_diff]
    titles_trace_diff = ["Real"] + [t[0].replace("(projected)", "(diff, projected)") for t in trace_diff]
    for part_i in range(0, len(seqs_trace_diff), chunk_size):
        chunk_seqs = seqs_trace_diff[part_i:part_i + chunk_size]
        chunk_titles = titles_trace_diff[part_i:part_i + chunk_size]
        part_num = part_i // chunk_size + 1
        render_skeleton_panels(
            out_dir / f"06_decoder_trace_diffused_z0_part{part_num}.png",
            chunk_seqs, chunk_titles,
        )
        print(f"  -> 06_decoder_trace_diffused_z0_part{part_num}.png")

    # ── Plot 7: Side-by-side decoder trace comparison (clean vs diffused) ──
    # Show: Real, clean_input, diff_input, clean_after_g0, diff_after_g0, ..., clean_final, diff_final
    key_layers = [0, 3, 6, 7]  # input, after_graph_1+temporal_1, after_graph_2+temporal_2, final
    seqs_compare = [x_np]
    titles_compare = ["Real"]
    for li in key_layers:
        if li < len(trace_clean):
            seqs_compare.append(trace_clean[li][1][idx].cpu().numpy())
            titles_compare.append(f"clean\n{trace_clean[li][0]}")
        if li < len(trace_diff):
            seqs_compare.append(trace_diff[li][1][idx].cpu().numpy())
            titles_compare.append(f"diffused\n{trace_diff[li][0]}")
    render_skeleton_panels(out_dir / "07_decoder_trace_comparison.png", seqs_compare, titles_compare)
    print("  -> 07_decoder_trace_comparison.png")

    # ── Save summary ─────────────────────────────────────────────────────
    summary_path = out_dir / "trace_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
