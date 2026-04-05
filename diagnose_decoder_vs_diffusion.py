"""Diagnose whether Stage-3 errors come from the decoder or the diffusion path.

Three tests:
  A) decoder(encoder(x))                     — pure autoencoder, no diffusion
  B) decoder(z0 from reverse diffusion)     — full IMU-conditioned generation path
  C) Compare A vs B in both pose and latent space

Interpretation:
  - If A is bad: the decoder is the bottleneck.
  - If A is good but B is bad: reverse diffusion / conditioning is the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_model.dataset import create_dataset
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint
from diffusion_model.training_eval import render_skeleton_panels, sample_stage3_latents
from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_NUM_CLASSES, DEFAULT_TIMESTEPS, DEFAULT_WINDOW


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def _infer_latent_dim(stage1_ckpt: str) -> int:
    state_dict = _load_state_dict(stage1_ckpt)
    return int(state_dict["encoder.in_proj.weight"].shape[0])


def _infer_d_shared(stage2_ckpt: str, default: int = 64) -> int:
    state_dict = _load_state_dict(stage2_ckpt)
    key = "shared_motion_layer.net.0.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    return int(default)


def _load_stage2_compat(path: str, model: Stage2Model) -> None:
    # Older checkpoints can miss newer shared-motion normalization parameters.
    load_checkpoint(path, model, strict=False)


def _load_stage3_compat(path: str, model: Stage3Model) -> None:
    # Keep this tolerant for checkpoint/code drift while preserving the core weights.
    load_checkpoint(path, model, strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decoder vs Diffusion diagnosis")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="/home/qsw26/smartfall/gait_loss/filtered_skeleton_data")
    parser.add_argument("--hip_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/phone")
    parser.add_argument("--wrist_folder", type=str, default="/home/qsw26/smartfall/gait_loss/SmartFallMM-Dataset/young/accelerometer/watch")
    parser.add_argument("--output_dir", type=str, default="outputs/decoder_vs_diffusion")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--latent_dim", type=int, default=0, help="0 means infer from Stage-1 checkpoint.")
    parser.add_argument("--d_shared", type=int, default=0, help="0 means infer from Stage-2 checkpoint when possible.")
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=45)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--imu_graph", type=str, default="multiscale", choices=["chain", "multiscale"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_graph_op, skeleton_graph_op = infer_graph_ops_from_checkpoint(args.stage1_ckpt)
    latent_dim = args.latent_dim or _infer_latent_dim(args.stage1_ckpt)
    d_shared = args.d_shared or _infer_d_shared(args.stage2_ckpt)
    print(
        "Resolved config:",
        f"latent_dim={latent_dim}",
        f"d_shared={d_shared}",
        f"encoder_graph_op={encoder_graph_op}",
        f"skeleton_graph_op={skeleton_graph_op}",
        f"imu_graph={args.imu_graph}",
    )

    print("Loading Stage 1...")
    stage1 = Stage1Model(
        latent_dim=latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        num_classes=args.num_classes,
        encoder_type=encoder_graph_op,
        skeleton_graph_op=skeleton_graph_op,
    ).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=False)
    stage1.eval()

    print("Loading Stage 2...")
    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
        num_classes=args.num_classes,
        imu_graph_type=args.imu_graph,
        d_shared=d_shared,
    ).to(device)
    _load_stage2_compat(args.stage2_ckpt, stage2)
    stage2.eval()

    print("Loading Stage 3...")
    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=latent_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        d_shared=d_shared,
        shared_motion_layer=stage2.shared_motion_layer,
    ).to(device)
    _load_stage3_compat(args.stage3_ckpt, stage3)
    stage3.eval()

    print("Loading dataset...")
    dataset = create_dataset(
        dataset_path=args.dataset_path or None,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.num_samples, shuffle=True, drop_last=False)
    batch = next(iter(loader))

    x = batch["skeleton"].to(device)
    a_hip = batch["A_hip"].to(device)
    a_wrist = batch["A_wrist"].to(device)

    print(f"Skeleton shape: {tuple(x.shape)}, range=[{x.min().item():.4f}, {x.max().item():.4f}]")

    with torch.no_grad():
        z0_real = stage3.encoder(x, gait_metrics=None)
        x_hat_autoenc = stage3.decoder(z0_real)
        mpjpe_autoenc = torch.linalg.norm(x_hat_autoenc - x, dim=-1).mean()

        h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip, a_wrist_stream=a_wrist)
        z0_diffused = sample_stage3_latents(
            stage3=stage3,
            shape=z0_real.shape,
            device=device,
            h_tokens=h_tokens,
            h_global=h_global,
            a_hip_stream=a_hip,
            a_wrist_stream=a_wrist,
            gait_metrics=None,
            sample_steps=args.sample_steps,
            sampler=args.sampler,
            sample_seed=args.sample_seed,
        )
        x_hat_diffused = stage3.decoder(z0_diffused)
        mpjpe_diffused = torch.linalg.norm(x_hat_diffused - x, dim=-1).mean()

        z0_l2 = torch.linalg.norm(z0_diffused - z0_real, dim=-1).mean()
        z0_real_mean = z0_real.mean()
        z0_real_std = z0_real.std()
        z0_diff_mean = z0_diffused.mean()
        z0_diff_std = z0_diffused.std()

        per_sample_rows: list[dict[str, float]] = []
        for i in range(x.shape[0]):
            ae = torch.linalg.norm(x_hat_autoenc[i] - x[i], dim=-1).mean()
            diff = torch.linalg.norm(x_hat_diffused[i] - x[i], dim=-1).mean()
            per_sample_rows.append(
                {
                    "sample_index": int(i),
                    "autoencoder_mpjpe_m": float(ae.item()),
                    "diffusion_mpjpe_m": float(diff.item()),
                }
            )

    print("\n" + "=" * 60)
    print("DECODER vs DIFFUSION DIAGNOSIS")
    print("=" * 60)
    print(f"\n[A] Autoencoder (encoder→decoder, no diffusion):")
    print(f"    MPJPE = {mpjpe_autoenc.item():.4f} m  ({mpjpe_autoenc.item() * 100:.1f} cm)")
    print(f"\n[B] Full pipeline (diffusion reverse → decoder):")
    print(f"    MPJPE = {mpjpe_diffused.item():.4f} m  ({mpjpe_diffused.item() * 100:.1f} cm)")
    print(f"\n[C] Latent space comparison:")
    print(f"    z0_real  — mean={z0_real_mean.item():.4f}, std={z0_real_std.item():.4f}")
    print(f"    z0_diff  — mean={z0_diff_mean.item():.4f}, std={z0_diff_std.item():.4f}")
    print(f"    L2(z0_diff, z0_real) = {z0_l2.item():.4f}")

    verdict: str
    if mpjpe_autoenc.item() > 0.3:
        verdict = "Decoder is broken — it cannot reconstruct even from clean encoder latents."
    elif mpjpe_diffused.item() > 2 * mpjpe_autoenc.item():
        verdict = "Decoder works, but diffusion produces poor latents."
    else:
        verdict = "Both paths have similar error — decoder is likely the limiting factor."
    print(f"\nVERDICT: {verdict}")
    print("=" * 60)

    summary = {
        "stage1_ckpt": args.stage1_ckpt,
        "stage2_ckpt": args.stage2_ckpt,
        "stage3_ckpt": args.stage3_ckpt,
        "latent_dim": latent_dim,
        "d_shared": d_shared,
        "sample_steps": args.sample_steps,
        "sampler": args.sampler,
        "sample_seed": args.sample_seed,
        "autoencoder_mpjpe_m": float(mpjpe_autoenc.item()),
        "diffusion_mpjpe_m": float(mpjpe_diffused.item()),
        "latent_l2_mean": float(z0_l2.item()),
        "z0_real_mean": float(z0_real_mean.item()),
        "z0_real_std": float(z0_real_std.item()),
        "z0_diff_mean": float(z0_diff_mean.item()),
        "z0_diff_std": float(z0_diff_std.item()),
        "verdict": verdict,
        "per_sample": per_sample_rows,
    }
    summary_path = out_dir / "diagnosis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    for i in range(min(args.num_samples, x.shape[0], 4)):
        panel_path = out_dir / f"comparison_sample_{i}.png"
        render_skeleton_panels(
            panel_path,
            [x[i].cpu().numpy(), x_hat_autoenc[i].cpu().numpy(), x_hat_diffused[i].cpu().numpy()],
            ["Real", "Autoencoder\n(enc→dec)", "Full Pipeline\n(diffusion→dec)"],
        )
        print(f"Saved: {panel_path}")

    print("\nPer-sample MPJPE:")
    for row in per_sample_rows:
        ae = row["autoencoder_mpjpe_m"]
        diff = row["diffusion_mpjpe_m"]
        print(
            f"  Sample {row['sample_index']}: "
            f"autoencoder={ae:.4f}m ({ae * 100:.1f}cm)  "
            f"diffusion={diff:.4f}m ({diff * 100:.1f}cm)"
        )


if __name__ == "__main__":
    main()
