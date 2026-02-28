"""Conditional generation script for stage-3 latent diffusion."""

from __future__ import annotations

import argparse

import torch

from diffusion_model.dataset import create_dataloader
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate skeletons with conditional latent diffusion")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--h_none", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run conditional sampling and optional classification."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)

    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=args.latent_dim, num_joints=args.joints).to(device)
    load_checkpoint(args.stage2_ckpt, stage2, strict=False)

    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
    ).to(device)

    loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        synthetic_length=args.batch_size,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        shuffle=False,
    )

    batch = next(iter(loader))
    skeleton = batch["skeleton"].to(device)
    hip = batch["hip"].to(device)
    wrist = batch["wrist"].to(device)

    if args.h_none:
        h = None
    else:
        h = stage2.aligner(hip, wrist)

    shape = (skeleton.shape[0], skeleton.shape[1], skeleton.shape[2], args.latent_dim)
    z0_gen = stage3.diffusion.p_sample_loop(stage3.denoiser, shape=torch.Size(shape), device=device, h=h)
    x_hat = stage3.decoder(z0_gen)

    print(f"h shape: {None if h is None else tuple(h.shape)}")
    print(f"z0_gen shape: {tuple(z0_gen.shape)}")
    print(f"x_hat shape: {tuple(x_hat.shape)}")

    if args.classify:
        logits = stage3.classifier(x_hat)
        print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
