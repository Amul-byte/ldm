"""Training entrypoint for 3-stage joint-aware latent diffusion."""

from __future__ import annotations

import argparse
import os

import torch
import torch.optim as optim

from diffusion_model.dataset import create_dataloader
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import load_checkpoint, save_checkpoint
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LAMBDA_CLS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train 3-stage joint-aware latent diffusion")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--lambda_cls", type=float, default=DEFAULT_LAMBDA_CLS)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--synthetic_length", type=int, default=32)
    parser.add_argument("--use_h_none", action="store_true")
    return parser.parse_args()


def train_stage1(args: argparse.Namespace, device: torch.device) -> None:
    """Train stage 1 model."""
    model = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        synthetic_length=args.synthetic_length,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        for batch in loader:
            x = batch["skeleton"].to(device)
            out = model(x)
            loss = out["loss_diff"] + out["loss_recon"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[Stage1] epoch={epoch + 1} loss_diff={out['loss_diff'].item():.6f} loss_recon={out['loss_recon'].item():.6f}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_checkpoint(os.path.join(args.save_dir, "stage1.pt"), model)


def train_stage2(args: argparse.Namespace, device: torch.device) -> None:
    """Train stage 2 model with frozen stage-1 encoder."""
    stage1 = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    if args.stage1_ckpt:
        load_checkpoint(args.stage1_ckpt, stage1, strict=False)

    model = Stage2Model(encoder=stage1.encoder, latent_dim=args.latent_dim, num_joints=args.joints).to(device)
    assert all(not p.requires_grad for p in model.encoder.parameters()), "stage2 encoder must be frozen"

    loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        synthetic_length=args.synthetic_length,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
    )
    optimizer = optim.Adam(model.aligner.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        for batch in loader:
            x = batch["skeleton"].to(device)
            s_hip = batch["hip"].to(device)
            s_wrist = batch["wrist"].to(device)
            out = model(x=x, s_hip=s_hip, s_wrist=s_wrist)
            loss = out["loss_align"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[Stage2] epoch={epoch + 1} loss_align={out['loss_align'].item():.6f}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_checkpoint(os.path.join(args.save_dir, "stage2.pt"), model)


def train_stage3(args: argparse.Namespace, device: torch.device) -> None:
    """Train stage 3 conditional diffusion + classification model."""
    stage1 = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    if args.stage1_ckpt:
        load_checkpoint(args.stage1_ckpt, stage1, strict=False)

    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=args.latent_dim, num_joints=args.joints).to(device)
    if args.stage2_ckpt:
        load_checkpoint(args.stage2_ckpt, stage2, strict=False)

    model = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        lambda_cls=args.lambda_cls,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        synthetic_length=args.synthetic_length,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
    )
    model.train()

    for epoch in range(args.epochs):
        for batch in loader:
            x = batch["skeleton"].to(device)
            y = batch["label"].to(device)
            s_hip = batch["hip"].to(device)
            s_wrist = batch["wrist"].to(device)
            h = None if args.use_h_none else stage2.aligner(s_hip=s_hip, s_wrist=s_wrist)
            out = model(x=x, y=y, h=h)
            loss = out["loss_total"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(
            f"[Stage3] epoch={epoch + 1} loss_total={out['loss_total'].item():.6f} "
            f"loss_diff={out['loss_diff'].item():.6f} loss_cls={out['loss_cls'].item():.6f}"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    save_checkpoint(os.path.join(args.save_dir, "stage3.pt"), model)


def main() -> None:
    """Run stage-specific training."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Synthetic fallback active: {not bool(args.dataset_path)}")

    if args.stage == 1:
        train_stage1(args, device)
    elif args.stage == 2:
        train_stage2(args, device)
    elif args.stage == 3:
        train_stage3(args, device)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
