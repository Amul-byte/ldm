"""Training entrypoint for 3-stage joint-aware latent diffusion."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict

import torch
import torch.optim as optim
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

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


LOGGER = logging.getLogger("train")


def _iter_with_progress(loader: torch.utils.data.DataLoader, desc: str):
    """Wrap loader with tqdm when available, otherwise return plain iterator."""
    if tqdm is None:
        return loader
    return tqdm(loader, total=len(loader), desc=desc, leave=False)


def _setup_logging() -> None:
    """Configure timestamped logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_data_mode(args: argparse.Namespace) -> str:
    """Return active input mode: torch-file, csv-folders, or synthetic."""
    if args.dataset_path:
        return "torch-file"
    if args.skeleton_folder and args.hip_folder and args.wrist_folder:
        return "csv-folders"
    return "synthetic"


def _print_run_summary(args: argparse.Namespace, device: torch.device) -> None:
    """Print a compact, readable training run summary."""
    data_mode = _resolve_data_mode(args)
    LOGGER.info("Run dir: %s", args.save_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info(
        "Config: stage=%s epochs=%s batch_size=%s lr=%s window=%s stride=%s joints=%s latent_dim=%s timesteps=%s",
        args.stage,
        args.epochs,
        args.batch_size,
        args.lr,
        args.window,
        args.stride,
        args.joints,
        args.latent_dim,
        args.timesteps,
    )
    LOGGER.info("Data mode: %s", data_mode)
    if data_mode == "torch-file":
        LOGGER.info("dataset_path=%s", args.dataset_path)
    elif data_mode == "csv-folders":
        LOGGER.info("skeleton_folder=%s", args.skeleton_folder)
        LOGGER.info("hip_folder=%s", args.hip_folder)
        LOGGER.info("wrist_folder=%s", args.wrist_folder)
    else:
        LOGGER.info("synthetic_length=%s", args.synthetic_length)


def _print_loader_summary(loader: torch.utils.data.DataLoader) -> None:
    """Print dataset and dataloader shape summary once at startup."""
    dataset = loader.dataset
    LOGGER.info(
        "Dataset: %s samples=%s batches_per_epoch=%s drop_last=%s",
        dataset.__class__.__name__,
        len(dataset),
        len(loader),
        loader.drop_last,
    )


def _print_epoch_summary(stage_name: str, epoch: int, total_epochs: int, metrics: Dict[str, float], sec: float) -> None:
    """Print one formatted epoch summary line."""
    items = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
    LOGGER.info("[%s] epoch=%s/%s %s epoch_time=%.1fs", stage_name, epoch, total_epochs, items, sec)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train 3-stage joint-aware latent diffusion")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--skeleton_folder", type=str, default="")
    parser.add_argument("--hip_folder", type=str, default="")
    parser.add_argument("--wrist_folder", type=str, default="")
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
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--disable_sensor_norm", action="store_true")
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
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    _print_loader_summary(loader)

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_loss = 0.0
        n_batches = 0
        pbar = _iter_with_progress(loader, f"Stage1 Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            x = batch["skeleton"].to(device)
            out = model(x)
            loss = out["loss_diff"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item())
            n_batches += 1
            if tqdm is not None:
                pbar.set_postfix(loss=f"{(sum_loss / n_batches):.4f}")
        avg_loss = sum_loss / max(n_batches, 1)
        _print_epoch_summary("Stage1", epoch + 1, args.epochs, {"loss_diff": avg_loss}, time.time() - t0)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, "stage1.pt")
    save_checkpoint(ckpt, model)
    LOGGER.info("[Stage1] saved checkpoint: %s", ckpt)


def train_stage2(args: argparse.Namespace, device: torch.device) -> None:
    """Train stage 2 model with frozen stage-1 encoder."""
    stage1 = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    if args.stage1_ckpt:
        load_checkpoint(args.stage1_ckpt, stage1, strict=True)

    model = Stage2Model(encoder=stage1.encoder, latent_dim=args.latent_dim, num_joints=args.joints).to(device)
    assert all(not p.requires_grad for p in model.encoder.parameters()), "stage2 encoder must be frozen"

    loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        synthetic_length=args.synthetic_length,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
    )
    optimizer = optim.Adam(model.aligner.parameters(), lr=args.lr)
    model.train()
    _print_loader_summary(loader)

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_loss = 0.0
        n_batches = 0
        pbar = _iter_with_progress(loader, f"Stage2 Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            x = batch["skeleton"].to(device)
            a_hip_stream = batch["A_hip"].to(device)
            a_wrist_stream = batch["A_wrist"].to(device)
            out = model(x=x, a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)
            loss = out["loss_align"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item())
            n_batches += 1
            if tqdm is not None:
                pbar.set_postfix(loss=f"{(sum_loss / n_batches):.4f}")
        avg_loss = sum_loss / max(n_batches, 1)
        _print_epoch_summary("Stage2", epoch + 1, args.epochs, {"loss_align": avg_loss}, time.time() - t0)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, "stage2.pt")
    save_checkpoint(ckpt, model)
    LOGGER.info("[Stage2] saved checkpoint: %s", ckpt)


def train_stage3(args: argparse.Namespace, device: torch.device) -> None:
    """Train stage 3 conditional diffusion + classification model."""
    stage1 = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    if args.stage1_ckpt:
        load_checkpoint(args.stage1_ckpt, stage1, strict=True)

    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=args.latent_dim, num_joints=args.joints).to(device)
    if args.stage2_ckpt:
        load_checkpoint(args.stage2_ckpt, stage2, strict=True)

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
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
    )
    model.train()
    _print_loader_summary(loader)

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_total = 0.0
        sum_diff = 0.0
        sum_cls = 0.0
        n_batches = 0
        pbar = _iter_with_progress(loader, f"Stage3 Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            x = batch["skeleton"].to(device)
            y = batch["label"].to(device)
            a_hip_stream = batch["A_hip"].to(device)
            a_wrist_stream = batch["A_wrist"].to(device)

            if args.use_h_none:
                h_global = None
                sensor_tokens = None
            else:
                h_global, sensor_tokens = stage2.aligner(
                    a_hip_stream=a_hip_stream,
                    a_wrist_stream=a_wrist_stream,
                )

            out = model(x=x, y=y, sensor_tokens=sensor_tokens, h_global=h_global)
            loss = out["loss_total"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_total += float(out["loss_total"].item())
            sum_diff += float(out["loss_diff"].item())
            sum_cls += float(out["loss_cls"].item())
            n_batches += 1
            if tqdm is not None:
                pbar.set_postfix(loss=f"{(sum_total / n_batches):.4f}")
        _print_epoch_summary(
            "Stage3",
            epoch + 1,
            args.epochs,
            {
                "loss_total": sum_total / max(n_batches, 1),
                "loss_diff": sum_diff / max(n_batches, 1),
                "loss_cls": sum_cls / max(n_batches, 1),
            },
            time.time() - t0,
        )

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, "stage3.pt")
    save_checkpoint(ckpt, model)
    LOGGER.info("[Stage3] saved checkpoint: %s", ckpt)


def main() -> None:
    """Run stage-specific training."""
    _setup_logging()
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print_run_summary(args, device)

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
