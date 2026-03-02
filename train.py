"""Training entrypoint for 3-stage joint-aware latent diffusion."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from diffusion_model.dataset import create_dataloader, create_dataset
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import load_checkpoint, save_checkpoint
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TIMESTEPS,
    DEFAULT_WINDOW,
    set_seed,
)


LOGGER = logging.getLogger("train")


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _sync_mean(value: float, device: torch.device) -> float:
    """Average scalar across ranks when DDP is active."""
    if not _is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _iter_with_progress(loader: torch.utils.data.DataLoader, desc: str, enabled: bool = True):
    """Wrap loader with tqdm when available, otherwise return plain iterator."""
    if tqdm is None or not enabled:
        return loader
    return tqdm(loader, total=len(loader), desc=desc, leave=False)


def _log_step_progress(
    stage_name: str,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    metric_name: str,
    metric_value: float,
    epoch_start_time: float,
) -> None:
    """Print plain step progress for non-interactive environments (e.g. notebooks/nohup)."""
    if not _is_main_process():
        return
    elapsed = max(time.time() - epoch_start_time, 1e-6)
    steps_per_sec = step / elapsed
    eta_sec = max(0.0, (total_steps - step) / max(steps_per_sec, 1e-6))
    LOGGER.info(
        "[%s] epoch=%s/%s step=%s/%s %s=%.6f eta=%.1fs",
        stage_name,
        epoch,
        total_epochs,
        step,
        total_steps,
        metric_name,
        metric_value,
        eta_sec,
    )


def _setup_logging() -> None:
    """Configure timestamped logging format."""
    logging.basicConfig(
        level=logging.INFO if _is_main_process() else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _init_distributed(args: argparse.Namespace) -> tuple[bool, int, int]:
    """Single-GPU mode: reject multi-process launches and skip DDP init."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise ValueError(
            "This training script is configured for single-GPU mode. "
            "Run with `python train.py ...` (do not use torchrun --nproc_per_node>1)."
        )
    if args.ddp:
        print("[train] --ddp is ignored in single-GPU mode.")
    return False, 0, 1


def _resolve_data_mode(args: argparse.Namespace) -> str:
    """Return active input mode: torch-file or csv-folders."""
    if args.dataset_path:
        return "torch-file"
    if args.skeleton_folder and args.hip_folder and args.wrist_folder:
        return "csv-folders"
    raise ValueError(
        "Strict proposal mode requires either --dataset_path or all CSV folders: "
        "--skeleton_folder, --hip_folder, --wrist_folder."
    )


def _print_run_summary(args: argparse.Namespace, device: torch.device) -> None:
    """Print a compact, readable training run summary."""
    if not _is_main_process():
        return
    data_mode = _resolve_data_mode(args)
    LOGGER.info("Run dir: %s", args.save_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info(
        "Config: stage=%s epochs=%s batch_size=%s lr=%s window=%s stride=%s joints=%s latent_dim=%s timesteps=%s val_split=%s",
        args.stage,
        args.epochs,
        args.batch_size,
        args.lr,
        args.window,
        args.stride,
        args.joints,
        args.latent_dim,
        args.timesteps,
        args.val_split,
    )
    LOGGER.info("Data mode: %s", data_mode)
    if data_mode == "torch-file":
        LOGGER.info("dataset_path=%s", args.dataset_path)
    else:
        LOGGER.info("skeleton_folder=%s", args.skeleton_folder)
        LOGGER.info("hip_folder=%s", args.hip_folder)
        LOGGER.info("wrist_folder=%s", args.wrist_folder)


def _print_loader_summary(loader: torch.utils.data.DataLoader) -> None:
    """Print dataset and dataloader shape summary once at startup."""
    if not _is_main_process():
        return
    dataset = loader.dataset
    LOGGER.info(
        "Dataset: %s samples=%s batches_per_epoch=%s drop_last=%s",
        dataset.__class__.__name__,
        len(dataset),
        len(loader),
        loader.drop_last,
    )


def _split_train_val_dataset(dataset: Dataset, val_split: float, seed: int) -> Tuple[Dataset, Optional[Dataset]]:
    """Deterministically split dataset into train/val subsets."""
    if val_split <= 0.0:
        return dataset, None
    total = len(dataset)
    if total < 2:
        raise ValueError("Validation split requires at least 2 samples.")
    n_val = max(1, int(round(total * val_split)))
    n_val = min(n_val, total - 1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=g).tolist()
    train_idx = perm[:-n_val]
    val_idx = perm[-n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _build_train_val_loaders(
    args: argparse.Namespace,
    dataset: Dataset,
    distributed: bool,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """Create train/val dataloaders from one dataset using deterministic split."""
    train_dataset, val_dataset = _split_train_val_dataset(dataset, val_split=args.val_split, seed=args.seed)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
    train_loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        sampler=train_sampler,
        dataset=train_dataset,
        shuffle=not distributed,
        drop_last=True,
    )

    if val_dataset is None:
        return train_loader, None

    val_sampler = None
    if distributed:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )
    val_loader = create_dataloader(
        dataset_path=args.dataset_path or None,
        batch_size=args.batch_size,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        sampler=val_sampler,
        dataset=val_dataset,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def _print_epoch_summary(stage_name: str, epoch: int, total_epochs: int, metrics: Dict[str, float], sec: float) -> None:
    """Print one formatted epoch summary line."""
    if not _is_main_process():
        return
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
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel (or auto-enabled under torchrun).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--disable_sensor_norm", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data used for validation (0 disables validation).")
    parser.add_argument("--log_every", type=int, default=20, help="Plain progress print interval (steps) when tqdm is not interactive.")
    return parser.parse_args()


def train_stage1(args: argparse.Namespace, device: torch.device, distributed: bool, local_rank: int) -> None:
    """Train stage 1 model."""
    model = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    dataset = create_dataset(
        dataset_path=args.dataset_path or None,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
    )
    train_loader, val_loader = _build_train_val_loaders(args, dataset=dataset, distributed=distributed)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    _print_loader_summary(train_loader)
    if val_loader is not None and _is_main_process():
        LOGGER.info(
            "Validation: samples=%s batches=%s drop_last=%s",
            len(val_loader.dataset),
            len(val_loader),
            val_loader.drop_last,
        )
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_loss = 0.0
        n_batches = 0
        total_steps = len(train_loader)
        progress_enabled = _is_main_process() and sys.stdout.isatty()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        pbar = _iter_with_progress(train_loader, f"Stage1 Epoch {epoch + 1}/{args.epochs}", enabled=progress_enabled)
        for batch in pbar:
            x = batch["skeleton"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                loss = out["loss_diff"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sum_loss += float(loss.item())
            n_batches += 1
            if tqdm is not None and progress_enabled:
                pbar.set_postfix(loss=f"{(sum_loss / n_batches):.4f}")
            elif _is_main_process() and (
                n_batches == 1
                or n_batches % max(1, args.log_every) == 0
                or n_batches == total_steps
            ):
                _log_step_progress(
                    "Stage1",
                    epoch + 1,
                    args.epochs,
                    n_batches,
                    total_steps,
                    "loss",
                    sum_loss / n_batches,
                    t0,
                )
        avg_train_loss = sum_loss / max(n_batches, 1)
        avg_train_loss = _sync_mean(avg_train_loss, device)

        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["skeleton"].to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(x)
                        loss = out["loss_diff"]
                    val_sum += float(loss.item())
                    val_n += 1
            model.train()
            avg_val_loss = val_sum / max(val_n, 1)
            avg_val_loss = _sync_mean(avg_val_loss, device)

        metrics = {"train_loss_diff": avg_train_loss}
        if avg_val_loss is not None:
            metrics["val_loss_diff"] = avg_val_loss
        _print_epoch_summary("Stage1", epoch + 1, args.epochs, metrics, time.time() - t0)

        monitor_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
        if _is_main_process() and monitor_loss < best_loss:
            best_loss = monitor_loss
            best_ckpt = os.path.join(args.save_dir, "stage1_best.pt")
            save_checkpoint(
                best_ckpt,
                _unwrap_model(model),
                extra={
                    "best_monitor_loss": best_loss,
                    "best_monitor_name": "val_loss_diff" if avg_val_loss is not None else "train_loss_diff",
                    "best_epoch": epoch + 1,
                },
            )
            LOGGER.info("[Stage1] new best checkpoint: %s (epoch=%s monitor_loss=%.6f)", best_ckpt, epoch + 1, best_loss)

    if _is_main_process():
        ckpt = os.path.join(args.save_dir, "stage1.pt")
        save_checkpoint(ckpt, _unwrap_model(model))
        LOGGER.info("[Stage1] saved checkpoint: %s", ckpt)


def train_stage2(args: argparse.Namespace, device: torch.device, distributed: bool, local_rank: int) -> None:
    """Train stage 2 model with frozen stage-1 encoder."""
    if not args.stage1_ckpt:
        raise ValueError("Stage 2 requires --stage1_ckpt (pretrained Stage 1 encoder).")
    stage1 = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    if args.stage1_ckpt:
        load_checkpoint(args.stage1_ckpt, stage1, strict=True)

    model = Stage2Model(encoder=stage1.encoder, latent_dim=args.latent_dim, num_joints=args.joints).to(device)
    assert all(not p.requires_grad for p in model.encoder.parameters()), "stage2 encoder must be frozen"
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    dataset = create_dataset(
        dataset_path=args.dataset_path or None,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
    )
    train_loader, val_loader = _build_train_val_loaders(args, dataset=dataset, distributed=distributed)
    trainable_params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    model.train()
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    _print_loader_summary(train_loader)
    if val_loader is not None and _is_main_process():
        LOGGER.info(
            "Validation: samples=%s batches=%s drop_last=%s",
            len(val_loader.dataset),
            len(val_loader),
            val_loader.drop_last,
        )
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_loss = 0.0
        n_batches = 0
        total_steps = len(train_loader)
        progress_enabled = _is_main_process() and sys.stdout.isatty()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        pbar = _iter_with_progress(train_loader, f"Stage2 Epoch {epoch + 1}/{args.epochs}", enabled=progress_enabled)
        for batch in pbar:
            x = batch["skeleton"].to(device)
            a_hip_stream = batch["A_hip"].to(device)
            a_wrist_stream = batch["A_wrist"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x=x, a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)
                loss = out["loss_align"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sum_loss += float(loss.item())
            n_batches += 1
            if tqdm is not None and progress_enabled:
                pbar.set_postfix(loss=f"{(sum_loss / n_batches):.4f}")
            elif _is_main_process() and (
                n_batches == 1
                or n_batches % max(1, args.log_every) == 0
                or n_batches == total_steps
            ):
                _log_step_progress(
                    "Stage2",
                    epoch + 1,
                    args.epochs,
                    n_batches,
                    total_steps,
                    "loss",
                    sum_loss / n_batches,
                    t0,
                )
        avg_train_loss = sum_loss / max(n_batches, 1)
        avg_train_loss = _sync_mean(avg_train_loss, device)

        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["skeleton"].to(device)
                    a_hip_stream = batch["A_hip"].to(device)
                    a_wrist_stream = batch["A_wrist"].to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(x=x, a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)
                        loss = out["loss_align"]
                    val_sum += float(loss.item())
                    val_n += 1
            model.train()
            avg_val_loss = val_sum / max(val_n, 1)
            avg_val_loss = _sync_mean(avg_val_loss, device)

        metrics = {"train_loss_align": avg_train_loss}
        if avg_val_loss is not None:
            metrics["val_loss_align"] = avg_val_loss
        _print_epoch_summary("Stage2", epoch + 1, args.epochs, metrics, time.time() - t0)

        monitor_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
        if _is_main_process() and monitor_loss < best_loss:
            best_loss = monitor_loss
            best_ckpt = os.path.join(args.save_dir, "stage2_best.pt")
            save_checkpoint(
                best_ckpt,
                _unwrap_model(model),
                extra={
                    "best_monitor_loss": best_loss,
                    "best_monitor_name": "val_loss_align" if avg_val_loss is not None else "train_loss_align",
                    "best_epoch": epoch + 1,
                },
            )
            LOGGER.info("[Stage2] new best checkpoint: %s (epoch=%s monitor_loss=%.6f)", best_ckpt, epoch + 1, best_loss)

    if _is_main_process():
        ckpt = os.path.join(args.save_dir, "stage2.pt")
        save_checkpoint(ckpt, _unwrap_model(model))
        LOGGER.info("[Stage2] saved checkpoint: %s", ckpt)


def train_stage3(args: argparse.Namespace, device: torch.device, distributed: bool, local_rank: int) -> None:
    """Train stage 3 conditional diffusion + classification model."""
    if not args.stage1_ckpt:
        raise ValueError("Stage 3 requires --stage1_ckpt (pretrained Stage 1 model).")
    if not args.stage2_ckpt:
        raise ValueError("Stage 3 requires --stage2_ckpt (pretrained Stage 2 aligner).")
    stage1 = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    if args.stage1_ckpt:
        load_checkpoint(args.stage1_ckpt, stage1, strict=True)

    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=args.latent_dim, num_joints=args.joints).to(device)
    if args.stage2_ckpt:
        load_checkpoint(args.stage2_ckpt, stage2, strict=True)
    stage2.eval()

    model = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    trainable_params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    dataset = create_dataset(
        dataset_path=args.dataset_path or None,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        hip_folder=args.hip_folder or None,
        wrist_folder=args.wrist_folder or None,
        stride=args.stride,
        normalize_sensors=not args.disable_sensor_norm,
    )
    train_loader, val_loader = _build_train_val_loaders(args, dataset=dataset, distributed=distributed)
    model.train()
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    _print_loader_summary(train_loader)
    if val_loader is not None and _is_main_process():
        LOGGER.info(
            "Validation: samples=%s batches=%s drop_last=%s",
            len(val_loader.dataset),
            len(val_loader),
            val_loader.drop_last,
        )
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_total = 0.0
        sum_diff = 0.0
        sum_cls = 0.0
        n_batches = 0
        total_steps = len(train_loader)
        progress_enabled = _is_main_process() and sys.stdout.isatty()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        pbar = _iter_with_progress(train_loader, f"Stage3 Epoch {epoch + 1}/{args.epochs}", enabled=progress_enabled)
        for batch in pbar:
            x = batch["skeleton"].to(device)
            y = batch["label"].to(device)
            a_hip_stream = batch["A_hip"].to(device)
            a_wrist_stream = batch["A_wrist"].to(device)

            # Stage2 is a fixed conditioner in stage3; avoid autograd through aligner.
            with torch.no_grad():
                h_tokens, h_global = stage2.aligner(
                    a_hip_stream=a_hip_stream,
                    a_wrist_stream=a_wrist_stream,
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x=x, y=y, h_tokens=h_tokens, h_global=h_global)
                loss = out["loss_total"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sum_total += float(out["loss_total"].item())
            sum_diff += float(out["loss_diff"].item())
            sum_cls += float(out["loss_cls"].item())
            n_batches += 1
            if tqdm is not None and progress_enabled:
                pbar.set_postfix(loss=f"{(sum_total / n_batches):.4f}")
            elif _is_main_process() and (
                n_batches == 1
                or n_batches % max(1, args.log_every) == 0
                or n_batches == total_steps
            ):
                _log_step_progress(
                    "Stage3",
                    epoch + 1,
                    args.epochs,
                    n_batches,
                    total_steps,
                    "loss",
                    sum_total / n_batches,
                    t0,
                )
        avg_train_total = sum_total / max(n_batches, 1)
        avg_train_diff = sum_diff / max(n_batches, 1)
        avg_train_cls = sum_cls / max(n_batches, 1)
        avg_train_total = _sync_mean(avg_train_total, device)
        avg_train_diff = _sync_mean(avg_train_diff, device)
        avg_train_cls = _sync_mean(avg_train_cls, device)

        avg_val_total = None
        avg_val_diff = None
        avg_val_cls = None
        if val_loader is not None:
            model.eval()
            val_total_sum = 0.0
            val_diff_sum = 0.0
            val_cls_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["skeleton"].to(device)
                    y = batch["label"].to(device)
                    a_hip_stream = batch["A_hip"].to(device)
                    a_wrist_stream = batch["A_wrist"].to(device)
                    h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(x=x, y=y, h_tokens=h_tokens, h_global=h_global)
                    val_total_sum += float(out["loss_total"].item())
                    val_diff_sum += float(out["loss_diff"].item())
                    val_cls_sum += float(out["loss_cls"].item())
                    val_n += 1
            model.train()
            avg_val_total = _sync_mean(val_total_sum / max(val_n, 1), device)
            avg_val_diff = _sync_mean(val_diff_sum / max(val_n, 1), device)
            avg_val_cls = _sync_mean(val_cls_sum / max(val_n, 1), device)

        metrics = {
            "train_loss_total": avg_train_total,
            "train_loss_diff": avg_train_diff,
            "train_loss_cls": avg_train_cls,
        }
        if avg_val_total is not None and avg_val_diff is not None and avg_val_cls is not None:
            metrics["val_loss_total"] = avg_val_total
            metrics["val_loss_diff"] = avg_val_diff
            metrics["val_loss_cls"] = avg_val_cls
        _print_epoch_summary(
            "Stage3",
            epoch + 1,
            args.epochs,
            metrics,
            time.time() - t0,
        )
        monitor_loss = avg_val_total if avg_val_total is not None else avg_train_total
        if _is_main_process() and monitor_loss < best_loss:
            best_loss = monitor_loss
            best_ckpt = os.path.join(args.save_dir, "stage3_best.pt")
            save_checkpoint(
                best_ckpt,
                _unwrap_model(model),
                extra={
                    "best_monitor_loss": best_loss,
                    "best_monitor_name": "val_loss_total" if avg_val_total is not None else "train_loss_total",
                    "best_train_loss_total": avg_train_total,
                    "best_train_loss_diff": avg_train_diff,
                    "best_train_loss_cls": avg_train_cls,
                    "best_val_loss_total": avg_val_total,
                    "best_val_loss_diff": avg_val_diff,
                    "best_val_loss_cls": avg_val_cls,
                    "best_epoch": epoch + 1,
                },
            )
            LOGGER.info("[Stage3] new best checkpoint: %s (epoch=%s monitor_loss=%.6f)", best_ckpt, epoch + 1, best_loss)

    if _is_main_process():
        ckpt = os.path.join(args.save_dir, "stage3.pt")
        save_checkpoint(ckpt, _unwrap_model(model))
        LOGGER.info("[Stage3] saved checkpoint: %s", ckpt)


def main() -> None:
    """Run stage-specific training."""
    args = parse_args()
    if args.val_split < 0.0 or args.val_split >= 1.0:
        raise ValueError("--val_split must be in [0.0, 1.0).")
    distributed, local_rank, world_size = _init_distributed(args)
    _setup_logging()
    set_seed(args.seed)
    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    if _is_main_process():
        LOGGER.info("DDP enabled: %s (world_size=%s)", distributed, world_size)
    _print_run_summary(args, device)

    if args.stage == 1:
        train_stage1(args, device, distributed=distributed, local_rank=local_rank)
    elif args.stage == 2:
        train_stage2(args, device, distributed=distributed, local_rank=local_rank)
    elif args.stage == 3:
        train_stage3(args, device, distributed=distributed, local_rank=local_rank)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")
    if _is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
