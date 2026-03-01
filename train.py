"""Training entrypoint for 3-stage joint-aware latent diffusion."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
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
    DEFAULT_LAMBDA_CLS,
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
    """Return active input mode: torch-file, csv-folders, or synthetic."""
    if args.dataset_path:
        return "torch-file"
    if args.skeleton_folder and args.hip_folder and args.wrist_folder:
        return "csv-folders"
    return "synthetic"


def _print_run_summary(args: argparse.Namespace, device: torch.device) -> None:
    """Print a compact, readable training run summary."""
    if not _is_main_process():
        return
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
    parser.add_argument("--lambda_cls", type=float, default=DEFAULT_LAMBDA_CLS)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--disable_sensor_norm", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--log_every", type=int, default=20, help="Plain progress print interval (steps) when tqdm is not interactive.")
    parser.add_argument("--synthetic_length", type=int, default=32)
    parser.add_argument("--use_h_none", action="store_true")
    return parser.parse_args()


def train_stage1(args: argparse.Namespace, device: torch.device, distributed: bool, local_rank: int) -> None:
    """Train stage 1 model."""
    model = Stage1Model(latent_dim=args.latent_dim, num_joints=args.joints, timesteps=args.timesteps).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    dataset = create_dataset(
        dataset_path=args.dataset_path or None,
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
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
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
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        sampler=sampler,
        dataset=dataset,
        shuffle=not distributed,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    _print_loader_summary(loader)
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_loss = 0.0
        n_batches = 0
        total_steps = len(loader)
        progress_enabled = _is_main_process() and sys.stdout.isatty()
        if distributed and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        pbar = _iter_with_progress(loader, f"Stage1 Epoch {epoch + 1}/{args.epochs}", enabled=progress_enabled)
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
        avg_loss = sum_loss / max(n_batches, 1)
        avg_loss = _sync_mean(avg_loss, device)
        _print_epoch_summary("Stage1", epoch + 1, args.epochs, {"loss_diff": avg_loss}, time.time() - t0)
        if _is_main_process() and avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(args.save_dir, "stage1_best.pt")
            save_checkpoint(best_ckpt, _unwrap_model(model), extra={"best_loss_diff": best_loss, "best_epoch": epoch + 1})
            LOGGER.info("[Stage1] new best checkpoint: %s (epoch=%s loss_diff=%.6f)", best_ckpt, epoch + 1, best_loss)

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
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )

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
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        sampler=sampler,
        dataset=dataset,
        shuffle=not distributed,
    )
    optimizer = optim.Adam(_unwrap_model(model).aligner.parameters(), lr=args.lr)
    model.train()
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    _print_loader_summary(loader)
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_loss = 0.0
        n_batches = 0
        total_steps = len(loader)
        progress_enabled = _is_main_process() and sys.stdout.isatty()
        if distributed and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        pbar = _iter_with_progress(loader, f"Stage2 Epoch {epoch + 1}/{args.epochs}", enabled=progress_enabled)
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
        avg_loss = sum_loss / max(n_batches, 1)
        avg_loss = _sync_mean(avg_loss, device)
        _print_epoch_summary("Stage2", epoch + 1, args.epochs, {"loss_align": avg_loss}, time.time() - t0)
        if _is_main_process() and avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(args.save_dir, "stage2_best.pt")
            save_checkpoint(best_ckpt, _unwrap_model(model), extra={"best_loss_align": best_loss, "best_epoch": epoch + 1})
            LOGGER.info("[Stage2] new best checkpoint: %s (epoch=%s loss_align=%.6f)", best_ckpt, epoch + 1, best_loss)

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
    if args.use_h_none:
        raise ValueError("Strict proposal behavior requires sensor conditioning in Stage 3. Remove --use_h_none.")
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
        lambda_cls=args.lambda_cls,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    trainable_params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    dataset = create_dataset(
        dataset_path=args.dataset_path or None,
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
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
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
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        sampler=sampler,
        dataset=dataset,
        shuffle=not distributed,
    )
    model.train()
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    _print_loader_summary(loader)
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_total = 0.0
        sum_diff = 0.0
        sum_cls = 0.0
        n_batches = 0
        total_steps = len(loader)
        progress_enabled = _is_main_process() and sys.stdout.isatty()
        if distributed and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        pbar = _iter_with_progress(loader, f"Stage3 Epoch {epoch + 1}/{args.epochs}", enabled=progress_enabled)
        for batch in pbar:
            x = batch["skeleton"].to(device)
            y = batch["label"].to(device)
            a_hip_stream = batch["A_hip"].to(device)
            a_wrist_stream = batch["A_wrist"].to(device)

            # Stage2 is a fixed conditioner in stage3; avoid autograd through aligner.
            with torch.no_grad():
                h_global, sensor_tokens = stage2.aligner(
                    a_hip_stream=a_hip_stream,
                    a_wrist_stream=a_wrist_stream,
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x=x, y=y, sensor_tokens=sensor_tokens, h_global=h_global)
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
        avg_total = sum_total / max(n_batches, 1)
        avg_diff = sum_diff / max(n_batches, 1)
        avg_cls = sum_cls / max(n_batches, 1)
        avg_total = _sync_mean(avg_total, device)
        avg_diff = _sync_mean(avg_diff, device)
        avg_cls = _sync_mean(avg_cls, device)
        _print_epoch_summary(
            "Stage3",
            epoch + 1,
            args.epochs,
            {
                "loss_total": avg_total,
                "loss_diff": avg_diff,
                "loss_cls": avg_cls,
            },
            time.time() - t0,
        )
        if _is_main_process() and avg_total < best_loss:
            best_loss = avg_total
            best_ckpt = os.path.join(args.save_dir, "stage3_best.pt")
            save_checkpoint(
                best_ckpt,
                _unwrap_model(model),
                extra={
                    "best_loss_total": best_loss,
                    "best_loss_diff": avg_diff,
                    "best_loss_cls": avg_cls,
                    "best_epoch": epoch + 1,
                },
            )
            LOGGER.info("[Stage3] new best checkpoint: %s (epoch=%s loss_total=%.6f)", best_ckpt, epoch + 1, best_loss)

    if _is_main_process():
        ckpt = os.path.join(args.save_dir, "stage3.pt")
        save_checkpoint(ckpt, _unwrap_model(model))
        LOGGER.info("[Stage3] saved checkpoint: %s", ckpt)


def main() -> None:
    """Run stage-specific training."""
    args = parse_args()
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
