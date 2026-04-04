"""Training entrypoint for 3-stage joint-aware latent diffusion."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from diffusion_model.dataset import create_dataloader, create_dataset, parse_subject_list, split_train_val_dataset
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM, GAIT_METRIC_NAMES
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import infer_graph_ops_from_checkpoint, load_checkpoint, save_checkpoint
from diffusion_model.sensor_model import IMU_FEATURE_NAMES
from diffusion_model.training_eval import (
    evaluate_stage1,
    evaluate_stage2,
    evaluate_stage2_reports,
    evaluate_stage3,
    save_run_manifest,
    write_history,
)
from diffusion_model.util import DEFAULT_FPS, DEFAULT_JOINTS, DEFAULT_LATENT_DIM, DEFAULT_NUM_CLASSES, DEFAULT_TIMESTEPS, DEFAULT_WINDOW, set_seed


LOGGER = logging.getLogger("train")
DEFAULT_TRAIN_SUBJECTS = [
    28, 29, 30, 31, 33, 35, 38, 39, 32, 36, 37, 43, 44, 45, 46, 49, 51, 56, 57, 58, 59, 61, 62
]


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
    ''''Suppose GPU 0 gets train loss 0.9 and GPU 1 gets 1.1.
    Then this function averages them to 1.0.'''
    if not _is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _sync_sum(value: float, device: torch.device) -> float:
    if not _is_distributed():
        return float(value)
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _sync_ratio(numerator: float, denominator: float, device: torch.device) -> float:
    total_num = _sync_sum(numerator, device)
    total_den = _sync_sum(denominator, device)
    return total_num / max(total_den, 1.0)


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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise ValueError("DDP requires CUDA/NCCL in this script.")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        torch.cuda.set_device(local_rank)

        if args.ddp or world_size > 1:
            print(f"[train] DDP initialized rank={rank} local_rank={local_rank} world_size={world_size}")

        return True, local_rank, world_size

    if args.ddp:
        print("[train] --ddp was passed, but WORLD_SIZE=1 so DDP is disabled.")
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
        LOGGER.info("gait_cache_dir=%s", args.gait_cache_dir)
    LOGGER.info("Gait metrics dim: %s", args.gait_metrics_dim)
    LOGGER.info("Gait metric names: %s", ", ".join(GAIT_METRIC_NAMES))
    LOGGER.info("IMU feature names: %s", ", ".join(IMU_FEATURE_NAMES))
    LOGGER.info(
        "Skeleton graph ops: encoder=%s full_skeleton=%s",
        getattr(args, "encoder_graph_op_resolved", args.encoder_type or "gat"),
        getattr(args, "skeleton_graph_op_resolved", args.skeleton_graph_op or "gat"),
    )
    LOGGER.info("One-to-one mode: %s", args.one_to_one)
    if args.stage == 3:
        LOGGER.info(
            "Stage3 objective: lambda_pose=%s lambda_latent=%s lambda_vel=%s lambda_gait=%s lambda_motion=%s fps=%s sample_steps=%s d_shared=%s",
            args.lambda_pose,
            args.lambda_latent,
            args.lambda_vel,
            args.lambda_motion,
            args.lambda_gait,
            args.fps,
            args.sample_steps,
            args.d_shared,
        )
    if args.stage == 2:
        LOGGER.info(
            "Stage2 objective: lambda_feature=%s lambda_align=%s lambda_cls=%s lambda_gait_pred=%s d_shared=%s",
            args.lambda_feature,
            args.lambda_align,
            args.lambda_cls_s2,
            args.lambda_gait_s2,
            args.d_shared,
        )


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


def _make_run_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or f"stage{args.stage}_{stamp}"
    run_dir = Path(args.report_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _normalize_graph_op_arg(value: str | None, default: str | None = None) -> str:
    graph_op = default if value in {None, ""} else str(value).lower()
    if graph_op not in {"gat", "gcn"}:
        raise ValueError(f"Unsupported graph op: {value}")
    return graph_op


def _resolve_graph_ops(args: argparse.Namespace, stage1_ckpt: str = "") -> tuple[str, str]:
    ckpt_encoder = ckpt_skeleton = None
    if stage1_ckpt:
        ckpt_encoder, ckpt_skeleton = infer_graph_ops_from_checkpoint(stage1_ckpt)
    skeleton_graph_op = _normalize_graph_op_arg(args.skeleton_graph_op, default=ckpt_skeleton or "gat")
    encoder_graph_op = _normalize_graph_op_arg(args.encoder_type, default=ckpt_encoder or skeleton_graph_op)
    return encoder_graph_op, skeleton_graph_op


def _build_train_val_loaders(
    args: argparse.Namespace,
    dataset: Dataset,
    distributed: bool,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """Create train/val dataloaders from one dataset using deterministic split."""
    train_subjects = parse_subject_list(args.train_subjects) if args.train_subjects else None
    train_dataset, val_dataset = split_train_val_dataset(
        dataset,
        val_split=args.val_split,
        seed=args.seed,
        train_subjects=train_subjects,
        logger=LOGGER if _is_main_process() else None,
    )

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
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
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
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
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
    parser.add_argument("--gait_cache_dir", type=str, default="")
    parser.add_argument("--disable_gait_cache", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel (or auto-enabled under torchrun).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--report_dir", type=str, default="outputs/training_reports")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument(
        "--gait_metrics_dim",
        type=int,
        default=DEFAULT_GAIT_METRICS_DIM,
        help="Number of scalar gait metrics provided per sample. Defaults to the fixed auto-computed gait-summary size.",
    )
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--stride", type=int, default=45)
    parser.add_argument("--disable_sensor_norm", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--one_to_one", dest="one_to_one", action="store_true", help="Enable one-to-one IMU->skeleton mode.")
    parser.add_argument("--no_one_to_one", dest="one_to_one", action="store_false", help="Use legacy conditioning behavior.")
    parser.set_defaults(one_to_one=True)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data used for validation (0 disables validation).")
    parser.add_argument(
        "--train_subjects",
        type=str,
        default="",
        help="Comma-separated subject ids used for training; all other subjects go to validation. Leave empty to use random val_split instead.",
    )
    parser.add_argument("--log_every", type=int, default=20, help="Plain progress print interval (steps) when tqdm is not interactive.")
    parser.add_argument("--skeleton_graph_op", choices=["gat", "gcn"], default="gcn",
                        help="Skeleton graph family for decoder/denoiser and the default encoder choice. Leave empty to use checkpoint/default behavior.")
    parser.add_argument("--encoder_type", choices=["gat", "gcn"], default="gcn",
                        help="Deprecated encoder-only override. Use with --skeleton_graph_op gat to reproduce the old mixed GCN-encoder/GAT-decoder-denoiser ablation.")
    parser.add_argument("--imu_graph", choices=["chain", "multiscale"], default="multiscale",
                        help="IMU temporal graph type for Stage-2 encoder: 'chain' (j→j+1 only, simpler) or 'multiscale' (gaps 1,5,15,30, wider receptive field).")
    parser.add_argument("--lambda_cls", type=float, default=0.01, help="Weight for classification loss (Stage-3 only).")
    parser.add_argument("--lambda_cls_s1", type=float, default=0.1, help="Weight for classification loss in Stage-1 (discriminative latent training).")
    parser.add_argument("--lambda_var", type=float, default=0.01, help="Weight for variance regulariser in Stage-1 (prevents embedding collapse).")
    parser.add_argument("--lambda_align", type=float, default=0.1, help="Weight for pooled IMU-vs-skeleton latent MSE in Stage-2 (weak auxiliary regulariser).")
    parser.add_argument("--lambda_cls_s2", type=float, default=1, help="Weight for CE classification loss in Stage-2.")
    parser.add_argument("--lambda_gait_s2", type=float, default=1, help="Weight for gait-metric prediction loss in Stage-2 (primary supervised signal).")
    parser.add_argument("--lambda_feature", type=float, default=1.0, help="Weight for shared-motion feature alignment loss in Stage-2.")
    parser.add_argument("--lambda_pose", type=float, default=1.0, help="Weight for paired skeleton reconstruction loss (Stage-3 only).")
    parser.add_argument("--lambda_latent", type=float, default=0.5, help="Weight for latent reconstruction loss (Stage-3 only).")
    parser.add_argument("--lambda_vel", type=float, default=0.5, help="Weight for velocity reconstruction loss (Stage-3 only).")
    parser.add_argument("--lambda_motion", type=float, default=1.0, help="Weight for motion regularization loss (Stage-3 only).")
    parser.add_argument("--lambda-gait", type=float, default=1.0, help="Weight for generated-vs-real gait-summary MSE loss (Stage-3 only).")
    parser.add_argument("--d_shared", type=int, default=64, help="Hidden dimension of the shared IMU/skeleton motion projection layer.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Skeleton frame-rate for temporal losses (Stage-3 only).")
    parser.add_argument("--sample_steps", type=int, default=50, help="DDIM sampling steps for evaluation/generation.")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddim", "ddpm"], help="Reverse-process sampler for evaluation/generation.")
    parser.add_argument("--sample_seed", type=int, default=0, help="Deterministic sampling seed used for evaluation/generation.")
    parser.add_argument("--eval_every_stage1", type=int, default=5, help="Epoch interval for Stage-1 evaluation artifacts.")
    parser.add_argument("--eval_every_stage2", type=int, default=500, help="Epoch interval for Stage-2 evaluation artifacts.")
    parser.add_argument("--eval_every_stage2_reports", type=int, default=10, help="Epoch interval for lightweight Stage-2 classifier/gait reports.")
    parser.add_argument("--eval_every_stage3", type=int, default=5, help="Epoch interval for Stage-3 evaluation artifacts.")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Cap batches per training epoch for smoke runs (0 disables).")
    parser.add_argument("--max_val_batches", type=int, default=0, help="Cap batches per validation epoch for smoke runs (0 disables).")
    return parser.parse_args()


def train_stage1(args: argparse.Namespace, device: torch.device, distributed: bool, local_rank: int) -> None:
    """Train stage 1 model."""
    model = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        num_classes=args.num_classes,
        encoder_type=args.encoder_graph_op_resolved,
        skeleton_graph_op=args.skeleton_graph_op_resolved,
    ).to(device)
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
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
    )
    train_loader, val_loader = _build_train_val_loaders(args, dataset=dataset, distributed=distributed)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    if _is_main_process():
        save_run_manifest(
            Path(args.run_dir),
            args,
            device,
            runtime={
                "optimizer": optimizer.__class__.__name__,
                "scheduler": "none",
                "sensor_modality": "accelerometer only",
                "sensor_locations": [Path(args.hip_folder).name if args.hip_folder else "", Path(args.wrist_folder).name if args.wrist_folder else ""],
            },
        )
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
    history: list[dict[str, float]] = []
    run_dir = Path(args.run_dir)

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_loss = 0.0
        sum_diff = 0.0
        sum_cls  = 0.0
        sum_var  = 0.0
        sum_cls_correct = 0.0
        sum_cls_count = 0.0
        n_batches = 0
        total_steps = len(train_loader)
        progress_enabled = _is_main_process() and sys.stdout.isatty()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        pbar = _iter_with_progress(train_loader, f"Stage1 Epoch {epoch + 1}/{args.epochs}", enabled=progress_enabled)
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            x = batch["skeleton"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x, gait_metrics=None, y=y)
                loss = (
                    out["loss_diff"]
                    + args.lambda_cls_s1 * out["loss_cls"]
                    + args.lambda_var    * out["loss_var"]
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            logits = _unwrap_model(model).cls_head(out["z0"].mean(dim=(1, 2)).float())
            sum_loss += float(loss.item())
            sum_diff += float(out["loss_diff"].item())
            sum_cls  += float(out["loss_cls"].item())
            sum_var  += float(out["loss_var"].item())
            sum_cls_correct += float((logits.argmax(dim=1) == y).sum().item())
            sum_cls_count += float(y.numel())
            n_batches += 1
            if tqdm is not None and progress_enabled:
                pbar.set_postfix(
                    total=f"{sum_loss/n_batches:.4f}",
                    diff=f"{sum_diff/n_batches:.4f}",
                    cls=f"{sum_cls/n_batches:.4f}",
                    var=f"{sum_var/n_batches:.4f}",
                )
            elif _is_main_process() and (
                n_batches == 1
                or n_batches % max(1, args.log_every) == 0
                or n_batches == total_steps
            ):
                LOGGER.info(
                    "[Stage1] epoch=%s/%s step=%s/%s  total=%.4f  diff=%.4f  cls=%.4f  var=%.4f",
                    epoch + 1, args.epochs, n_batches, total_steps,
                    sum_loss / n_batches, sum_diff / n_batches,
                    sum_cls  / n_batches, sum_var  / n_batches,
                )
        avg_train_total = sum_loss / max(n_batches, 1)
        avg_train_diff  = sum_diff / max(n_batches, 1)
        avg_train_cls   = sum_cls  / max(n_batches, 1)
        avg_train_var   = sum_var  / max(n_batches, 1)
        avg_train_total = _sync_mean(avg_train_total, device)
        avg_train_diff  = _sync_mean(avg_train_diff,  device)
        avg_train_cls   = _sync_mean(avg_train_cls,   device)
        avg_train_var   = _sync_mean(avg_train_var,   device)
        avg_train_acc_latent_cls = _sync_ratio(sum_cls_correct, sum_cls_count, device)
        avg_train_loss  = avg_train_total  # kept for checkpoint monitor compat

        avg_val_loss = None
        avg_val_total = avg_val_diff = avg_val_cls = avg_val_var = avg_val_acc_latent_cls = None
        if val_loader is not None:
            model.eval()
            val_sum = val_diff = val_cls = val_var = 0.0
            val_cls_correct = 0.0
            val_cls_count = 0.0
            val_n = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    x = batch["skeleton"].to(device)
                    gait_metrics = batch["gait_metrics"].to(device)
                    y = batch["label"].to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(x, gait_metrics=None, y=y)
                        loss = (
                            out["loss_diff"]
                            + args.lambda_cls_s1 * out["loss_cls"]
                            + args.lambda_var    * out["loss_var"]
                        )
                    logits = _unwrap_model(model).cls_head(out["z0"].mean(dim=(1, 2)).float())
                    val_sum  += float(loss.item())
                    val_diff += float(out["loss_diff"].item())
                    val_cls  += float(out["loss_cls"].item())
                    val_var  += float(out["loss_var"].item())
                    val_cls_correct += float((logits.argmax(dim=1) == y).sum().item())
                    val_cls_count += float(y.numel())
                    val_n += 1
            model.train()
            avg_val_total = _sync_mean(val_sum  / max(val_n, 1), device)
            avg_val_diff  = _sync_mean(val_diff / max(val_n, 1), device)
            avg_val_cls   = _sync_mean(val_cls  / max(val_n, 1), device)
            avg_val_var   = _sync_mean(val_var  / max(val_n, 1), device)
            avg_val_acc_latent_cls = _sync_ratio(val_cls_correct, val_cls_count, device)
            avg_val_loss  = avg_val_total

        metrics = {
            "train_loss_total": avg_train_total,
            "train_loss_diff":  avg_train_diff,
            "train_loss_cls":   avg_train_cls,
            "train_loss_var":   avg_train_var,
            "train_acc_latent_cls": avg_train_acc_latent_cls,
        }
        if avg_val_total is not None:
            metrics["val_loss_total"] = avg_val_total
            metrics["val_loss_diff"]  = avg_val_diff
            metrics["val_loss_cls"]   = avg_val_cls
            metrics["val_loss_var"]   = avg_val_var
            metrics["val_acc_latent_cls"] = avg_val_acc_latent_cls
        history.append({"epoch": float(epoch + 1), **metrics})
        if _is_main_process():
            write_history(run_dir, "stage1", history)
        _print_epoch_summary("Stage1", epoch + 1, args.epochs, metrics, time.time() - t0)

        if _is_main_process() and (epoch + 1) % max(1, args.eval_every_stage1) == 0:
            model.eval()
            eval_loader = val_loader or train_loader
            evaluate_stage1(
                _unwrap_model(model),
                eval_loader,
                device,
                run_dir / "stage1" / f"epoch_{epoch + 1:03d}",
                timestep_values=[0, 50, 100, 200, 300, 400, args.timesteps - 1],
            )
            model.train()

        monitor_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
        if _is_main_process() and monitor_loss < best_loss:
            best_loss = monitor_loss
            best_ckpt = os.path.join(args.save_dir, "stage1_best.pt")
            save_checkpoint(
                best_ckpt,
                _unwrap_model(model),
                extra={
                    "best_monitor_loss": best_loss,
                    "best_monitor_name": "val_loss_total" if avg_val_loss is not None else "train_loss_total",
                    "best_epoch": epoch + 1,
                    "encoder_graph_op": args.encoder_graph_op_resolved,
                    "skeleton_graph_op": args.skeleton_graph_op_resolved,
                    "gait_metric_names": list(GAIT_METRIC_NAMES),
                    "imu_feature_names": list(IMU_FEATURE_NAMES),
                    "run_dir": args.run_dir,
                },
            )
            LOGGER.info("[Stage1] new best checkpoint: %s (epoch=%s monitor_loss=%.6f)", best_ckpt, epoch + 1, best_loss)

    if _is_main_process():
        ckpt = os.path.join(args.save_dir, "stage1.pt")
        save_checkpoint(
            ckpt,
            _unwrap_model(model),
            extra={
                "encoder_graph_op": args.encoder_graph_op_resolved,
                "skeleton_graph_op": args.skeleton_graph_op_resolved,
                "gait_metric_names": list(GAIT_METRIC_NAMES),
                "imu_feature_names": list(IMU_FEATURE_NAMES),
                "run_dir": args.run_dir,
            },
        )
        LOGGER.info("[Stage1] saved checkpoint: %s", ckpt)


def train_stage2(args: argparse.Namespace, device: torch.device, distributed: bool, local_rank: int) -> None:
    """Train stage 2 model with frozen stage-1 encoder."""
    if not args.stage1_ckpt:
        raise ValueError("Stage 2 requires --stage1_ckpt (pretrained Stage 1 encoder).")
    stage1 = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        encoder_type=args.encoder_graph_op_resolved,
        skeleton_graph_op=args.skeleton_graph_op_resolved,
    ).to(device)
    if args.stage1_ckpt:
        # strict=False: checkpoint may predate Stage-1 cls_head; only encoder weights matter here
        load_checkpoint(args.stage1_ckpt, stage1, strict=False)

    model = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
        num_classes=args.num_classes,
        imu_graph_type=args.imu_graph,
        d_shared=args.d_shared,
    ).to(device)
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
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
    )
    train_loader, val_loader = _build_train_val_loaders(args, dataset=dataset, distributed=distributed)
    trainable_params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    if _is_main_process():
        save_run_manifest(
            Path(args.run_dir),
            args,
            device,
            runtime={
                "optimizer": optimizer.__class__.__name__,
                "scheduler": "ReduceLROnPlateau(factor=0.5, patience=5)",
                "sensor_modality": "accelerometer only",
                "sensor_locations": [Path(args.hip_folder).name if args.hip_folder else "", Path(args.wrist_folder).name if args.wrist_folder else ""],
            },
        )
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
    history: list[dict[str, float]] = []
    run_dir = Path(args.run_dir)

    for epoch in range(args.epochs):
        t0 = time.time()
        sum_loss = 0.0
        sum_align = 0.0
        sum_feature = 0.0
        sum_cls = 0.0
        sum_gait_pred = 0.0
        sum_cls_correct = 0.0
        sum_cls_count = 0.0
        sum_gait_abs = 0.0
        sum_gait_count = 0.0
        n_batches = 0
        total_steps = len(train_loader)
        progress_enabled = _is_main_process() and sys.stdout.isatty()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        pbar = _iter_with_progress(train_loader, f"Stage2 Epoch {epoch + 1}/{args.epochs}", enabled=progress_enabled)
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            x = batch["skeleton"].to(device)
            a_hip_stream = batch["A_hip"].to(device)
            a_wrist_stream = batch["A_wrist"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    x=x,
                    a_hip_stream=a_hip_stream,
                    a_wrist_stream=a_wrist_stream,
                    gait_metrics=gait_metrics,
                    y=y,
                )
                loss = (args.lambda_feature  * out["loss_feature"]
                        + args.lambda_align   * out["loss_align"]
                        + args.lambda_cls_s2  * out["loss_cls"]
                        + args.lambda_gait_s2 * out["loss_gait_pred"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            cls_logits = _unwrap_model(model).cls_head(out["h_global"].float())
            sum_loss += float(loss.item())
            sum_align += float(out["loss_align"].item())
            sum_feature += float(out["loss_feature"].item())
            sum_cls += float(out["loss_cls"].item())
            sum_gait_pred += float(out["loss_gait_pred"].item())
            sum_cls_correct += float((cls_logits.argmax(dim=1) == y).sum().item())
            sum_cls_count += float(y.numel())
            if _unwrap_model(model).gait_pred_head is not None:
                gait_pred = _unwrap_model(model).gait_pred_head(out["h_global"].float())
                sum_gait_abs += float(torch.abs(gait_pred - gait_metrics).sum().item())
                sum_gait_count += float(gait_metrics.numel())
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

        avg_train_align = sum_align / max(n_batches, 1)
        avg_train_feature = sum_feature / max(n_batches, 1)
        avg_train_cls = sum_cls / max(n_batches, 1)
        avg_train_gait_pred = sum_gait_pred / max(n_batches, 1)
        avg_train_acc_cls_aux = _sync_ratio(sum_cls_correct, sum_cls_count, device)
        avg_train_gait_mae = _sync_ratio(sum_gait_abs, sum_gait_count, device) if sum_gait_count > 0 else None

        avg_val_loss = None
        avg_val_align = None
        avg_val_feature = None
        avg_val_cls = None
        avg_val_gait_pred = None
        avg_val_acc_cls_aux = None
        avg_val_gait_mae = None
        if val_loader is not None:
            model.eval()
            val_sum = 0.0
            val_align_sum = 0.0
            val_feature_sum = 0.0
            val_cls_sum = 0.0
            val_gait_pred_sum = 0.0
            val_cls_correct = 0.0
            val_cls_count = 0.0
            val_gait_abs = 0.0
            val_gait_count = 0.0
            val_n = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    x = batch["skeleton"].to(device)
                    a_hip_stream = batch["A_hip"].to(device)
                    a_wrist_stream = batch["A_wrist"].to(device)
                    gait_metrics = batch["gait_metrics"].to(device)
                    y = batch["label"].to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(
                            x=x,
                            a_hip_stream=a_hip_stream,
                            a_wrist_stream=a_wrist_stream,
                            gait_metrics=gait_metrics,
                            y=y,
                        )
                        loss = (args.lambda_feature  * out["loss_feature"]
                                + args.lambda_align   * out["loss_align"]
                                + args.lambda_cls_s2  * out["loss_cls"]
                                + args.lambda_gait_s2 * out["loss_gait_pred"])
                    cls_logits = _unwrap_model(model).cls_head(out["h_global"].float())
                    val_sum += float(loss.item())
                    val_align_sum += float(out["loss_align"].item())
                    val_feature_sum += float(out["loss_feature"].item())
                    val_cls_sum += float(out["loss_cls"].item())
                    val_gait_pred_sum += float(out["loss_gait_pred"].item())
                    val_cls_correct += float((cls_logits.argmax(dim=1) == y).sum().item())
                    val_cls_count += float(y.numel())
                    if _unwrap_model(model).gait_pred_head is not None:
                        gait_pred = _unwrap_model(model).gait_pred_head(out["h_global"].float())
                        val_gait_abs += float(torch.abs(gait_pred - gait_metrics).sum().item())
                        val_gait_count += float(gait_metrics.numel())
                    val_n += 1
            model.train()
            avg_val_loss = val_sum / max(val_n, 1)
            avg_val_loss = _sync_mean(avg_val_loss, device)
            avg_val_align = val_align_sum / max(val_n, 1)
            avg_val_feature = val_feature_sum / max(val_n, 1)
            avg_val_cls = val_cls_sum / max(val_n, 1)
            avg_val_gait_pred = val_gait_pred_sum / max(val_n, 1)
            avg_val_acc_cls_aux = _sync_ratio(val_cls_correct, val_cls_count, device)
            avg_val_gait_mae = _sync_ratio(val_gait_abs, val_gait_count, device) if val_gait_count > 0 else None

        if avg_val_loss is not None:
            scheduler.step(avg_val_loss)

        metrics = {
            "train_loss_total": avg_train_loss,
            "train_loss_feature": avg_train_feature,
            "train_loss_align": avg_train_align,
            "train_loss_cls": avg_train_cls,
            "train_loss_gait_pred": avg_train_gait_pred,
            "train_acc_cls_aux": avg_train_acc_cls_aux,
        }
        if avg_train_gait_mae is not None:
            metrics["train_gait_mae"] = avg_train_gait_mae
        if avg_val_loss is not None:
            metrics["val_loss_total"] = avg_val_loss
            metrics["val_loss_feature"] = avg_val_feature
            metrics["val_loss_align"] = avg_val_align
            metrics["val_loss_cls"] = avg_val_cls
            metrics["val_loss_gait_pred"] = avg_val_gait_pred
            metrics["val_acc_cls_aux"] = avg_val_acc_cls_aux
            if avg_val_gait_mae is not None:
                metrics["val_gait_mae"] = avg_val_gait_mae
        history.append({"epoch": float(epoch + 1), **metrics})
        if _is_main_process():
            write_history(run_dir, "stage2", history)
        _print_epoch_summary("Stage2", epoch + 1, args.epochs, metrics, time.time() - t0)

        if _is_main_process() and (epoch + 1) % max(1, args.eval_every_stage2) == 0:
            model.eval()
            evaluate_stage2(
                stage1,
                _unwrap_model(model),
                val_loader or train_loader,
                device,
                run_dir / "stage2" / f"epoch_{epoch + 1:03d}",
                epoch=epoch + 1,
            )
            model.train()
        if _is_main_process() and (epoch + 1) % max(1, args.eval_every_stage2_reports) == 0:
            model.eval()
            evaluate_stage2_reports(
                _unwrap_model(model),
                val_loader or train_loader,
                device,
                run_dir / "stage2" / f"epoch_{epoch + 1:03d}",
            )
            model.train()

        monitor_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
        if _is_main_process() and monitor_loss < best_loss:
            best_loss = monitor_loss
            best_ckpt = os.path.join(args.save_dir, "stage2_best.pt")
            save_checkpoint(
                best_ckpt,
                _unwrap_model(model),
                extra={
                    "best_monitor_loss": best_loss,
                    "best_monitor_name": "val_loss_total" if avg_val_loss is not None else "train_loss_total",
                    "best_epoch": epoch + 1,
                    "encoder_graph_op": args.encoder_graph_op_resolved,
                    "skeleton_graph_op": args.skeleton_graph_op_resolved,
                    "gait_metric_names": list(GAIT_METRIC_NAMES),
                    "imu_feature_names": list(IMU_FEATURE_NAMES),
                    "run_dir": args.run_dir,
                },
            )
            LOGGER.info("[Stage2] new best checkpoint: %s (epoch=%s monitor_loss=%.6f)", best_ckpt, epoch + 1, best_loss)

    if _is_main_process():
        ckpt = os.path.join(args.save_dir, "stage2.pt")
        save_checkpoint(
            ckpt,
            _unwrap_model(model),
            extra={
                "encoder_graph_op": args.encoder_graph_op_resolved,
                "skeleton_graph_op": args.skeleton_graph_op_resolved,
                "gait_metric_names": list(GAIT_METRIC_NAMES),
                "imu_feature_names": list(IMU_FEATURE_NAMES),
                "run_dir": args.run_dir,
            },
        )
        LOGGER.info("[Stage2] saved checkpoint: %s", ckpt)


def train_stage3(args: argparse.Namespace, device: torch.device, distributed: bool, local_rank: int) -> None:
    """Train stage 3 conditional diffusion + classification model."""
    if not args.stage1_ckpt:
        raise ValueError("Stage 3 requires --stage1_ckpt (pretrained Stage 1 model).")
    if not args.stage2_ckpt:
        raise ValueError("Stage 3 requires --stage2_ckpt (pretrained Stage 2 aligner).")
    stage1 = Stage1Model(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        encoder_type=args.encoder_graph_op_resolved,
        skeleton_graph_op=args.skeleton_graph_op_resolved,
    ).to(device)
    if args.stage1_ckpt:
        load_checkpoint(args.stage1_ckpt, stage1, strict=True)

    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
        imu_graph_type=args.imu_graph,
        d_shared=args.d_shared,
    ).to(device)
    if args.stage2_ckpt:
        load_checkpoint(args.stage2_ckpt, stage2, strict=True)
    stage2.eval()

    # Decoder was never called during Stage 1 (no reconstruction loss), so its
    # weights are at initialisation.  Leave it trainable so Stage-3 reconstruction
    # losses (pose, velocity, motion) can train it to map latents → skeletons.
    # This matches the paper: D_ψ(z0) is part of the Stage-3 generation path.

    # Freeze the denoiser backbone: blocks (GAT graph layers), temporal_blocks, time_mlp, out.
    # Leave cross_attn_blocks and global_cond_proj trainable — both were never
    # called during Stage-1 training (h_tokens=None, h_global=None there), so
    # their weights are random and must be learned in Stage 3.
    # NOTE: the graph blocks are named "blocks" in GraphDenoiserMasked, NOT "graph_blocks".
    _denoiser_backbone = {"blocks", "temporal_blocks", "time_mlp", "out"}
    for name, p in stage1.denoiser.named_parameters():
        p.requires_grad = not any(name.startswith(b) for b in _denoiser_backbone)

    # Zero-init the output projections of cross_attn_blocks AND global_cond_proj
    # after loading the Stage-1 checkpoint.  Both modules have random weights
    # (never updated in Stage 1); zeroing the output projections makes them
    # mathematical no-ops at the start of Stage 3, so the denoiser behaves
    # identically to the Stage-1 unconditional denoiser and generates valid
    # human skeletons from epoch 0.  Conditioning then grows in gradually.
    for block in stage1.denoiser.cross_attn_blocks:
        torch.nn.init.zeros_(block.attn.out_proj.weight)
        torch.nn.init.zeros_(block.attn.out_proj.bias)
        torch.nn.init.zeros_(block.ff[2].weight)
        torch.nn.init.zeros_(block.ff[2].bias)
    for layer in stage1.denoiser.global_cond_proj:
        if hasattr(layer, "weight"):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    model = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        gait_metrics_dim=args.gait_metrics_dim,
        use_gait_conditioning=False,
        d_shared=args.d_shared,
        shared_motion_layer=stage2.shared_motion_layer,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Include the Stage-2 aligner in the Stage-3 optimizer so Stage-3
    # reconstruction losses (pose, velocity, gait, motion) can refine it.
    # Stage-2 pre-training warm-starts the aligner; Stage-3 fine-tunes it.
    trainable_params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    trainable_params += list(stage2.aligner.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3)
    if _is_main_process():
        save_run_manifest(
            Path(args.run_dir),
            args,
            device,
            runtime={
                "optimizer": optimizer.__class__.__name__,
                "scheduler": "none",
                "sensor_modality": "accelerometer only",
                "sensor_locations": [Path(args.hip_folder).name if args.hip_folder else "", Path(args.wrist_folder).name if args.wrist_folder else ""],
            },
        )
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
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
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
    history: list[dict[str, float]] = []
    run_dir = Path(args.run_dir)
    for epoch in range(args.epochs):
        t0 = time.time()
        sum_total = 0.0
        sum_diff = 0.0
        sum_cls = 0.0
        sum_pose = 0.0
        sum_latent = 0.0
        sum_vel = 0.0
        sum_gait = 0.0
        sum_motion = 0.0
        sum_bone = 0.0
        sum_skate = 0.0
        sum_smooth = 0.0
        sum_instab = 0.0
        n_batches = 0
        total_steps = len(train_loader)
        progress_enabled = _is_main_process() and sys.stdout.isatty()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        pbar = _iter_with_progress(train_loader, f"Stage3 Epoch {epoch + 1}/{args.epochs}", enabled=progress_enabled)
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            x = batch["skeleton"].to(device)
            a_hip_stream = batch["A_hip"].to(device)
            a_wrist_stream = batch["A_wrist"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            y = batch["label"].to(device)

            # Aligner is trainable in Stage 3: reconstruction losses provide
            # stronger gradient signal than Stage-2 gait-metric prediction alone.
            h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    x=x,
                    h_tokens=h_tokens,
                    h_global=h_global,
                    gait_target=gait_metrics,
                    gait_metrics=None,
                    fps=args.fps,
                    a_hip_stream=a_hip_stream,
                    a_wrist_stream=a_wrist_stream,
                )
                loss_diff = out["loss_diff"]
                loss_pose = out["loss_pose"]
                loss_latent = out["loss_latent"]
                loss_vel = out["loss_vel"]
                loss_gait = out["loss_gait"]
                loss_motion = out["loss_motion"]
                loss_cls = F.cross_entropy(
                    _unwrap_model(model).classifier(out["x_hat"].float()), y.long()
                )
                loss = (
                    loss_diff
                    + args.lambda_cls    * loss_cls
                    + args.lambda_pose   * loss_pose
                    + args.lambda_latent * loss_latent
                    + args.lambda_vel    * loss_vel
                    + args.lambda_motion * loss_motion
                    + args.lambda_gait   * loss_gait
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sum_total += float(loss.item())
            sum_diff += float(loss_diff.item())
            sum_cls += float(loss_cls.item())
            sum_pose += float(loss_pose.item())
            sum_latent += float(loss_latent.item())
            sum_vel += float(loss_vel.item())
            sum_gait += float(loss_gait.item())
            sum_motion += float(loss_motion.item())
            sum_bone += float(out["loss_bone"].item())
            sum_skate += float(out["loss_skate"].item())
            sum_smooth += float(out["loss_smooth"].item())
            sum_instab += float(out["loss_instab"].item())
            n_batches += 1
            if tqdm is not None and progress_enabled:
                pbar.set_postfix(
                    total=f"{(sum_total / n_batches):.4f}",
                    diff=f"{(sum_diff / n_batches):.4f}",
                    gait=f"{(sum_gait / n_batches):.4f}",
                )
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
                    "loss_total",
                    sum_total / n_batches,
                    t0,
                )
                LOGGER.info(
                    "[Stage3] step_losses diff=%.6f cls=%.6f gait=%.6f motion=%.6f",
                    sum_diff / n_batches,
                    sum_cls / n_batches,
                    sum_gait / n_batches,
                    sum_motion / n_batches,
                )
        avg_train_total = sum_total / max(n_batches, 1)
        avg_train_diff = sum_diff / max(n_batches, 1)
        avg_train_cls = sum_cls / max(n_batches, 1)
        avg_train_pose = sum_pose / max(n_batches, 1)
        avg_train_latent = sum_latent / max(n_batches, 1)
        avg_train_vel = sum_vel / max(n_batches, 1)
        avg_train_gait = sum_gait / max(n_batches, 1)
        avg_train_motion = sum_motion / max(n_batches, 1)
        avg_train_bone = sum_bone / max(n_batches, 1)
        avg_train_skate = sum_skate / max(n_batches, 1)
        avg_train_smooth = sum_smooth / max(n_batches, 1)
        avg_train_instab = sum_instab / max(n_batches, 1)
        avg_train_total = _sync_mean(avg_train_total, device)
        avg_train_diff = _sync_mean(avg_train_diff, device)
        avg_train_cls = _sync_mean(avg_train_cls, device)
        avg_train_pose = _sync_mean(avg_train_pose, device)
        avg_train_latent = _sync_mean(avg_train_latent, device)
        avg_train_vel = _sync_mean(avg_train_vel, device)
        avg_train_gait = _sync_mean(avg_train_gait, device)
        avg_train_motion = _sync_mean(avg_train_motion, device)
        avg_train_bone = _sync_mean(avg_train_bone, device)
        avg_train_skate = _sync_mean(avg_train_skate, device)
        avg_train_smooth = _sync_mean(avg_train_smooth, device)
        avg_train_instab = _sync_mean(avg_train_instab, device)

        avg_val_total = None
        avg_val_diff = None
        avg_val_cls = None
        avg_val_pose = None
        avg_val_latent = None
        avg_val_vel = None
        avg_val_gait = None
        avg_val_motion = None
        avg_val_bone = None
        avg_val_skate = None
        avg_val_smooth = None
        avg_val_instab = None
        if val_loader is not None:
            model.eval()
            val_total_sum = 0.0
            val_diff_sum = 0.0
            val_cls_sum = 0.0
            val_pose_sum = 0.0
            val_latent_sum = 0.0
            val_vel_sum = 0.0
            val_gait_sum = 0.0
            val_motion_sum = 0.0
            val_bone_sum = 0.0
            val_skate_sum = 0.0
            val_smooth_sum = 0.0
            val_instab_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    x = batch["skeleton"].to(device)
                    a_hip_stream = batch["A_hip"].to(device)
                    a_wrist_stream = batch["A_wrist"].to(device)
                    gait_metrics = batch["gait_metrics"].to(device)
                    y = batch["label"].to(device)
                    h_tokens, h_global = stage2.aligner(a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(
                            x=x,
                            h_tokens=h_tokens,
                            h_global=h_global,
                            gait_target=gait_metrics,
                            gait_metrics=None,
                            fps=args.fps,
                            a_hip_stream=a_hip_stream,
                            a_wrist_stream=a_wrist_stream,
                        )
                        val_loss_diff = out["loss_diff"]
                        val_loss_pose = out["loss_pose"]
                        val_loss_latent = out["loss_latent"]
                        val_loss_vel = out["loss_vel"]
                        val_loss_gait = out["loss_gait"]
                        val_loss_motion = out["loss_motion"]
                        val_loss_cls = F.cross_entropy(
                            _unwrap_model(model).classifier(out["x_hat"].float()), y.long()
                        )
                        val_total = (
                            val_loss_diff
                            + val_loss_cls
                            + args.lambda_pose   * val_loss_pose
                            + args.lambda_latent * val_loss_latent
                            + args.lambda_vel    * val_loss_vel
                            + args.lambda_motion * val_loss_motion
                            + args.lambda_gait   * val_loss_gait
                        )
                    val_total_sum += float(val_total.item())
                    val_diff_sum += float(val_loss_diff.item())
                    val_cls_sum += float(val_loss_cls.item())
                    val_pose_sum += float(val_loss_pose.item())
                    val_latent_sum += float(val_loss_latent.item())
                    val_vel_sum += float(val_loss_vel.item())
                    val_gait_sum += float(val_loss_gait.item())
                    val_motion_sum += float(val_loss_motion.item())
                    val_bone_sum += float(out["loss_bone"].item())
                    val_skate_sum += float(out["loss_skate"].item())
                    val_smooth_sum += float(out["loss_smooth"].item())
                    val_instab_sum += float(out["loss_instab"].item())
                    val_n += 1
            model.train()
            avg_val_total = _sync_mean(val_total_sum / max(val_n, 1), device)
            avg_val_diff = _sync_mean(val_diff_sum / max(val_n, 1), device)
            avg_val_cls = _sync_mean(val_cls_sum / max(val_n, 1), device)
            avg_val_pose = _sync_mean(val_pose_sum / max(val_n, 1), device)
            avg_val_latent = _sync_mean(val_latent_sum / max(val_n, 1), device)
            avg_val_vel = _sync_mean(val_vel_sum / max(val_n, 1), device)
            avg_val_gait = _sync_mean(val_gait_sum / max(val_n, 1), device)
            avg_val_motion = _sync_mean(val_motion_sum / max(val_n, 1), device)
            avg_val_bone = _sync_mean(val_bone_sum / max(val_n, 1), device)
            avg_val_skate = _sync_mean(val_skate_sum / max(val_n, 1), device)
            avg_val_smooth = _sync_mean(val_smooth_sum / max(val_n, 1), device)
            avg_val_instab = _sync_mean(val_instab_sum / max(val_n, 1), device)

        metrics = {
            "train_loss_total":  avg_train_total,
            "train_loss_diff":   avg_train_diff,
            "train_loss_cls":    avg_train_cls,
            "train_loss_pose":   avg_train_pose,
            "train_loss_latent": avg_train_latent,
            "train_loss_vel":    avg_train_vel,
            "train_loss_gait":   avg_train_gait,
            "train_loss_motion": avg_train_motion,
            "train_loss_bone":   avg_train_bone,
            "train_loss_skate":  avg_train_skate,
        }
        if avg_val_total is not None and avg_val_gait is not None:
            metrics["val_loss_total"]  = avg_val_total
            metrics["val_loss_diff"]   = avg_val_diff
            metrics["val_loss_cls"]    = avg_val_cls
            metrics["val_loss_pose"]   = avg_val_pose
            metrics["val_loss_latent"] = avg_val_latent
            metrics["val_loss_vel"]    = avg_val_vel
            metrics["val_loss_gait"]   = avg_val_gait
            metrics["val_loss_motion"] = avg_val_motion
            metrics["val_loss_bone"]   = avg_val_bone
            metrics["val_loss_skate"]  = avg_val_skate
        history.append({"epoch": float(epoch + 1), **metrics})
        if _is_main_process():
            write_history(run_dir, "stage3", history)
        _print_epoch_summary(
            "Stage3",
            epoch + 1,
            args.epochs,
            metrics,
            time.time() - t0,
        )
        if _is_main_process() and (epoch + 1) % max(1, args.eval_every_stage3) == 0:
            model.eval()
            evaluate_stage3(
                stage2,
                _unwrap_model(model),
                val_loader or train_loader,
                device,
                run_dir / "stage3" / f"epoch_{epoch + 1:03d}",
                args.sample_steps,
                fps=args.fps,
                epoch=epoch + 1,
                sampler=args.sampler,
                sample_seed=args.sample_seed,
            )
            model.train()
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
                    "best_train_loss_gait": avg_train_gait,
                    "best_train_loss_motion": avg_train_motion,
                    "best_val_loss_total": avg_val_total,
                    "best_val_loss_diff": avg_val_diff,
                    "best_val_loss_cls": avg_val_cls,
                    "best_val_loss_gait": avg_val_gait,
                    "best_val_loss_motion": avg_val_motion,
                    "best_epoch": epoch + 1,
                    "encoder_graph_op": args.encoder_graph_op_resolved,
                    "skeleton_graph_op": args.skeleton_graph_op_resolved,
                    "gait_metric_names": list(GAIT_METRIC_NAMES),
                    "imu_feature_names": list(IMU_FEATURE_NAMES),
                    "run_dir": args.run_dir,
                    "conditioning_mode": "imu_only" if args.one_to_one else "legacy",
                    "gait_role": "auxiliary_supervision" if args.one_to_one else "conditioning",
                },
            )
            LOGGER.info("[Stage3] new best checkpoint: %s (epoch=%s monitor_loss=%.6f)", best_ckpt, epoch + 1, best_loss)

    if _is_main_process():
        ckpt = os.path.join(args.save_dir, "stage3.pt")
        save_checkpoint(
            ckpt,
            _unwrap_model(model),
            extra={
                "encoder_graph_op": args.encoder_graph_op_resolved,
                "skeleton_graph_op": args.skeleton_graph_op_resolved,
                "gait_metric_names": list(GAIT_METRIC_NAMES),
                "imu_feature_names": list(IMU_FEATURE_NAMES),
                "run_dir": args.run_dir,
                "conditioning_mode": "imu_only" if args.one_to_one else "legacy",
                "gait_role": "auxiliary_supervision" if args.one_to_one else "conditioning",
            },
        )
        LOGGER.info("[Stage3] saved checkpoint: %s", ckpt)


def main() -> None:
    """Run stage-specific training."""
    args = parse_args()
    if args.val_split < 0.0 or args.val_split >= 1.0:
        raise ValueError("--val_split must be in [0.0, 1.0).")
    if args.gait_metrics_dim != DEFAULT_GAIT_METRICS_DIM:
        raise ValueError(
            f"--gait_metrics_dim must equal the fixed auto-computed gait-summary size ({DEFAULT_GAIT_METRICS_DIM})."
        )
    if args.stage == 3 and args.fps <= 0:
        raise ValueError("--fps must be positive.")
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
    graph_ckpt = args.stage1_ckpt if args.stage in {2, 3} else ""
    args.encoder_graph_op_resolved, args.skeleton_graph_op_resolved = _resolve_graph_ops(args, stage1_ckpt=graph_ckpt)
    args.run_dir = str(_make_run_dir(args))
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
