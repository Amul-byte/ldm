"""Checkpoint save/load helpers."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def save_checkpoint(path: str, model: nn.Module, extra: Dict[str, Any] | None = None) -> None:
    """Save model state dict and optional metadata to checkpoint path."""
    payload: Dict[str, Any] = {"state_dict": model.state_dict()}
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path: str, model: nn.Module, strict: bool = True) -> Dict[str, Any]:
    """Load checkpoint into model and print missing/unexpected keys."""
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded checkpoint: {path}")
    print(f"missing keys ({len(missing)}): {missing}")
    print(f"unexpected keys ({len(unexpected)}): {unexpected}")
    return checkpoint


def _infer_graph_op_from_state_dict(state_dict: Dict[str, Any], prefix: str) -> str | None:
    has_gcn = any(key.startswith(prefix) and ".graph_op.gcn" in key for key in state_dict.keys())
    has_gat = any(key.startswith(prefix) and ".graph_op.gat" in key for key in state_dict.keys())
    if has_gcn and not has_gat:
        return "gcn"
    if has_gat and not has_gcn:
        return "gat"
    return None


def infer_graph_ops_from_checkpoint(
    path: str,
    default_encoder_graph_op: str = "gat",
    default_skeleton_graph_op: str = "gat",
) -> tuple[str, str]:
    """Infer encoder and full-skeleton graph families from checkpoint metadata or weights."""
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    extra = checkpoint.get("extra", {}) if isinstance(checkpoint, dict) else {}

    encoder_graph_op = str(extra.get("encoder_graph_op", extra.get("encoder_type", ""))).lower() or None
    skeleton_graph_op = str(extra.get("skeleton_graph_op", "")).lower() or None

    if encoder_graph_op is None:
        encoder_graph_op = _infer_graph_op_from_state_dict(state_dict, "encoder.graph_blocks")
    if skeleton_graph_op is None:
        decoder_graph_op = _infer_graph_op_from_state_dict(state_dict, "decoder.graph_blocks")
        denoiser_graph_op = _infer_graph_op_from_state_dict(state_dict, "denoiser.blocks")
        skeleton_graph_op = decoder_graph_op or denoiser_graph_op

    encoder_graph_op = encoder_graph_op or default_encoder_graph_op
    skeleton_graph_op = skeleton_graph_op or default_skeleton_graph_op

    if encoder_graph_op not in {"gat", "gcn"}:
        raise ValueError(f"Unsupported encoder graph op inferred from {path}: {encoder_graph_op}")
    if skeleton_graph_op not in {"gat", "gcn"}:
        raise ValueError(f"Unsupported skeleton graph op inferred from {path}: {skeleton_graph_op}")
    return encoder_graph_op, skeleton_graph_op
