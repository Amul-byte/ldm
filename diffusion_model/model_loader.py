"""Checkpoint save/load helpers."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from diffusion_model.util import DEFAULT_JOINTS, SKELETON_LAYOUT_VERSION


def _layout_metadata_from_model(model: nn.Module) -> Dict[str, Any]:
    num_joints = getattr(model, "num_joints", DEFAULT_JOINTS)
    return {
        "skeleton_layout_version": SKELETON_LAYOUT_VERSION,
        "num_joints": int(num_joints),
    }


def _validate_checkpoint_layout(checkpoint: Dict[str, Any], path: str, model: nn.Module | None = None) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} is not a structured dict; retrain under the 16-joint layout.")
    extra = checkpoint.get("extra", {})
    layout_version = extra.get("skeleton_layout_version", None)
    num_joints = extra.get("num_joints", None)
    if layout_version != SKELETON_LAYOUT_VERSION or int(num_joints or -1) != DEFAULT_JOINTS:
        raise ValueError(
            f"Checkpoint {path} uses an incompatible skeleton layout "
            f"(version={layout_version!r}, num_joints={num_joints!r}). "
            f"Expected version={SKELETON_LAYOUT_VERSION!r}, num_joints={DEFAULT_JOINTS}. Retrain from scratch."
        )
    if model is not None and getattr(model, "num_joints", DEFAULT_JOINTS) != DEFAULT_JOINTS:
        raise ValueError(
            f"Model expects num_joints={getattr(model, 'num_joints', None)}, but canonical layout requires {DEFAULT_JOINTS}."
        )


def save_checkpoint(path: str, model: nn.Module, extra: Dict[str, Any] | None = None) -> None:
    """Save model state dict and optional metadata to checkpoint path."""
    payload: Dict[str, Any] = {"state_dict": model.state_dict()}
    merged_extra = {**_layout_metadata_from_model(model), **(extra or {})}
    payload["extra"] = merged_extra
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path: str, model: nn.Module, strict: bool = True) -> Dict[str, Any]:
    """Load checkpoint into model and print missing/unexpected keys."""
    checkpoint = torch.load(path, map_location="cpu")
    _validate_checkpoint_layout(checkpoint, path, model=model)
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
    _validate_checkpoint_layout(checkpoint, path, model=None)
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


def infer_temporal_block_type_from_checkpoint(path: str, default_temporal_block_type: str = "conv") -> str:
    """Infer the Stage-1/3 denoiser temporal block type from checkpoint metadata or weights."""
    checkpoint = torch.load(path, map_location="cpu")
    _validate_checkpoint_layout(checkpoint, path, model=None)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    extra = checkpoint.get("extra", {}) if isinstance(checkpoint, dict) else {}

    temporal_block_type = str(extra.get("temporal_block_type", "")).lower() or None
    if temporal_block_type is None:
        has_attention = any(
            key.startswith("denoiser.temporal_blocks.") and ".attn." in key
            for key in state_dict.keys()
        )
        has_conv = any(
            key.startswith("denoiser.temporal_blocks.") and ".conv." in key
            for key in state_dict.keys()
        )
        if has_attention and not has_conv:
            temporal_block_type = "attention"
        elif has_conv and not has_attention:
            temporal_block_type = "conv"

    temporal_block_type = temporal_block_type or default_temporal_block_type
    if temporal_block_type not in {"conv", "attention"}:
        raise ValueError(f"Unsupported temporal block type inferred from {path}: {temporal_block_type}")
    return temporal_block_type
