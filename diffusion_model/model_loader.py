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
