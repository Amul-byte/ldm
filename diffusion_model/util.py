"""Utility functions and constants for joint-aware latent diffusion."""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch

DEFAULT_TIMESTEPS = 500
DEFAULT_LATENT_DIM = 256
DEFAULT_WINDOW = 90
DEFAULT_JOINTS = 32
DEFAULT_NUM_CLASSES = 14
DEFAULT_LAMBDA_CLS = 0.1
DEFAULT_FPS = 30.0
EPS = 1e-6

# Joint order provided from skeleton analysis notebook.
JOINT_LABELS: Tuple[str, ...] = (
    "PELVIS",
    "SPINE_NAVAL",
    "SPINE_CHEST",
    "NECK",
    "CLAVICLE_LEFT",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "HAND_LEFT",
    "HANDTIP_LEFT",
    "THUMB_LEFT",
    "CLAVICLE_RIGHT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HAND_RIGHT",
    "HANDTIP_RIGHT",
    "THUMB_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "HEAD",
    "NOSE",
    "EYE_LEFT",
    "EAR_LEFT",
    "EYE_RIGHT",
    "EAR_RIGHT",
)

_SKELETON_CONNECTIONS_BY_NAME: Tuple[Tuple[str, str], ...] = (
    ("PELVIS", "SPINE_NAVAL"),
    ("SPINE_NAVAL", "SPINE_CHEST"),
    ("SPINE_CHEST", "NECK"),
    ("NECK", "HEAD"),
    ("HEAD", "NOSE"),
    ("NOSE", "EYE_LEFT"),
    ("NOSE", "EYE_RIGHT"),
    ("EYE_LEFT", "EAR_LEFT"),
    ("EYE_RIGHT", "EAR_RIGHT"),
    ("NECK", "CLAVICLE_LEFT"),
    ("CLAVICLE_LEFT", "SHOULDER_LEFT"),
    ("SHOULDER_LEFT", "ELBOW_LEFT"),
    ("ELBOW_LEFT", "WRIST_LEFT"),
    ("WRIST_LEFT", "HAND_LEFT"),
    ("HAND_LEFT", "HANDTIP_LEFT"),
    ("WRIST_LEFT", "THUMB_LEFT"),
    ("NECK", "CLAVICLE_RIGHT"),
    ("CLAVICLE_RIGHT", "SHOULDER_RIGHT"),
    ("SHOULDER_RIGHT", "ELBOW_RIGHT"),
    ("ELBOW_RIGHT", "WRIST_RIGHT"),
    ("WRIST_RIGHT", "HAND_RIGHT"),
    ("HAND_RIGHT", "HANDTIP_RIGHT"),
    ("WRIST_RIGHT", "THUMB_RIGHT"),
    ("PELVIS", "HIP_LEFT"),
    ("HIP_LEFT", "KNEE_LEFT"),
    ("KNEE_LEFT", "ANKLE_LEFT"),
    ("ANKLE_LEFT", "FOOT_LEFT"),
    ("PELVIS", "HIP_RIGHT"),
    ("HIP_RIGHT", "KNEE_RIGHT"),
    ("KNEE_RIGHT", "ANKLE_RIGHT"),
    ("ANKLE_RIGHT", "FOOT_RIGHT"),
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_skeleton_edges() -> List[Tuple[int, int]]:
    """Return adjacency edges derived from the fixed JOINT_LABELS ordering."""
    label_to_index = {label: idx for idx, label in enumerate(JOINT_LABELS)}
    return [(label_to_index[a], label_to_index[b]) for a, b in _SKELETON_CONNECTIONS_BY_NAME]


def get_joint_labels() -> Tuple[str, ...]:
    """Return the exact expected joint label order."""
    return JOINT_LABELS


def validate_joint_labels(labels: Sequence[str]) -> None:
    """Validate that dataset joint labels match the expected exact order."""
    expected = list(JOINT_LABELS)
    got = list(labels)
    assert got == expected, (
        "Joint label order mismatch.\n"
        f"Expected: {expected}\n"
        f"Got:      {got}"
    )


def build_adjacency_matrix(num_joints: int, device: torch.device | None = None) -> torch.Tensor:
    """Build a binary adjacency matrix with self-loops for the fixed skeleton graph."""
    adj = torch.zeros((num_joints, num_joints), dtype=torch.float32, device=device)
    for i, j in get_skeleton_edges():
        if i < num_joints and j < num_joints:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    adj.fill_diagonal_(1.0)
    return adj


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    assert timesteps.ndim == 1, "timesteps must have shape [B]"
    half_dim = dim // 2
    scale = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=emb.device)], dim=1)
    assert emb.shape == (timesteps.shape[0], dim), "invalid timestep embedding shape"
    return emb


def assert_shape(tensor: torch.Tensor, expected: Sequence[int | None], name: str) -> None:
    """Assert tensor rank and dimensions, where None means any size."""
    assert tensor.ndim == len(expected), f"{name} rank mismatch: got {tensor.ndim}, expected {len(expected)}"
    for idx, (actual, exp) in enumerate(zip(tensor.shape, expected)):
        if exp is not None:
            assert actual == exp, f"{name} dim {idx} mismatch: got {actual}, expected {exp}"
