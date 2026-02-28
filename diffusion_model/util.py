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


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_skeleton_edges() -> List[Tuple[int, int]]:
    """Return a fixed 32-joint undirected skeleton adjacency list."""
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (0, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (9, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (10, 26),
        (26, 27),
        (27, 28),
        (28, 29),
        (11, 30),
        (12, 31),
    ]
    return edges


def build_adjacency_matrix(num_joints: int, device: torch.device | None = None) -> torch.Tensor:
    """Build a binary adjacency matrix with self-loops for the fixed skeleton graph."""
    edges = get_skeleton_edges()
    adj = torch.zeros((num_joints, num_joints), dtype=torch.float32, device=device)
    for i, j in edges:
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
