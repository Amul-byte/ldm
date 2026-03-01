"""Sensor models for IMU-to-latent alignment using temporal GNN branches."""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_model.graph_modules import HAS_TORCH_GEOMETRIC, TemporalGraphBlock, build_edge_index_from_adjacency
from diffusion_model.util import assert_shape


def build_temporal_adjacency(length: int, device: torch.device) -> torch.Tensor:
    """Build temporal adjacency with self-loops and immediate neighbors."""
    adj = torch.zeros((length, length), dtype=torch.float32, device=device)
    idx = torch.arange(length, device=device)
    adj[idx, idx] = 1.0
    if length > 1:
        adj[idx[:-1], idx[1:]] = 1.0
        adj[idx[1:], idx[:-1]] = 1.0
    assert_shape(adj, [length, length], "build_temporal_adjacency.adj")
    return adj


class SensorTGNNBranch(nn.Module):
    """Temporal GNN branch for one IMU stream with input shape [B, T, 3]."""

    def __init__(self, input_dim: int = 3, latent_dim: int = 256, depth: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.ModuleList([TemporalGraphBlock(dim=latent_dim, num_heads=8, use_pyg=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Encode one IMU sequence into temporal latent tokens [B, T, latent_dim]."""
        assert_shape(s, [None, None, self.input_dim], "SensorTGNNBranch.s")
        b, t, _ = s.shape
        adjacency = build_temporal_adjacency(length=t, device=s.device)
        edge_index = build_edge_index_from_adjacency(adjacency) if HAS_TORCH_GEOMETRIC else None

        h = self.in_proj(s)
        for block in self.blocks:
            h = block(h, adjacency=adjacency, edge_index=edge_index)
        h = self.norm(h)
        assert_shape(h, [b, t, self.latent_dim], "SensorTGNNBranch.h")
        return h


class IMULatentAligner(nn.Module):
    """Two-branch IMU aligner for A and Omega streams, each shaped [B, T, 3]."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.a_branch = SensorTGNNBranch(input_dim=3, latent_dim=latent_dim)
        self.omega_branch = SensorTGNNBranch(input_dim=3, latent_dim=latent_dim)
        self.fuse_tokens = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, a_stream: torch.Tensor, omega_stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global embedding h_global [B,D] and temporal sensor tokens [B,T,D]."""
        assert_shape(a_stream, [None, None, 3], "IMULatentAligner.a_stream")
        assert_shape(omega_stream, [None, None, 3], "IMULatentAligner.omega_stream")
        assert a_stream.shape[:2] == omega_stream.shape[:2], "A and Omega must share [B,T]"

        a_tokens = self.a_branch(a_stream)
        omega_tokens = self.omega_branch(omega_stream)
        fused = torch.cat([a_tokens, omega_tokens], dim=-1)
        sensor_tokens = self.fuse_tokens(fused)
        h_global = sensor_tokens.mean(dim=1)

        assert_shape(sensor_tokens, [a_stream.shape[0], a_stream.shape[1], self.latent_dim], "IMULatentAligner.sensor_tokens")
        assert_shape(h_global, [a_stream.shape[0], self.latent_dim], "IMULatentAligner.h_global")
        return h_global, sensor_tokens
