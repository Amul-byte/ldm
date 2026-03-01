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
    """Temporal GNN branch for one IMU stream with input shape [B, T, 6]."""

    def __init__(self, input_dim: int = 6, latent_dim: int = 256, depth: int = 3) -> None:
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
    """Two-branch IMU aligner for right-hip and left-wrist accel+gyro streams."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hip_branch = SensorTGNNBranch(input_dim=6, latent_dim=latent_dim)
        self.wrist_branch = SensorTGNNBranch(input_dim=6, latent_dim=latent_dim)
        self.fuse_tokens = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, s_hip: torch.Tensor, s_wrist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global embedding h_global [B,D] and temporal tokens h_tokens [B,T,D]."""
        assert_shape(s_hip, [None, None, 6], "IMULatentAligner.s_hip")
        assert_shape(s_wrist, [None, None, 6], "IMULatentAligner.s_wrist")
        assert s_hip.shape[:2] == s_wrist.shape[:2], "hip and wrist must share [B,T]"

        hip_tokens = self.hip_branch(s_hip)
        wrist_tokens = self.wrist_branch(s_wrist)
        fused = torch.cat([hip_tokens, wrist_tokens], dim=-1)
        h_tokens = self.fuse_tokens(fused)
        h_global = h_tokens.mean(dim=1)

        assert_shape(h_tokens, [s_hip.shape[0], s_hip.shape[1], self.latent_dim], "IMULatentAligner.h_tokens")
        assert_shape(h_global, [s_hip.shape[0], self.latent_dim], "IMULatentAligner.h_global")
        return h_global, h_tokens
