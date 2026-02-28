"""Sensor models for IMU-to-latent alignment."""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_model.util import assert_shape


class SensorTGNNBranch(nn.Module):
    """Temporal branch for one IMU stream with shape [B, T, 6]."""

    def __init__(self, input_dim: int = 6, latent_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(input_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Encode sensor sequence and return global pooled embedding [B, latent_dim]."""
        assert_shape(s, [None, None, self.input_dim], "SensorTGNNBranch.s")
        h = self.in_proj(s)
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h.mean(dim=1)
        assert_shape(pooled, [s.shape[0], self.latent_dim], "SensorTGNNBranch.pooled")
        return pooled


class IMULatentAligner(nn.Module):
    """Two-branch IMU aligner for right-hip and left-wrist accel+gyro."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hip_branch = SensorTGNNBranch(input_dim=6, latent_dim=latent_dim)
        self.wrist_branch = SensorTGNNBranch(input_dim=6, latent_dim=latent_dim)
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, s_hip: torch.Tensor, s_wrist: torch.Tensor) -> torch.Tensor:
        """Fuse hip and wrist embeddings into global latent condition h [B, latent_dim]."""
        assert_shape(s_hip, [None, None, 6], "IMULatentAligner.s_hip")
        assert_shape(s_wrist, [None, None, 6], "IMULatentAligner.s_wrist")
        assert s_hip.shape[:2] == s_wrist.shape[:2], "hip and wrist must share [B,T]"
        h_hip = self.hip_branch(s_hip)
        h_wrist = self.wrist_branch(s_wrist)
        fused = torch.cat([h_hip, h_wrist], dim=-1)
        h = self.fuse(fused)
        assert_shape(h, [s_hip.shape[0], self.latent_dim], "IMULatentAligner.h")
        return h
