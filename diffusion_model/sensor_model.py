"""Sensor models for IMU-to-latent alignment using temporal GNN branches."""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_model.graph_modules import HAS_TORCH_GEOMETRIC, TemporalGraphBlock, build_edge_index_from_adjacency
from diffusion_model.util import assert_shape


IMU_FEATURE_NAMES: tuple[str, ...] = ("ax", "ay", "az", "magnitude", "pitch", "roll")


def build_imu_features(accel: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Expand raw accelerometer streams [B,T,3] into [B,T,6]."""
    assert_shape(accel, [None, None, 3], "build_imu_features.accel")
    ax = accel[..., 0]
    ay = accel[..., 1]
    az = accel[..., 2]
    magnitude = torch.sqrt(torch.clamp(ax * ax + ay * ay + az * az, min=eps))
    pitch = torch.atan2(ax, torch.sqrt(torch.clamp(ay * ay + az * az, min=eps)))
    az_safe = torch.where(az.abs() < eps, torch.full_like(az, eps), az)
    roll = torch.atan2(ay, az_safe)
    features = torch.stack([ax, ay, az, magnitude, pitch, roll], dim=-1)
    assert_shape(features, [accel.shape[0], accel.shape[1], len(IMU_FEATURE_NAMES)], "build_imu_features.features")
    return features


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
        # Disable the sensor-branch output normalization and keep the forward path shape-compatible.
        self.norm = nn.Identity()

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
    """Two-branch accelerometer aligner for hip and wrist streams, each [B, T, 3]."""

    def __init__(self, latent_dim: int = 256, gait_metrics_dim: int = 0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.gait_metrics_dim = gait_metrics_dim
        self.imu_feature_dim = len(IMU_FEATURE_NAMES)
        self.a_branch = SensorTGNNBranch(input_dim=self.imu_feature_dim, latent_dim=latent_dim)
        self.omega_branch = SensorTGNNBranch(input_dim=self.imu_feature_dim, latent_dim=latent_dim)
        fuse_in_dim = latent_dim * 2
        if gait_metrics_dim > 0:
            self.gait_metrics_proj = nn.Sequential(
                nn.Linear(gait_metrics_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
            fuse_in_dim += latent_dim
        else:
            self.gait_metrics_proj = None
        self.fuse_tokens = nn.Sequential(
            nn.Linear(fuse_in_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        a_hip_stream: torch.Tensor,
        a_wrist_stream: torch.Tensor,
        gait_metrics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return proposal-style temporal/global embeddings: (h_tokens [B,T,D], h_global [B,D])."""
        assert_shape(a_hip_stream, [None, None, 3], "IMULatentAligner.a_hip_stream")
        assert_shape(a_wrist_stream, [None, None, 3], "IMULatentAligner.a_wrist_stream")
        assert a_hip_stream.shape[:2] == a_wrist_stream.shape[:2], "Hip and wrist streams must share [B,T]"

        hip_features = build_imu_features(a_hip_stream)
        wrist_features = build_imu_features(a_wrist_stream)
        hip_tokens = self.a_branch(hip_features)
        wrist_tokens = self.omega_branch(wrist_features)
        token_parts = [hip_tokens, wrist_tokens]
        gait_global = None
        if self.gait_metrics_proj is not None:
            if gait_metrics is None:
                raise ValueError("gait_metrics must be provided when gait_metrics_dim > 0")
            assert_shape(gait_metrics, [a_hip_stream.shape[0], self.gait_metrics_dim], "IMULatentAligner.gait_metrics")
            gait_global = self.gait_metrics_proj(gait_metrics)
            token_parts.append(gait_global.unsqueeze(1).expand(-1, a_hip_stream.shape[1], -1))
        fused = torch.cat(token_parts, dim=-1)
        sensor_tokens = self.fuse_tokens(fused)
        h_global = sensor_tokens.mean(dim=1)
        if gait_global is not None:
            h_global = h_global + gait_global

        assert_shape(
            sensor_tokens,
            [a_hip_stream.shape[0], a_hip_stream.shape[1], self.latent_dim],
            "IMULatentAligner.sensor_tokens",
        )
        assert_shape(h_global, [a_hip_stream.shape[0], self.latent_dim], "IMULatentAligner.h_global")
        return sensor_tokens, h_global
