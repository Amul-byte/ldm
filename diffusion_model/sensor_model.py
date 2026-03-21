"""Sensor models for IMU-to-latent alignment using temporal GNN branches."""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_model.graph_modules import HAS_TORCH_GEOMETRIC, TemporalGCNBlock, build_edge_index_from_adjacency
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


_IMU_GRAPH_SCALES: tuple[int, ...] = (1, 5, 15, 30)
"""Multi-scale temporal hop distances for the IMU graph.

With 3 stacked GCNConv layers and max distance 30, the receptive field is
3 × 30 + 1 = 91 frames — enough to cover a full gait cycle at 30 fps.
Distances chosen to capture:
  1  → local motion continuity (consecutive frames)
  5  → within-step oscillation (~0.17 s at 30 fps)
  15 → half a stride (~0.5 s)
  30 → full stride (~1 s, matches walking cadence at ~1 Hz)
"""


def build_imu_graph_adjacency(window_len: int, device: torch.device) -> torch.Tensor:
    """Build [2T, 2T] multi-scale adjacency for a combined hip+wrist IMU graph.

    Node layout:
      - Nodes 0 .. T-1      : hip timesteps
      - Nodes T .. 2T-1     : wrist timesteps

    Edge types:
      - Self-loops on all 2T nodes
      - Multi-scale temporal edges within each stream at distances in _IMU_GRAPH_SCALES
      - Cross-sensor edges between aligned timesteps: (i, T+i) and (T+i, i)

    The original ±1-only graph limited GCN receptive field to 7 frames with
    3 layers — structurally blind to gait patterns (~30 frames).  Multi-scale
    edges give a 91-frame receptive field with the same 3 layers.
    """
    n = 2 * window_len
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)

    # Self-loops
    idx_all = torch.arange(n, device=device)
    adj[idx_all, idx_all] = 1.0

    # Multi-scale intra-stream temporal edges
    for scale in _IMU_GRAPH_SCALES:
        if scale >= window_len:
            continue
        idx = torch.arange(window_len - scale, device=device)
        # Hip stream
        adj[idx, idx + scale] = 1.0
        adj[idx + scale, idx] = 1.0
        # Wrist stream
        adj[window_len + idx, window_len + idx + scale] = 1.0
        adj[window_len + idx + scale, window_len + idx] = 1.0

    # Cross-sensor edges (aligned timesteps)
    idx_t = torch.arange(window_len, device=device)
    adj[idx_t, window_len + idx_t] = 1.0
    adj[window_len + idx_t, idx_t] = 1.0

    assert_shape(adj, [n, n], "build_imu_graph_adjacency.adj")
    return adj


class IMUGraphEncoder(nn.Module):
    """Encode hip and wrist IMU streams jointly as a single bipartite temporal graph.

    Nodes 0..T-1 represent hip timesteps, nodes T..2T-1 represent wrist timesteps.
    Cross-sensor edges allow information to flow between the two streams at each
    aligned timestep, so the TGNN can capture hip-wrist correlations explicitly.

    Args:
        input_dim:  Feature dimension of each IMU stream after build_imu_features (default 6).
        latent_dim: Output embedding dimension per node (default 256).
        depth:      Number of TemporalGraphBlock layers applied to the combined graph.
    """

    def __init__(self, input_dim: int = 6, latent_dim: int = 256, depth: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.depth = depth
        # Separate projections for each sensor so the model can learn sensor-specific
        # initial embeddings before the shared graph processing.
        self.hip_proj = nn.Linear(input_dim, latent_dim)
        self.wrist_proj = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [TemporalGCNBlock(dim=latent_dim, num_layers=3) for _ in range(depth)]
        )
        # Lazy cache for IMU graph structures keyed by window length.
        # The IMU graph topology (adjacency + edge_index) is fixed for a given T,
        # so rebuilding it every forward call wastes ~2ms per step.
        self._imu_graph_cache: dict = {}

    def forward(self, hip_feat: torch.Tensor, wrist_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run TGNN on the combined hip+wrist graph and split outputs back by sensor.

        Args:
            hip_feat:   [B, T, input_dim]  (hip IMU features)
            wrist_feat: [B, T, input_dim]  (wrist IMU features)

        Returns:
            hip_tokens:   [B, T, latent_dim]
            wrist_tokens: [B, T, latent_dim]
        """
        assert_shape(hip_feat, [None, None, self.input_dim], "IMUGraphEncoder.hip_feat")
        assert_shape(wrist_feat, [None, None, self.input_dim], "IMUGraphEncoder.wrist_feat")
        assert hip_feat.shape[:2] == wrist_feat.shape[:2], "hip and wrist must share [B, T]"

        b, t, _ = hip_feat.shape

        # Project each stream to latent dim
        hip_latent = self.hip_proj(hip_feat)    # [B, T, D]
        wrist_latent = self.wrist_proj(wrist_feat)  # [B, T, D]

        # Concatenate along the time axis to form a single [B, 2T, D] node sequence.
        # Node i < T corresponds to hip timestep i; node T+i to wrist timestep i.
        combined = torch.cat([hip_latent, wrist_latent], dim=1)  # [B, 2T, D]

        # Retrieve or build the [2T, 2T] adjacency and edge_index for this window length.
        if t not in self._imu_graph_cache:
            adj = build_imu_graph_adjacency(window_len=t, device=hip_feat.device)
            edge_index = build_edge_index_from_adjacency(adj) if HAS_TORCH_GEOMETRIC else None
            self._imu_graph_cache[t] = (adj, edge_index)
        adj, edge_index = self._imu_graph_cache[t]
        # Move cached tensors to current device if needed (e.g. after model.to(device))
        if adj.device != hip_feat.device:
            adj = adj.to(hip_feat.device)
            edge_index = edge_index.to(hip_feat.device) if edge_index is not None else None
            self._imu_graph_cache[t] = (adj, edge_index)

        for block in self.blocks:
            combined = block(combined, adjacency=adj, edge_index=edge_index)

        # Split back into per-sensor outputs
        hip_out = combined[:, :t, :]    # [B, T, D]
        wrist_out = combined[:, t:, :]  # [B, T, D]

        assert_shape(hip_out, [b, t, self.latent_dim], "IMUGraphEncoder.hip_out")
        assert_shape(wrist_out, [b, t, self.latent_dim], "IMUGraphEncoder.wrist_out")
        return hip_out, wrist_out


class IMULatentAligner(nn.Module):
    """Two-stream accelerometer aligner for hip and wrist, each [B, T, 3].

    IMU windows are first converted to explicit graphs (build_imu_graph_adjacency)
    and then processed jointly by a shared TGNN (IMUGraphEncoder).  Cross-sensor
    edges in the graph allow the encoder to capture correlations between hip and
    wrist motion before the two streams are fused.

    Checkpoint note: this replaces the former dual-SensorTGNNBranch design, so
    Stage-2 checkpoints trained with the old architecture must be retrained.
    Stage-1 checkpoints are unaffected (they do not contain the aligner).
    """

    def __init__(self, latent_dim: int = 256, gait_metrics_dim: int = 0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.gait_metrics_dim = gait_metrics_dim
        self.imu_feature_dim = len(IMU_FEATURE_NAMES)
        # Single encoder that processes both streams as one graph
        self.graph_encoder = IMUGraphEncoder(
            input_dim=self.imu_feature_dim,
            latent_dim=latent_dim,
            depth=3,
        )
        fuse_in_dim = latent_dim * 2
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
        """Return temporal/global sensor embeddings from the IMU graph encoding.

        Each IMU window is converted to an explicit bipartite graph (hip nodes +
        wrist nodes + cross-sensor edges) before being processed by the TGNN.

        The optional ``gait_metrics`` argument is ignored and kept only for
        backward compatibility with older call sites.

        Returns:
            sensor_tokens: [B, T, latent_dim]  temporal conditioning tokens
            h_global:      [B, latent_dim]     global conditioning vector
        """
        assert_shape(a_hip_stream, [None, None, 3], "IMULatentAligner.a_hip_stream")
        assert_shape(a_wrist_stream, [None, None, 3], "IMULatentAligner.a_wrist_stream")
        assert a_hip_stream.shape[:2] == a_wrist_stream.shape[:2], "Hip and wrist streams must share [B,T]"

        hip_features = build_imu_features(a_hip_stream)
        wrist_features = build_imu_features(a_wrist_stream)

        # Graph-based joint encoding: cross-sensor edges are included in the graph
        hip_tokens, wrist_tokens = self.graph_encoder(hip_features, wrist_features)

        # Fuse hip and wrist token sequences
        fused = torch.cat([hip_tokens, wrist_tokens], dim=-1)
        sensor_tokens = self.fuse_tokens(fused)
        h_global = sensor_tokens.mean(dim=1)

        assert_shape(
            sensor_tokens,
            [a_hip_stream.shape[0], a_hip_stream.shape[1], self.latent_dim],
            "IMULatentAligner.sensor_tokens",
        )
        assert_shape(h_global, [a_hip_stream.shape[0], self.latent_dim], "IMULatentAligner.h_global")
        return sensor_tokens, h_global
