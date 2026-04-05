"""Sensor models for IMU-to-latent alignment using temporal GNN branches."""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_model.graph_modules import HAS_TORCH_GEOMETRIC, TemporalGCNBlock, build_edge_index_from_adjacency
from diffusion_model.shared_features import SHARED_FEATURE_NAMES, build_shared_motion_features
if HAS_TORCH_GEOMETRIC:
    from torch_geometric.nn import GCNConv
from diffusion_model.util import assert_shape


IMU_FEATURE_NAMES: tuple[str, ...] = SHARED_FEATURE_NAMES


def build_imu_features(accel: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Expand raw accelerometer streams [B,T,3] into shared [B,T,10] motion features."""
    assert_shape(accel, [None, None, 3], "build_imu_features.accel")
    features = build_shared_motion_features(accel, eps=eps)
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


def build_chain_imu_graph_adjacency(window_len: int, device: torch.device) -> torch.Tensor:
    """Build [2T, 2T] simple chain adjacency for a combined hip+wrist IMU graph.

    Node layout:
      - Nodes 0 .. T-1      : hip timesteps
      - Nodes T .. 2T-1     : wrist timesteps

    Edge types:
      - Self-loops on all 2T nodes
      - Sequential edges (gap=1 only) within each stream: node[t] ↔ node[t+1]
      - Cross-sensor edges between aligned timesteps: (i, T+i) and (T+i, i)

    Receptive field with 3 stacked TemporalGCNBlock (3 internal GCN layers each):
    3 × 3 × 1 + 1 = ~19 frames — simpler than multiscale but still captures
    short-range motion patterns. Mirrors the graph topology of GCNN_v2.ipynb.
    """
    n = 2 * window_len
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)

    # Self-loops
    idx_all = torch.arange(n, device=device)
    adj[idx_all, idx_all] = 1.0

    # Sequential edges (gap=1 only) within each stream
    idx = torch.arange(window_len - 1, device=device)
    adj[idx, idx + 1] = 1.0
    adj[idx + 1, idx] = 1.0
    adj[window_len + idx, window_len + idx + 1] = 1.0
    adj[window_len + idx + 1, window_len + idx] = 1.0

    # Cross-sensor edges (aligned timesteps)
    idx_t = torch.arange(window_len, device=device)
    adj[idx_t, window_len + idx_t] = 1.0
    adj[window_len + idx_t, idx_t] = 1.0

    assert_shape(adj, [n, n], "build_chain_imu_graph_adjacency.adj")
    return adj


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


class SensorGCNEncoder(nn.Module):
    """Per-sensor temporal GCN encoder matching the saved checkpoint architecture.

    Parameter names: conv1/conv2/conv3, norm1/norm2/norm3, out_proj.
    Each conv is a GCNConv over a chain temporal graph on T nodes.
    """

    def __init__(
        self,
        input_dim: int = len(IMU_FEATURE_NAMES),
        latent_dim: int = 256,
        graph_type: str = "chain",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.graph_type = graph_type
        self.dropout = float(dropout)
        # Hidden dims match the saved Stage-2 branch family: input_dim→32→32→64→latent_dim
        self.conv1 = GCNConv(input_dim, 12, add_self_loops=False)
        self.conv2 = GCNConv(12, 12, add_self_loops=False)
        self.conv3 = GCNConv(12, 24, add_self_loops=False)
        self.norm1 = nn.LayerNorm(12)
        self.norm2 = nn.LayerNorm(12)
        self.norm3 = nn.LayerNorm(24)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.out_proj = nn.Linear(24, latent_dim)
        self._edge_cache: dict = {}

    def _get_batched_edge_index(self, b: int, t: int, device: torch.device) -> torch.Tensor:
        cache_key = (b, t, self.graph_type)
        if cache_key not in self._edge_cache:
            self_idx = torch.arange(t, device=device)
            srcs = [self_idx]
            dsts = [self_idx]
            scales = [1, 5, 15, 30] if self.graph_type == "multiscale" else [1]
            for scale in scales:
                if scale >= t:
                    continue
                idx = torch.arange(t - scale, device=device)
                srcs += [idx, idx + scale]
                dsts += [idx + scale, idx]
            src = torch.cat(srcs)
            dst = torch.cat(dsts)
            single_ei = torch.stack([src, dst], dim=0)
            batched = torch.cat([single_ei + i * t for i in range(b)], dim=1)
            self._edge_cache[cache_key] = batched
        ei = self._edge_cache[cache_key]
        if ei.device != device:
            ei = ei.to(device)
            self._edge_cache[cache_key] = ei
        return ei

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim] -> [B, T, latent_dim]"""
        b, t, _ = x.shape
        ei = self._get_batched_edge_index(b, t, x.device)
        h = x.reshape(b * t, -1)
        h = self.drop1(torch.relu(self.norm1(self.conv1(h, ei).reshape(b, t, -1)))).reshape(b * t, -1)
        h = self.drop2(torch.relu(self.norm2(self.conv2(h, ei).reshape(b, t, -1)))).reshape(b * t, -1)
        h = self.drop3(torch.relu(self.norm3(self.conv3(h, ei).reshape(b, t, -1)))).reshape(b * t, -1)
        return self.out_proj(h).reshape(b, t, -1)


class IMUGraphEncoder(nn.Module):
    """Encode hip and wrist IMU streams jointly as a single bipartite temporal graph.

    Nodes 0..T-1 represent hip timesteps, nodes T..2T-1 represent wrist timesteps.
    Cross-sensor edges allow information to flow between the two streams at each
    aligned timestep, so the TGNN can capture hip-wrist correlations explicitly.

    Args:
        input_dim:  Feature dimension of each IMU stream after build_imu_features (default 10).
        latent_dim: Output embedding dimension per node (default 256).
        depth:      Number of TemporalGraphBlock layers applied to the combined graph.
    """

    def __init__(self, input_dim: int = len(IMU_FEATURE_NAMES), latent_dim: int = 256, depth: int = 3,
                 graph_type: str = "chain") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.depth = depth
        self.graph_type = graph_type
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
            if self.graph_type == "chain":
                adj = build_chain_imu_graph_adjacency(window_len=t, device=hip_feat.device)
            else:
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

    def __init__(
        self,
        latent_dim: int = 256,
        gait_metrics_dim: int = 0,
        graph_type: str = "chain",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.gait_metrics_dim = gait_metrics_dim
        self.imu_feature_dim = len(IMU_FEATURE_NAMES)
        self.dropout = float(dropout)
        # Separate per-sensor encoders matching checkpoint architecture
        self.hip_encoder = SensorGCNEncoder(
            input_dim=self.imu_feature_dim,
            latent_dim=latent_dim,
            graph_type=graph_type,
            dropout=self.dropout,
        )
        self.wrist_encoder = SensorGCNEncoder(
            input_dim=self.imu_feature_dim,
            latent_dim=latent_dim,
            graph_type=graph_type,
            dropout=self.dropout,
        )
        self.fuse_tokens = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(p=0.5),
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

        hip_features = build_imu_features(a_hip_stream)      # [B, T, 10]
        wrist_features = build_imu_features(a_wrist_stream)  # [B, T, 10]

        hip_tokens = self.hip_encoder(hip_features)        # [B, T, D]
        wrist_tokens = self.wrist_encoder(wrist_features)  # [B, T, D]

        fused = torch.cat([hip_tokens, wrist_tokens], dim=-1)  # [B, T, 2D]
        sensor_tokens = self.fuse_tokens(fused)                # [B, T, D]
        h_global = sensor_tokens.mean(dim=1)                   # [B, D]

        assert_shape(
            sensor_tokens,
            [a_hip_stream.shape[0], a_hip_stream.shape[1], self.latent_dim],
            "IMULatentAligner.sensor_tokens",
        )
        assert_shape(h_global, [a_hip_stream.shape[0], self.latent_dim], "IMULatentAligner.h_global")
        return sensor_tokens, h_global
