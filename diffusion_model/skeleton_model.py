"""Skeleton graph models for latent diffusion stages."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from diffusion_model.graph_modules import (
    CrossAttentionBlock,
    GraphBlock,
    GraphBlockGCN,
    TemporalConvBlock,
    HAS_TORCH_GEOMETRIC,
    build_edge_index,
)
from diffusion_model.shared_features import (
    SHARED_FEATURE_NAMES,
    build_shared_motion_features,
    compute_skeleton_acceleration,
)
from diffusion_model.util import assert_shape, build_adjacency_matrix, sinusoidal_timestep_embedding


# Skeleton features now combine raw position with shared acceleration-derived motion features.
SKELETON_FEATURE_NAMES: tuple[str, ...] = ("x", "y", "z") + SHARED_FEATURE_NAMES
SKELETON_FEATURE_DIM: int = len(SKELETON_FEATURE_NAMES)  # 13


def build_skeleton_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Expand raw skeleton coords [B, T, J, 3] into [B, T, J, 13]."""
    assert_shape(x, [None, None, None, 3], "build_skeleton_features.x")
    accel = compute_skeleton_acceleration(x)
    shared = build_shared_motion_features(accel, eps=eps)
    features = torch.cat([x, shared], dim=-1)
    assert_shape(
        features,
        [x.shape[0], x.shape[1], x.shape[2], SKELETON_FEATURE_DIM],
        "build_skeleton_features.features",
    )
    return features


def _normalize_skeleton_graph_op(graph_op: str) -> str:
    graph_op_name = str(graph_op).lower()
    if graph_op_name not in {"gat", "gcn"}:
        raise ValueError(f"Unsupported skeleton graph op: {graph_op}")
    return graph_op_name


def _build_skeleton_graph_block(graph_op: str, dim: int, num_joints: int) -> nn.Module:
    graph_op_name = _normalize_skeleton_graph_op(graph_op)
    if graph_op_name == "gcn":
        return GraphBlockGCN(dim=dim, num_joints=num_joints)
    return GraphBlock(dim=dim, num_heads=8, num_joints=num_joints, use_pyg=True)


class GraphEncoder(nn.Module):
    """Graph encoder mapping skeleton coordinates to joint-aware latent tokens."""

    def __init__(
        self,
        input_dim: int = SKELETON_FEATURE_DIM,
        latent_dim: int = 256,
        num_joints: int = 32,
        depth: int = 4,
        gait_metrics_dim: int = 0,
        use_gait_conditioning: bool = True,
        graph_op: str = "gat",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.graph_op_name = _normalize_skeleton_graph_op(graph_op)
        self.use_gait_conditioning = False  # disabled: gait metrics used only as eval/loss, not conditioning
        self.in_proj = nn.Linear(input_dim, latent_dim)
        if gait_metrics_dim > 0:
            self.gait_proj = nn.Sequential(
                nn.Linear(gait_metrics_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
        else:
            self.gait_proj = None
        self.graph_blocks = nn.ModuleList(
            [_build_skeleton_graph_block(self.graph_op_name, dim=latent_dim, num_joints=num_joints) for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList([TemporalConvBlock(dim=latent_dim) for _ in range(depth)])
        # Pre-register adjacency and edge_index as buffers so they are constructed
        # once at init and moved to the correct device with the model, instead of
        # being rebuilt on every forward call (saves ~8ms per forward on GPU).
        _adj = build_adjacency_matrix(num_joints=num_joints, device=torch.device("cpu"))
        self.register_buffer("_skel_adjacency", _adj, persistent=False)
        _ei = build_edge_index(num_joints, torch.device("cpu")) if HAS_TORCH_GEOMETRIC else None
        if _ei is not None:
            self.register_buffer("_skel_edge_index", _ei, persistent=False)
        else:
            self._skel_edge_index = None

    def forward(self, x: torch.Tensor, gait_metrics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode x with shape [B, T, J, 3] into z0 with shape [B, T, J, latent_dim]."""
        x = build_skeleton_features(x)  # [B, T, J, 3] → [B, T, J, 13]
        assert_shape(x, [None, None, self.num_joints, self.input_dim], "GraphEncoder.x")
        adjacency = self._skel_adjacency
        edge_index = self._skel_edge_index
        z = self.in_proj(x)
        if self.gait_proj is not None and self.use_gait_conditioning:
            if gait_metrics is None:
                raise ValueError("gait_metrics must be provided when gait_metrics_dim > 0")
            assert_shape(gait_metrics, [x.shape[0], self.gait_metrics_dim], "GraphEncoder.gait_metrics")
            gait_bias = self.gait_proj(gait_metrics).unsqueeze(1).unsqueeze(1)
            z = z + gait_bias
        for g_block, t_block in zip(self.graph_blocks, self.temporal_blocks):
            z = g_block(z, adjacency=adjacency, edge_index=edge_index)
            z = t_block(z)
        assert_shape(z, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "GraphEncoder.z")
        return z


class GraphEncoderGCN(GraphEncoder):
    """GCN-based skeleton encoder — drop-in replacement for GraphEncoder.

    Uses GCNConv (fixed degree-normalized aggregation) instead of GATConv
    (learned per-edge attention weights). Lighter and faster; useful for
    comparing whether attention is necessary for the skeleton latent space.
    Output shape is identical: [B, T, J, latent_dim].
    """

    def __init__(
        self,
        input_dim: int = SKELETON_FEATURE_DIM,
        latent_dim: int = 256,
        num_joints: int = 32,
        depth: int = 4,
        gait_metrics_dim: int = 0,
        use_gait_conditioning: bool = True,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_joints=num_joints,
            depth=depth,
            gait_metrics_dim=gait_metrics_dim,
            use_gait_conditioning=use_gait_conditioning,
            graph_op="gcn",
        )


class GraphDecoder(nn.Module):
    """Graph decoder mapping latent joint tokens back to 3D skeleton coordinates."""

    def __init__(
        self,
        latent_dim: int = 256,
        output_dim: int = 3,
        num_joints: int = 32,
        depth: int = 3,
        graph_op: str = "gat",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_joints = num_joints
        self.graph_op_name = _normalize_skeleton_graph_op(graph_op)
        self.graph_blocks = nn.ModuleList(
            [_build_skeleton_graph_block(self.graph_op_name, dim=latent_dim, num_joints=num_joints) for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList([TemporalConvBlock(dim=latent_dim) for _ in range(depth)])
        self.out_proj = nn.Linear(latent_dim, output_dim)
        _adj = build_adjacency_matrix(num_joints=num_joints, device=torch.device("cpu"))
        self.register_buffer("_skel_adjacency", _adj, persistent=False)
        _ei = build_edge_index(num_joints, torch.device("cpu")) if HAS_TORCH_GEOMETRIC else None
        if _ei is not None:
            self.register_buffer("_skel_edge_index", _ei, persistent=False)
        else:
            self._skel_edge_index = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode z with shape [B, T, J, latent_dim] to skeleton with shape [B, T, J, 3]."""
        assert_shape(z, [None, None, self.num_joints, self.latent_dim], "GraphDecoder.z")
        adjacency = self._skel_adjacency
        edge_index = self._skel_edge_index
        h = z
        for g_block, t_block in zip(self.graph_blocks, self.temporal_blocks):
            h = g_block(h, adjacency=adjacency, edge_index=edge_index)
            h = t_block(h)
        x_hat = self.out_proj(h)
        assert_shape(x_hat, [z.shape[0], z.shape[1], self.num_joints, self.output_dim], "GraphDecoder.x_hat")
        return x_hat


class GraphDecoderGCN(GraphDecoder):
    """GCN-based skeleton decoder — same temporal stack, GCN spatial blocks."""

    def __init__(self, latent_dim: int = 256, output_dim: int = 3, num_joints: int = 32, depth: int = 3) -> None:
        super().__init__(
            latent_dim=latent_dim,
            output_dim=output_dim,
            num_joints=num_joints,
            depth=depth,
            graph_op="gcn",
        )


class GraphDenoiserMasked(nn.Module):
    """Adjacency-masked graph denoiser with cross-attention sensor conditioning."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_joints: int = 32,
        depth: int = 6,
        gait_metrics_dim: int = 0,
        use_gait_conditioning: bool = True,
        graph_op: str = "gat",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.graph_op_name = _normalize_skeleton_graph_op(graph_op)
        self.use_gait_conditioning = False  # disabled: gait metrics used only as eval/loss, not conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.global_cond_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        if gait_metrics_dim > 0:
            self.gait_proj = nn.Sequential(
                nn.Linear(gait_metrics_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
        else:
            self.gait_proj = None
        self.blocks = nn.ModuleList(
            [_build_skeleton_graph_block(self.graph_op_name, dim=latent_dim, num_joints=num_joints) for _ in range(depth)]
        )
        self.cross_attn_blocks = nn.ModuleList([CrossAttentionBlock(dim=latent_dim, num_heads=8) for _ in range(depth)])
        self.temporal_blocks = nn.ModuleList([TemporalConvBlock(dim=latent_dim) for _ in range(depth)])
        self.out = nn.Linear(latent_dim, latent_dim)
        _adj = build_adjacency_matrix(num_joints=num_joints, device=torch.device("cpu"))
        self.register_buffer("_skel_adjacency", _adj, persistent=False)
        _ei = build_edge_index(num_joints, torch.device("cpu")) if HAS_TORCH_GEOMETRIC else None
        if _ei is not None:
            self.register_buffer("_skel_edge_index", _ei, persistent=False)
        else:
            self._skel_edge_index = None

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        sensor_tokens: Optional[torch.Tensor] = None,
        h_tokens: Optional[torch.Tensor] = None,
        h_global: Optional[torch.Tensor] = None,
        gait_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict epsilon for z_t with optional temporal sensor tokens [B,T,D]."""
        assert_shape(z_t, [None, None, self.num_joints, self.latent_dim], "GraphDenoiserMasked.z_t")
        assert_shape(t, [z_t.shape[0]], "GraphDenoiserMasked.t")
        if h_global is None and h is not None:
            h_global = h
        if h_tokens is None and sensor_tokens is not None:
            h_tokens = sensor_tokens
        if h_tokens is not None:
            assert_shape(h_tokens, [z_t.shape[0], z_t.shape[1], self.latent_dim], "GraphDenoiserMasked.h_tokens")
        if h_global is not None:
            assert_shape(h_global, [z_t.shape[0], self.latent_dim], "GraphDenoiserMasked.h_global")
        if self.gait_proj is not None and self.use_gait_conditioning:
            if gait_metrics is None:
                raise ValueError("gait_metrics must be provided when gait_metrics_dim > 0")
            assert_shape(gait_metrics, [z_t.shape[0], self.gait_metrics_dim], "GraphDenoiserMasked.gait_metrics")

        adjacency = self._skel_adjacency
        edge_index = self._skel_edge_index

        t_emb = sinusoidal_timestep_embedding(t, self.latent_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1).unsqueeze(1)
        x = z_t + t_emb

        gait_tokens = None
        if self.gait_proj is not None and self.use_gait_conditioning and gait_metrics is not None:
            gait_embed = self.gait_proj(gait_metrics)
            gait_tokens = gait_embed.unsqueeze(1).expand(-1, z_t.shape[1], -1)
            h_global = gait_embed if h_global is None else (h_global + gait_embed)
            h_tokens = gait_tokens if h_tokens is None else (h_tokens + gait_tokens)

        if h_global is not None:
            g = self.global_cond_proj(h_global).unsqueeze(1).unsqueeze(1)
            x = x + g

        for g_block, c_block, t_block in zip(self.blocks, self.cross_attn_blocks, self.temporal_blocks):
            x = g_block(x, adjacency=adjacency, edge_index=edge_index)
            if h_tokens is not None:
                x = c_block(x, h_tokens)
            x = t_block(x)

        eps = self.out(x)
        assert_shape(eps, [z_t.shape[0], z_t.shape[1], self.num_joints, self.latent_dim], "GraphDenoiserMasked.eps")
        return eps


class GraphDenoiserMaskedGCN(GraphDenoiserMasked):
    """GCN-based skeleton denoiser with the same conditioning and temporal path."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_joints: int = 32,
        depth: int = 6,
        gait_metrics_dim: int = 0,
        use_gait_conditioning: bool = True,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            num_joints=num_joints,
            depth=depth,
            gait_metrics_dim=gait_metrics_dim,
            use_gait_conditioning=use_gait_conditioning,
            graph_op="gcn",
        )
