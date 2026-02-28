"""Skeleton graph models for latent diffusion stages."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from diffusion_model.graph_modules import GraphBlock, build_edge_index, HAS_TORCH_GEOMETRIC, TemporalConvBlock
from diffusion_model.util import assert_shape, build_adjacency_matrix, sinusoidal_timestep_embedding


class GraphEncoder(nn.Module):
    """Graph encoder mapping skeleton coordinates to joint-aware latent tokens."""

    def __init__(self, input_dim: int = 3, latent_dim: int = 256, num_joints: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.in_proj = nn.Linear(input_dim, latent_dim)
        self.graph_blocks = nn.ModuleList(
            [GraphBlock(dim=latent_dim, num_heads=8, num_joints=num_joints, use_pyg=True) for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList([TemporalConvBlock(dim=latent_dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x with shape [B, T, J, 3] into z0 with shape [B, T, J, latent_dim]."""
        assert_shape(x, [None, None, self.num_joints, self.input_dim], "GraphEncoder.x")
        adjacency = build_adjacency_matrix(num_joints=self.num_joints, device=x.device)
        edge_index = build_edge_index(self.num_joints, x.device) if HAS_TORCH_GEOMETRIC else None
        z = self.in_proj(x)
        for g_block, t_block in zip(self.graph_blocks, self.temporal_blocks):
            z = g_block(z, adjacency=adjacency, edge_index=edge_index)
            z = t_block(z)
        assert_shape(z, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "GraphEncoder.z")
        return z


class GraphDecoder(nn.Module):
    """Graph decoder mapping latent joint tokens back to 3D skeleton coordinates."""

    def __init__(self, latent_dim: int = 256, output_dim: int = 3, num_joints: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_joints = num_joints
        self.graph_blocks = nn.ModuleList(
            [GraphBlock(dim=latent_dim, num_heads=8, num_joints=num_joints, use_pyg=True) for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList([TemporalConvBlock(dim=latent_dim) for _ in range(depth)])
        self.out_proj = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode z with shape [B, T, J, latent_dim] to skeleton with shape [B, T, J, 3]."""
        assert_shape(z, [None, None, self.num_joints, self.latent_dim], "GraphDecoder.z")
        adjacency = build_adjacency_matrix(num_joints=self.num_joints, device=z.device)
        edge_index = build_edge_index(self.num_joints, z.device) if HAS_TORCH_GEOMETRIC else None
        h = z
        for g_block, t_block in zip(self.graph_blocks, self.temporal_blocks):
            h = g_block(h, adjacency=adjacency, edge_index=edge_index)
            h = t_block(h)
        x_hat = self.out_proj(h)
        assert_shape(x_hat, [z.shape[0], z.shape[1], self.num_joints, self.output_dim], "GraphDecoder.x_hat")
        return x_hat


class GraphDenoiserMasked(nn.Module):
    """Adjacency-masked graph denoiser with optional FiLM conditioning."""

    def __init__(self, latent_dim: int = 256, num_joints: int = 32, depth: int = 6) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )
        self.blocks = nn.ModuleList(
            [GraphBlock(dim=latent_dim, num_heads=8, num_joints=num_joints, use_pyg=True) for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList([TemporalConvBlock(dim=latent_dim) for _ in range(depth)])
        self.out = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict epsilon for z_t using timestep t and optional global condition h."""
        assert_shape(z_t, [None, None, self.num_joints, self.latent_dim], "GraphDenoiserMasked.z_t")
        assert_shape(t, [z_t.shape[0]], "GraphDenoiserMasked.t")
        if h is not None:
            assert_shape(h, [z_t.shape[0], self.latent_dim], "GraphDenoiserMasked.h")

        adjacency = build_adjacency_matrix(num_joints=self.num_joints, device=z_t.device)
        edge_index = build_edge_index(self.num_joints, z_t.device) if HAS_TORCH_GEOMETRIC else None

        t_emb = sinusoidal_timestep_embedding(t, self.latent_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1).unsqueeze(1)
        x = z_t + t_emb

        if h is not None:
            film = self.cond_mlp(h)
            gamma, beta = film.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1).unsqueeze(1)
            beta = beta.unsqueeze(1).unsqueeze(1)
            x = x * (1.0 + gamma) + beta

        for g_block, t_block in zip(self.blocks, self.temporal_blocks):
            x = g_block(x, adjacency=adjacency, edge_index=edge_index)
            x = t_block(x)

        eps = self.out(x)
        assert_shape(eps, [z_t.shape[0], z_t.shape[1], self.num_joints, self.latent_dim], "GraphDenoiserMasked.eps")
        return eps
