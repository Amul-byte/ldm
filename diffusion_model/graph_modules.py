"""Graph modules for fixed-adjacency skeleton processing."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.util import assert_shape, build_adjacency_matrix

try:
    from torch_geometric.nn import GATConv

    HAS_TORCH_GEOMETRIC = True
except Exception:
    HAS_TORCH_GEOMETRIC = False


class MaskedGraphAttention(nn.Module):
    """Adjacency-masked multi-head attention over joints."""

    def __init__(self, dim: int, num_heads: int, num_joints: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_joints = num_joints
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply masked attention to x with shape [B, T, J, D]."""
        b, t, j, d = x.shape
        assert_shape(x, [None, None, self.num_joints, self.dim], "MaskedGraphAttention.x")
        assert_shape(adjacency, [self.num_joints, self.num_joints], "MaskedGraphAttention.adjacency")
        x_bt = x.reshape(b * t, j, d)
        attn_mask = (adjacency <= 0)
        y, _ = self.attn(x_bt, x_bt, x_bt, attn_mask=attn_mask)
        y = y.reshape(b, t, j, d)
        return y


class PyGGraphLayer(nn.Module):
    """Graph layer using torch_geometric GATConv when available."""

    def __init__(self, dim: int, num_joints: int, heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        self.gat = GATConv(in_channels=dim, out_channels=dim // heads, heads=heads, concat=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply graph convolution to x with shape [B, T, J, D]."""
        b, t, j, d = x.shape
        assert_shape(x, [None, None, self.num_joints, self.dim], "PyGGraphLayer.x")
        assert_shape(edge_index, [2, None], "PyGGraphLayer.edge_index")
        x_flat = x.reshape(b * t * j, d)
        offsets = torch.arange(0, b * t, device=x.device).repeat_interleave(edge_index.shape[1]) * j
        repeated_edge_index = edge_index.repeat(1, b * t) + offsets.unsqueeze(0)
        y = self.gat(x_flat, repeated_edge_index)
        y = y.reshape(b, t, j, d)
        return y


class GraphBlock(nn.Module):
    """Residual graph block with adjacency masking fallback."""

    def __init__(self, dim: int, num_heads: int, num_joints: int, use_pyg: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        self.use_pyg = bool(use_pyg and HAS_TORCH_GEOMETRIC)
        if self.use_pyg:
            self.graph_op = PyGGraphLayer(dim=dim, num_joints=num_joints, heads=num_heads)
            self.masked_attn = None
        else:
            self.masked_attn = MaskedGraphAttention(dim=dim, num_heads=num_heads, num_joints=num_joints)
            self.graph_op = None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run graph op then feed-forward with residual paths."""
        assert_shape(x, [None, None, self.num_joints, self.dim], "GraphBlock.x")
        assert_shape(adjacency, [self.num_joints, self.num_joints], "GraphBlock.adjacency")
        h = self.norm1(x)
        if self.use_pyg:
            assert edge_index is not None, "edge_index is required when torch_geometric path is used"
            y = self.graph_op(h, edge_index)
        else:
            y = self.masked_attn(h, adjacency)
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


class TemporalConvBlock(nn.Module):
    """Temporal convolution block applied per joint."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal convolution to x with shape [B, T, J, D]."""
        b, t, j, d = x.shape
        assert_shape(x, [None, None, None, self.dim], "TemporalConvBlock.x")
        x_perm = x.permute(0, 2, 3, 1).reshape(b * j, d, t)
        y = self.conv(x_perm)
        y = y.reshape(b, j, d, t).permute(0, 3, 1, 2)
        y = F.gelu(y)
        y = self.norm(y)
        return x + y


def build_edge_index(num_joints: int, device: torch.device) -> torch.Tensor:
    """Build edge_index tensor for torch_geometric from fixed adjacency."""
    adj = build_adjacency_matrix(num_joints=num_joints, device=device)
    idx = adj.nonzero(as_tuple=False).t().contiguous()
    assert_shape(idx, [2, None], "build_edge_index.idx")
    return idx
