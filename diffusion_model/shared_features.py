"""Shared acceleration-derived motion features for skeleton and IMU inputs."""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_model.util import assert_shape


SHARED_FEATURE_NAMES: tuple[str, ...] = (
    "ax",
    "ay",
    "az",
    "magnitude",
    "pitch",
    "roll",
    "jerk_x",
    "jerk_y",
    "jerk_z",
    "jerk_magnitude",
)
SHARED_FEATURE_DIM: int = len(SHARED_FEATURE_NAMES)


def build_shared_motion_features(accel_3d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build acceleration-derived motion features for [B, T, ..., 3] inputs."""
    if accel_3d.ndim < 3 or accel_3d.shape[-1] != 3:
        raise ValueError(
            f"build_shared_motion_features expects shape [B, T, ..., 3], got {tuple(accel_3d.shape)}"
        )

    ax = accel_3d[..., 0]
    ay = accel_3d[..., 1]
    az = accel_3d[..., 2]

    magnitude = torch.sqrt(torch.clamp(ax * ax + ay * ay + az * az, min=eps))
    pitch = torch.atan2(ax, torch.sqrt(torch.clamp(ay * ay + az * az, min=eps)))
    az_safe = torch.where(az.abs() < eps, torch.full_like(az, eps), az)
    roll = torch.atan2(ay, az_safe)

    if accel_3d.shape[1] > 1:
        jerk_valid = accel_3d[:, 1:] - accel_3d[:, :-1]
        jerk = torch.cat([jerk_valid[:, :1], jerk_valid], dim=1)
    else:
        jerk = torch.zeros_like(accel_3d)

    jerk_x = jerk[..., 0]
    jerk_y = jerk[..., 1]
    jerk_z = jerk[..., 2]
    jerk_magnitude = torch.sqrt(torch.clamp(jerk_x * jerk_x + jerk_y * jerk_y + jerk_z * jerk_z, min=eps))

    features = torch.stack(
        [ax, ay, az, magnitude, pitch, roll, jerk_x, jerk_y, jerk_z, jerk_magnitude],
        dim=-1,
    )

    if features.shape[:-1] != accel_3d.shape[:-1] or features.shape[-1] != SHARED_FEATURE_DIM:
        raise AssertionError(
            f"Shared feature shape mismatch: expected {accel_3d.shape[:-1]} + ({SHARED_FEATURE_DIM},), "
            f"got {tuple(features.shape)}"
        )
    return features


def compute_skeleton_acceleration(positions: torch.Tensor) -> torch.Tensor:
    """Compute a causal finite-difference acceleration tensor with exact [B, T, J, 3] output."""
    assert_shape(positions, [None, None, None, 3], "compute_skeleton_acceleration.positions")
    b, t, j, _ = positions.shape
    if t <= 1:
        accel = torch.zeros_like(positions)
        assert_shape(accel, [b, t, j, 3], "compute_skeleton_acceleration.accel")
        return accel

    vel_valid = positions[:, 1:] - positions[:, :-1]          # [B, T-1, J, 3]
    vel = torch.cat([vel_valid[:, :1], vel_valid], dim=1)     # [B, T,   J, 3]

    accel_valid = vel[:, 1:] - vel[:, :-1]                    # [B, T-1, J, 3]
    accel = torch.cat([accel_valid[:, :1], accel_valid], dim=1)  # [B, T, J, 3]

    assert_shape(accel, [b, t, j, 3], "compute_skeleton_acceleration.accel")
    return accel


class ModalityNorm(nn.Module):
    """Per-modality learned normalization: scale and shift each feature independently.

    Unlike z-score (which discards magnitude), this learns what scale/offset
    each modality needs so that skeleton-derived and IMU-derived features
    land in a comparable range for the shared layer.
    """

    def __init__(self, num_features: int = SHARED_FEATURE_DIM) -> None:
        super().__init__()
        self.num_features = num_features
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class SharedMotionLayer(nn.Module):
    """Shared projection applied to acceleration-derived features from both modalities.

    Each modality gets its own ``ModalityNorm`` (``skel_norm`` / ``imu_norm``)
    that learns to rescale its raw 10-dim features before the shared MLP.
    This compensates for the scale mismatch between skeleton-derived
    acceleration (gravity-free, small) and IMU acceleration (includes
    gravity, large).
    """

    def __init__(self, input_dim: int = SHARED_FEATURE_DIM, d_shared: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_shared = d_shared
        self.skel_norm = ModalityNorm(input_dim)
        self.imu_norm = ModalityNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_shared),
            nn.GELU(),
            nn.LayerNorm(d_shared),
            nn.Linear(d_shared, d_shared),
        )

    def forward(self, x: torch.Tensor, modality: str = "imu") -> torch.Tensor:
        """Project the last dimension of x while preserving all leading dimensions.

        Args:
            x: input tensor with last dim == input_dim.
            modality: ``"skel"`` or ``"imu"`` — selects the per-modality normalization.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"SharedMotionLayer expects last dim {self.input_dim}, got {x.shape[-1]}"
            )
        if modality == "skel":
            x = self.skel_norm(x)
        else:
            x = self.imu_norm(x)
        out = self.net(x)
        if out.shape[:-1] != x.shape[:-1] or out.shape[-1] != self.d_shared:
            raise AssertionError(
                f"SharedMotionLayer output shape mismatch: input={tuple(x.shape)} output={tuple(out.shape)}"
            )
        return out
