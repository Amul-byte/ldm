"""Motion regularizers for generated skeleton trajectories."""

from __future__ import annotations

import torch

from diffusion_model.util import EPS, get_skeleton_edges


FOOT_JOINT_INDICES: tuple[int, ...] = (20, 21, 24, 25)
ANKLE_JOINT_INDICES: tuple[int, ...] = (20, 24)
PELVIS_INDEX = 0


def bone_length_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalize temporal bone-length drift within each generated sequence."""
    lengths = []
    for i, j in get_skeleton_edges():
        bone = torch.linalg.norm(x[:, :, i, :] - x[:, :, j, :], dim=-1)
        lengths.append(bone)
    bone_lengths = torch.stack(lengths, dim=-1)
    return bone_lengths.std(dim=1, unbiased=False).mean()


def foot_skating_loss(x: torch.Tensor, contact_threshold: float = 0.03) -> torch.Tensor:
    """Penalize horizontal foot motion when feet are close to the ground."""
    feet = x[:, :, FOOT_JOINT_INDICES, :]
    ground = feet[..., 2].amin(dim=1, keepdim=True)
    near_ground = (feet[..., 2] - ground) < contact_threshold
    horizontal_velocity = torch.linalg.norm(feet[:, 1:, :, :2] - feet[:, :-1, :, :2], dim=-1)
    contact_mask = near_ground[:, 1:, :].to(x.dtype)
    denom = contact_mask.sum().clamp_min(1.0)
    return (horizontal_velocity * contact_mask).sum() / denom


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalize large temporal accelerations across all joints."""
    velocity = x[:, 1:, :, :] - x[:, :-1, :, :]
    acceleration = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]
    return torch.mean(acceleration.pow(2))


def instability_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalize pelvis drift outside the ankle support width in the lateral axis."""
    pelvis_x = x[:, :, PELVIS_INDEX, 0]
    left_ankle_x = x[:, :, ANKLE_JOINT_INDICES[0], 0]
    right_ankle_x = x[:, :, ANKLE_JOINT_INDICES[1], 0]
    center = 0.5 * (left_ankle_x + right_ankle_x)
    support_half_width = 0.5 * (left_ankle_x - right_ankle_x).abs().clamp_min(EPS)
    overflow = torch.relu((pelvis_x - center).abs() - support_half_width)
    return overflow.mean()


def motion_losses(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute all motion regularizers and their summed loss."""
    loss_bone = bone_length_loss(x)
    loss_skate = foot_skating_loss(x)
    loss_smooth = smoothness_loss(x)
    loss_instab = instability_loss(x)
    loss_motion = loss_bone + loss_skate + loss_smooth + loss_instab
    return {
        "loss_bone": loss_bone,
        "loss_skate": loss_skate,
        "loss_smooth": loss_smooth,
        "loss_instab": loss_instab,
        "loss_motion": loss_motion,
    }
