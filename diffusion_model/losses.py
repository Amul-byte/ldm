"""Motion regularizers for generated skeleton trajectories."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from diffusion_model.util import EPS, get_joint_index, get_skeleton_edges


CONTACT_JOINT_INDICES: tuple[int, ...] = (get_joint_index("ANKLE_LEFT"), get_joint_index("ANKLE_RIGHT"))
ANKLE_JOINT_INDICES: tuple[int, ...] = CONTACT_JOINT_INDICES
PELVIS_INDEX = get_joint_index("PELVIS")
ANGLE_TRIPLETS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("left_knee", (get_joint_index("HIP_LEFT"), get_joint_index("KNEE_LEFT"), get_joint_index("ANKLE_LEFT"))),
    ("right_knee", (get_joint_index("HIP_RIGHT"), get_joint_index("KNEE_RIGHT"), get_joint_index("ANKLE_RIGHT"))),
    ("left_elbow", (get_joint_index("SHOULDER_LEFT"), get_joint_index("ELBOW_LEFT"), get_joint_index("WRIST_LEFT"))),
    ("right_elbow", (get_joint_index("SHOULDER_RIGHT"), get_joint_index("ELBOW_RIGHT"), get_joint_index("WRIST_RIGHT"))),
)
ANGLE_LIMITS_RAD: dict[str, tuple[float, float]] = {
    # Soft ranges: permissive enough for the current dataset, but still penalize
    # degenerate spider-like hyper-folded or locked poses.
    "left_knee": (math.radians(15.0), math.radians(175.0)),
    "right_knee": (math.radians(15.0), math.radians(175.0)),
    "left_elbow": (math.radians(10.0), math.radians(175.0)),
    "right_elbow": (math.radians(10.0), math.radians(175.0)),
}


def joint_angles(x: torch.Tensor) -> torch.Tensor:
    """Return key hinge-joint angles [B, T, K] in radians from skeletons [B, T, J, 3]."""
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"joint_angles expects [B, T, J, 3], got {tuple(x.shape)}")
    angles = []
    for _, (a, b, c) in ANGLE_TRIPLETS:
        v1 = x[:, :, a, :] - x[:, :, b, :]
        v2 = x[:, :, c, :] - x[:, :, b, :]
        denom = torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1)
        cos = (v1 * v2).sum(dim=-1) / denom.clamp_min(EPS)
        angles.append(torch.acos(cos.clamp(-1.0, 1.0)))
    return torch.stack(angles, dim=-1)


def joint_angle_limit_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalize key joints when their angles leave permissive human ranges."""
    angles = joint_angles(x)
    lower = torch.tensor(
        [ANGLE_LIMITS_RAD[name][0] for name, _ in ANGLE_TRIPLETS],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, -1)
    upper = torch.tensor(
        [ANGLE_LIMITS_RAD[name][1] for name, _ in ANGLE_TRIPLETS],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, -1)
    below = F.relu(lower - angles)
    above = F.relu(angles - upper)
    return (below + above).mean()


def angular_reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Match generated and target key-joint angles."""
    return F.smooth_l1_loss(joint_angles(x_hat), joint_angles(x))


def angular_velocity_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Match temporal changes in key-joint angles."""
    angle_hat = joint_angles(x_hat)
    angle_real = joint_angles(x)
    if angle_hat.shape[1] <= 1:
        return torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
    vel_hat = angle_hat[:, 1:] - angle_hat[:, :-1]
    vel_real = angle_real[:, 1:] - angle_real[:, :-1]
    return F.smooth_l1_loss(vel_hat, vel_real)


def bone_length_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalize temporal bone-length drift within each generated sequence."""
    lengths = []
    for i, j in get_skeleton_edges():
        bone = torch.linalg.norm(x[:, :, i, :] - x[:, :, j, :], dim=-1)
        lengths.append(bone)
    bone_lengths = torch.stack(lengths, dim=-1)
    return bone_lengths.std(dim=1, unbiased=False).mean()


def foot_skating_loss(x: torch.Tensor, contact_threshold: float = 0.03) -> torch.Tensor:
    """Penalize horizontal ankle motion when ankle contacts are close to the ground."""
    contacts = x[:, :, CONTACT_JOINT_INDICES, :]
    ground = contacts[..., 2].amin(dim=1, keepdim=True)
    near_ground = (contacts[..., 2] - ground) < contact_threshold
    horizontal_velocity = torch.linalg.norm(contacts[:, 1:, :, :2] - contacts[:, :-1, :, :2], dim=-1)
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


def motion_losses(x: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Compute motion regularizers and optional anatomy-aware supervision."""
    loss_bone = bone_length_loss(x)
    loss_skate = foot_skating_loss(x)
    loss_smooth = smoothness_loss(x)
    loss_instab = instability_loss(x)
    loss_motion = loss_bone + loss_skate + loss_smooth + loss_instab
    loss_angle_limit = joint_angle_limit_loss(x)
    if target is not None:
        loss_angle_recon = angular_reconstruction_loss(x, target)
        loss_angvel = angular_velocity_loss(x, target)
    else:
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        loss_angle_recon = zero
        loss_angvel = zero
    loss_angle = loss_angle_recon + loss_angle_limit
    return {
        "loss_bone": loss_bone,
        "loss_skate": loss_skate,
        "loss_smooth": loss_smooth,
        "loss_instab": loss_instab,
        "loss_motion": loss_motion,
        "loss_angle_limit": loss_angle_limit,
        "loss_angle_recon": loss_angle_recon,
        "loss_angle": loss_angle,
        "loss_angvel": loss_angvel,
    }
