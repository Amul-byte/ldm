"""Utility functions and constants for joint-aware latent diffusion."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_TIMESTEPS = 500
DEFAULT_LATENT_DIM = 256
DEFAULT_WINDOW = 90
DEFAULT_JOINTS = 32
DEFAULT_NUM_CLASSES = 14
DEFAULT_LAMBDA_CLS = 0.1
DEFAULT_FPS = 30.0
EPS = 1e-6

# Joint order provided from skeleton analysis notebook.
JOINT_LABELS: Tuple[str, ...] = (
    "PELVIS",
    "SPINE_NAVAL",
    "SPINE_CHEST",
    "NECK",
    "CLAVICLE_LEFT",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "HAND_LEFT",
    "HANDTIP_LEFT",
    "THUMB_LEFT",
    "CLAVICLE_RIGHT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HAND_RIGHT",
    "HANDTIP_RIGHT",
    "THUMB_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "HEAD",
    "NOSE",
    "EYE_LEFT",
    "EAR_LEFT",
    "EYE_RIGHT",
    "EAR_RIGHT",
)

_SKELETON_CONNECTIONS_BY_NAME: Tuple[Tuple[str, str], ...] = (
    ("PELVIS", "SPINE_NAVAL"),
    ("SPINE_NAVAL", "SPINE_CHEST"),
    ("SPINE_CHEST", "NECK"),
    ("NECK", "HEAD"),
    ("HEAD", "NOSE"),
    ("NOSE", "EYE_LEFT"),
    ("NOSE", "EYE_RIGHT"),
    ("EYE_LEFT", "EAR_LEFT"),
    ("EYE_RIGHT", "EAR_RIGHT"),
    ("NECK", "CLAVICLE_LEFT"),
    ("CLAVICLE_LEFT", "SHOULDER_LEFT"),
    ("SHOULDER_LEFT", "ELBOW_LEFT"),
    ("ELBOW_LEFT", "WRIST_LEFT"),
    ("WRIST_LEFT", "HAND_LEFT"),
    ("HAND_LEFT", "HANDTIP_LEFT"),
    ("WRIST_LEFT", "THUMB_LEFT"),
    ("NECK", "CLAVICLE_RIGHT"),
    ("CLAVICLE_RIGHT", "SHOULDER_RIGHT"),
    ("SHOULDER_RIGHT", "ELBOW_RIGHT"),
    ("ELBOW_RIGHT", "WRIST_RIGHT"),
    ("WRIST_RIGHT", "HAND_RIGHT"),
    ("HAND_RIGHT", "HANDTIP_RIGHT"),
    ("WRIST_RIGHT", "THUMB_RIGHT"),
    ("PELVIS", "HIP_LEFT"),
    ("HIP_LEFT", "KNEE_LEFT"),
    ("KNEE_LEFT", "ANKLE_LEFT"),
    ("ANKLE_LEFT", "FOOT_LEFT"),
    ("PELVIS", "HIP_RIGHT"),
    ("HIP_RIGHT", "KNEE_RIGHT"),
    ("KNEE_RIGHT", "ANKLE_RIGHT"),
    ("ANKLE_RIGHT", "FOOT_RIGHT"),
)

_JOINT_TO_INDEX = {label: idx for idx, label in enumerate(JOINT_LABELS)}
DEFAULT_BONES: Tuple[Tuple[int, int], ...] = tuple((_JOINT_TO_INDEX[a], _JOINT_TO_INDEX[b]) for a, b in _SKELETON_CONNECTIONS_BY_NAME)


@dataclass(frozen=True)
class JointIndexConfig:
    """Joint indices used by biomechanics-aware losses.

    The defaults follow the fixed 32-joint ordering in `JOINT_LABELS`.
    `left_foot_idx`/`right_foot_idx` are optional and fall back to ankles when absent.
    """

    left_ankle_idx: int
    right_ankle_idx: int
    left_foot_idx: Optional[int] = None
    right_foot_idx: Optional[int] = None
    pelvis_idx: Optional[int] = None
    torso_indices: Optional[Tuple[int, ...]] = None

    def resolved_left_foot_idx(self) -> int:
        """Return left foot index, with ankle fallback."""
        return self.left_foot_idx if self.left_foot_idx is not None else self.left_ankle_idx

    def resolved_right_foot_idx(self) -> int:
        """Return right foot index, with ankle fallback."""
        return self.right_foot_idx if self.right_foot_idx is not None else self.right_ankle_idx


@dataclass(frozen=True)
class InstabilityWeights:
    """Coefficients for instability curve terms.

    I(t) = a * margin + b * avgW(vel) + c * avgW(acc) + d * stdW(margin)
    """

    a: float = 1.0
    b: float = 0.35
    c: float = 0.20
    d: float = 0.30


DEFAULT_JOINT_CONFIG = JointIndexConfig(
    left_ankle_idx=_JOINT_TO_INDEX["ANKLE_LEFT"],
    right_ankle_idx=_JOINT_TO_INDEX["ANKLE_RIGHT"],
    left_foot_idx=_JOINT_TO_INDEX["FOOT_LEFT"],
    right_foot_idx=_JOINT_TO_INDEX["FOOT_RIGHT"],
    pelvis_idx=_JOINT_TO_INDEX["PELVIS"],
    torso_indices=(
        _JOINT_TO_INDEX["PELVIS"],
        _JOINT_TO_INDEX["SPINE_NAVAL"],
        _JOINT_TO_INDEX["SPINE_CHEST"],
        _JOINT_TO_INDEX["NECK"],
        _JOINT_TO_INDEX["HEAD"],
    ),
)


def make_joint_index_config(
    left_ankle_idx: Optional[int] = None,
    right_ankle_idx: Optional[int] = None,
    left_foot_idx: Optional[int] = None,
    right_foot_idx: Optional[int] = None,
    pelvis_idx: Optional[int] = None,
    torso_indices: Optional[Sequence[int]] = None,
) -> JointIndexConfig:
    """Build a joint config with optional CLI overrides."""
    base = DEFAULT_JOINT_CONFIG
    return JointIndexConfig(
        left_ankle_idx=base.left_ankle_idx if left_ankle_idx is None else int(left_ankle_idx),
        right_ankle_idx=base.right_ankle_idx if right_ankle_idx is None else int(right_ankle_idx),
        left_foot_idx=base.left_foot_idx if left_foot_idx is None else int(left_foot_idx),
        right_foot_idx=base.right_foot_idx if right_foot_idx is None else int(right_foot_idx),
        pelvis_idx=base.pelvis_idx if pelvis_idx is None else int(pelvis_idx),
        torso_indices=base.torso_indices if torso_indices is None else tuple(int(x) for x in torso_indices),
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_skeleton_edges() -> List[Tuple[int, int]]:
    """Return adjacency edges derived from the fixed JOINT_LABELS ordering."""
    label_to_index = {label: idx for idx, label in enumerate(JOINT_LABELS)}
    edges = [(label_to_index[a], label_to_index[b]) for a, b in _SKELETON_CONNECTIONS_BY_NAME]
    return edges


def get_joint_labels() -> Tuple[str, ...]:
    """Return the exact expected joint label order."""
    return JOINT_LABELS


def validate_joint_labels(labels: Sequence[str]) -> None:
    """Validate that dataset joint labels match the expected exact order."""
    expected = list(JOINT_LABELS)
    got = list(labels)
    assert got == expected, (
        "Joint label order mismatch.\n"
        f"Expected: {expected}\n"
        f"Got:      {got}"
    )


def build_adjacency_matrix(num_joints: int, device: torch.device | None = None) -> torch.Tensor:
    """Build a binary adjacency matrix with self-loops for the fixed skeleton graph."""
    edges = get_skeleton_edges()
    adj = torch.zeros((num_joints, num_joints), dtype=torch.float32, device=device)
    for i, j in edges:
        if i < num_joints and j < num_joints:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    adj.fill_diagonal_(1.0)
    return adj


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    assert timesteps.ndim == 1, "timesteps must have shape [B]"
    half_dim = dim // 2
    scale = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=emb.device)], dim=1)
    assert emb.shape == (timesteps.shape[0], dim), "invalid timestep embedding shape"
    return emb


def assert_shape(tensor: torch.Tensor, expected: Sequence[int | None], name: str) -> None:
    """Assert tensor rank and dimensions, where None means any size."""
    assert tensor.ndim == len(expected), f"{name} rank mismatch: got {tensor.ndim}, expected {len(expected)}"
    for idx, (actual, exp) in enumerate(zip(tensor.shape, expected)):
        if exp is not None:
            assert actual == exp, f"{name} dim {idx} mismatch: got {actual}, expected {exp}"


def _validate_joint_index(idx: int, num_joints: int, name: str) -> None:
    if idx < 0 or idx >= num_joints:
        raise ValueError(
            f"Invalid {name}={idx} for skeleton with J={num_joints}. "
            "Override via train.py flags such as --left-ankle-idx / --right-ankle-idx / --left-foot-idx / --right-foot-idx."
        )


def _resolve_joint_config(joint_config: Optional[JointIndexConfig], num_joints: int) -> JointIndexConfig:
    cfg = joint_config or DEFAULT_JOINT_CONFIG
    _validate_joint_index(cfg.left_ankle_idx, num_joints, "left_ankle_idx")
    _validate_joint_index(cfg.right_ankle_idx, num_joints, "right_ankle_idx")
    if cfg.left_foot_idx is not None:
        _validate_joint_index(cfg.left_foot_idx, num_joints, "left_foot_idx")
    if cfg.right_foot_idx is not None:
        _validate_joint_index(cfg.right_foot_idx, num_joints, "right_foot_idx")
    if cfg.pelvis_idx is not None:
        _validate_joint_index(cfg.pelvis_idx, num_joints, "pelvis_idx")
    if cfg.torso_indices is not None:
        for tidx in cfg.torso_indices:
            _validate_joint_index(tidx, num_joints, "torso_indices")
    return cfg


def infer_binary_fall_labels(y: torch.Tensor, fall_class_start: int = 9) -> torch.Tensor:
    """Convert labels to binary fall labels (0=non-fall, 1=fall).

    If labels are already binary, this is a no-op. Otherwise we assume 14-class
    activities and treat class ids >= `fall_class_start` as falls (default: A10-A14).
    """

    assert_shape(y, [None], "infer_binary_fall_labels.y")
    if y.numel() == 0:
        return y.float()
    if y.dtype.is_floating_point:
        yf = y.float()
        if float(yf.min().item()) >= 0.0 and float(yf.max().item()) <= 1.0:
            return yf
        return (yf >= float(fall_class_start)).float()
    yi = y.long()
    if int(yi.min().item()) >= 0 and int(yi.max().item()) <= 1:
        return yi.float()
    return (yi >= int(fall_class_start)).float()


def compute_com(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    joint_config: Optional[JointIndexConfig] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute center-of-mass proxy from skeleton sequence [B,T,J,3].

    Default weighting is normalized and intentionally simple:
    pelvis + torso joints are heavier; distal joints are lighter.
    """

    assert_shape(x, [None, None, None, 3], "compute_com.x")
    b, t, j, _ = x.shape
    cfg = _resolve_joint_config(joint_config, j)

    if weights is None:
        w = torch.ones(j, device=x.device, dtype=x.dtype)
        if cfg.torso_indices is not None:
            torso_idx = torch.tensor(list(cfg.torso_indices), device=x.device, dtype=torch.long)
            w.index_fill_(0, torso_idx, 1.5)
        if cfg.pelvis_idx is not None:
            w[cfg.pelvis_idx] = 2.0
        w = w / w.sum().clamp_min(eps)
    else:
        w = torch.as_tensor(weights, dtype=x.dtype, device=x.device)
        assert_shape(w, [j], "compute_com.weights")
        w = w / w.sum().clamp_min(eps)

    com = (x * w.view(1, 1, j, 1)).sum(dim=2)
    assert_shape(com, [b, t, 3], "compute_com.com")
    return com


def _select_base_of_support_joints(x: torch.Tensor, cfg: JointIndexConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    left_idx = cfg.resolved_left_foot_idx()
    right_idx = cfg.resolved_right_foot_idx()
    left = x[:, :, left_idx]
    right = x[:, :, right_idx]
    return left, right


def compute_bos_proxy(
    x: torch.Tensor,
    joint_config: Optional[JointIndexConfig] = None,
    alpha: float = 0.60,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute BoS proxy in x-z plane using feet (or ankles as fallback).

    Returns:
    - bos_center_xz: [B,T,2]
    - bos_radius: [B,T]
    """

    assert_shape(x, [None, None, None, 3], "compute_bos_proxy.x")
    b, t, j, _ = x.shape
    cfg = _resolve_joint_config(joint_config, j)
    left, right = _select_base_of_support_joints(x, cfg)
    left_xz = left[..., (0, 2)]
    right_xz = right[..., (0, 2)]
    bos_center_xz = 0.5 * (left_xz + right_xz)
    inter_foot = torch.norm(left_xz - right_xz, dim=-1)
    bos_radius = alpha * inter_foot + eps
    assert_shape(bos_center_xz, [b, t, 2], "compute_bos_proxy.bos_center_xz")
    assert_shape(bos_radius, [b, t], "compute_bos_proxy.bos_radius")
    return bos_center_xz, bos_radius


def stability_margin_soft(com: torch.Tensor, bos_center_xz: torch.Tensor, bos_radius: torch.Tensor) -> torch.Tensor:
    """Soft instability margin based on CoM-to-BoS distance in x-z plane."""

    assert_shape(com, [None, None, 3], "stability_margin_soft.com")
    assert_shape(bos_center_xz, [com.shape[0], com.shape[1], 2], "stability_margin_soft.bos_center_xz")
    assert_shape(bos_radius, [com.shape[0], com.shape[1]], "stability_margin_soft.bos_radius")
    com_xz = com[..., (0, 2)]
    dist = torch.norm(com_xz - bos_center_xz, dim=-1)
    margin = F.softplus(dist - bos_radius)
    assert_shape(margin, [com.shape[0], com.shape[1]], "stability_margin_soft.margin")
    return margin


def temporal_derivatives(com: torch.Tensor, dt: float, eps: float = EPS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute CoM speed and acceleration magnitudes with zero-padding at start."""

    assert_shape(com, [None, None, 3], "temporal_derivatives.com")
    safe_dt = max(float(dt), eps)
    vel_core = torch.norm(com[:, 1:] - com[:, :-1], dim=-1) / safe_dt
    vel = F.pad(vel_core, (1, 0), mode="constant", value=0.0)
    acc_core = torch.abs(vel[:, 1:] - vel[:, :-1]) / safe_dt
    acc = F.pad(acc_core, (1, 0), mode="constant", value=0.0)
    assert_shape(vel, [com.shape[0], com.shape[1]], "temporal_derivatives.vel")
    assert_shape(acc, [com.shape[0], com.shape[1]], "temporal_derivatives.acc")
    return vel, acc


def _avg_pool_same_1d(x: torch.Tensor, window_frames: int) -> torch.Tensor:
    """Apply AvgPool1d with approximately same-length output."""

    assert_shape(x, [None, None], "_avg_pool_same_1d.x")
    w = max(1, int(window_frames))
    if w % 2 == 0:
        w += 1
    if w == 1:
        return x
    x_1 = x.unsqueeze(1)
    pooled = F.avg_pool1d(x_1, kernel_size=w, stride=1, padding=w // 2, count_include_pad=False).squeeze(1)
    if pooled.shape[1] > x.shape[1]:
        pooled = pooled[:, : x.shape[1]]
    elif pooled.shape[1] < x.shape[1]:
        pad_count = x.shape[1] - pooled.shape[1]
        pad_tail = pooled[:, -1:].expand(-1, pad_count)
        pooled = torch.cat((pooled, pad_tail), dim=1)
    return pooled


def window_avg(x: torch.Tensor, window_frames: int) -> torch.Tensor:
    """Differentiable moving average over time for [B,T] inputs."""

    return _avg_pool_same_1d(x, window_frames=window_frames)


def window_std(x: torch.Tensor, window_frames: int, eps: float = EPS) -> torch.Tensor:
    """Differentiable moving std over time using pooled moments."""

    mean = _avg_pool_same_1d(x, window_frames=window_frames)
    second_moment = _avg_pool_same_1d(x * x, window_frames=window_frames)
    var = torch.clamp(second_moment - mean * mean, min=0.0)
    return torch.sqrt(var + eps)


def bone_length_loss(
    x: torch.Tensor,
    bones: Optional[Sequence[Tuple[int, int]]] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Penalize temporal variance of canonical bone lengths.

    Lower values enforce kinematic consistency over time.
    """

    assert_shape(x, [None, None, None, 3], "bone_length_loss.x")
    _, _, j, _ = x.shape
    bone_pairs = list(DEFAULT_BONES if bones is None else bones)
    if len(bone_pairs) == 0:
        raise ValueError("bone_length_loss requires at least one bone pair.")
    idx_a = torch.tensor([a for a, _ in bone_pairs], device=x.device, dtype=torch.long)
    idx_b = torch.tensor([b for _, b in bone_pairs], device=x.device, dtype=torch.long)
    if int(idx_a.min().item()) < 0 or int(idx_b.min().item()) < 0 or int(idx_a.max().item()) >= j or int(idx_b.max().item()) >= j:
        raise ValueError(
            f"Bone indices out of range for J={j}. "
            "If your joint order differs, override ankle/foot indices via CLI and pass a compatible bones list."
        )
    seg = x[:, :, idx_a] - x[:, :, idx_b]
    lengths = torch.sqrt((seg * seg).sum(dim=-1) + eps)
    length_var = lengths.var(dim=1, unbiased=False)
    return length_var.mean()


def _soft_contact_prob_from_foot(
    foot_xyz: torch.Tensor,
    fps: float,
    height_scale: float = 0.03,
    vel_threshold: float = 0.05,
    vel_scale: float = 0.03,
    detach_ground_height: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute soft foot contact probability [B,T] from height and vertical velocity."""

    assert_shape(foot_xyz, [None, None, 3], "_soft_contact_prob_from_foot.foot_xyz")
    dt = 1.0 / max(float(fps), eps)
    h = foot_xyz[..., 1]
    h0 = h.amin(dim=1, keepdim=True)
    if detach_ground_height:
        # Ground reference as a sequence-level anchor; detaching avoids unstable gradients through min().
        h0 = h0.detach()
    vy = (foot_xyz[:, 1:, 1] - foot_xyz[:, :-1, 1]) / dt
    vy = F.pad(vy, (1, 0), mode="constant", value=0.0)
    p_height = torch.sigmoid(-(h - h0) / max(height_scale, eps))
    p_vy = torch.sigmoid(-(vy.abs() - vel_threshold) / max(vel_scale, eps))
    p_contact = p_height * p_vy
    assert_shape(p_contact, [foot_xyz.shape[0], foot_xyz.shape[1]], "_soft_contact_prob_from_foot.p_contact")
    return p_contact


def soft_contact_prob(
    x: torch.Tensor,
    joint_config: Optional[JointIndexConfig] = None,
    fps: float = DEFAULT_FPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return soft contact probabilities for left/right foot, each [B,T]."""

    assert_shape(x, [None, None, None, 3], "soft_contact_prob.x")
    _, _, j, _ = x.shape
    cfg = _resolve_joint_config(joint_config, j)
    left, right = _select_base_of_support_joints(x, cfg)
    p_left = _soft_contact_prob_from_foot(left, fps=fps)
    p_right = _soft_contact_prob_from_foot(right, fps=fps)
    return p_left, p_right


def foot_skating_loss(
    x: torch.Tensor,
    contact_prob: Tuple[torch.Tensor, torch.Tensor],
    joint_config: Optional[JointIndexConfig] = None,
    fps: float = DEFAULT_FPS,
    eps: float = EPS,
) -> torch.Tensor:
    """Penalize horizontal foot motion when soft contact probability is high."""

    assert_shape(x, [None, None, None, 3], "foot_skating_loss.x")
    _, _, j, _ = x.shape
    cfg = _resolve_joint_config(joint_config, j)
    p_left, p_right = contact_prob
    assert_shape(p_left, [x.shape[0], x.shape[1]], "foot_skating_loss.p_left")
    assert_shape(p_right, [x.shape[0], x.shape[1]], "foot_skating_loss.p_right")

    left, right = _select_base_of_support_joints(x, cfg)
    dt = 1.0 / max(float(fps), eps)

    left_v = (left[:, 1:, (0, 2)] - left[:, :-1, (0, 2)]) / dt
    right_v = (right[:, 1:, (0, 2)] - right[:, :-1, (0, 2)]) / dt
    left_speed = F.pad(torch.norm(left_v, dim=-1), (1, 0), mode="constant", value=0.0)
    right_speed = F.pad(torch.norm(right_v, dim=-1), (1, 0), mode="constant", value=0.0)

    loss_left = (p_left * left_speed).mean()
    loss_right = (p_right * right_speed).mean()
    return 0.5 * (loss_left + loss_right)


def smoothness_loss(x: torch.Tensor, fps: float = DEFAULT_FPS, eps: float = EPS) -> torch.Tensor:
    """Small acceleration penalty to reduce frame-to-frame jitter."""

    assert_shape(x, [None, None, None, 3], "smoothness_loss.x")
    if x.shape[1] < 3:
        return x.new_zeros(())
    dt2 = (1.0 / max(float(fps), eps)) ** 2
    second_diff = (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]) / dt2
    acc_mag = torch.norm(second_diff, dim=-1)
    return acc_mag.mean()


def instability_curve(
    x: torch.Tensor,
    y_label: Optional[torch.Tensor] = None,
    fps: float = DEFAULT_FPS,
    window_sec: float = 3.0,
    joint_config: Optional[JointIndexConfig] = None,
    weights: Optional[InstabilityWeights | Mapping[str, float]] = None,
) -> torch.Tensor:
    """Compute instability energy curve I(t) from soft stability and dynamics."""

    del y_label  # Reserved for future label-conditional variants.
    assert_shape(x, [None, None, None, 3], "instability_curve.x")
    b, t, j, _ = x.shape
    cfg = _resolve_joint_config(joint_config, j)
    w = weights if isinstance(weights, InstabilityWeights) else InstabilityWeights(**(weights or {}))
    dt = 1.0 / max(float(fps), EPS)
    window_frames = max(1, int(round(float(window_sec) * float(fps))))

    com = compute_com(x, joint_config=cfg)
    bos_center_xz, bos_radius = compute_bos_proxy(x, joint_config=cfg)
    margin = stability_margin_soft(com, bos_center_xz, bos_radius)
    vel, acc = temporal_derivatives(com, dt=dt)

    avg_vel = window_avg(vel, window_frames=window_frames)
    avg_acc = window_avg(acc, window_frames=window_frames)
    std_margin = window_std(margin, window_frames=window_frames)

    instab = w.a * margin + w.b * avg_vel + w.c * avg_acc + w.d * std_margin
    assert_shape(instab, [b, t], "instability_curve.instab")
    return instab


def instability_loss(
    x_gen: torch.Tensor,
    x_gt: Optional[torch.Tensor],
    y: torch.Tensor,
    fps: float = DEFAULT_FPS,
    window_sec: float = 3.0,
    joint_config: Optional[JointIndexConfig] = None,
    weights: Optional[InstabilityWeights | Mapping[str, float]] = None,
) -> torch.Tensor:
    """Activity-aware instability loss.

    - Non-fall: penalize high generated instability mean.
    - Fall: align generated instability curve to GT curve (L1 over time).
    If GT is unavailable, the fall-alignment term is disabled and non-fall regularization is used for all samples.
    """

    assert_shape(x_gen, [None, None, None, 3], "instability_loss.x_gen")
    assert_shape(y, [x_gen.shape[0]], "instability_loss.y")
    y_bin = infer_binary_fall_labels(y).to(device=x_gen.device, dtype=x_gen.dtype)

    i_gen = instability_curve(
        x_gen,
        y_label=y_bin,
        fps=fps,
        window_sec=window_sec,
        joint_config=joint_config,
        weights=weights,
    )
    non_fall_term = i_gen.mean(dim=1)

    if x_gt is None:
        fall_term = non_fall_term
    else:
        assert_shape(x_gt, [x_gen.shape[0], x_gen.shape[1], x_gen.shape[2], 3], "instability_loss.x_gt")
        i_gt = instability_curve(
            x_gt.detach(),
            y_label=y_bin,
            fps=fps,
            window_sec=window_sec,
            joint_config=joint_config,
            weights=weights,
        )
        fall_term = torch.abs(i_gen - i_gt).mean(dim=1)

    per_sample = (1.0 - y_bin) * non_fall_term + y_bin * fall_term
    return per_sample.mean()


def run_biomech_sanity_checks(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Run two lightweight unit-like checks for the biomechanics losses."""

    dev = device or torch.device("cpu")
    cfg = DEFAULT_JOINT_CONFIG
    b, t, j = 2, 90, len(JOINT_LABELS)
    x_const = torch.zeros((b, t, j, 3), dtype=torch.float32, device=dev)
    x_const[:, :, cfg.resolved_left_foot_idx(), 0] = -0.2
    x_const[:, :, cfg.resolved_right_foot_idx(), 0] = 0.2
    x_const[:, :, cfg.pelvis_idx or 0, 1] = 0.9

    com_const = compute_com(x_const, joint_config=cfg)
    bos_center_const, bos_radius_const = compute_bos_proxy(x_const, joint_config=cfg)
    margin_const = stability_margin_soft(com_const, bos_center_const, bos_radius_const)
    vel_const, acc_const = temporal_derivatives(com_const, dt=1.0 / DEFAULT_FPS)
    smooth_const = smoothness_loss(x_const, fps=DEFAULT_FPS)
    instab_const = instability_curve(x_const, fps=DEFAULT_FPS, window_sec=3.0, joint_config=cfg)

    x_slide = x_const.clone()
    slide = torch.linspace(0.0, 0.5, t, device=dev, dtype=torch.float32).view(1, t)
    x_slide[:, :, cfg.resolved_left_foot_idx(), 0] += slide
    x_slide[:, :, cfg.resolved_right_foot_idx(), 0] += slide
    p_left, p_right = soft_contact_prob(x_slide, joint_config=cfg, fps=DEFAULT_FPS)
    skate_slide = foot_skating_loss(x_slide, (p_left, p_right), joint_config=cfg, fps=DEFAULT_FPS)
    skate_const = foot_skating_loss(
        x_const,
        soft_contact_prob(x_const, joint_config=cfg, fps=DEFAULT_FPS),
        joint_config=cfg,
        fps=DEFAULT_FPS,
    )

    const_stationary_ok = (
        float(vel_const.max().item()) < 1e-6
        and float(acc_const.max().item()) < 1e-6
        and float(smooth_const.item()) < 1e-6
        and float(instab_const.mean().item()) < 1.0
    )
    skate_response_ok = float(skate_slide.item()) > float(skate_const.item()) + 1e-4

    return {
        "const_vel_max": float(vel_const.max().item()),
        "const_acc_max": float(acc_const.max().item()),
        "const_smooth_loss": float(smooth_const.item()),
        "const_instability_mean": float(instab_const.mean().item()),
        "const_margin_min": float(margin_const.min().item()),
        "const_margin_max": float(margin_const.max().item()),
        "skating_loss_const": float(skate_const.item()),
        "skating_loss_slide": float(skate_slide.item()),
        "check_const_stationary_ok": float(const_stationary_ok),
        "check_skate_response_ok": float(skate_response_ok),
    }


if __name__ == "__main__":
    report = run_biomech_sanity_checks()
    print("[util] Biomech sanity checks:")
    for key, value in report.items():
        print(f"  {key}: {value:.6f}")
