"""Stage-specific models for the 3-stage joint-aware latent diffusion pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.diffusion import DiffusionProcess, extract
from diffusion_model.gait_metrics import compute_gait_metrics_torch
from diffusion_model.losses import motion_losses
from diffusion_model.shared_features import (
    SHARED_FEATURE_DIM,
    SharedMotionLayer,
    build_shared_motion_features,
    compute_skeleton_acceleration,
)
from diffusion_model.sensor_model import IMULatentAligner
from diffusion_model.skeleton_model import (
    GraphDecoder,
    GraphDecoderGCN,
    GraphDenoiserMasked,
    GraphDenoiserMaskedGCN,
    GraphEncoder,
    GraphEncoderGCN,
    SKELETON_FEATURE_DIM,
)
from diffusion_model.util import assert_shape


def _normalize_graph_op_name(graph_op: str | None, default: str = "gat") -> str:
    name = default if graph_op in {None, ""} else str(graph_op).lower()
    if name not in {"gat", "gcn"}:
        raise ValueError(f"Unsupported graph op: {graph_op}")
    return name


class SkeletonTransformerClassifier(nn.Module):
    """Transformer classifier over decoded skeleton trajectories."""

    def __init__(self, num_joints: int = 32, num_classes: int = 14, d_model: int = 256) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.d_model = d_model
        self.in_proj = nn.Linear(num_joints * 3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify x with shape [B, T, J, 3] and return logits [B, C]."""
        assert_shape(x, [None, None, self.num_joints, 3], "SkeletonTransformerClassifier.x")
        b, t, j, c = x.shape
        x_flat = x.reshape(b, t, j * c)
        h = self.in_proj(x_flat)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        logits = self.head(pooled)
        assert_shape(logits, [b, self.num_classes], "SkeletonTransformerClassifier.logits")
        return logits


class Stage1Model(nn.Module):
    """Stage 1 model: skeleton latent diffusion pre-training."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_joints: int = 32,
        timesteps: int = 500,
        gait_metrics_dim: int = 0,
        use_gait_conditioning: bool = True,
        num_classes: int = 14,
        encoder_type: str | None = None,
        skeleton_graph_op: str = "gat",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.gait_metrics_dim = gait_metrics_dim
        self.use_gait_conditioning = bool(use_gait_conditioning)
        self.skeleton_graph_op = _normalize_graph_op_name(skeleton_graph_op)
        self.encoder_graph_op = _normalize_graph_op_name(encoder_type, default=self.skeleton_graph_op)
        encoder_cls = GraphEncoderGCN if self.encoder_graph_op == "gcn" else GraphEncoder
        decoder_cls = GraphDecoderGCN if self.skeleton_graph_op == "gcn" else GraphDecoder
        denoiser_cls = GraphDenoiserMaskedGCN if self.skeleton_graph_op == "gcn" else GraphDenoiserMasked
        self.encoder = encoder_cls(
            input_dim=SKELETON_FEATURE_DIM,
            latent_dim=latent_dim,
            num_joints=num_joints,
            gait_metrics_dim=gait_metrics_dim,
            use_gait_conditioning=self.use_gait_conditioning,
        )
        self.denoiser = denoiser_cls(
            latent_dim=latent_dim,
            num_joints=num_joints,
            gait_metrics_dim=gait_metrics_dim,
            use_gait_conditioning=self.use_gait_conditioning,
        )
        self.decoder = decoder_cls(latent_dim=latent_dim, output_dim=3, num_joints=num_joints)
        self.diffusion = DiffusionProcess(timesteps=timesteps)
        # Auxiliary classification head — used only during Stage-1 training to
        # force discriminative latent structure; not used in Stage-2/3.
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        gait_metrics: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute stage-1 losses: diffusion + optional classification + variance regularizer."""
        assert_shape(x, [None, None, self.num_joints, 3], "Stage1Model.x")
        # Gait metrics no longer used as conditioning — only as eval/loss targets
        z0 = self.encoder(x, gait_metrics=None)
        t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        loss_diff = self.diffusion.predict_noise_loss(
            self.denoiser,
            z0,
            t,
            h_tokens=None,
            h_global=None,
            gait_metrics=None,
        )

        # Pool z0 over T and J → [B, D] for classification and variance losses
        z_pooled = z0.mean(dim=(1, 2))

        # Classification loss: CE on pooled encoder latents
        if y is not None:
            loss_cls = F.cross_entropy(self.cls_head(z_pooled), y)
        else:
            loss_cls = torch.zeros(1, device=x.device).squeeze()

        # Variance regulariser: penalise any latent dimension with var < 1
        # relu(1 - var) → 0 when spread out, positive when collapsed
        var_per_dim = z_pooled.var(dim=0,unbiased=False)                        # [D]
        loss_var = torch.relu(1.0 - var_per_dim).mean()

        return {
            "loss_diff": loss_diff,
            "loss_cls": loss_cls,
            "loss_var": loss_var,
            "z0": z0,
        }


class Stage2Model(nn.Module):
    """Stage 2 model: IMU encoder trained via gait-metric prediction.

    The Stage-1 encoder produces latents z0 that are non-discriminative
    (all samples cluster at cosine ~1.0) because the diffusion objective
    only requires invertibility, not discriminability.  Aligning h_tokens
    to z0 (MSE or InfoNCE) therefore provides no learning signal.

    Instead, Stage 2 trains the aligner with two supervised objectives:
      1. Gait-metric prediction: h_global → MLP → predicted gait metrics
         (walking speed, stride width, CoM variation, …).  These vary
         significantly across subjects and activities, giving a strong
         non-collapsing gradient.
      2. Latent projection: a learned linear head projects h_global into
         the z0 subspace so Stage 3 cross-attention is already warm-started.

    After Stage 2, h_global encodes motion-relevant information derived
    purely from IMU that is useful for conditioning the denoiser in Stage 3.
    """

    def __init__(
        self,
        encoder: GraphEncoder,
        latent_dim: int = 256,
        num_joints: int = 32,
        gait_metrics_dim: int = 0,
        num_classes: int = 14,
        imu_graph_type: str = "chain",
        d_shared: int = 64,
        stage2_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.num_classes = num_classes
        self.d_shared = d_shared
        self.stage2_dropout = float(stage2_dropout)
        self.encoder = encoder
        self.aligner = IMULatentAligner(
            latent_dim=latent_dim,
            gait_metrics_dim=0,
            graph_type=imu_graph_type,
            dropout=self.stage2_dropout,
        )
        self.shared_motion_layer = SharedMotionLayer(input_dim=SHARED_FEATURE_DIM, d_shared=d_shared)
        self.cls_head = nn.Linear(latent_dim, num_classes)
        # Gait-metric prediction head: the primary supervised signal for Stage 2.
        # Predicts the 9 gait metrics from h_global, giving a strong non-collapsing
        # gradient because gait metrics vary significantly across subjects/activities.
        self.gait_pred_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, gait_metrics_dim),
        ) if gait_metrics_dim > 0 else None
        self.use_gait_conditioning = False
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze stage-1 encoder, disable its gait conditioning, and verify."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        # Disable gait conditioning on the encoder: Stage 2 targets should be
        # derived from skeleton geometry only, not leak ground-truth gait labels
        # into z0 (which the aligner would then merely need to memorise).
        self.encoder.use_gait_conditioning = False
        assert all(not p.requires_grad for p in self.encoder.parameters()), "encoder freeze verification failed"

    def _encode_skeleton_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Project skeleton-derived acceleration features to the shared motion space."""
        skel_accel = compute_skeleton_acceleration(x)
        skel_shared = build_shared_motion_features(skel_accel)
        skel_embed = self.shared_motion_layer(skel_shared, modality="skel").mean(dim=2)
        assert_shape(skel_embed, [x.shape[0], x.shape[1], self.d_shared], "Stage2Model.skel_embed")
        return skel_embed

    def _encode_imu_shared(self, a_hip_stream: torch.Tensor, a_wrist_stream: torch.Tensor) -> torch.Tensor:
        """Project raw hip/wrist acceleration streams into the shared motion space."""
        hip_shared = build_shared_motion_features(a_hip_stream)
        wrist_shared = build_shared_motion_features(a_wrist_stream)
        hip_embed = self.shared_motion_layer(hip_shared, modality="imu")
        wrist_embed = self.shared_motion_layer(wrist_shared, modality="imu")
        imu_embed = 0.5 * (hip_embed + wrist_embed)
        assert_shape(
            imu_embed,
            [a_hip_stream.shape[0], a_hip_stream.shape[1], self.d_shared],
            "Stage2Model.imu_embed",
        )
        return imu_embed

    def forward(
        self,
        x: torch.Tensor,
        a_hip_stream: torch.Tensor,
        a_wrist_stream: torch.Tensor,
        gait_metrics: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute Stage-2 losses: activity classification + latent projection."""
        assert_shape(x, [None, None, self.num_joints, 3], "Stage2Model.x")
        assert_shape(a_hip_stream, [x.shape[0], x.shape[1], 3], "Stage2Model.a_hip_stream")
        assert_shape(a_wrist_stream, [x.shape[0], x.shape[1], 3], "Stage2Model.a_wrist_stream")

        with torch.no_grad():
            z0 = self.encoder(x, gait_metrics=None)
            skel_embed = self._encode_skeleton_shared(x)
        assert_shape(z0, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "Stage2Model.z0")

        h_tokens, h_global = self.aligner(a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)
        imu_embed = self._encode_imu_shared(a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)

        # MSE alignment: push h_global toward the pooled skeleton latent z0_global.
        # MSE is used instead of cosine because the gait_pred_head anchors h_global
        # to a meaningful magnitude scale, so magnitude collapse is no longer a risk.
        # MSE penalises both direction and magnitude, giving a stronger gradient signal.
        z0_global = z0.mean(dim=(1, 2)).detach()     # [B, D] — pool over T and J
        loss_align = F.mse_loss(h_global, z0_global)
        loss_feature = F.mse_loss(imu_embed, skel_embed)

        # CE classification — forces linear separability of 14 classes in h_global
        if y is not None:
            loss_cls = F.cross_entropy(self.cls_head(h_global), y)
        else:
            loss_cls = torch.zeros((), device=x.device, dtype=x.dtype)

        # Gait-metric prediction: the primary supervised signal that forces h_global
        # to encode motion-relevant information from IMU.  Unlike the cosine alignment
        # target (z0, which is non-discriminative), gait metrics vary strongly across
        # subjects and activities, providing a reliable gradient.
        if self.gait_pred_head is not None and gait_metrics is not None:
            loss_gait_pred = F.mse_loss(self.gait_pred_head(h_global), gait_metrics)
        else:
            loss_gait_pred = torch.zeros((), device=x.device, dtype=x.dtype)

        return {
            "loss_align": loss_align,
            "loss_feature": loss_feature,
            "loss_cls": loss_cls,
            "loss_gait_pred": loss_gait_pred,
            "h_global": h_global,
            "h_tokens": h_tokens,
            "sensor_tokens": h_tokens,
            "z0_target": z0,
        }


class Stage3Model(nn.Module):
    """Stage 3 model: IMU-conditioned latent diffusion with paired reconstruction losses."""

    def __init__(
        self,
        encoder: GraphEncoder,
        decoder: GraphDecoder,
        denoiser: GraphDenoiserMasked,
        latent_dim: int = 256,
        num_joints: int = 32,
        num_classes: int = 14,
        timesteps: int = 500,
        gait_metrics_dim: int = 0,
        use_gait_conditioning: bool = True,
        d_shared: int = 64,
        shared_motion_layer: SharedMotionLayer | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.num_classes = num_classes
        self.use_gait_conditioning = bool(use_gait_conditioning)
        self.encoder = encoder
        self.decoder = decoder
        self.denoiser = denoiser
        self.diffusion = DiffusionProcess(timesteps=timesteps)
        self.class_embed = nn.Embedding(num_classes, latent_dim)
        self.classifier = SkeletonTransformerClassifier(num_joints=num_joints, num_classes=num_classes, d_model=latent_dim)
        self.shared_motion_layer = shared_motion_layer or SharedMotionLayer(
            input_dim=SHARED_FEATURE_DIM,
            d_shared=d_shared,
        )
        self.d_shared = self.shared_motion_layer.d_shared
        self.shared_to_latent = nn.Linear(self.d_shared, latent_dim)
        nn.init.zeros_(self.shared_to_latent.weight)
        nn.init.zeros_(self.shared_to_latent.bias)

    def condition_with_labels(
        self,
        h_tokens: torch.Tensor,
        h_global: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return IMU conditioning unchanged.

        Labels are ignored in the one-to-one path. This compatibility method is
        retained so older scripts do not break while the repo transitions away
        from class-conditioned generation.
        """
        del y
        return h_tokens, h_global

    def augment_conditioning(
        self,
        h_tokens: torch.Tensor,
        h_global: torch.Tensor,
        a_hip_stream: torch.Tensor | None = None,
        a_wrist_stream: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment Stage-2 conditioning tokens with shared IMU motion features."""
        assert_shape(h_tokens, [None, None, self.latent_dim], "Stage3Model.augment_conditioning.h_tokens")
        assert_shape(h_global, [h_tokens.shape[0], self.latent_dim], "Stage3Model.augment_conditioning.h_global")
        if a_hip_stream is None or a_wrist_stream is None:
            return h_tokens, h_global

        assert_shape(
            a_hip_stream,
            [h_tokens.shape[0], h_tokens.shape[1], 3],
            "Stage3Model.augment_conditioning.a_hip_stream",
        )
        assert_shape(
            a_wrist_stream,
            [h_tokens.shape[0], h_tokens.shape[1], 3],
            "Stage3Model.augment_conditioning.a_wrist_stream",
        )
        hip_shared = build_shared_motion_features(a_hip_stream)
        wrist_shared = build_shared_motion_features(a_wrist_stream)
        hip_embed = self.shared_motion_layer(hip_shared, modality="imu")
        wrist_embed = self.shared_motion_layer(wrist_shared, modality="imu")
        shared_proj = self.shared_to_latent(0.5 * (hip_embed + wrist_embed))
        aug_tokens = h_tokens + shared_proj
        assert_shape(
            aug_tokens,
            [h_tokens.shape[0], h_tokens.shape[1], self.latent_dim],
            "Stage3Model.augment_conditioning.aug_tokens",
        )
        return aug_tokens, h_global

    def forward(
        self,
        x: torch.Tensor,
        h_tokens: torch.Tensor,
        h_global: torch.Tensor,
        gait_target: torch.Tensor | None = None,
        fps: float = 30.0,
        y: torch.Tensor | None = None,
        gait_metrics: torch.Tensor | None = None,
        a_hip_stream: torch.Tensor | None = None,
        a_wrist_stream: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Run Stage 3 forward pass and return all loss-ready tensors."""
        assert_shape(x, [None, None, self.num_joints, 3], "Stage3Model.x")

        assert_shape(h_tokens, [x.shape[0], x.shape[1], self.latent_dim], "Stage3Model.h_tokens")
        assert_shape(h_global, [x.shape[0], self.latent_dim], "Stage3Model.h_global")
        h_tokens, h_global = self.augment_conditioning(
            h_tokens=h_tokens,
            h_global=h_global,
            a_hip_stream=a_hip_stream,
            a_wrist_stream=a_wrist_stream,
        )
        if gait_target is None and gait_metrics is not None:
            gait_target = gait_metrics
        if gait_target is not None:
            assert_shape(gait_target, [x.shape[0], self.gait_metrics_dim], "Stage3Model.gait_target")

        with torch.no_grad():
            z0 = self.encoder(x, gait_metrics=None)

        t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = self.diffusion.q_sample(z0=z0, t=t, noise=noise)
        pred_noise = self.denoiser(
            zt,
            t,
            sensor_tokens=h_tokens,
            h_tokens=h_tokens,
            h_global=h_global,
            gait_metrics=None,
        )

        sqrt_alpha_bar_t = extract(self.diffusion.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.diffusion.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        z0_gen = (zt - sqrt_one_minus_alpha_bar_t * pred_noise) / torch.clamp(sqrt_alpha_bar_t, min=1e-20)

        amp_off_ctx = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if x.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with amp_off_ctx:
            x_hat = self.decoder(z0_gen.float())
        gait_gen = compute_gait_metrics_torch(x_hat.float(), fps=fps)
        loss_diff = F.mse_loss(pred_noise, noise)
        loss_latent = F.mse_loss(z0_gen, z0.detach())
        loss_pose = F.smooth_l1_loss(x_hat, x)
        loss_vel = F.smooth_l1_loss(x_hat[:, 1:] - x_hat[:, :-1], x[:, 1:] - x[:, :-1])
        loss_gait = F.mse_loss(gait_gen, gait_target) if gait_target is not None else torch.zeros((), device=x.device, dtype=x.dtype)
        loss_terms = motion_losses(x_hat.float())

        return {
            "x_hat": x_hat,
            "z0_gen": z0_gen,
            "z0_target": z0.detach(),
            "gait_gen": gait_gen,
            "pred_noise": pred_noise,
            "noise": noise,
            "loss_diff": loss_diff,
            "loss_latent": loss_latent,
            "loss_pose": loss_pose,
            "loss_vel": loss_vel,
            "loss_gait": loss_gait,
            **loss_terms,
        }
