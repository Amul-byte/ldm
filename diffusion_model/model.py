"""Stage-specific models for the 3-stage joint-aware latent diffusion pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.diffusion import DiffusionProcess, extract
from diffusion_model.gait_metrics import compute_gait_metrics_torch
from diffusion_model.losses import motion_losses, supervised_contrastive_loss
from diffusion_model.sensor_model import IMULatentAligner
from diffusion_model.skeleton_model import GraphDecoder, GraphDenoiserMasked, GraphEncoder
from diffusion_model.util import assert_shape


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
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.gait_metrics_dim = gait_metrics_dim
        self.use_gait_conditioning = bool(use_gait_conditioning)
        self.encoder = GraphEncoder(
            input_dim=3,
            latent_dim=latent_dim,
            num_joints=num_joints,
            gait_metrics_dim=gait_metrics_dim,
            use_gait_conditioning=self.use_gait_conditioning,
        )
        self.denoiser = GraphDenoiserMasked(
            latent_dim=latent_dim,
            num_joints=num_joints,
            gait_metrics_dim=gait_metrics_dim,
            use_gait_conditioning=self.use_gait_conditioning,
        )
        self.decoder = GraphDecoder(latent_dim=latent_dim, output_dim=3, num_joints=num_joints)
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
        if self.gait_metrics_dim > 0 and self.use_gait_conditioning:
            assert gait_metrics is not None, "gait_metrics are required when gait_metrics_dim > 0"
            assert_shape(gait_metrics, [x.shape[0], self.gait_metrics_dim], "Stage1Model.gait_metrics")
        active_gait = gait_metrics if self.use_gait_conditioning else None
        z0 = self.encoder(x, gait_metrics=active_gait)
        t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        loss_diff = self.diffusion.predict_noise_loss(
            self.denoiser,
            z0,
            t,
            h_tokens=None,
            h_global=None,
            gait_metrics=active_gait,
        )

        # Pool z0 over T and J → [B, D] for classification and variance losses
        z_pooled = z0.mean(dim=(1, 2))

        # Classification loss: forces inter-class separation in latent space
        if y is not None:
            loss_cls = F.cross_entropy(self.cls_head(z_pooled), y)
        else:
            loss_cls = torch.zeros(1, device=x.device).squeeze()

        # Variance regulariser: penalise any latent dimension with var < 1
        # relu(1 - var) → 0 when spread out, positive when collapsed
        var_per_dim = z_pooled.var(dim=0)                        # [D]
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
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.num_classes = num_classes
        self.encoder = encoder
        self.aligner = IMULatentAligner(latent_dim=latent_dim, gait_metrics_dim=0, graph_type=imu_graph_type)
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
        assert_shape(z0, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "Stage2Model.z0")

        h_tokens, h_global = self.aligner(a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)
        h = h_tokens.unsqueeze(2).expand(-1, -1, self.num_joints, -1)
        assert_shape(h, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "Stage2Model.h")

        # Class-conditional alignment: target = mean z0 over all same-class samples in the batch.
        # Per-sample z0_mean encodes subject-specific skeleton geometry, causing the IMU encoder
        # to memorise subject identity (train subjects fit, val subjects don't).
        # Using the within-batch class mean removes subject-specific signal while still grounding
        # h_tokens in the skeleton latent space — the same direction SupCon pushes h_global.
        z0_joint_mean = z0.mean(dim=2).detach()    # [B, T, D]  joint-mean per sample
        if y is not None:
            # Build class-conditional target: for each sample, average z0 across same-class samples
            unique_classes = y.unique()
            z0_class_target = z0_joint_mean.clone()
            for c in unique_classes:
                mask = (y == c)                    # [B] bool
                z0_class_target[mask] = z0_joint_mean[mask].mean(dim=0, keepdim=True)
        else:
            z0_class_target = z0_joint_mean
        loss_align = F.mse_loss(h_tokens, z0_class_target)

        # SupCon loss — pulls same-activity IMU embeddings together across subjects
        if y is not None:
            h_norm = F.normalize(h_global, dim=-1)   # [B, D] — L2 normalise
            loss_supcon = supervised_contrastive_loss(h_norm, y)
        else:
            loss_supcon = torch.zeros((), device=x.device, dtype=x.dtype)

        return {
            "loss_align": loss_align,
            "loss_supcon": loss_supcon,
            "h": h,
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

    def forward(
        self,
        x: torch.Tensor,
        h_tokens: torch.Tensor,
        h_global: torch.Tensor,
        gait_target: torch.Tensor | None = None,
        fps: float = 30.0,
        y: torch.Tensor | None = None,
        gait_metrics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Run Stage 3 forward pass and return all loss-ready tensors."""
        assert_shape(x, [None, None, self.num_joints, 3], "Stage3Model.x")

        assert_shape(h_tokens, [x.shape[0], x.shape[1], self.latent_dim], "Stage3Model.h_tokens")
        assert_shape(h_global, [x.shape[0], self.latent_dim], "Stage3Model.h_global")
        if gait_target is None and gait_metrics is not None:
            gait_target = gait_metrics
        if gait_target is not None:
            assert_shape(gait_target, [x.shape[0], self.gait_metrics_dim], "Stage3Model.gait_target")

        with torch.no_grad():
            z0 = self.encoder(x, gait_metrics=gait_metrics if self.use_gait_conditioning else None)

        t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = self.diffusion.q_sample(z0=z0, t=t, noise=noise)
        pred_noise = self.denoiser(
            zt,
            t,
            sensor_tokens=h_tokens,
            h_tokens=h_tokens,
            h_global=h_global,
            gait_metrics=gait_metrics if self.use_gait_conditioning else None,
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
