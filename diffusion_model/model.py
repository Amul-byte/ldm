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

    def __init__(self, latent_dim: int = 256, num_joints: int = 32, timesteps: int = 500, gait_metrics_dim: int = 0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.encoder = GraphEncoder(input_dim=3, latent_dim=latent_dim, num_joints=num_joints, gait_metrics_dim=gait_metrics_dim)
        self.denoiser = GraphDenoiserMasked(latent_dim=latent_dim, num_joints=num_joints, gait_metrics_dim=gait_metrics_dim)
        self.decoder = GraphDecoder(latent_dim=latent_dim, output_dim=3, num_joints=num_joints)
        self.diffusion = DiffusionProcess(timesteps=timesteps)

    def forward(self, x: torch.Tensor, gait_metrics: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Compute stage-1 diffusion loss and latent outputs."""
        assert_shape(x, [None, None, self.num_joints, 3], "Stage1Model.x")
        if self.gait_metrics_dim > 0:
            assert gait_metrics is not None, "gait_metrics are required when gait_metrics_dim > 0"
            assert_shape(gait_metrics, [x.shape[0], self.gait_metrics_dim], "Stage1Model.gait_metrics")
        z0 = self.encoder(x, gait_metrics=gait_metrics)
        t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        loss_diff = self.diffusion.predict_noise_loss(
            self.denoiser,
            z0,
            t,
            h_tokens=None,
            h_global=None,
            gait_metrics=gait_metrics,
        )
        return {"loss_diff": loss_diff, "z0": z0}


class Stage2Model(nn.Module):
    """Stage 2 model: IMU to latent embedding alignment with frozen encoder."""

    def __init__(
        self,
        encoder: GraphEncoder,
        latent_dim: int = 256,
        num_joints: int = 32,
        gait_metrics_dim: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.encoder = encoder
        self.aligner = IMULatentAligner(latent_dim=latent_dim, gait_metrics_dim=0)
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze stage-1 encoder and verify all params are frozen."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        assert all(not p.requires_grad for p in self.encoder.parameters()), "encoder freeze verification failed"

    def forward(
        self,
        x: torch.Tensor,
        a_hip_stream: torch.Tensor,
        a_wrist_stream: torch.Tensor,
        gait_metrics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute alignment loss to frozen latent target z0."""
        assert_shape(x, [None, None, self.num_joints, 3], "Stage2Model.x")
        assert_shape(a_hip_stream, [x.shape[0], x.shape[1], 3], "Stage2Model.a_hip_stream")
        assert_shape(a_wrist_stream, [x.shape[0], x.shape[1], 3], "Stage2Model.a_wrist_stream")
        if self.gait_metrics_dim > 0:
            assert gait_metrics is not None, "gait_metrics are required when gait_metrics_dim > 0"
            assert_shape(gait_metrics, [x.shape[0], self.gait_metrics_dim], "Stage2Model.gait_metrics")

        with torch.no_grad():
            z0 = self.encoder(x, gait_metrics=gait_metrics)
        assert_shape(z0, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "Stage2Model.z0")

        h_tokens, h_global = self.aligner(
            a_hip_stream=a_hip_stream,
            a_wrist_stream=a_wrist_stream,
            gait_metrics=gait_metrics,
        )
        h = h_tokens.unsqueeze(2).expand(-1, -1, self.num_joints, -1)
        assert_shape(h, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "Stage2Model.h")
        loss_align = F.mse_loss(h, z0)
        return {
            "loss_align": loss_align,
            "h": h,
            "h_global": h_global,
            "h_tokens": h_tokens,
            "sensor_tokens": h_tokens,
            "z0_target": z0,
        }


class Stage3Model(nn.Module):
    """Stage 3 model: conditional latent diffusion and downstream classification."""

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
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.num_classes = num_classes
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
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse class labels into the temporal/global conditioning streams."""
        assert_shape(h_tokens, [None, None, self.latent_dim], "Stage3Model.condition_with_labels.h_tokens")
        assert_shape(h_global, [h_tokens.shape[0], self.latent_dim], "Stage3Model.condition_with_labels.h_global")
        assert_shape(y, [h_tokens.shape[0]], "Stage3Model.condition_with_labels.y")
        class_global = self.class_embed(y)
        class_tokens = class_global.unsqueeze(1).expand(-1, h_tokens.shape[1], -1)
        return h_tokens + class_tokens, h_global + class_global

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        h_tokens: torch.Tensor,
        h_global: torch.Tensor,
        gait_metrics: torch.Tensor | None = None,
        fps: float = 30.0,
    ) -> Dict[str, torch.Tensor]:
        """Run Stage 3 forward pass and return all loss-ready tensors."""
        assert_shape(x, [None, None, self.num_joints, 3], "Stage3Model.x")
        assert_shape(y, [x.shape[0]], "Stage3Model.y")

        assert_shape(h_tokens, [x.shape[0], x.shape[1], self.latent_dim], "Stage3Model.h_tokens")
        assert_shape(h_global, [x.shape[0], self.latent_dim], "Stage3Model.h_global")
        if self.gait_metrics_dim > 0:
            assert gait_metrics is not None, "gait_metrics are required when gait_metrics_dim > 0"
            assert_shape(gait_metrics, [x.shape[0], self.gait_metrics_dim], "Stage3Model.gait_metrics")
        cond_tokens, cond_global = self.condition_with_labels(h_tokens=h_tokens, h_global=h_global, y=y)

        with torch.no_grad():
            z0 = self.encoder(x, gait_metrics=gait_metrics)

        t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = self.diffusion.q_sample(z0=z0, t=t, noise=noise)
        pred_noise = self.denoiser(
            zt,
            t,
            sensor_tokens=cond_tokens,
            h_tokens=cond_tokens,
            h_global=cond_global,
            gait_metrics=gait_metrics,
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
        logits = self.classifier(x_hat)
        loss_diff = F.mse_loss(pred_noise, noise)
        loss_cls = F.cross_entropy(logits, y)
        loss_terms = motion_losses(x_hat.float())

        return {
            "x_hat": x_hat,
            "z0_gen": z0_gen,
            "gait_gen": gait_gen,
            "logits": logits,
            "pred_noise": pred_noise,
            "noise": noise,
            "loss_diff": loss_diff,
            "loss_cls": loss_cls,
            **loss_terms,
        }
