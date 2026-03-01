"""Stage-specific models for the 3-stage joint-aware latent diffusion pipeline."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.diffusion import DiffusionProcess
from diffusion_model.sensor_model import IMULatentAligner
from diffusion_model.skeleton_model import GraphDecoder, GraphDenoiserMasked, GraphEncoder
from diffusion_model.util import DEFAULT_LAMBDA_CLS, assert_shape


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

    def __init__(self, latent_dim: int = 256, num_joints: int = 32, timesteps: int = 500) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.encoder = GraphEncoder(input_dim=3, latent_dim=latent_dim, num_joints=num_joints)
        self.denoiser = GraphDenoiserMasked(latent_dim=latent_dim, num_joints=num_joints)
        self.decoder = GraphDecoder(latent_dim=latent_dim, output_dim=3, num_joints=num_joints)
        self.diffusion = DiffusionProcess(timesteps=timesteps)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute stage-1 diffusion loss and latent outputs."""
        assert_shape(x, [None, None, self.num_joints, 3], "Stage1Model.x")
        z0 = self.encoder(x)
        t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        loss_diff = self.diffusion.predict_noise_loss(self.denoiser, z0, t, h_tokens=None, h_global=None)
        return {"loss_diff": loss_diff, "z0": z0}


class Stage2Model(nn.Module):
    """Stage 2 model: IMU to latent global embedding alignment with frozen encoder."""

    def __init__(self, encoder: GraphEncoder, latent_dim: int = 256, num_joints: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.encoder = encoder
        self.aligner = IMULatentAligner(latent_dim=latent_dim)
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze stage-1 encoder and verify all params are frozen."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        assert all(not p.requires_grad for p in self.encoder.parameters()), "encoder freeze verification failed"

    def forward(self, x: torch.Tensor, a_hip_stream: torch.Tensor, a_wrist_stream: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute alignment loss to frozen latent target z0."""
        assert_shape(x, [None, None, self.num_joints, 3], "Stage2Model.x")
        assert_shape(a_hip_stream, [x.shape[0], x.shape[1], 3], "Stage2Model.a_hip_stream")
        assert_shape(a_wrist_stream, [x.shape[0], x.shape[1], 3], "Stage2Model.a_wrist_stream")

        with torch.no_grad():
            z0 = self.encoder(x)
        assert_shape(z0, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "Stage2Model.z0")

        h_global, sensor_tokens = self.aligner(a_hip_stream=a_hip_stream, a_wrist_stream=a_wrist_stream)
        h_latent = sensor_tokens.unsqueeze(2).expand(-1, -1, self.num_joints, -1)
        assert_shape(h_latent, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "Stage2Model.h_latent")
        loss_align = F.mse_loss(h_latent, z0)
        return {
            "loss_align": loss_align,
            "h_global": h_global,
            "h_tokens": sensor_tokens,
            "sensor_tokens": sensor_tokens,
            "h_latent": h_latent,
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
        lambda_cls: float = DEFAULT_LAMBDA_CLS,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.lambda_cls = lambda_cls
        self.encoder = encoder
        self.decoder = decoder
        self.denoiser = denoiser
        self.diffusion = DiffusionProcess(timesteps=timesteps)
        self.classifier = SkeletonTransformerClassifier(num_joints=num_joints, num_classes=num_classes, d_model=latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sensor_tokens: Optional[torch.Tensor] = None,
        h_tokens: Optional[torch.Tensor] = None,
        h_global: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute stage-3 loss with temporal sensor tokens [B,T,D] for cross-attention."""
        if h_tokens is None and sensor_tokens is not None:
            h_tokens = sensor_tokens
        assert_shape(x, [None, None, self.num_joints, 3], "Stage3Model.x")
        assert_shape(y, [x.shape[0]], "Stage3Model.y")
        if h_tokens is not None:
            assert_shape(h_tokens, [x.shape[0], x.shape[1], self.latent_dim], "Stage3Model.h_tokens")
        if h_global is not None:
            assert_shape(h_global, [x.shape[0], self.latent_dim], "Stage3Model.h_global")

        with torch.no_grad():
            z0 = self.encoder(x)

        t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        loss_diff = self.diffusion.predict_noise_loss(
            self.denoiser,
            z0,
            t,
            h_tokens=h_tokens,
            h_global=h_global,
        )

        z0_gen = self.diffusion.p_sample_loop(
            denoiser=self.denoiser,
            shape=z0.shape,
            device=x.device,
            h_tokens=h_tokens,
            h_global=h_global,
        )
        x_hat = self.decoder(z0_gen)
        logits = self.classifier(x_hat)
        loss_cls = F.cross_entropy(logits, y)
        loss_total = loss_diff + self.lambda_cls * loss_cls

        return {
            "loss_total": loss_total,
            "loss_diff": loss_diff,
            "loss_cls": loss_cls,
            "x_hat": x_hat,
            "logits": logits,
            "z0_gen": z0_gen,
        }
