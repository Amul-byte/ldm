"""Diffusion utilities for latent skeleton denoising."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.util import DEFAULT_TIMESTEPS, assert_shape


def linear_beta_schedule(timesteps: int = DEFAULT_TIMESTEPS, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Return linear beta schedule of shape [T]."""
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    assert_shape(betas, [timesteps], "linear_beta_schedule.betas")
    return betas


def extract(coeff: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract timestep coefficients and reshape for broadcast over latent tensors."""
    out = coeff.gather(0, t)
    out = out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))
    return out


class DiffusionProcess(nn.Module):
    """DDPM process with optional conditioning for latent denoising."""

    def __init__(self, timesteps: int = DEFAULT_TIMESTEPS) -> None:
        super().__init__()
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps=timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample forward process q(z_t | z_0)."""
        assert_shape(z0, [None, None, None, None], "DiffusionProcess.q_sample.z0")
        assert_shape(t, [z0.shape[0]], "DiffusionProcess.q_sample.t")
        if noise is None:
            noise = torch.randn_like(z0)
        assert noise.shape == z0.shape, "noise and z0 shape mismatch"
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, z0.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        zt = sqrt_alpha * z0 + sqrt_one_minus * noise
        assert zt.shape == z0.shape, "q_sample output shape mismatch"
        return zt

    def predict_noise_loss(
        self,
        denoiser: nn.Module,
        z0: torch.Tensor,
        t: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute epsilon prediction MSE loss for a denoiser."""
        assert_shape(z0, [None, None, None, None], "DiffusionProcess.predict_noise_loss.z0")
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0=z0, t=t, noise=noise)
        pred_noise = denoiser(zt, t, h)
        assert pred_noise.shape == noise.shape, "predicted noise shape mismatch"
        return F.mse_loss(pred_noise, noise)

    def p_sample(self, denoiser: nn.Module, zt: torch.Tensor, t: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample one reverse step p(z_{t-1}|z_t, h)."""
        assert_shape(zt, [None, None, None, None], "DiffusionProcess.p_sample.zt")
        assert_shape(t, [zt.shape[0]], "DiffusionProcess.p_sample.t")
        betas_t = extract(self.betas, t, zt.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, zt.shape)

        pred_noise = denoiser(zt, t, h)
        model_mean = sqrt_recip_alphas_t * (zt - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

        nonzero_mask = (t != 0).float().reshape(zt.shape[0], *([1] * (zt.ndim - 1)))
        posterior_var_t = extract(self.posterior_variance, t, zt.shape)
        noise = torch.randn_like(zt)
        z_prev = model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_var_t, min=1e-20)) * noise
        assert z_prev.shape == zt.shape, "p_sample output shape mismatch"
        return z_prev

    @torch.no_grad()
    def p_sample_loop(
        self,
        denoiser: nn.Module,
        shape: torch.Size,
        device: torch.device,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run full reverse process from z_T ~ N(0, I) to z_0."""
        assert len(shape) == 4, "shape must be [B,T,J,D]"
        z = torch.randn(shape, device=device)
        if h is not None:
            assert_shape(h, [shape[0], shape[3]], "DiffusionProcess.p_sample_loop.h")
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            z = self.p_sample(denoiser=denoiser, zt=z, t=t, h=h)
        assert_shape(z, [shape[0], shape[1], shape[2], shape[3]], "DiffusionProcess.p_sample_loop.z")
        return z
