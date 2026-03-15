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
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        # Kept for backward compatibility with older checkpoints that stored this buffer.
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

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
        sensor_tokens: Optional[torch.Tensor] = None,
        h_tokens: Optional[torch.Tensor] = None,
        h_global: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        gait_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute epsilon prediction MSE loss for a denoiser."""
        assert_shape(z0, [None, None, None, None], "DiffusionProcess.predict_noise_loss.z0")
        if h_global is None and h is not None:
            h_global = h
        if h_tokens is None and sensor_tokens is not None:
            h_tokens = sensor_tokens
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0=z0, t=t, noise=noise)
        pred_noise = denoiser(zt, t, sensor_tokens=h_tokens, h_tokens=h_tokens, h_global=h_global, gait_metrics=gait_metrics)
        assert pred_noise.shape == noise.shape, "predicted noise shape mismatch"
        return F.mse_loss(pred_noise, noise)

    def p_sample(
        self,
        denoiser: nn.Module,
        zt: torch.Tensor,
        t: torch.Tensor,
        sensor_tokens: Optional[torch.Tensor] = None,
        h_tokens: Optional[torch.Tensor] = None,
        h_global: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        gait_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample one reverse step p(z_{t-1}|z_t, h)."""
        assert_shape(zt, [None, None, None, None], "DiffusionProcess.p_sample.zt")
        assert_shape(t, [zt.shape[0]], "DiffusionProcess.p_sample.t")
        if h_global is None and h is not None:
            h_global = h
        if h_tokens is None and sensor_tokens is not None:
            h_tokens = sensor_tokens

        betas_t = extract(self.betas, t, zt.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, zt.shape)

        pred_noise = denoiser(zt, t, sensor_tokens=h_tokens, h_tokens=h_tokens, h_global=h_global, gait_metrics=gait_metrics)
        model_mean = sqrt_recip_alphas_t * (zt - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

        nonzero_mask = (t != 0).float().reshape(zt.shape[0], *([1] * (zt.ndim - 1)))
        noise = torch.randn_like(zt)
        sigma_t = torch.sqrt(torch.clamp(betas_t, min=1e-20))
        z_prev = model_mean + nonzero_mask * sigma_t * noise
        assert z_prev.shape == zt.shape, "p_sample output shape mismatch"
        return z_prev

    @torch.no_grad()
    def p_sample_loop(
        self,
        denoiser: nn.Module,
        shape: torch.Size,
        device: torch.device,
        sensor_tokens: Optional[torch.Tensor] = None,
        h_tokens: Optional[torch.Tensor] = None,
        h_global: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        gait_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run full reverse process from z_T ~ N(0, I) to z_0."""
        assert len(shape) == 4, "shape must be [B,T,J,D]"
        if h_global is None and h is not None:
            h_global = h
        if h_tokens is None and sensor_tokens is not None:
            h_tokens = sensor_tokens
        z = torch.randn(shape, device=device)

        if h_tokens is not None:
            assert_shape(h_tokens, [shape[0], shape[1], shape[3]], "DiffusionProcess.p_sample_loop.h_tokens")
        if h_global is not None:
            assert_shape(h_global, [shape[0], shape[3]], "DiffusionProcess.p_sample_loop.h_global")

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            z = self.p_sample(
                denoiser=denoiser,
                zt=z,
                t=t,
                sensor_tokens=h_tokens,
                h_tokens=h_tokens,
                h_global=h_global,
                gait_metrics=gait_metrics,
            )
        assert_shape(z, [shape[0], shape[1], shape[2], shape[3]], "DiffusionProcess.p_sample_loop.z")
        return z

    def _build_sampling_schedule(self, sample_steps: int, device: torch.device) -> torch.Tensor:
        """Build descending timestep schedule for accelerated sampling."""
        if sample_steps < 1 or sample_steps > self.timesteps:
            raise ValueError(f"sample_steps must be in [1, {self.timesteps}], got {sample_steps}.")
        if sample_steps == self.timesteps:
            return torch.arange(self.timesteps - 1, -1, -1, device=device, dtype=torch.long)
        times = torch.linspace(self.timesteps - 1, 0, sample_steps, device=device, dtype=torch.float32)
        schedule = torch.round(times).long()
        schedule = torch.unique_consecutive(schedule, dim=0)
        if schedule[0].item() != self.timesteps - 1:
            schedule = torch.cat([torch.tensor([self.timesteps - 1], device=device, dtype=torch.long), schedule], dim=0)
        if schedule[-1].item() != 0:
            schedule = torch.cat([schedule, torch.tensor([0], device=device, dtype=torch.long)], dim=0)
        return schedule

    @torch.no_grad()
    def p_sample_loop_ddim(
        self,
        denoiser: nn.Module,
        shape: torch.Size,
        device: torch.device,
        sample_steps: int = 50,
        eta: float = 0.0,
        sensor_tokens: Optional[torch.Tensor] = None,
        h_tokens: Optional[torch.Tensor] = None,
        h_global: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        gait_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run DDIM sampling with configurable number of reverse steps.

        `eta=0` gives deterministic DDIM sampling.
        """
        assert len(shape) == 4, "shape must be [B,T,J,D]"
        if eta < 0.0:
            raise ValueError(f"eta must be >= 0, got {eta}")
        if h_global is None and h is not None:
            h_global = h
        if h_tokens is None and sensor_tokens is not None:
            h_tokens = sensor_tokens
        z = torch.randn(shape, device=device)

        if h_tokens is not None:
            assert_shape(h_tokens, [shape[0], shape[1], shape[3]], "DiffusionProcess.p_sample_loop_ddim.h_tokens")
        if h_global is not None:
            assert_shape(h_global, [shape[0], shape[3]], "DiffusionProcess.p_sample_loop_ddim.h_global")

        schedule = self._build_sampling_schedule(sample_steps=sample_steps, device=device)
        next_schedule = torch.cat([schedule[1:], torch.tensor([-1], device=device, dtype=torch.long)], dim=0)

        for t_scalar, t_next_scalar in zip(schedule, next_schedule):
            t = torch.full((shape[0],), int(t_scalar.item()), device=device, dtype=torch.long)
            pred_noise = denoiser(z, t, sensor_tokens=h_tokens, h_tokens=h_tokens, h_global=h_global, gait_metrics=gait_metrics)

            alpha_bar_t = extract(self.alphas_cumprod, t, z.shape)
            sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-20))
            sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-20))
            x0_pred = (z - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t

            if int(t_next_scalar.item()) < 0:
                alpha_bar_next = torch.ones_like(alpha_bar_t)
            else:
                t_next = torch.full((shape[0],), int(t_next_scalar.item()), device=device, dtype=torch.long)
                alpha_bar_next = extract(self.alphas_cumprod, t_next, z.shape)

            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t), min=0.0)
                * torch.clamp(1.0 - alpha_bar_t / torch.clamp(alpha_bar_next, min=1e-20), min=0.0)
            )
            direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma * sigma, min=0.0)) * pred_noise
            if eta > 0.0 and int(t_next_scalar.item()) >= 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            z = torch.sqrt(torch.clamp(alpha_bar_next, min=1e-20)) * x0_pred + direction + sigma * noise

        assert_shape(z, [shape[0], shape[1], shape[2], shape[3]], "DiffusionProcess.p_sample_loop_ddim.z")
        return z
