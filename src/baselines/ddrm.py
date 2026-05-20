"""DDRM-style (Kawar et al. 2022) posterior sampling on a pretrained DDPM.

For a linear A (Gaussian blur with circular convolution → diagonal in FFT
basis), DDRM modifies the standard DDIM reverse step to project Tweedie's
clean-image estimate onto the measurement constraint, with the trade-off
between prior and measurement controlled by each spectral component's
singular value vs. the noise scale at timestep t.

This implementation is a deterministic, simplified DDRM:
- η_a = 0 (no extra noise in the measurement-dominated subspace)
- η_b = 1 (standard DDIM in the prior-dominated subspace)
- Spectral mask: components where |A_FFT| > thr(t) use the pseudo-inverse;
  others fall back to the DDPM prior estimate.

It captures the qualitative behavior of full DDRM (range-space projection,
spectral selectivity) without the full per-mode noise calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DDIMScheduler, UNet2DModel


def _gaussian_kernel_torch(sigma: float, size: int, device, dtype) -> torch.Tensor:
    if size % 2 == 0:
        size += 1
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _otf(kernel: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    H, W = shape
    kh, kw = kernel.shape
    padded = torch.zeros(H, W, device=kernel.device, dtype=kernel.dtype)
    padded[:kh, :kw] = kernel
    padded = torch.roll(padded, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.fft2(padded)


@dataclass
class DDRMResult:
    x_hat: torch.Tensor
    nfe: int
    elapsed_s: float


class DDRM:
    """DDRM-lite around a HuggingFace `diffusers` DDPM `UNet2DModel`.

    Args:
        unet: pretrained DDPM UNet, output is predicted noise.
        scheduler_config: DDPM scheduler config (`pipe.scheduler.config`).
        device: 'cuda' or 'cpu'.
    """

    def __init__(self, unet: UNet2DModel, scheduler_config, device="cuda"):
        self.device = torch.device(device)
        self.unet = unet.to(self.device).eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.scheduler = DDIMScheduler.from_config(scheduler_config)

    def sample(
        self,
        y: torch.Tensor,
        sigma_blur: float,
        sigma_noise: float = 0.05,
        num_steps: int = 100,
        eta: float = 0.85,
        seed: Optional[int] = 0,
    ) -> DDRMResult:
        """Run DDRM-lite to recover x from y = blur(x; sigma_blur) + n.

        Args:
            y: degraded observation in [0, 1], shape (B, C, H, W).
            sigma_blur: known Gaussian blur sigma (the inverse problem's A).
            sigma_noise: assumed measurement-noise std (sets spectral threshold).
            num_steps: number of DDIM steps (NFE).
            eta: convex combination weight between DDPM prior estimate and
                measurement pseudo-inverse for the well-conditioned subspace.
                eta=1.0 → fully trust DDPM prior; eta=0.0 → fully trust
                measurement pseudo-inverse. Typical: 0.85.
        """
        import time

        B, C, H, W = y.shape
        y = y.to(self.device)

        # Build the OTF for the blur. Periodic-convolution approximation; for
        # large kernels relative to image size this would matter, but for
        # sigma_b ≤ 5 the kernels (≤ 31 px) are much smaller than 256 px.
        kernel_size = max(3, int(2 * math.ceil(3 * sigma_blur) + 1))
        k = _gaussian_kernel_torch(sigma_blur, kernel_size, self.device, torch.float32)
        Hf = _otf(k, (H, W))
        Hf_conj = Hf.conj()
        Hf_mag = Hf.abs().real           # singular values
        Hf_mag2 = Hf_mag ** 2

        # Measurement in FFT basis. y is in [0,1] but the diffusion model
        # operates in [-1,1]; we'll work in [-1,1] internally and shift y too.
        y_dm = y * 2.0 - 1.0
        Yf = torch.fft.fft2(y_dm)

        # Initialize x_T ~ N(0, I) in [-1,1] space.
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn(y.shape, generator=gen, device=self.device, dtype=torch.float32)

        self.scheduler.set_timesteps(num_steps, device=self.device)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        t0 = time.time()

        # Tikhonov regularization λ. Smaller = trust measurement more;
        # larger = trust prior more. eta in [0,1] interpolates:
        # eta=0 → λ=0 (pure pseudo-inverse), eta=1 → λ=∞ (pure prior).
        # We use λ = max(σ_n², 1e-4) scaled by (1-η)/η-ish heuristic.
        lam_base = max(sigma_noise ** 2, 1e-4)
        lam = lam_base * (eta / max(1 - eta, 1e-3))

        timesteps = list(self.scheduler.timesteps)
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                eps = self.unet(x, t).sample
                alpha_bar = alphas_cumprod[t]
                # Tweedie's clean-image estimate (in [-1,1]).
                x_hat_0 = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()

                # Tikhonov-regularized projection in FFT basis:
                # X_proj = (conj(H)·Y + λ·X) / (|H|² + λ)
                # — interpolates pseudo-inverse (λ=0) to prior (λ=∞).
                Xf_prior = torch.fft.fft2(x_hat_0)
                Xf_proj = (Hf_conj * Yf + lam * Xf_prior) / (Hf_mag2 + lam)
                x_hat_0_proj = torch.real(torch.fft.ifft2(Xf_proj))

                # Manual DDIM step using the projected x_hat_0 and ORIGINAL eps
                # (so the noise-direction stays consistent with what the model
                # predicted, only the clean-image estimate is corrected).
                if i + 1 < len(timesteps):
                    t_prev = timesteps[i + 1]
                    alpha_bar_prev = alphas_cumprod[t_prev]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=self.device)
                x = alpha_bar_prev.sqrt() * x_hat_0_proj + (1 - alpha_bar_prev).sqrt() * eps

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        elapsed = time.time() - t0

        x_hat_pix = ((x.clamp(-1, 1) + 1) / 2).clamp(0, 1)
        return DDRMResult(x_hat=x_hat_pix, nfe=num_steps, elapsed_s=elapsed)
