"""Plug-and-Play ADMM for Gaussian deblurring, with DnCNN as the denoiser.

For y = A x + n with A = Gaussian blur (sigma_b) and n ~ N(0, sigma_n^2 I),
the ADMM splits the deconvolution objective into a data-fidelity step and
a denoising step:

    x ← arg min_x  ||y - A x||² + ρ ||x - (z - u)||²       # Tikhonov, FFT-solvable
    z ← Denoiser(x + u,  sigma = sqrt(sigma_y_inner/ρ))     # plug-and-play
    u ← u + x - z

With Gaussian blur the x-update is a single FFT-domain solve:
    x = IFFT[ (conj(H)·Y + ρ·FFT(z - u)) / (|H|² + ρ) ]

The denoiser sigma schedule descends across iterations to mimic a
multi-noise-level prior (Zhang 2017, Zhang 2021 DPIR convention).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch


def _gaussian_kernel_torch(sigma: float, size: int, device, dtype) -> torch.Tensor:
    if size % 2 == 0:
        size += 1
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _otf(kernel: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Optical-transfer-function: shift kernel to top-left and FFT2."""
    H, W = shape
    kh, kw = kernel.shape
    padded = torch.zeros(H, W, device=kernel.device, dtype=kernel.dtype)
    padded[:kh, :kw] = kernel
    padded = torch.roll(padded, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.fft2(padded)


def pnp_admm_deblur(
    y: torch.Tensor,
    sigma_blur: float,
    sigma_noise: float,
    denoiser: Callable[[torch.Tensor, float], torch.Tensor],
    num_iters: int = 20,
    rho: float = 0.1,
    sigma_schedule: tuple[float, float] = (0.10, 0.01),
    kernel_size: int | None = None,
) -> torch.Tensor:
    """Run Plug-and-Play ADMM with a generic denoiser.

    Args:
        y: degraded observation in `[0, 1]`, shape (B, C, H, W).
        sigma_blur: known Gaussian blur sigma used in A.
        sigma_noise: assumed measurement-noise std (not strictly used; included
            for API symmetry with DPS).
        denoiser: callable mapping `(noisy_img_in_[0,1], sigma)` → cleaner img.
        num_iters: ADMM outer iterations (20–30 typical).
        rho: ADMM penalty (~0.01–1.0). Larger = stronger coupling to denoiser.
        sigma_schedule: (sigma_max, sigma_min) — denoiser sigma linearly
            interpolated across iterations from max to min.
        kernel_size: spatial kernel size; defaults to `6·sigma_b + 1` (odd).

    Returns:
        Reconstruction in `[0, 1]`, same shape as `y`.
    """
    if kernel_size is None:
        kernel_size = max(3, int(2 * math.ceil(3 * sigma_blur) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1

    device, dtype = y.device, y.dtype
    B, C, H, W = y.shape

    # Build OTF (per-channel identical).
    k = _gaussian_kernel_torch(sigma_blur, kernel_size, device, dtype)
    Hf = _otf(k, (H, W))                # (H, W) complex
    Hf_conj = Hf.conj()
    Hf_mag2 = (Hf.abs() ** 2).real      # (H, W) real

    # FFT of y.
    Yf = torch.fft.fft2(y)              # (B, C, H, W) complex

    # State.
    x = y.clone()
    z = y.clone()
    u = torch.zeros_like(y)

    sigma_max, sigma_min = sigma_schedule
    for t in range(num_iters):
        # Linearly schedule denoiser sigma from sigma_max → sigma_min.
        alpha = t / max(num_iters - 1, 1)
        s_t = sigma_max * (1 - alpha) + sigma_min * alpha

        # ----- x-update: FFT-domain Tikhonov solve -----
        with torch.no_grad():
            ZUf = torch.fft.fft2(z - u)
            num = Hf_conj * Yf + rho * ZUf
            den = Hf_mag2 + rho
            x = torch.real(torch.fft.ifft2(num / den))

            # ----- z-update: denoiser proximal step -----
            z_in = (x + u).clamp(0.0, 1.0)
            z = denoiser(z_in, s_t).clamp(0.0, 1.0)

            # ----- dual update -----
            u = u + x - z

    return z.clamp(0.0, 1.0)
