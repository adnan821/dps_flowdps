"""Forward degradation operators for inverse-problem experiments.

All tensors are float, shape (B, C, H, W), value range [0, 1] unless stated.
Operators are differentiable so they can be used inside DPS / FlowDPS
likelihood gradients.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def gaussian_kernel_2d(
    sigma: float,
    kernel_size: Optional[int] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a normalized 2D Gaussian kernel of shape (1, 1, k, k)."""
    if kernel_size is None:
        # 6*sigma covers >99.7% mass; force odd.
        kernel_size = max(3, int(2 * math.ceil(3 * sigma) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def motion_blur_kernel(
    length: int,
    angle_deg: float,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a normalized motion-blur kernel of shape (1, 1, L, L).

    The kernel is a unit-width line of `length` pixels rotated by `angle_deg`
    inside an `L x L` canvas, then bilinearly sampled to be subpixel-accurate.
    """
    L = max(3, length | 1)  # force odd canvas size at least 3
    center = (L - 1) / 2

    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    kernel = torch.zeros(L, L, device=device, dtype=dtype)
    # Sample along the line at sub-pixel positions, then bilinearly splat.
    n_samples = length * 4
    for i in range(n_samples):
        t = (i / (n_samples - 1) - 0.5) * (length - 1)
        x = center + t * cos_a
        y = center + t * sin_a
        if not (0 <= x <= L - 1 and 0 <= y <= L - 1):
            continue
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, L - 1), min(y0 + 1, L - 1)
        dx, dy = x - x0, y - y0
        kernel[y0, x0] += (1 - dx) * (1 - dy)
        kernel[y0, x1] += dx * (1 - dy)
        kernel[y1, x0] += (1 - dx) * dy
        kernel[y1, x1] += dx * dy
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, L, L)


def _apply_kernel(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a single-channel kernel to each channel of x with reflect padding."""
    if x.dim() != 4:
        raise ValueError(f"expected (B,C,H,W), got {tuple(x.shape)}")
    C = x.shape[1]
    k = kernel.expand(C, 1, *kernel.shape[-2:]).to(device=x.device, dtype=x.dtype)
    pad = kernel.shape[-1] // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_padded, k, groups=C)


def gaussian_blur(
    x: torch.Tensor,
    sigma: float,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    kernel = gaussian_kernel_2d(sigma, kernel_size, device=x.device, dtype=x.dtype)
    return _apply_kernel(x, kernel)


def motion_blur(
    x: torch.Tensor,
    length: int,
    angle_deg: float,
) -> torch.Tensor:
    kernel = motion_blur_kernel(length, angle_deg, device=x.device, dtype=x.dtype)
    return _apply_kernel(x, kernel)


def sr_bicubic(
    x: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    """Bicubic downsampling super-resolution forward operator.

    Maps a high-res image x of shape (B, C, H, W) to its low-res
    measurement y of shape (B, C, H/factor, W/factor) via bilinear
    pre-filter (Gaussian sigma ~ factor/2 anti-aliasing) followed by
    bicubic interpolation. Differentiable so DPS likelihood gradient
    backpropagates through. The sampler's reconstruction lives in the
    original (H, W) image space; only the measurement y is low-res.
    """
    import torch.nn.functional as F
    # Anti-aliasing blur (sigma scales with the downsample ratio).
    anti_alias = gaussian_blur(x, sigma=max(0.5, factor / 2.0))
    return F.interpolate(
        anti_alias, scale_factor=1.0 / float(factor),
        mode="bicubic", align_corners=False, antialias=False,
    )


def add_noise(
    y: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if sigma == 0.0:
        return y
    noise = torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype)
    return y + sigma * noise


def degrade(
    x: torch.Tensor,
    blur_type: str = "gaussian",
    blur_sigma: float = 3.0,
    motion_length: int = 21,
    motion_angle_deg: float = 45.0,
    sr_factor: int = 4,
    noise_sigma: float = 0.05,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Apply forward model y = A x + eps.

    `blur_type` selects the operator; `seed` makes noise deterministic per call.
    Returns y on the same device/dtype as x.
    """
    if blur_type == "gaussian":
        y = gaussian_blur(x, blur_sigma)
    elif blur_type == "motion":
        y = motion_blur(x, motion_length, motion_angle_deg)
    elif blur_type == "sr":
        y = sr_bicubic(x, sr_factor)
    elif blur_type == "identity":
        y = x
    else:
        raise ValueError(f"unknown blur_type: {blur_type}")

    if noise_sigma > 0.0:
        gen = None
        if seed is not None:
            gen = torch.Generator(device=y.device).manual_seed(int(seed))
        y = add_noise(y, noise_sigma, generator=gen)
    return y
