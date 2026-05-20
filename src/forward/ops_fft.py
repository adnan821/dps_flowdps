"""Shared FFT / optical-transfer-function (OTF) utilities for circulant
forward operators.

For periodic-convolution forward models, an operator A becomes diagonal
in the Fourier basis with diagonal entries `H(f) = FFT(kernel)`. These
helpers exist so the Wiener filter (`src/baselines/wiener.py`),
PnP-ADMM (`src/baselines/pnp_admm.py`), DDRM (`src/baselines/ddrm.py`),
and the upcoming spectral / Pi-GDM samplers can all share one
implementation rather than duplicating it.

All kernels are normalized to sum to 1 (so the DC component of H is 1).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def gaussian_kernel_torch(
    sigma: float,
    size: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a normalized 2D Gaussian kernel of shape `(size, size)`.

    Mirrors `src/forward/degradation.py:gaussian_kernel_2d` but returns
    a `(size, size)` tensor (no leading `(1, 1)` batch dims) since OTF
    consumers want the bare spatial kernel.
    """
    if size is None:
        size = max(3, int(2 * math.ceil(3 * sigma) + 1))
    if size % 2 == 0:
        size += 1
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def otf(kernel: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Optical-transfer-function: shift the kernel to be top-left-centered
    and FFT to `shape`. The shift is the standard `np.fft.ifftshift`-style
    correction so the OTF has unit DC.
    """
    H, W = shape
    kh, kw = kernel.shape
    padded = torch.zeros(H, W, device=kernel.device, dtype=kernel.dtype)
    padded[:kh, :kw] = kernel
    padded = torch.roll(padded, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.fft2(padded)


def gaussian_otf(
    sigma: float,
    shape: Tuple[int, int],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convenience: kernel from `gaussian_kernel_torch` then `otf` to `shape`."""
    k = gaussian_kernel_torch(sigma, device=device, dtype=dtype)
    return otf(k, shape)


def motion_blur_kernel_torch(
    length: int,
    angle_deg: float,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Motion-blur kernel: a unit-width line of `length` pixels rotated
    by `angle_deg`, bilinearly splatted onto an `L x L` canvas. Mirrors
    `src/forward/degradation.py:motion_blur_kernel` but returns the bare
    `(L, L)` tensor (no batch dims)."""
    L = max(3, length | 1)  # force odd, at least 3
    center = (L - 1) / 2
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    kernel = torch.zeros(L, L, device=device, dtype=dtype)
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
    return kernel / kernel.sum()


def motion_blur_otf(
    length: int,
    angle_deg: float,
    shape: Tuple[int, int],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """OTF of a motion-blur kernel — the missing FFT counterpart to
    `src/forward/degradation.py:motion_blur`. Needed for spectral-aware
    guidance under operator mismatch."""
    k = motion_blur_kernel_torch(length, angle_deg, device=device, dtype=dtype)
    return otf(k, shape)
