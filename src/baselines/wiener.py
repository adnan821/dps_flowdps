"""Wiener filter — classical analytical baseline for Gaussian deblurring.

Operates in the Fourier domain: given y = A*x + n with A = Gaussian blur
of std sigma_b and n ~ N(0, sigma_n^2 I), the optimal linear MMSE estimate
under signal model |X|^2 ~ S is:

    X_hat(f) = conj(H(f)) / (|H(f)|^2 + sigma_n^2 / S(f)) * Y(f)

We approximate S as a flat spectrum (white-image prior) → the term becomes
a single scalar `K = sigma_n^2 / S` typically tuned per noise level.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _gaussian_kernel_np(sigma: float, size: int) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    ax = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _build_optical_transfer_function(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Place the kernel at the top-left, zero-pad, and FFT.

    `scipy.signal.fftconvolve`'s circular convolution convention requires the
    kernel to be top-left-centered before FFT for the deconvolution to match.
    """
    H, W = shape
    kh, kw = kernel.shape
    padded = np.zeros((H, W), dtype=np.float32)
    padded[:kh, :kw] = kernel
    # Center the kernel via fftshift inverse so the output isn't translated.
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return np.fft.fft2(padded)


def wiener_deblur(
    y: torch.Tensor,
    sigma_blur: float,
    K: float = 1e-3,
    kernel_size: int | None = None,
) -> torch.Tensor:
    """Apply Wiener deconvolution per channel.

    Args:
        y: (B, C, H, W) tensor in `[0, 1]` on CPU or CUDA.
        sigma_blur: Gaussian blur std the sampler should invert.
        K: noise-to-signal scalar regularization. Typical: 1e-4 for
           low-noise, 1e-2 for moderate, 1e-1 for high.
        kernel_size: spatial kernel size; defaults to `6*sigma + 1` (odd).

    Returns:
        x_hat: same shape/device/dtype as `y`, clamped to [0, 1].
    """
    if kernel_size is None:
        kernel_size = max(3, int(2 * math.ceil(3 * sigma_blur) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = _gaussian_kernel_np(sigma_blur, kernel_size)

    y_np = y.detach().cpu().numpy()  # (B, C, H, W)
    B, C, H, W = y_np.shape

    Hf = _build_optical_transfer_function(k, (H, W))  # (H, W) complex
    wiener_fr = np.conj(Hf) / (np.abs(Hf) ** 2 + K)

    out = np.empty_like(y_np)
    for b in range(B):
        for c in range(C):
            Y = np.fft.fft2(y_np[b, c])
            X_hat = wiener_fr * Y
            out[b, c] = np.real(np.fft.ifft2(X_hat))

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(out).to(device=y.device, dtype=y.dtype)
