"""Π-GDM-style covariance-corrected likelihood for posterior samplers.

The naïve DPS / FlowDPS likelihood term `||y - A(x_hat_0)||_2` treats
the Tweedie clean-image estimate `x_hat_0` as a point estimate. The
proper posterior (Song et al. 2023 "Pi-GDM", Kawar et al. 2022 DDRM)
gives `x_hat_0` covariance `r_t · I` where

    r_t = (1 - alpha_bar_t) / alpha_bar_t       (diffusion DDPM)
    r_t = (1 - t)^2                              (rectified flow analog)

For a linear forward operator `A` with periodic Gaussian-blur kernel,
`A A^T` is diagonal in the Fourier basis with entries `|H(f)|^2`. The
proper likelihood under this covariance is a Gaussian with covariance
`sigma_n^2 I + r_t * A A^T`, which is diagonal in FFT basis:

    log p(y | x_hat_0) ∝ -sum_f |Y(f) - H(f) X(f)|^2 /
                                   (sigma_n^2 + r_t * |H(f)|^2)

The per-frequency weight is therefore `1 / (sigma_n^2 + r_t * |H(f)|^2)`.
We expose it as `pigdm_weight(H_otf, sigma_n, r_t)` so the sampler
loops can call it once per timestep.
"""
from __future__ import annotations

import math

import torch


def pigdm_weight(
    H_otf: torch.Tensor,
    sigma_n: float,
    r_t: float,
    normalize_mean: bool = True,
    r_t_cap: float = 1e3,
) -> torch.Tensor:
    """Per-frequency weight for the Tweedie-corrected likelihood.

    Args:
        H_otf: complex `(H, W)` OTF of the assumed forward operator.
        sigma_n: assumed measurement-noise std.
        r_t: Tweedie variance scale at the current timestep (see module
            docstring for diffusion vs. RF formulas).
        normalize_mean: rescale so the mean weight is 1; keeps the
            effective gradient magnitude comparable to the no-correction
            path so `zeta` doesn't need re-tuning.
        r_t_cap: clip very large `r_t` (happens near t=0 of diffusion's
            VP schedule where `(1 - alpha_bar_t) / alpha_bar_t -> infty`).
            DDRM's reference impl applies a similar guard.

    Returns:
        Real-valued `(H, W)` weight tensor. Sqrt of the inverse-variance
        weight (so that `||sqrt(W) * residual_fft||_2^2` equals the
        log-likelihood up to a constant).
    """
    r_t = float(min(r_t, r_t_cap))
    Hmag2 = (H_otf.abs() ** 2).real  # |H(f)|^2 in (H, W)
    inv_var = 1.0 / (sigma_n ** 2 + r_t * Hmag2 + 1e-12)
    W = torch.sqrt(inv_var)
    if normalize_mean:
        W = W / W.mean().clamp_min(1e-8)
    return W


def diffusion_r_t(alpha_bar_t: float) -> float:
    """r_t for a VP DDPM: `(1 - alpha_bar_t) / alpha_bar_t`. Clipped at
    1e3 by `pigdm_weight` when applied."""
    return (1.0 - alpha_bar_t) / max(alpha_bar_t, 1e-12)


def rf_r_t(t: float) -> float:
    """r_t for Rectified Flow: `(1 - t)^2`. (1 - t) is the noise-side
    coefficient in `x_t = (1 - t) z_0 + t z_1`."""
    return (1.0 - t) ** 2
