"""Frequency-adaptive likelihood-gradient weighting for posterior samplers.

Both `PixelDPS` and `FlowDPSRF` compute a likelihood loss
`||y - A(x_hat_0)||_2` and backpropagate the resulting gradient. With
fixed `zeta`, the gradient is amplified by the same scalar at every
spatial frequency. But for circulant linear operators (Gaussian blur,
motion blur), the operator's response `|H(f)|` varies drastically with
`f`: near-zero at high frequencies and exactly 1 at DC. Under operator
mismatch (true = motion blur, assumed = Gaussian), the spectra of
`H_true(f)` and `H_assumed(f)` disagree in particular at the
near-zero frequencies of the true operator — exactly where the
likelihood gradient is most noisy and least informative.

This module provides per-frequency weights `W(f)` that can be applied
to the residual `y_hat - y` in Fourier space before the L2 norm is
computed. The default heuristic `noise_floor_weight` downweights
frequencies where the assumed operator's spectral magnitude is below
a noise threshold; these frequencies are dominated by measurement
noise (under both the assumed and the true model) and their gradient
contribution is mostly noise amplification.

Default value `spectral_weight=None` in the samplers reproduces the
pre-Tier-B behavior bit-for-bit (see tests/test_sampler_backcompat.py).
"""
from __future__ import annotations

import torch


def noise_floor_weight(
    H_otf: torch.Tensor,
    eps: float = 0.05,
    alpha: float = 10.0,
    normalize_mean: bool = True,
) -> torch.Tensor:
    """A frequency-adaptive weight that smoothly downweights frequencies
    where the assumed operator's magnitude is below a noise floor.

    `W(f) = 1 / (1 + alpha * relu(eps - |H(f)|))`.

    For frequencies with `|H(f)| >= eps` (well-conditioned subspace),
    `W(f) = 1` (no change). For frequencies with `|H(f)| < eps`,
    `W(f)` falls smoothly toward `1 / (1 + alpha * eps)` — i.e. those
    components contribute less to the likelihood gradient.

    Args:
        H_otf: complex `(H, W)` optical-transfer-function (e.g. from
            `src.forward.ops_fft.gaussian_otf`).
        eps: threshold below which we consider the operator
            "ill-conditioned". `0.05` works empirically for Gaussian
            blur at the project's sigma range.
        alpha: steepness of the downweighting. Higher = more aggressive.
        normalize_mean: if True, rescale `W` so its mean is 1.0. This
            keeps the effective gradient magnitude comparable to the
            no-weighting baseline so `zeta` doesn't need re-tuning.

    Returns:
        Real-valued `(H, W)` tensor of weights in `(0, 1]`.
    """
    Hmag = H_otf.abs().real
    deficit = torch.clamp(eps - Hmag, min=0.0)  # 0 where well-conditioned
    W = 1.0 / (1.0 + alpha * deficit)
    if normalize_mean:
        W = W / W.mean().clamp_min(1e-8)
    return W


def spectral_residual_l2(
    residual: torch.Tensor,
    W: torch.Tensor | None,
) -> torch.Tensor:
    """Compute `||IFFT(W * FFT(residual))||_2` per batch element, summed.

    If `W is None`, falls back to the spatial-domain `||residual||_2`,
    bit-identical to the pre-Tier-B path (note: torch.fft is unitary
    in the `norm="ortho"` mode, but we don't use ortho-norm; with
    `norm="backward"` (default) FFT and IFFT both have a 1/(H*W)
    factor folded in such that `IFFT(FFT(x)) == x`, so applying
    `W = 1` everywhere yields `IFFT(FFT(residual)) == residual`,
    matching the spatial-domain norm).

    Returns a scalar.
    """
    if W is None:
        return torch.linalg.norm(residual.flatten(1), dim=1).sum()
    # FFT, multiply by W (broadcast over batch+channels), IFFT, then
    # L2 norm in spatial domain. We do the L2 norm AFTER IFFT (not in
    # FFT-domain via Parseval) because IFFT loses the imaginary part
    # we don't care about (residual is real), and we want autograd to
    # see a real-valued residual for stable backward.
    R = torch.fft.fft2(residual)  # (B, C, H, W) complex
    W_b = W.to(R.device, dtype=R.dtype)  # broadcast
    R_w = R * W_b
    r_weighted = torch.real(torch.fft.ifft2(R_w))
    return torch.linalg.norm(r_weighted.flatten(1), dim=1).sum()
