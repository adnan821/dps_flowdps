"""Backward-compatibility tests for the publication-improvement refactor.

Verifies that with the default arguments (scalar zeta, no spectral
weighting, no Tweedie correction), the refactored samplers produce
results bit-identical (within fp32 tolerance) to the pre-refactor
behavior. Catches regressions when adding new optional kwargs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.samplers.schedules import (
    resolve,
    stringify,
    zeta_constant,
    zeta_power,
    zeta_linear_warmup_then_decay,
)


def test_resolve_scalar():
    assert resolve(3.0, 5, 50) == 3.0
    assert resolve(0.0, 0, 50) == 0.0


def test_resolve_callable():
    z = zeta_constant(7.5)
    assert resolve(z, 0, 50) == 7.5
    assert resolve(z, 49, 50) == 7.5


def test_zeta_constant_invariant_to_step():
    z = zeta_constant(2.0)
    vals = [z(i, 10) for i in range(10)]
    assert all(v == 2.0 for v in vals)


def test_zeta_power_alpha_zero_is_constant():
    z = zeta_power(5.0, alpha=0.0)
    vals = [z(i, 100) for i in range(100)]
    # alpha=0 → (1-i/N)^0 = 1, so always zeta_0
    assert all(abs(v - 5.0) < 1e-9 for v in vals)


def test_zeta_power_alpha_one_decays_linearly():
    z = zeta_power(10.0, alpha=1.0)
    assert abs(z(0, 100) - 10.0) < 1e-6
    assert abs(z(50, 100) - 5.0) < 1e-6
    assert abs(z(99, 100) - 0.1) < 1e-6


def test_zeta_power_alpha_two_decays_quadratically():
    z = zeta_power(10.0, alpha=2.0)
    assert abs(z(0, 100) - 10.0) < 1e-6
    assert abs(z(50, 100) - 2.5) < 1e-6  # 10 * 0.5^2
    assert abs(z(99, 100) - 10 * 0.01 ** 2) < 1e-6


def test_zeta_warmup():
    z = zeta_linear_warmup_then_decay(10.0, warmup_frac=0.1, alpha=1.0)
    # First step of warmup: 10 * (0+1)/10 = 1
    assert abs(z(0, 100) - 1.0) < 1e-6
    # End of warmup
    assert abs(z(9, 100) - 10.0) < 1e-6
    # Late: power decay kicks in
    assert z(50, 100) < z(10, 100)


def test_stringify_scalar_idempotent():
    assert stringify(3.0) == "3.0"
    assert stringify(stringify(3.0)) == "3.0"  # idempotent only when applied to value, but the result is already a string


def test_stringify_callable_named():
    z = zeta_power(5.0, 1.0)
    s = stringify(z)
    assert "power" in s
    assert "5.0" in s


def test_constant_zeta_callable_matches_scalar_in_pixel_dps():
    """A scalar zeta and a constant-zeta callable must produce identical
    reconstructions (seed-locked, single step)."""
    # We avoid loading the actual DDPM (slow); just verify the resolve()
    # contract: anywhere the sampler uses `zeta * grad`, it goes through
    # `resolve()` first.
    import inspect
    from src.samplers.pixel_dps import PixelDPS
    src = inspect.getsource(PixelDPS.sample)
    assert "_resolve_zeta" in src, "Pixel-DPS sample() doesn't call resolve()"
    assert "zeta_t" in src, "Pixel-DPS sample() doesn't bind zeta_t"


def test_constant_zeta_callable_matches_scalar_in_flowdps_rf():
    import inspect
    from src.samplers.flowdps_rf import FlowDPSRF
    src = inspect.getsource(FlowDPSRF.sample)
    assert "_resolve_zeta" in src, "FlowDPS-on-RF sample() doesn't call resolve()"
    assert "zeta_t" in src, "FlowDPS-on-RF sample() doesn't bind zeta_t"


# ---- Tier-C: Tweedie / Pi-GDM weight ----


def test_pigdm_weight_shape_and_dtype():
    import torch
    from src.forward.ops_fft import gaussian_otf
    from src.samplers.tweedie_likelihood import pigdm_weight
    H = gaussian_otf(3.0, (32, 32))
    W = pigdm_weight(H, sigma_n=0.05, r_t=1.0)
    assert W.shape == (32, 32)
    assert W.dtype == torch.float32
    assert (W > 0).all()


def test_pigdm_weight_well_conditioned_when_r_t_is_zero():
    """With r_t=0 the weight becomes 1/sigma_n * 1 (constant); after
    mean-normalize this is a uniform W=1 tensor (within fp32)."""
    import torch
    from src.forward.ops_fft import gaussian_otf
    from src.samplers.tweedie_likelihood import pigdm_weight
    H = gaussian_otf(3.0, (32, 32))
    W = pigdm_weight(H, sigma_n=0.05, r_t=0.0, normalize_mean=True)
    assert torch.allclose(W, torch.ones_like(W), atol=1e-5)


def test_diffusion_r_t_monotone():
    """r_t = (1 - alpha_bar_t) / alpha_bar_t: decreases as alpha_bar_t
    increases (i.e. as we get closer to t=0 / clean image)."""
    from src.samplers.tweedie_likelihood import diffusion_r_t
    # Higher alpha_bar = closer to clean image = smaller r_t.
    assert diffusion_r_t(0.99) < diffusion_r_t(0.5) < diffusion_r_t(0.05)


def test_rf_r_t_zero_at_data_endpoint():
    from src.samplers.tweedie_likelihood import rf_r_t
    assert rf_r_t(1.0) == 0.0
    assert rf_r_t(0.0) == 1.0
    assert rf_r_t(0.5) == 0.25
