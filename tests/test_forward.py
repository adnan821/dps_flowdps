"""Tests for forward degradation operators."""
import math

import pytest
import torch

from src.forward import (
    add_noise,
    degrade,
    gaussian_blur,
    gaussian_kernel_2d,
    motion_blur,
    motion_blur_kernel,
)


def _fixture(B=1, C=3, H=64, W=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, C, H, W, generator=g)


def test_gaussian_kernel_sums_to_one():
    for sigma in [0.5, 1.5, 3.0, 5.0]:
        k = gaussian_kernel_2d(sigma)
        assert torch.allclose(k.sum(), torch.tensor(1.0), atol=1e-6), f"sigma={sigma}"


def test_motion_kernel_sums_to_one():
    for length, angle in [(5, 0), (15, 45), (21, 90), (11, 30)]:
        k = motion_blur_kernel(length, angle)
        assert torch.allclose(k.sum(), torch.tensor(1.0), atol=1e-6), f"L={length} a={angle}"


def test_gaussian_blur_preserves_shape():
    x = _fixture()
    for sigma in [1.5, 3.0, 5.0]:
        y = gaussian_blur(x, sigma)
        assert y.shape == x.shape


def test_motion_blur_preserves_shape():
    x = _fixture()
    y = motion_blur(x, length=15, angle_deg=45.0)
    assert y.shape == x.shape


def test_forward_is_linear_in_x():
    """A(a*x + b*y) == a*A(x) + b*A(y) up to fp tolerance."""
    x = _fixture(seed=1)
    y = _fixture(seed=2)
    a, b = 0.3, 0.7
    lhs = gaussian_blur(a * x + b * y, sigma=3.0)
    rhs = a * gaussian_blur(x, sigma=3.0) + b * gaussian_blur(y, sigma=3.0)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_motion_blur_is_linear_in_x():
    x = _fixture(seed=1)
    y = _fixture(seed=2)
    a, b = 0.3, 0.7
    lhs = motion_blur(a * x + b * y, length=15, angle_deg=30.0)
    rhs = a * motion_blur(x, length=15, angle_deg=30.0) + b * motion_blur(y, length=15, angle_deg=30.0)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_add_noise_deterministic_with_seed():
    y = _fixture()
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    n1 = add_noise(y, 0.05, generator=g1)
    n2 = add_noise(y, 0.05, generator=g2)
    assert torch.allclose(n1, n2)


def test_add_noise_zero_sigma_is_identity():
    y = _fixture()
    assert torch.equal(add_noise(y, 0.0), y)


def test_degrade_deterministic_with_seed():
    x = _fixture()
    y1 = degrade(x, blur_type="gaussian", blur_sigma=3.0, noise_sigma=0.05, seed=7)
    y2 = degrade(x, blur_type="gaussian", blur_sigma=3.0, noise_sigma=0.05, seed=7)
    assert torch.allclose(y1, y2)


def test_degrade_supports_gradient():
    """DPS-style likelihood gradient must flow through the forward operator."""
    x = _fixture().requires_grad_(True)
    y = degrade(x, blur_type="gaussian", blur_sigma=3.0, noise_sigma=0.0)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_dc_preserved_no_noise():
    """For a normalized blur kernel, mean of A(x) should equal mean of x (within edge effects)."""
    x = _fixture(H=128, W=128)
    y = gaussian_blur(x, sigma=3.0)
    # Compare central region to avoid reflect-padding edge bias on a tiny crop.
    cx = x[:, :, 16:-16, 16:-16].mean()
    cy = y[:, :, 16:-16, 16:-16].mean()
    assert (cx - cy).abs() < 1e-3
