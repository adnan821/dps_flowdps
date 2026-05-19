"""Pixel-level + perceptual image metrics.

Conventions:
- Inputs are float tensors in **[0, 1]** with shape (B, C, H, W) (or (C, H, W)).
- Higher PSNR / SSIM is better; lower LPIPS is better.
- LPIPS is wrapped lazily so importing this module doesn't pull in lpips/torch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _to_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"expected 3D or 4D tensor, got shape {tuple(x.shape)}")


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR in dB; assumes inputs in [0, max_val]. Returns a Python float."""
    pred = _to_4d(pred).to(torch.float64)
    target = _to_4d(target).to(torch.float64)
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * float(np.log10((max_val**2) / mse))


def batch_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> list[float]:
    """Per-image PSNR for a batch."""
    pred = _to_4d(pred).to(torch.float64)
    target = _to_4d(target).to(torch.float64)
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    out = []
    for m in mse:
        m = m.item()
        out.append(float("inf") if m <= 1e-12 else 10.0 * float(np.log10(max_val**2 / m)))
    return out


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean SSIM over channels and batch. Differentiable PyTorch implementation."""
    pred = _to_4d(pred).to(torch.float32)
    target = _to_4d(target).to(torch.float32)
    C = pred.shape[1]
    w = _gaussian_window(window_size, sigma, pred.device, pred.dtype).expand(C, 1, -1, -1)
    pad = window_size // 2

    mu_x = F.conv2d(pred, w, padding=pad, groups=C)
    mu_y = F.conv2d(target, w, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, w, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, w, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, w, padding=pad, groups=C) - mu_xy

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (num / den).mean().item()


def batch_ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> list[float]:
    """Per-image SSIM."""
    out = []
    pred = _to_4d(pred)
    target = _to_4d(target)
    for i in range(pred.shape[0]):
        out.append(ssim(pred[i : i + 1], target[i : i + 1], max_val=max_val))
    return out


class LPIPSMetric:
    """Wrapper around the lpips package, AlexNet backbone, with [0,1] -> [-1,1] rescaling."""

    def __init__(self, net: str = "alex", device: Optional[str | torch.device] = None):
        import lpips  # lazy import

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = lpips.LPIPS(net=net).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> list[float]:
        pred = _to_4d(pred).to(self.device).float()
        target = _to_4d(target).to(self.device).float()
        # lpips expects [-1, 1].
        pred = pred.clamp(0, 1) * 2.0 - 1.0
        target = target.clamp(0, 1) * 2.0 - 1.0
        d = self.model(pred, target).view(-1)
        return [v.item() for v in d]
