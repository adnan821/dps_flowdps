"""Pure Π-GDM posterior sampler on the Rectified Flow CelebA-HQ-256
backbone — analytical Wiener-form gradient, no autograd through the
velocity field, no free guidance scalar ζ.

Rectified Flow analogue of `pixel_dps_pigdm_pure.PixelDPSPiGDMPure`.
Key differences from the diffusion version:

  - Tweedie clean-data endpoint:  z₁̂ = x_t + (1 − t) · v_θ(x_t, t)
  - Tweedie covariance scale:     r_t = (1 − t)²                  (RF analog)
  - drop-Jacobian approximation:  ∂z₁̂/∂x_t ≈ I  (no √ᾱ_t scaling)

Correction at each Euler step (derivation identical to the DDPM case,
A circulant ⇒ Σ_y diagonal in Fourier):

    correction(x_t)  =  ℱ⁻¹[ conj(H) / (σ_n² + r_t |H|²) · ℱ(y − A z₁̂) ]

The guided Euler step is then

    v_eff(x_t, t)  =  v_θ(x_t, t)  +  η · correction
    x_{t+dt}       =  x_t  +  dt · v_eff

with η = 1 (Song 2023 canonical, no free knob).

σ_n floor = 0.01, identical to the diffusion version.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from src.models.rf_celebahq import RFVelocityModel


@dataclass
class FlowDPSPiGDMPureResult:
    x_hat: torch.Tensor
    nfe: int
    elapsed_s: float
    log: list[dict]


class FlowDPSRFPiGDMPure:
    def __init__(
        self,
        rf_model: RFVelocityModel,
        device: str | torch.device = "cuda",
        sigma_n_floor: float = 0.01,
    ):
        self.rf = rf_model
        self.device = torch.device(device)
        self.sigma_n_floor = float(sigma_n_floor)

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        H_otf: torch.Tensor,
        num_steps: int = 100,
        sigma_n: float = 0.05,
        eps_t: float = 1e-3,
        seed: Optional[int] = 0,
        verbose: bool = False,
        eta: float = 1.0,
        # Accepted for grid-runner symmetry — ignored:
        zeta: float = 0.0,
        spectral_weight: Optional[torch.Tensor] = None,
        pigdm_otf: Optional[torch.Tensor] = None,
        pigdm_sigma_n: Optional[float] = None,
        integrator: str = "euler",
    ) -> FlowDPSPiGDMPureResult:
        import time

        sn = max(float(sigma_n), self.sigma_n_floor)
        B, C, H_, W = y.shape
        y = y.to(self.device)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn((B, C, H_, W), generator=gen, device=self.device, dtype=torch.float32)

        T = 1.0
        dt = (T - eps_t) / num_steps

        H = H_otf.to(self.device)
        Hmag2 = (H.abs() ** 2).real
        H_conj = H.conj()

        log: list[dict] = []
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        for i in range(num_steps):
            num_t = (i / num_steps) * (T - eps_t) + eps_t
            t_batch = torch.full((B,), num_t, device=self.device, dtype=x.dtype)

            v = self.rf.velocity(x, t_batch)             # (B, C, H, W)
            z1_hat = x + (1.0 - num_t) * v                # Tweedie clean
            z1_hat_pix = (z1_hat + 1.0) / 2.0             # [0, 1]

            y_hat = forward_op(z1_hat_pix)
            residual = y - y_hat                          # (B, C, H, W)

            # Analytical Π-GDM gradient (RF: r_t = (1-t)^2).
            r_t_val = (1.0 - num_t) ** 2
            R = torch.fft.fft2(residual)
            denom = sn ** 2 + r_t_val * Hmag2
            filtered_fft = R * (H_conj / denom).view(1, 1, *H.shape)
            grad_pix = torch.fft.ifft2(filtered_fft).real

            # Bayesian update for z₁̂ (Kalman form, prior cov = r_t · I):
            #     δz₁̂ = r_t · A^T (Σ_y)^{-1} (y - A z₁̂)
            # 0.5 factor comes from the [-1,1] ↔ [0,1] convention.
            delta_z1_dpr = 0.5 * r_t_val * grad_pix

            # Chain to velocity: v = (z₁̂ - x) / (1 - t), so a δz₁̂ shift
            # corresponds to δv = δz₁̂ / (1 - t). With r_t = (1-t)², the
            # net velocity-space scaling is (1-t) on the raw FFT term.
            correction_v = delta_z1_dpr / max(1.0 - num_t, 1e-6)

            v_eff = v + eta * correction_v
            x = (x + dt * v_eff).detach()

            log.append({
                "step": i, "t": float(num_t),
                "r_t": float(r_t_val),
                "sigma_n_eff": float(sn),
                "grad_norm": float(correction_v.flatten(1).norm(dim=1).mean().item()),
            })
            if verbose and (i % max(1, num_steps // 5) == 0):
                print(f"  [{i+1}/{num_steps}] t={num_t:.3f} r_t={r_t_val:.3f} "
                      f"|corr_v|={correction_v.flatten(1).norm(dim=1).mean():.3e}")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        x_hat_pix = ((x + 1.0) / 2.0).clamp(0, 1)
        return FlowDPSPiGDMPureResult(x_hat=x_hat_pix, nfe=num_steps, elapsed_s=elapsed, log=log)
