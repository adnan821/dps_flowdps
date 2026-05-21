"""FlowDPS-style posterior sampling on a Rectified Flow CelebA-HQ-256 model.

Reimplements the Kim et al. 2025 FlowDPS algorithm on top of the Liu 2023
Rectified Flow CelebA-HQ-256 checkpoint (pixel-space NCSN++ U-Net), since
the official FlowDPS repo is SD3-only.

Rectified Flow convention (Liu 2023):
- Continuous time t ∈ [0, 1].
- x_t = (1 - t) * z_0 + t * z_1 where z_0 ~ N(0, I) is noise and z_1 is data.
- Velocity v_theta(x_t, t) = dz_1/dt - dz_0/dt = z_1 - z_0.
- Tweedie-style estimates of the endpoints:
    z_1_hat (clean data) = x_t + (1 - t) * v_theta(x_t, t)
    z_0_hat (noise)      = x_t - t * v_theta(x_t, t)
- Sampling: Euler integration FORWARD in time, t = eps → T (typically 1 - eps).
  Start from x = z_0 ~ N(0, I), step x_{t+dt} = x_t + dt * v_theta(x_t, t).

FlowDPS posterior update modifies the velocity field with a measurement
likelihood gradient at each step:
    v_eff(x_t, t) = v_theta(x_t, t) - lambda * grad_x_t ||y - A(z_1_hat)||_2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

import torch

from src.models.rf_celebahq import RFVelocityModel

if TYPE_CHECKING:
    from src.samplers.schedules import ZetaSchedule


@dataclass
class FlowDPSResult:
    x_hat: torch.Tensor    # final reconstruction in [0, 1], shape (B, C, H, W)
    nfe: int               # number of velocity-field evaluations
    elapsed_s: float
    log: list[dict]


class FlowDPSRF:
    """FlowDPS-style guided sampler on a Liu 2023 Rectified Flow backbone."""

    def __init__(self, rf_model: RFVelocityModel, device: str | torch.device = "cuda"):
        self.rf = rf_model
        self.device = torch.device(device)

    def sample(
        self,
        y: torch.Tensor,
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int = 100,
        zeta: "float | ZetaSchedule" = 1.0,
        eps_t: float = 1e-3,
        seed: Optional[int] = 0,
        verbose: bool = False,
        spectral_weight: Optional[torch.Tensor] = None,
        pigdm_otf: Optional[torch.Tensor] = None,
        pigdm_sigma_n: float = 0.05,
        integrator: str = "euler",
    ) -> FlowDPSResult:
        """Sample x ~ p(x | y) with FlowDPS-on-RF.

        Args:
            y: observed degraded image in [0, 1], shape (B, C, H, W).
            forward_op: differentiable forward operator on [0, 1] images.
            num_steps: NFE budget (number of velocity-field evaluations).
                For `integrator="euler"` this is the step count directly.
                For `integrator="heun"` each step costs 2 velocity calls,
                so we run `num_steps // 2` Heun steps — the `nfe` reported
                stays equal to `num_steps`, keeping the budget comparable.
            zeta: likelihood-gradient guidance scale. Either a scalar
                (constant ζ across all steps, the original behavior) or
                a `ZetaSchedule` callable `(step_idx, num_steps) -> float`
                from `src.samplers.schedules`. Default 1.0 (scalar).
            eps_t: integration start time (avoid t=0 singularity in 1/t terms).
            seed: per-call RNG seed for the initial noise draw.
            integrator: "euler" (1st-order, original) or "heun" (2nd-order
                predictor-corrector). Heun applies the 2nd-order correction
                to the model velocity field; the likelihood-guidance term
                is computed once per step (at the predictor point) and
                reused for the corrector — the ODE backbone benefits most
                from higher-order integration, the guidance is a slowly
                varying correction.
        """
        from src.samplers.schedules import resolve as _resolve_zeta
        from src.samplers.spectral_weight import spectral_residual_l2
        from src.samplers.tweedie_likelihood import pigdm_weight, rf_r_t

        use_pigdm = pigdm_otf is not None
        if integrator not in ("euler", "heun"):
            raise ValueError(f"Unknown integrator: {integrator}")
        import time

        B, C, H, W = y.shape
        y = y.to(self.device)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        # Initial state: pure noise (this is z_0 ~ N(0, I)).
        x = torch.randn((B, C, H, W), generator=gen, device=self.device, dtype=torch.float32)

        T = 1.0
        # Heun spends 2 velocity calls per step; halve the step count so
        # the NFE budget matches a Euler run with the same `num_steps`.
        n_steps_eff = num_steps if integrator == "euler" else max(1, num_steps // 2)
        dt = (T - eps_t) / n_steps_eff

        def _likelihood_grad(x_in, t_scalar, step_idx):
            """Compute the FlowDPS likelihood-gradient at (x_in, t). Returns
            (grad, loss, grad_norm)."""
            x_in = x_in.detach().requires_grad_(True)
            t_b = torch.full((B,), t_scalar, device=self.device, dtype=x_in.dtype)
            v_local = self.rf.velocity(x_in, t_b)
            z1_hat = x_in + (1.0 - t_scalar) * v_local
            # NOTE: no clamp inside the gradient path — clamping zeros the
            # gradient for any out-of-range pixel at early t, which kills DPS.
            z1_hat_pix = (z1_hat + 1.0) / 2.0
            residual = forward_op(z1_hat_pix) - y
            if use_pigdm:
                # Π-GDM: weight shape = 1/sqrt(σ_n² + r_t |H|²), un-normalized.
                r_t = rf_r_t(t_scalar)
                W = pigdm_weight(pigdm_otf, pigdm_sigma_n, r_t, normalize_mean=False)
                loss_local = spectral_residual_l2(residual, W, squared=False)
            else:
                loss_local = spectral_residual_l2(residual, spectral_weight)
            g = torch.autograd.grad(loss_local, x_in, retain_graph=False)[0]
            return v_local.detach(), g.detach(), float(loss_local.item())

        log: list[dict] = []
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0_wall = time.time()

        for i in range(n_steps_eff):
            num_t = (i / n_steps_eff) * (T - eps_t) + eps_t  # in [eps, T - eps]
            zeta_t = _resolve_zeta(zeta, i, n_steps_eff)

            v, grad, loss_val = _likelihood_grad(x, num_t, i)
            grad_norm = grad.flatten(1).norm(dim=1).mean().item()

            with torch.no_grad():
                v_eff = v - zeta_t * grad
                if integrator == "euler":
                    x = (x.detach() + dt * v_eff).detach()
                else:
                    # Heun: predictor, extra velocity eval, corrector. The
                    # guidance grad is reused (computed once at the start).
                    x_pred = x.detach() + dt * v_eff
                    t_next = min(num_t + dt, T - eps_t)
                    t_b_next = torch.full((B,), t_next, device=self.device, dtype=x.dtype)
                    v_next = self.rf.velocity(x_pred, t_b_next)
                    v_avg = 0.5 * (v + v_next)
                    x = (x.detach() + dt * (v_avg - zeta_t * grad)).detach()

            log.append({
                "step": i,
                "t": float(num_t),
                "loss": loss_val,
                "grad_norm": float(grad_norm),
            })
            if verbose and (i % max(1, n_steps_eff // 5) == 0):
                print(f"  [{i+1}/{n_steps_eff}] t={num_t:.3f} loss={loss_val:.4f} |grad|={grad_norm:.4e}")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0_wall

        x_hat_pix = ((x.detach() + 1) / 2).clamp(0, 1)
        return FlowDPSResult(x_hat=x_hat_pix, nfe=num_steps, elapsed_s=elapsed, log=log)
