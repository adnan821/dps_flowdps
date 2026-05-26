"""Gradient-free posterior sampling via Sequential Monte Carlo (SMC) on a
pixel-space DDPM.

Contrast to `pixel_dps.PixelDPS`:
- DPS injects a likelihood-gradient term `-zeta * grad_x ||y - A(x_hat_0)||`
  into the DDIM update. Computing that gradient requires backprop through
  the U-Net (≈ 2x forward cost) AND through the forward operator.
- Particle-DPS (this file) keeps `P` particles of `x_t`, runs the DDIM
  reverse step on each (no gradient), and reweights / resamples them by
  their measurement likelihood `w_p ∝ exp(-‖y - A(x_hat_0_p)‖² / 2σ_y²)`.
  No autograd through the U-Net, no autograd through the forward
  operator. This is the closest pixel-space analogue to the SMC-style
  family that includes Dou et al. 2026 "Constrained Particle Seeking".

The trade-off the report calls out:
- Pro: forward-only, so cheaper per model call; multiple particles
  naturally produce a posterior distribution.
- Con: gradient-based DPS carries more information per step than
  reweighting + resampling; on tight noise budgets DPS typically wins
  PSNR while particle methods often win sample diversity.

For grid-runner compatibility this sampler mirrors `PixelDPS.sample(...)`:
the public reconstruction returned in `DPSResult.x_hat` is the
weighted posterior mean of the final particle cloud. The per-particle
final states are kept in `log[-1]['particles']` for diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from diffusers import DDIMScheduler, UNet2DModel


@dataclass
class ParticleDPSResult:
    x_hat: torch.Tensor      # weighted posterior mean, [0, 1], (B, C, H, W)
    nfe: int                 # number of U-Net evaluations per particle
    elapsed_s: float
    log: list[dict]          # per-step diagnostics (loss, ESS, resample events)


class ParticleDPS:
    """SMC-style posterior sampler on a pixel-space DDPM. Forward-only —
    no gradient is taken through the U-Net or the forward operator."""

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler_config: dict | None = None,
        device: str | torch.device = "cuda",
        num_particles: int = 8,
        ess_threshold: float = 0.5,
    ):
        """
        Args:
            unet: pretrained DDPM U-Net.
            scheduler_config: passed to `DDIMScheduler.from_config` to
                set up the reverse schedule. Defaults to a fresh DDPM
                scheduler if None.
            device: cuda or cpu.
            num_particles: P. 8 is a sane default; the report's runtime
                budget assumed P=8.
            ess_threshold: fraction of P below which we resample. 0.5
                means: resample whenever effective sample size drops
                below P/2. Standard SMC heuristic.
        """
        self.device = torch.device(device)
        self.unet = unet.to(self.device).eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        if scheduler_config is None:
            from diffusers import DDPMScheduler
            scheduler_config = DDPMScheduler().config
        self.scheduler = DDIMScheduler.from_config(scheduler_config)
        self.num_particles = int(num_particles)
        self.ess_threshold = float(ess_threshold)

    def _tweedie_x0(self, x_t: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        """Same Tweedie identity as DPS — clean-image estimate from the
        current noisy state."""
        return (x_t - (1.0 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int = 100,
        sigma_y: float = 0.05,
        seed: Optional[int] = 0,
        verbose: bool = False,
        num_particles: Optional[int] = None,
        # Kept for run_grid.py interface symmetry — accepted and ignored:
        zeta: float = 0.0,
        spectral_weight: Optional[torch.Tensor] = None,
        pigdm_otf: Optional[torch.Tensor] = None,
        pigdm_sigma_n: Optional[float] = None,
    ) -> ParticleDPSResult:
        """Sample x ~ p(x | y) via SMC reweighting on the DDIM trajectory.

        Args:
            y: degraded observation in [0, 1], shape (B, C, H, W). For
                this gradient-free sampler we assume B=1 — particles are
                batched along the leading dim INSIDE the sampler.
            forward_op: differentiable not required, just callable
                f: [0,1] -> y-space.
            num_steps: NFE budget per particle. Each step costs one
                U-Net forward pass batched across particles, so the
                wall-clock per image is roughly `num_steps × P` U-Net
                forward-pass equivalents.
            sigma_y: assumed measurement-noise std in the likelihood
                `‖y - A(x̂_0)‖² / (2 σ_y²)`.
            num_particles: optionally override the constructor default.
            zeta/spectral_weight/pigdm_*: accepted for grid-runner
                compatibility; ignored (this sampler has no guidance
                gradient).
        """
        import time

        if y.shape[0] != 1:
            raise ValueError("ParticleDPS expects B=1; batches the P particles internally.")
        P = int(num_particles) if num_particles is not None else self.num_particles
        y = y.to(self.device)  # (1, C, H, W)
        _, C, H, W = y.shape

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        # P particles of initial Gaussian noise in [-1, 1] DDPM space.
        x = torch.randn((P, C, H, W), generator=gen, device=self.device, dtype=torch.float32)

        # log-weights of each particle. Uniform at init.
        log_w = torch.zeros(P, device=self.device, dtype=torch.float32)

        self.scheduler.set_timesteps(num_steps, device=self.device)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        log: list[dict] = []
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        # Broadcast y to all particles for the likelihood evaluation.
        y_rep = y.expand(P, -1, -1, -1)  # (P, C, H, W)

        for i, t in enumerate(self.scheduler.timesteps):
            # One batched U-Net forward pass over the P particles.
            eps = self.unet(x, t).sample             # (P, C, H, W)
            alpha_bar_t = alphas_cumprod[t]
            x_hat_0 = self._tweedie_x0(x, eps, alpha_bar_t)  # in [-1, 1]
            x_hat_0_pix = (x_hat_0 + 1.0) / 2.0              # in [0, 1]

            # Likelihood weight per particle (Gaussian, sigma_y).
            y_hat = forward_op(x_hat_0_pix)
            # Sum-of-squares per particle.
            sq = (y_hat - y_rep).flatten(1).pow(2).sum(dim=1)  # (P,)
            log_lik = -0.5 * sq / (sigma_y ** 2 + 1e-12)

            # Bayesian update: posterior log-weight ∝ prior log-weight + log-lik.
            log_w = log_w + log_lik
            # Normalize for numerical stability.
            log_w = log_w - log_w.logsumexp(0)
            w = log_w.exp()                                          # (P,)
            ess = 1.0 / (w.pow(2).sum() + 1e-12)
            ess_frac = ess.item() / P

            resampled = False
            if ess_frac < self.ess_threshold:
                # Multinomial resample with replacement; reset log-weights.
                idx = torch.multinomial(w, P, replacement=True, generator=gen)
                x = x[idx]
                log_w = torch.zeros(P, device=self.device, dtype=torch.float32)
                resampled = True

            # DDIM reverse step — no gradient needed. The scheduler.step
            # call works on each particle independently because eps
            # carries no inter-particle coupling.
            x = self.scheduler.step(eps, t, x).prev_sample            # (P, C, H, W)

            log.append({
                "step": i,
                "t": int(t),
                "ess_frac": float(ess_frac),
                "resampled": bool(resampled),
                "log_lik_mean": float(log_lik.mean().item()),
            })
            if verbose and (i % max(1, num_steps // 5) == 0):
                print(f"  [{i+1}/{num_steps}] t={int(t)} ess={ess_frac:.3f} "
                      f"resampled={resampled} log_lik_mean={log_lik.mean():.3f}")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        # Final reconstruction = weighted posterior mean of the particle cloud.
        # x is in [-1, 1]; convert to [0, 1] at the boundary.
        log_w = log_w - log_w.logsumexp(0)
        w = log_w.exp().view(P, 1, 1, 1)
        x_mean = (x * w).sum(dim=0, keepdim=True)                    # (1, C, H, W)
        x_hat_pix = ((x_mean.clamp(-1, 1) + 1) / 2).clamp(0, 1)

        # Keep the final particles + weights in the last log entry for
        # downstream diversity / std-map analysis.
        log.append({
            "step": "final",
            "particles_shape": list(x.shape),
            "final_weights": w.flatten().tolist(),
        })

        return ParticleDPSResult(x_hat=x_hat_pix, nfe=num_steps, elapsed_s=elapsed, log=log)
