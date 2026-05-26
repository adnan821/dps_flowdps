"""Pure (parameter-free) Π-GDM posterior sampler on a pixel-space DDPM.

This is the analytical / closed-form implementation of Song, Sun &
Saremi 2023 "Pseudoinverse-Guided Diffusion Models", in contrast to
the EXP-016/017 attempt which mixed Π-GDM weights with a free
guidance scalar ζ in an L2-norm DPS loss (and collapsed).

Key conventions, derived in `docs/TEMPERED_PARTICLE_DPS.md`-style
notation but applied to Π-GDM (this file's docstring is the source of
truth):

For a linear circulant forward operator A (Gaussian blur, motion blur
on the pixel torus), the score correction at reverse-step t is

    ∇_{x_t} log p(y | x_t)  ≈  (∂x̂₀/∂x_t)ᵀ · Aᵀ · Σ_y⁻¹ · (y − A x̂₀)

with
    Σ_y  =  σ_n² I  +  r_t · A Aᵀ
    r_t  =  (1 − ᾱ_t) / ᾱ_t           (diffusion Tweedie covariance)
    ∂x̂₀/∂x_t  ≈  I / √ᾱ_t              (Song 2023 drop-Jacobian approx)

Because A is circulant, Σ_y is diagonal in the Fourier basis with
entries σ_n² + r_t · |H(f)|², so the entire correction is one FFT,
one pointwise complex multiply, one IFFT:

    correction(x_t)  =  (1 / √ᾱ_t) · ℱ⁻¹[ conj(H) / (σ_n² + r_t |H|²) · ℱ(y − A x̂₀) ]

The reverse step then becomes the standard DDIM update plus this
correction with η = 1 (no free knob):

    x_{t-1}  =  x_{t-1}^{DDIM}  +  η · correction(x_t)

Two things to note vs the EXP-017 attempt:

  1. **No free ζ.** Magnitude is data-determined. The previous attempt
     mixed Π-GDM weights into a DPS L2-norm loss and used ζ=100 as a
     free knob; that broke catastrophically because the effective
     gradient scale depends on σ_n (1/σ_n explosion at σ_n=0 cells).
     Here we keep the math exact end-to-end.

  2. **σ_n floor = 0.01.** The 1/(σ_n² + r_t|H|²) denominator is
     numerically degenerate at the (σ_n=0, |H|≈0) corner. We use
     σ_n_eff = max(σ_n, 0.01) — small enough not to perturb noisy
     cells, large enough to keep σ_n=0 cells bounded. This is
     standard practice in Π-GDM / DDRM implementations.

Public API mirrors `PixelDPS.sample()` for grid-runner compatibility,
EXCEPT it requires an `H_otf` tensor passed explicitly (precomputed
once per σ_b cell by the grid runner).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from diffusers import DDIMScheduler, UNet2DModel


@dataclass
class PiGDMPureResult:
    x_hat: torch.Tensor   # final reconstruction in [0, 1], shape (B, C, H, W)
    nfe: int
    elapsed_s: float
    log: list[dict]


class PixelDPSPiGDMPure:
    """Analytical Π-GDM sampler — no autograd, no free ζ."""

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler_config: dict | None = None,
        device: str | torch.device = "cuda",
        sigma_n_floor: float = 0.01,
        r_t_cap: float = 1e3,
    ):
        self.device = torch.device(device)
        self.unet = unet.to(self.device).eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        if scheduler_config is None:
            from diffusers import DDPMScheduler
            scheduler_config = DDPMScheduler().config
        self.scheduler = DDIMScheduler.from_config(scheduler_config)
        self.sigma_n_floor = float(sigma_n_floor)
        self.r_t_cap = float(r_t_cap)

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        H_otf: torch.Tensor,
        num_steps: int = 100,
        sigma_n: float = 0.05,
        seed: Optional[int] = 0,
        verbose: bool = False,
        eta: float = 1.0,
        # Accepted for grid-runner symmetry — ignored:
        zeta: float = 0.0,
        spectral_weight: Optional[torch.Tensor] = None,
        pigdm_otf: Optional[torch.Tensor] = None,
        pigdm_sigma_n: Optional[float] = None,
        sigma_y: Optional[float] = None,
    ) -> PiGDMPureResult:
        """Run pure Π-GDM to recover x from y = A(x) + ε.

        Args:
            y: measurement in [0, 1], shape (B, C, H, W).
            forward_op: circulant forward operator on [0, 1] images.
            H_otf: precomputed complex-valued (H, W) OTF of `forward_op`.
                Created via `src.forward.ops_fft.gaussian_otf(σ_b, ...)`.
            num_steps: NFE budget.
            sigma_n: measurement-noise std. Floored at self.sigma_n_floor.
            eta: Π-GDM step scale. Default 1.0 (Song 2023 canonical).
            zeta/spectral_weight/pigdm_otf/pigdm_sigma_n/sigma_y:
                accepted-and-ignored for run_grid.py compatibility.
        """
        import time

        sn = max(float(sigma_n), self.sigma_n_floor)
        B = y.shape[0]
        y = y.to(self.device)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn(y.shape, generator=gen, device=self.device, dtype=torch.float32)

        self.scheduler.set_timesteps(num_steps, device=self.device)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # Precompute frequency-domain quantities (constant across t).
        H = H_otf.to(self.device)                      # (H, W) complex
        Hmag2 = (H.abs() ** 2).real                    # (H, W) real
        H_conj = H.conj()                              # (H, W) complex

        log: list[dict] = []
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        for i, t in enumerate(self.scheduler.timesteps):
            # 1. One U-Net forward (no autograd).
            eps = self.unet(x, t).sample                # (B, C, H, W)
            alpha_bar_t = alphas_cumprod[t]
            x_hat_0 = (x - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
            x_hat_0_pix = (x_hat_0 + 1.0) / 2.0         # [0, 1] for forward_op

            # 2. Residual in y-space.
            y_hat = forward_op(x_hat_0_pix)
            residual = y - y_hat                        # (B, C, H, W)

            # 3. Analytical Π-GDM gradient via FFT (Song 2023 Eq. 6).
            r_t_val = (1.0 - alpha_bar_t.item()) / max(alpha_bar_t.item(), 1e-12)
            r_t_val = min(r_t_val, self.r_t_cap)
            R = torch.fft.fft2(residual)                # (B, C, H, W) complex
            denom = sn ** 2 + r_t_val * Hmag2            # (H, W)
            # broadcast (H, W) → (1, 1, H, W)
            filtered_fft = R * (H_conj / denom).view(1, 1, *H.shape)
            grad_pix = torch.fft.ifft2(filtered_fft).real  # (B, C, H, W)

            # 4. Proper Bayesian update for x̂₀ | x_t, y (Kalman form).
            #    With prior x̂₀ | x_t ~ N(x̂₀(x_t), r_t · I), the posterior
            #    mean has a leading r_t factor:
            #        x̂₀_corr = x̂₀ + r_t · A^T (Σ_y)^{-1} (y - A x̂₀)
            #    Reference: standard Gaussian Bayes update, Π-GDM paper
            #    Eq. 5-6, DDRM Eq. 8. Dropping the r_t factor (as my
            #    first attempt did) produced 5-9 dB; including it
            #    rescales the correction so it's large early (high r_t,
            #    trust data more than uncertain x̂₀) and small late
            #    (low r_t, x̂₀ already close to truth).
            delta_x_hat_dpr = 0.5 * r_t_val * grad_pix    # [-1,1]-space delta
            x_hat_0_corrected = x_hat_0 + eta * delta_x_hat_dpr

            # 5. DDIM step (deterministic, σ_t=0) with corrected x̂₀ in
            #    the data term and ORIGINAL eps in the noise direction.
            #    Computing eps_corrected and feeding back through
            #    scheduler.step would double-count the correction
            #    (scheduler internally re-derives x̂₀ from the eps it's
            #    given). This is DDRM Eq. 9 form.
            if i + 1 < num_steps:
                t_next = self.scheduler.timesteps[i + 1]
                alpha_bar_next = alphas_cumprod[t_next]
            else:
                alpha_bar_next = torch.tensor(1.0, device=self.device, dtype=x.dtype)
            x = alpha_bar_next.sqrt() * x_hat_0_corrected + (1.0 - alpha_bar_next).sqrt() * eps

            log.append({
                "step": i, "t": int(t),
                "r_t": float(r_t_val),
                "sigma_n_eff": float(sn),
                "delta_norm": float(delta_x_hat_dpr.flatten(1).norm(dim=1).mean().item()),
            })
            if verbose and (i % max(1, num_steps // 5) == 0):
                print(f"  [{i+1}/{num_steps}] t={int(t)} r_t={r_t_val:.2f} "
                      f"|δx̂₀|={delta_x_hat_dpr.flatten(1).norm(dim=1).mean():.3e}")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        x_hat_pix = ((x.clamp(-1, 1) + 1) / 2).clamp(0, 1)
        return PiGDMPureResult(x_hat=x_hat_pix, nfe=num_steps, elapsed_s=elapsed, log=log)
