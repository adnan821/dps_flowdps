"""Diffusion Posterior Sampling (DPS) on a pixel-space DDPM.

Following Chung et al. 2023, "Diffusion Posterior Sampling for General Noisy
Inverse Problems". Uses Tweedie's formula to get a clean-image estimate
x_hat_0 from a noisy x_t, then injects a measurement-likelihood gradient
into the DDIM reverse update.

Conventions:
- Diffusion model lives in `[-1, 1]` space (standard for DDPM checkpoints).
- The user-facing forward operator and measurements `y` live in `[0, 1]`.
- The sampler converts between them as needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from diffusers import DDIMScheduler, UNet2DModel


@dataclass
class DPSResult:
    x_hat: torch.Tensor    # final reconstruction, [0, 1], shape (B, C, H, W)
    nfe: int               # number of UNet forward+backward calls
    elapsed_s: float       # GPU wall time
    log: list[dict]        # per-step diagnostics (loss, grad-norm, etc.)


class PixelDPS:
    """DPS sampler around a HuggingFace `diffusers` DDPM `UNet2DModel`."""

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler_config: dict | None = None,
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device)
        self.unet = unet.to(self.device).eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)

        # We use DDIM rather than DDPM for the reverse pass — fewer steps
        # at equivalent quality, and the deterministic update lets us write
        # a clean Tweedie + guidance loop.
        if scheduler_config is None:
            from diffusers import DDPMScheduler
            scheduler_config = DDPMScheduler().config
        self.scheduler = DDIMScheduler.from_config(scheduler_config)

    # ---- internals -----------------------------------------------------

    def _tweedie_x0(self, x_t: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        """Predict the clean image x_0 from x_t and the model's epsilon."""
        return (x_t - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()

    # ---- public API ----------------------------------------------------

    @torch.no_grad()
    def encode_decode_passthrough(self, x_pix: torch.Tensor) -> torch.Tensor:
        """Used only for sanity tests — no-op since this sampler is pixel-space."""
        return x_pix

    def sample(
        self,
        y: torch.Tensor,
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int = 100,
        zeta: float = 1.0,
        sigma_y: float = 0.05,
        seed: Optional[int] = 0,
        verbose: bool = False,
    ) -> DPSResult:
        """Run DPS to recover `x` from a measurement `y = A(x) + noise`.

        Args:
            y: degraded observation in `[0, 1]`, shape (B, C, H, W).
            forward_op: differentiable function mapping `[0,1]` images to `y`-space
                (same `A` used to generate `y`).
            num_steps: number of DDIM reverse steps (NFE budget per image).
            zeta: likelihood-gradient guidance scale.
            sigma_y: assumed measurement-noise std for the Gaussian likelihood
                (we use `0.5 * ||y - A(x_hat_0)||^2 / sigma_y^2`).
            seed: per-image RNG seed for reproducibility.
        """
        import time

        B = y.shape[0]
        y = y.to(self.device)

        # Initialize x_T ~ N(0, I) in [-1, 1] space (standard DDPM initial).
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn(y.shape, generator=gen, device=self.device, dtype=torch.float32)

        # Set up the reverse schedule.
        self.scheduler.set_timesteps(num_steps, device=self.device)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        log = []
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        t0 = time.time()

        for i, t in enumerate(self.scheduler.timesteps):
            x = x.detach().requires_grad_(True)

            # Forward pass through UNet (frozen).
            eps = self.unet(x, t).sample
            alpha_bar_t = alphas_cumprod[t]
            x_hat_0 = self._tweedie_x0(x, eps, alpha_bar_t)

            # Likelihood gradient. Forward op operates on [0, 1].
            # IMPORTANT: do NOT clamp inside the gradient path — clamp() zeros
            # the gradient for any pixel outside [-1, 1], which kills DPS at
            # early timesteps when x_hat_0 is far from the data manifold.
            x_hat_0_pix = (x_hat_0 + 1) / 2
            y_hat = forward_op(x_hat_0_pix)
            residual = (y_hat - y)
            # Chung et al. 2023 use the L2 norm (not squared) for the
            # likelihood gradient. This keeps the gradient magnitude bounded
            # and lets a fixed zeta work across all timesteps.
            loss = torch.linalg.norm(residual.flatten(1), dim=1).sum()
            grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
            grad_norm = grad.flatten(1).norm(dim=1).mean().item()

            # DDIM update (no guidance) — using the same eps we computed.
            with torch.no_grad():
                x_prev = self.scheduler.step(eps.detach(), t, x.detach()).prev_sample
                # DPS likelihood-gradient injection on top of the DDIM step.
                x = x_prev - zeta * grad

            log.append({
                "step": i,
                "t": int(t),
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm),
            })
            if verbose and (i % max(1, num_steps // 5) == 0):
                print(f"  [{i+1}/{num_steps}] t={int(t)} loss={loss.item():.4f} |grad|={grad_norm:.4e}")

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        elapsed = time.time() - t0

        x_hat_pix = ((x.detach().clamp(-1, 1) + 1) / 2).clamp(0, 1)
        return DPSResult(x_hat=x_hat_pix, nfe=num_steps, elapsed_s=elapsed, log=log)
