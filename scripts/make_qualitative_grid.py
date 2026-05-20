"""Generate a qualitative comparison grid (one row per test image,
columns: clean | y | Wiener | DnCNN+PnP | FlowDPS-on-RF | Pixel-DPS).

Re-runs each method on a small subset since the main-grid scripts only
saved metrics, not images. Picks a middle-difficulty condition by default
(sigma_b=3.0, sigma_n=0.05, NFE=50) so all four methods produce
sensible-looking outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers import DDPMPipeline

from src.baselines.pnp_admm import pnp_admm_deblur
from src.baselines.wiener import wiener_deblur
from src.forward import degrade
from src.forward.degradation import gaussian_blur
from src.metrics.image_metrics import psnr, ssim, LPIPSMetric
from src.models.rf_celebahq import load_rf_model
from src.samplers.flowdps_rf import FlowDPSRF
from src.samplers.pixel_dps import PixelDPS


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [int(x) for x in args.image_ids.split(",")]
    paths = sorted(Path(args.test_dir).glob("*.png"))
    images = torch.stack([load_image(str(paths[i])) for i in image_ids]).to(device)
    print(f"Loaded {len(images)} images: ids {image_ids}")

    # Forward degradation (same y for all methods).
    def forward_op(x, sb=args.sigma_blur):
        return gaussian_blur(x, sigma=sb)

    y = degrade(
        images, blur_type="gaussian",
        blur_sigma=args.sigma_blur, noise_sigma=args.sigma_noise,
        seed=args.seed,
    )

    # ---- Load samplers (once, reused per image) ----
    print("Loading DDPM (Pixel-DPS, DDRM)...")
    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache"
    )
    pixel_dps = PixelDPS(pipe.unet, scheduler_config=pipe.scheduler.config, device=device)

    print("Loading RF model (FlowDPS-on-RF)...")
    rf = load_rf_model(device=device)
    flowdps = FlowDPSRF(rf, device=device)

    print("Loading DnCNN (PnP-ADMM)...")
    from deepinv.models import DnCNN
    dncnn = DnCNN(in_channels=3, out_channels=3, pretrained="download").to(device).eval()
    for p in dncnn.parameters():
        p.requires_grad_(False)

    def denoise(img, sigma):
        return dncnn(img, sigma)

    # ---- Reconstruct per method (batched where possible) ----
    print("\nWiener...")
    x_hat_wiener = wiener_deblur(y, sigma_blur=args.sigma_blur, K=5e-2)

    print("DnCNN+PnP-ADMM...")
    sigma_max = max(0.10, args.sigma_noise * 4 + 0.05)
    sigma_min = max(0.02, args.sigma_noise)
    x_hat_pnp = pnp_admm_deblur(
        y, sigma_blur=args.sigma_blur, sigma_noise=args.sigma_noise,
        denoiser=denoise, num_iters=24, rho=2.0,
        sigma_schedule=(sigma_max, sigma_min),
    )

    # Per-image sampling for the diffusion / flow methods (their samplers
    # are written for batch=1 in the smoke scripts; running per-image is
    # fine for 4-6 images).
    x_hat_pixel = torch.zeros_like(images)
    x_hat_flow = torch.zeros_like(images)

    print(f"Pixel-DPS @ NFE={args.nfe}, zeta={args.zeta_pixel_dps}...")
    for i in range(len(images)):
        y_i = y[i : i + 1]
        res = pixel_dps.sample(
            y=y_i,
            forward_op=lambda z: gaussian_blur(z, sigma=args.sigma_blur),
            num_steps=args.nfe, zeta=args.zeta_pixel_dps,
            sigma_y=max(args.sigma_noise, 1e-3),
            seed=args.seed + i, verbose=False,
        )
        x_hat_pixel[i] = res.x_hat[0]
        print(f"  img {image_ids[i]}: PSNR={psnr(res.x_hat[0], images[i]):.2f}")

    print(f"FlowDPS-on-RF @ NFE={args.nfe}, zeta={args.zeta_flowdps_rf}...")
    for i in range(len(images)):
        y_i = y[i : i + 1]
        res = flowdps.sample(
            y=y_i,
            forward_op=lambda z: gaussian_blur(z, sigma=args.sigma_blur),
            num_steps=args.nfe, zeta=args.zeta_flowdps_rf,
            seed=args.seed + i, verbose=False,
        )
        x_hat_flow[i] = res.x_hat[0]
        print(f"  img {image_ids[i]}: PSNR={psnr(res.x_hat[0], images[i]):.2f}")

    # ---- Compute per-image metrics for the figure caption ----
    lp = LPIPSMetric(device=device)
    psnr_vals = {
        "Wiener": [psnr(x_hat_wiener[i], images[i]) for i in range(len(images))],
        "DnCNN+PnP": [psnr(x_hat_pnp[i], images[i]) for i in range(len(images))],
        "FlowDPS-on-RF": [psnr(x_hat_flow[i], images[i]) for i in range(len(images))],
        "Pixel-DPS": [psnr(x_hat_pixel[i], images[i]) for i in range(len(images))],
    }

    # ---- Render the grid ----
    cols = ["clean", f"y\n(σ_b={args.sigma_blur}, σ_n={args.sigma_noise})",
            "Wiener", "DnCNN+PnP", "FlowDPS-on-RF", "Pixel-DPS"]
    col_tensors = [images, y, x_hat_wiener, x_hat_pnp, x_hat_flow, x_hat_pixel]
    n_rows = len(image_ids)
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(to_np(col_tensors[c][r]))
            ax.set_axis_off()
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            # Add per-method PSNR under the image (skip clean/y columns).
            method = ["", "", "Wiener", "DnCNN+PnP", "FlowDPS-on-RF", "Pixel-DPS"][c]
            if method:
                p_val = psnr_vals[method][r]
                ax.text(
                    0.5, -0.05, f"{p_val:.1f} dB",
                    transform=ax.transAxes,
                    ha="center", va="top", fontsize=9,
                )
    plt.tight_layout()

    out_png = out_dir / args.out_basename
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out_png.with_suffix('.png')} + {out_pdf}")

    print("\nPer-image PSNR summary:")
    print(f"  {'img':>4} {'Wiener':>9} {'PnP':>9} {'FlowDPS':>9} {'PixelDPS':>9}")
    for r in range(n_rows):
        print(
            f"  {image_ids[r]:>4} "
            f"{psnr_vals['Wiener'][r]:>9.2f} "
            f"{psnr_vals['DnCNN+PnP'][r]:>9.2f} "
            f"{psnr_vals['FlowDPS-on-RF'][r]:>9.2f} "
            f"{psnr_vals['Pixel-DPS'][r]:>9.2f}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--image_ids", default="0,3,7,12,18,25",
                   help="comma-separated indices into the sorted test dir")
    p.add_argument("--sigma_blur", type=float, default=3.0)
    p.add_argument("--sigma_noise", type=float, default=0.05)
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--zeta_pixel_dps", type=float, default=10.0)
    p.add_argument("--zeta_flowdps_rf", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="outputs/figures")
    p.add_argument("--out_basename", default="qualitative_grid")
    args = p.parse_args()
    main(args)
