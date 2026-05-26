"""Uncertainty calibration analysis: does posterior sample variance
correlate with image structure (edges) or is it uniform?

For each of the diversity-study images, draws N=8 samples per method,
then for each (image, method) computes:
  - the per-pixel std map across the 8 samples
  - the Sobel gradient magnitude of the CLEAN image (the structure
    reference)
  - the Spearman rank correlation between the two flattened maps

A well-calibrated posterior sampler should have HIGH std where the
clean image has HIGH gradient (edges, hair, eyes, mouth boundaries)
and LOW std in flat regions (cheek, forehead). A high positive
Spearman correlation means uncertainty is image-structure-aware
rather than uniform / noise-like.

Outputs:
  - outputs/results/uncertainty_calibration.csv (one row per
    (image_id, method) with the Spearman rho + p-value)
  - outputs/figures/uncertainty_calibration.{pdf,png} — a 3-row x
    4-column panel: clean | edge map | DPS std | FlowDPS std for the
    same 3 images used by the diversity study.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forward import degrade
from src.forward.degradation import gaussian_blur


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def luminance(rgb: torch.Tensor) -> np.ndarray:
    """ITU-R BT.601 luminance from (3, H, W) [0,1] tensor → (H, W) np array."""
    r, g, b = rgb[0].cpu().numpy(), rgb[1].cpu().numpy(), rgb[2].cpu().numpy()
    return 0.299 * r + 0.587 * g + 0.114 * b


def edge_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude (H, W)."""
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Same images as the diversity study (default: 0, 7, 25).
    image_ids = [int(x) for x in args.image_ids.split(",")]
    test_dir = Path(args.test_dir)
    images = [load_image(str(test_dir / f"{i:05d}.png")) for i in image_ids]
    print(f"Images: {image_ids}")

    # Forward degradation (fixed measurement across seeds and methods).
    images_dev = torch.stack(images).to(device)
    y = degrade(images_dev, blur_type="gaussian",
                blur_sigma=args.sigma_blur, noise_sigma=args.sigma_noise,
                seed=args.measurement_seed)

    # Load samplers.
    from diffusers import DDPMPipeline
    from src.samplers.pixel_dps import PixelDPS
    from src.models.rf_celebahq import load_rf_model
    from src.samplers.flowdps_rf import FlowDPSRF

    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache"
    )
    pixel = PixelDPS(pipe.unet, scheduler_config=pipe.scheduler.config, device=device)
    rf = load_rf_model(device=device, use_non_ema=True)
    flow = FlowDPSRF(rf, device=device)

    # Draw N samples per (image, method).
    n_imgs = len(image_ids)
    n_samples = args.num_samples
    samples = {
        "pixel_dps": np.zeros((n_imgs, n_samples, 3, 256, 256), dtype=np.float32),
        "flowdps_rf": np.zeros((n_imgs, n_samples, 3, 256, 256), dtype=np.float32),
    }
    total = n_imgs * 2 * n_samples
    done = 0
    t0 = time.time()
    for i in range(n_imgs):
        y_i = y[i : i + 1]
        for method, sampler, zeta, is_pixel in (
            ("pixel_dps", pixel, args.zeta_pixel_dps, True),
            ("flowdps_rf", flow, args.zeta_flowdps_rf, False),
        ):
            for s in range(n_samples):
                seed = args.base_seed + s
                if is_pixel:
                    res = sampler.sample(
                        y=y_i, forward_op=lambda z: gaussian_blur(z, sigma=args.sigma_blur),
                        num_steps=args.nfe, zeta=zeta,
                        sigma_y=max(args.sigma_noise, 1e-3),
                        seed=seed, verbose=False,
                    )
                else:
                    res = sampler.sample(
                        y=y_i, forward_op=lambda z: gaussian_blur(z, sigma=args.sigma_blur),
                        num_steps=args.nfe, zeta=zeta,
                        seed=seed, verbose=False,
                    )
                samples[method][i, s] = res.x_hat[0].cpu().numpy()
                done += 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done else float("inf")
                print(f"[{done:>3}/{total}] img={image_ids[i]} method={method} seed={seed} "
                      f"({elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m)", flush=True)

    # Compute per-image edge map + per-(image, method) std map + Spearman rho.
    Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fig_path).parent.mkdir(parents=True, exist_ok=True)
    edge_maps = []
    std_maps = {"pixel_dps": [], "flowdps_rf": []}
    rows = []

    for i, img_id in enumerate(image_ids):
        gray = luminance(images[i])
        edge = edge_magnitude(gray)
        edge_maps.append(edge)
        edge_flat = edge.flatten()

        for method in ("pixel_dps", "flowdps_rf"):
            S = samples[method][i]  # (N, 3, H, W)
            std_per_ch = S.std(axis=0)  # (3, H, W)
            std_gray = 0.299 * std_per_ch[0] + 0.587 * std_per_ch[1] + 0.114 * std_per_ch[2]
            std_maps[method].append(std_gray)
            rho, p = spearmanr(edge_flat, std_gray.flatten())
            rows.append({
                "image_id": img_id, "method": method,
                "spearman_rho": round(float(rho), 4),
                "spearman_p": float(f"{p:.3e}"),
                "n_pixels": int(edge_flat.size),
                "edge_mean": round(float(edge_flat.mean()), 5),
                "std_map_mean": round(float(std_gray.mean()), 5),
                "std_map_max": round(float(std_gray.max()), 5),
            })

    # CSV.
    with open(args.csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {args.csv_path}")

    print("\n=== Spearman ρ (edge-map vs sample std-map) ===")
    for r in rows:
        print(f"  img {r['image_id']:>3} {r['method']:<13}  ρ = {r['spearman_rho']:+.4f}  (p = {r['spearman_p']:.2e})")

    # 4-column figure: clean | edge | DPS std | FlowDPS std, n_imgs rows.
    fig, axes = plt.subplots(n_imgs, 4, figsize=(11, 2.7 * n_imgs))
    if n_imgs == 1:
        axes = axes.reshape(1, -1)
    col_titles = ["Clean image", "Edge magnitude (Sobel)",
                  "Pixel-DPS sample std", "FlowDPS-on-RF sample std"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10)

    for i, img_id in enumerate(image_ids):
        axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i, 1].imshow(edge_maps[i], cmap="magma")
        # Match dynamic range across the two std maps for visual comparability.
        smax = max(std_maps["pixel_dps"][i].max(), std_maps["flowdps_rf"][i].max())
        axes[i, 2].imshow(std_maps["pixel_dps"][i], cmap="hot", vmin=0, vmax=smax)
        axes[i, 3].imshow(std_maps["flowdps_rf"][i], cmap="hot", vmin=0, vmax=smax)
        # Annotate Spearman ρ on the std panels.
        for c, m in ((2, "pixel_dps"), (3, "flowdps_rf")):
            r = next(x for x in rows if x["image_id"] == img_id and x["method"] == m)
            axes[i, c].text(8, 22, f"ρ={r['spearman_rho']:+.2f}",
                            color="white", fontsize=10, weight="bold",
                            bbox=dict(facecolor="black", alpha=0.5, pad=2))
        axes[i, 0].set_ylabel(f"img {img_id}", fontsize=9)
        for c in range(4):
            axes[i, c].set_xticks([]); axes[i, c].set_yticks([])

    fig.suptitle(
        f"Per-pixel sample std vs edge magnitude  "
        f"(σ_b={args.sigma_blur}, σ_n={args.sigma_noise}, NFE={args.nfe}, N={n_samples})",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(args.fig_path, bbox_inches="tight")
    fig.savefig(str(args.fig_path).replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    print(f"Wrote {args.fig_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--image_ids", default="0,7,25",
                   help="Comma-separated image IDs (default matches the diversity study).")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--sigma_blur", type=float, default=3.0)
    p.add_argument("--sigma_noise", type=float, default=0.05)
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--zeta_pixel_dps", type=float, default=10.0)
    p.add_argument("--zeta_flowdps_rf", type=float, default=100.0)
    p.add_argument("--base_seed", type=int, default=100)
    p.add_argument("--measurement_seed", type=int, default=0)
    p.add_argument("--csv_path", default="outputs/results/uncertainty_calibration.csv")
    p.add_argument("--fig_path", default="outputs/figures/uncertainty_calibration.pdf")
    args = p.parse_args()
    main(args)
