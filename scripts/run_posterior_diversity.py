"""Posterior-diversity study.

For each of (Pixel-DPS, FlowDPS-on-RF), on a small set of representative
test images, draw N samples with different seeds at fixed (sigma_b,
sigma_n, NFE). Report:

- Single-sample PSNR (one seed, what we normally call the "reconstruction")
- Mean-of-N PSNR (PSNR of the per-pixel mean of N samples)
- Best-of-N PSNR (max PSNR across the N samples)
- Pairwise mean LPIPS (a perceptual measure of posterior diversity)

Also save a visual: per-pixel std map across the N samples, alongside the
ground truth, the measurement, four sample reconstructions, and the
sample mean. One row per method per chosen image.

Outputs:
- outputs/results/posterior_diversity.csv (one row per (image, method, sample))
- outputs/results/posterior_diversity_summary.csv (one row per (image, method))
- outputs/figures/posterior_diversity.{png,pdf}
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers import DDPMPipeline

from src.forward import degrade
from src.forward.degradation import gaussian_blur
from src.metrics.image_metrics import psnr, LPIPSMetric
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
    image_ids = [int(x) for x in args.image_ids.split(",")]
    paths = sorted(Path(args.test_dir).glob("*.png"))
    images = torch.stack([load_image(str(paths[i])) for i in image_ids]).to(device)

    # Forward degradation (identical across seeds and methods).
    y = degrade(
        images, blur_type="gaussian",
        blur_sigma=args.sigma_blur, noise_sigma=args.sigma_noise,
        seed=args.measurement_seed,
    )

    # ---- Load samplers ----
    print("Loading Pixel-DPS...")
    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache"
    )
    pixel_dps = PixelDPS(pipe.unet, scheduler_config=pipe.scheduler.config, device=device)

    print("Loading FlowDPS-on-RF...")
    rf = load_rf_model(device=device)
    flowdps = FlowDPSRF(rf, device=device)

    lpips_metric = LPIPSMetric(device=device)

    # ---- Draw N samples per (image, method) ----
    # Shape: (n_images, n_methods, n_samples, C, H, W)
    n_imgs = len(image_ids)
    n_samples = args.num_samples
    samples = {
        "pixel_dps": torch.zeros(n_imgs, n_samples, 3, 256, 256),
        "flowdps_rf": torch.zeros(n_imgs, n_samples, 3, 256, 256),
    }
    per_seed_psnr = {"pixel_dps": np.zeros((n_imgs, n_samples)),
                     "flowdps_rf": np.zeros((n_imgs, n_samples))}

    t_start = time.time()
    total = n_imgs * 2 * n_samples
    done = 0

    for img_idx in range(n_imgs):
        x_clean = images[img_idx]
        y_i = y[img_idx : img_idx + 1]

        for method, sampler, zeta in (
            ("pixel_dps", pixel_dps, args.zeta_pixel_dps),
            ("flowdps_rf", flowdps, args.zeta_flowdps_rf),
        ):
            for s in range(n_samples):
                seed = args.base_seed + s
                if method == "pixel_dps":
                    res = sampler.sample(
                        y=y_i,
                        forward_op=lambda z: gaussian_blur(z, sigma=args.sigma_blur),
                        num_steps=args.nfe, zeta=zeta,
                        sigma_y=max(args.sigma_noise, 1e-3),
                        seed=seed, verbose=False,
                    )
                else:
                    res = sampler.sample(
                        y=y_i,
                        forward_op=lambda z: gaussian_blur(z, sigma=args.sigma_blur),
                        num_steps=args.nfe, zeta=zeta,
                        seed=seed, verbose=False,
                    )
                samples[method][img_idx, s] = res.x_hat[0].cpu()
                per_seed_psnr[method][img_idx, s] = psnr(res.x_hat[0], x_clean)
                done += 1
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done) if done else float("inf")
                print(
                    f"[{done:>3}/{total}] img={image_ids[img_idx]} method={method} seed={seed} "
                    f"PSNR={per_seed_psnr[method][img_idx, s]:.2f} "
                    f"(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)",
                    flush=True,
                )

    # ---- Diversity / accuracy summary ----
    Path(args.csv_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    summary_rows = []
    for img_idx in range(n_imgs):
        x_clean = images[img_idx]
        for method in ("pixel_dps", "flowdps_rf"):
            S = samples[method][img_idx].to(device)  # (N, 3, 256, 256)
            psnrs = per_seed_psnr[method][img_idx]
            single = float(psnrs[0])
            best = float(psnrs.max())
            # Mean-of-N: pixel-wise average across samples, then PSNR.
            mean_img = S.mean(dim=0)
            mean_psnr = psnr(mean_img, x_clean)
            # Per-pixel std map (averaged across channels).
            std_map = S.std(dim=0).mean(dim=0)  # (H, W)
            # Pairwise mean LPIPS over n*(n-1)/2 pairs.
            n = S.shape[0]
            pairs = []
            for i, j in itertools.combinations(range(n), 2):
                d = lpips_metric(S[i : i + 1], S[j : j + 1])
                pairs.append(float(d[0]))
            div_lpips = float(np.mean(pairs))

            summary_rows.append({
                "image_id": image_ids[img_idx],
                "method": method,
                "n_samples": n_samples,
                "psnr_single": round(single, 3),
                "psnr_mean_of_n": round(mean_psnr, 3),
                "psnr_best_of_n": round(best, 3),
                "pairwise_lpips_mean": round(div_lpips, 4),
                "std_map_mean": round(float(std_map.mean()), 5),
                "std_map_max": round(float(std_map.max()), 5),
            })
            for s in range(n_samples):
                rows.append({
                    "image_id": image_ids[img_idx],
                    "method": method,
                    "seed": args.base_seed + s,
                    "psnr": round(float(psnrs[s]), 3),
                })

    # Save CSVs.
    raw_path = Path(args.csv_dir) / "posterior_diversity.csv"
    sum_path = Path(args.csv_dir) / "posterior_diversity_summary.csv"
    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys()); w.writeheader(); w.writerows(summary_rows)
    print(f"\nWrote {raw_path}")
    print(f"Wrote {sum_path}")

    # ---- Pretty print summary ----
    print(f"\n{'img':>4} {'method':<14} {'single':>7} {'mean-N':>7} {'best-N':>7} "
          f"{'div(LPIPS)':>11} {'std_mean':>9}")
    for r in summary_rows:
        print(f"{r['image_id']:>4} {r['method']:<14} {r['psnr_single']:>7.2f} "
              f"{r['psnr_mean_of_n']:>7.2f} {r['psnr_best_of_n']:>7.2f} "
              f"{r['pairwise_lpips_mean']:>11.4f} {r['std_map_mean']:>9.5f}")

    # ---- Render figure ----
    # Layout: one BIG row per (image, method).
    # Columns: clean | y | sample 1 | sample 2 | sample 3 | sample 4 | mean | std_map
    rows_fig = []
    titles = ["clean", "y", "sample 1", "sample 2", "sample 3", "sample 4", "mean", "std map"]
    for img_idx in range(n_imgs):
        for method in ("pixel_dps", "flowdps_rf"):
            S = samples[method][img_idx]  # (N, 3, 256, 256) cpu
            row = [
                images[img_idx],   # clean
                y[img_idx],        # y
                S[0], S[1], S[2], S[3],   # 4 samples
                S.mean(dim=0),     # mean
                S.std(dim=0).mean(dim=0, keepdim=True),  # std map (1, H, W)
            ]
            rows_fig.append((image_ids[img_idx], method, row))

    n_rows_fig = len(rows_fig)
    n_cols_fig = len(titles)
    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(2 * n_cols_fig, 2.2 * n_rows_fig))
    if n_rows_fig == 1:
        axes = axes[None, :]
    for r_idx, (iid, method, row) in enumerate(rows_fig):
        for c_idx, tensor in enumerate(row):
            ax = axes[r_idx, c_idx]
            if c_idx == n_cols_fig - 1:  # std map column
                img = tensor.detach().cpu().clamp(0, None).numpy()
                if img.ndim == 3:
                    img = img.squeeze(0)
                im = ax.imshow(img, cmap="hot")
                # Per-axis colorbar makes the figure cluttered; skip and rely on cbars in caption.
            else:
                ax.imshow(to_np(tensor))
            ax.set_axis_off()
            if r_idx == 0:
                ax.set_title(titles[c_idx], fontsize=10)
        # Row label on the left.
        axes[r_idx, 0].set_ylabel(
            f"img {iid}\n{method}", fontsize=9, rotation=0, ha="right", va="center",
            labelpad=30,
        )
        # Restore axis off effects but keep ylabel by re-enabling ticks=off only on data axes.
        axes[r_idx, 0].axis("on")
        axes[r_idx, 0].set_xticks([])
        axes[r_idx, 0].set_yticks([])
        for spine in axes[r_idx, 0].spines.values():
            spine.set_visible(False)
    plt.tight_layout()

    out_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "posterior_diversity.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "posterior_diversity.pdf", bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out_dir / 'posterior_diversity.png'} + .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--image_ids", default="0,7,25",
                   help="comma-separated indices into the sorted test dir")
    p.add_argument("--num_samples", type=int, default=8,
                   help="N samples per (image, method)")
    p.add_argument("--sigma_blur", type=float, default=3.0)
    p.add_argument("--sigma_noise", type=float, default=0.05)
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--zeta_pixel_dps", type=float, default=10.0)
    p.add_argument("--zeta_flowdps_rf", type=float, default=100.0)
    p.add_argument("--base_seed", type=int, default=100,
                   help="first sampler seed; samples 0..N-1 use base+0..base+N-1")
    p.add_argument("--measurement_seed", type=int, default=0,
                   help="seed used to draw the measurement noise (held fixed)")
    p.add_argument("--csv_dir", default="outputs/results")
    p.add_argument("--fig_dir", default="outputs/figures")
    args = p.parse_args()
    main(args)
