"""Run a (method × σ_blur × σ_noise × NFE) grid on a fixed image subset.

Resumable + incremental CSV writes. One row per (method, image, NFE, blur, noise).
Designed for overnight runs: re-launching the script skips rows already done.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forward import degrade
from src.forward.degradation import gaussian_blur, motion_blur
from src.metrics.image_metrics import psnr, ssim, LPIPSMetric
from src.metrics.results_logger import ResultsLogger
from src.metrics.timing import CUDATimer


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def make_pixel_dps_sampler(device):
    from diffusers import DDPMPipeline
    from src.samplers.pixel_dps import PixelDPS
    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache"
    )
    return PixelDPS(pipe.unet, scheduler_config=pipe.scheduler.config, device=device)


def make_flowdps_rf_sampler(device):
    from src.models.rf_celebahq import load_rf_model
    from src.samplers.flowdps_rf import FlowDPSRF
    rf = load_rf_model(device=device)
    return FlowDPSRF(rf, device=device)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve grid axes.
    methods = args.methods.split(",")
    sigma_blurs = [float(x) for x in args.sigma_blurs.split(",")]
    sigma_noises = [float(x) for x in args.sigma_noises.split(",")]
    nfes = [int(x) for x in args.nfes.split(",")]
    zeta_by_method = {
        "pixel_dps": args.zeta_pixel_dps,
        "flowdps_rf": args.zeta_flowdps_rf,
    }

    # Load test images.
    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    print(f"Loading {len(img_paths)} test images from {args.test_dir}")
    images = [load_image(str(p)) for p in img_paths]

    # CSV logger.
    logger = ResultsLogger(args.csv_path)
    key_fields = ("method", "blur_type", "sigma_blur", "sigma_noise", "nfe", "zeta", "image_id")
    already_done = logger.done_keys(key_fields)
    print(f"Resume-skip: {len(already_done)} rows already in {args.csv_path}")

    lpips = LPIPSMetric(device=device)
    samplers = {}

    total_rows = len(methods) * len(sigma_blurs) * len(sigma_noises) * len(nfes) * len(images)
    done = 0
    t_start = time.time()

    for method, sb, sn, nfe in itertools.product(methods, sigma_blurs, sigma_noises, nfes):
        zeta = zeta_by_method[method]
        # Build sampler once per method.
        if method not in samplers:
            print(f"\n=== Building sampler: {method} ===")
            samplers[method] = (
                make_pixel_dps_sampler(device) if method == "pixel_dps"
                else make_flowdps_rf_sampler(device)
            )
        sampler = samplers[method]

        for img_idx, x in enumerate(images):
            key = (method, "gaussian", f"{sb}", f"{sn}", f"{nfe}", f"{zeta}", f"{img_idx}")
            if key in already_done:
                done += 1
                continue

            x_dev = x.unsqueeze(0).to(device)

            # Forward op closure (Gaussian blur).
            def forward_op(z, sb_=sb):
                return gaussian_blur(z, sigma=sb_)

            y = degrade(
                x_dev, blur_type="gaussian",
                blur_sigma=sb, noise_sigma=sn, seed=args.seed + img_idx,
            )

            with CUDATimer() as timer:
                if method == "pixel_dps":
                    res = sampler.sample(
                        y=y, forward_op=forward_op,
                        num_steps=nfe, zeta=zeta, sigma_y=max(sn, 1e-3),
                        seed=args.seed + img_idx, verbose=False,
                    )
                else:
                    res = sampler.sample(
                        y=y, forward_op=forward_op,
                        num_steps=nfe, zeta=zeta,
                        seed=args.seed + img_idx, verbose=False,
                    )
            x_hat = res.x_hat
            p = psnr(x_hat[0], x_dev[0])
            s = ssim(x_hat[0], x_dev[0])
            (l,) = lpips(x_hat, x_dev)

            logger.log({
                "method": method,
                "blur_type": "gaussian",
                "sigma_blur": sb,
                "sigma_noise": sn,
                "motion_length": "",
                "motion_angle": "",
                "nfe": nfe,
                "zeta": zeta,
                "seed": args.seed + img_idx,
                "image_id": img_idx,
                "psnr": round(p, 4),
                "ssim": round(s, 4),
                "lpips": round(float(l), 4),
                "time_s": round(timer.elapsed_s, 3),
                "resolution": 256,
                "notes": "",
            })

            done += 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (total_rows - done) if done else float("inf")
            print(
                f"[{done:>5}/{total_rows}] {method} sb={sb} sn={sn} nfe={nfe} img={img_idx} "
                f"PSNR={p:.2f} SSIM={s:.3f} LPIPS={l:.3f} "
                f"dt={timer.elapsed_s:.1f}s "
                f"(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)"
            )

    logger.close()
    print(f"\nGrid done. {done} rows written to {args.csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="pixel_dps,flowdps_rf")
    p.add_argument("--sigma_blurs", default="1.5,3.0,5.0")
    p.add_argument("--sigma_noises", default="0.0,0.05")
    p.add_argument("--nfes", default="25,50,100")
    p.add_argument("--zeta_pixel_dps", type=float, default=5.0)
    p.add_argument("--zeta_flowdps_rf", type=float, default=10.0)
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--csv_path", default="outputs/results/main_grid.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
