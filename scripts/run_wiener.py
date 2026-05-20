"""Run Wiener-filter deblurring on the same 18-condition grid as run_grid.py.

Writes outputs/results/baselines.csv with method='wiener'. CPU-only and fast
(~10 ms/image), so the whole grid finishes in well under a minute.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.wiener import wiener_deblur
from src.forward import degrade
from src.metrics.image_metrics import psnr, ssim, LPIPSMetric
from src.metrics.results_logger import ResultsLogger
from src.metrics.timing import CUDATimer


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# K (noise-to-signal scalar) — slightly tuned per noise level. Small K is
# closer to pure inversion (sharp but noisy); large K is closer to identity
# (smooth but blurry).
WIENER_K = {0.0: 1e-3, 0.05: 5e-2}


def main(args):
    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    print(f"Loading {len(img_paths)} test images from {args.test_dir}")
    images = [load_image(str(p)) for p in img_paths]

    sigma_blurs = [float(x) for x in args.sigma_blurs.split(",")]
    sigma_noises = [float(x) for x in args.sigma_noises.split(",")]

    logger = ResultsLogger(args.csv_path)
    key_fields = ("method", "blur_type", "sigma_blur", "sigma_noise", "nfe", "zeta", "image_id")
    already_done = logger.done_keys(key_fields)
    print(f"Resume-skip: {len(already_done)} rows already in {args.csv_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_metric = LPIPSMetric(device=device)

    total = len(sigma_blurs) * len(sigma_noises) * len(images)
    done = 0
    t_start = time.time()

    for sb, sn in itertools.product(sigma_blurs, sigma_noises):
        K = WIENER_K.get(sn, 1e-3)
        for img_idx, x in enumerate(images):
            key = ("wiener", "gaussian", f"{sb}", f"{sn}", "0", f"{K}", f"{img_idx}")
            if key in already_done:
                done += 1
                continue

            x_dev = x.unsqueeze(0).to(device)
            y = degrade(
                x_dev, blur_type="gaussian",
                blur_sigma=sb, noise_sigma=sn, seed=args.seed + img_idx,
            )

            with CUDATimer() as timer:
                x_hat = wiener_deblur(y, sigma_blur=sb, K=K)

            p = psnr(x_hat[0], x_dev[0])
            s = ssim(x_hat[0], x_dev[0])
            (l,) = lpips_metric(x_hat, x_dev)

            logger.log({
                "method": "wiener",
                "blur_type": "gaussian",
                "sigma_blur": sb,
                "sigma_noise": sn,
                "motion_length": "",
                "motion_angle": "",
                "nfe": 0,
                "zeta": K,
                "seed": args.seed + img_idx,
                "image_id": img_idx,
                "psnr": round(p, 4),
                "ssim": round(s, 4),
                "lpips": round(float(l), 4),
                "time_s": round(timer.elapsed_s, 4),
                "resolution": 256,
                "notes": f"wiener K={K}",
            })

            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done) if done else float("inf")
                print(
                    f"[{done:>5}/{total}] wiener sb={sb} sn={sn} img={img_idx} "
                    f"PSNR={p:.2f} SSIM={s:.3f} LPIPS={l:.3f} "
                    f"dt={timer.elapsed_s*1000:.0f}ms "
                    f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)",
                    flush=True,
                )

    logger.close()
    print(f"\nWiener baseline done. {done} rows in {args.csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sigma_blurs", default="1.5,3.0,5.0")
    p.add_argument("--sigma_noises", default="0.0,0.05")
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--csv_path", default="outputs/results/baselines.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
