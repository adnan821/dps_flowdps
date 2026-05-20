"""Run DDRM-lite on the main 6-condition Gaussian-deblur grid + 3 NFE."""
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

from diffusers import DDPMPipeline

from src.baselines.ddrm import DDRM
from src.forward import degrade
from src.metrics.image_metrics import psnr, ssim, LPIPSMetric
from src.metrics.results_logger import ResultsLogger
from src.metrics.timing import CUDATimer


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache"
    )
    sampler = DDRM(pipe.unet, pipe.scheduler.config, device=device)
    print(f"DDPM params: {sum(p.numel() for p in pipe.unet.parameters()):,}")

    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    images = [load_image(str(p)) for p in img_paths]
    print(f"Loaded {len(images)} images")

    sigma_blurs = [float(x) for x in args.sigma_blurs.split(",")]
    sigma_noises = [float(x) for x in args.sigma_noises.split(",")]
    nfes = [int(x) for x in args.nfes.split(",")]

    logger = ResultsLogger(args.csv_path)
    key_fields = ("method", "blur_type", "sigma_blur", "sigma_noise", "nfe", "zeta", "image_id")
    already_done = logger.done_keys(key_fields)
    print(f"Resume-skip: {len(already_done)} rows already in {args.csv_path}")

    lpips_metric = LPIPSMetric(device=device)

    total = len(sigma_blurs) * len(sigma_noises) * len(nfes) * len(images)
    done = 0
    t_start = time.time()

    for sb, sn, nfe in itertools.product(sigma_blurs, sigma_noises, nfes):
        for img_idx, x in enumerate(images):
            key = ("ddrm", "gaussian", f"{sb}", f"{sn}", f"{nfe}", f"{args.eta}", f"{img_idx}")
            if key in already_done:
                done += 1
                continue

            x_dev = x.unsqueeze(0).to(device)
            y = degrade(
                x_dev, blur_type="gaussian",
                blur_sigma=sb, noise_sigma=sn, seed=args.seed + img_idx,
            )

            with CUDATimer() as timer:
                res = sampler.sample(
                    y=y, sigma_blur=sb, sigma_noise=sn,
                    num_steps=nfe, eta=args.eta, seed=args.seed + img_idx,
                )
            x_hat = res.x_hat
            p = psnr(x_hat[0], x_dev[0])
            s = ssim(x_hat[0], x_dev[0])
            (l,) = lpips_metric(x_hat, x_dev)

            logger.log({
                "method": "ddrm",
                "blur_type": "gaussian",
                "sigma_blur": sb,
                "sigma_noise": sn,
                "motion_length": "",
                "motion_angle": "",
                "nfe": nfe,
                "zeta": args.eta,
                "seed": args.seed + img_idx,
                "image_id": img_idx,
                "psnr": round(p, 4),
                "ssim": round(s, 4),
                "lpips": round(float(l), 4),
                "time_s": round(timer.elapsed_s, 3),
                "resolution": 256,
                "notes": f"DDRM-lite eta={args.eta}",
            })

            done += 1
            if done % 25 == 0 or done == total:
                elapsed = time.time() - t_start
                eta_s = elapsed / done * (total - done) if done else float("inf")
                print(
                    f"[{done:>4}/{total}] ddrm sb={sb} sn={sn} nfe={nfe} img={img_idx} "
                    f"PSNR={p:.2f} SSIM={s:.3f} LPIPS={l:.3f} "
                    f"dt={timer.elapsed_s:.1f}s "
                    f"(elapsed {elapsed/60:.1f}m, ETA {eta_s/60:.1f}m)",
                    flush=True,
                )

    logger.close()
    print(f"\nDDRM done. {done} rows in {args.csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sigma_blurs", default="1.5,3.0,5.0")
    p.add_argument("--sigma_noises", default="0.0,0.05")
    p.add_argument("--nfes", default="25,50,100")
    p.add_argument("--eta", type=float, default=0.85)
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--csv_path", default="outputs/results/baselines.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
