"""Run DnCNN + PnP-ADMM on the main 6-condition Gaussian-deblur grid.

Writes to outputs/results/baselines.csv with method='dncnn_pnp_admm'.
Same 6-condition shape as Wiener (no NFE axis, but we record num_iters).
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

from src.baselines.pnp_admm import pnp_admm_deblur
from src.forward import degrade
from src.metrics.image_metrics import psnr, ssim, LPIPSMetric
from src.metrics.results_logger import ResultsLogger
from src.metrics.timing import CUDATimer


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def build_denoiser(name: str, device):
    from deepinv.models import DnCNN, DRUNet
    if name == "dncnn":
        net = DnCNN(in_channels=3, out_channels=3, pretrained="download")
    elif name == "drunet":
        net = DRUNet(in_channels=3, out_channels=3, pretrained="download")
    else:
        raise ValueError(name)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    def denoise(img: torch.Tensor, sigma: float) -> torch.Tensor:
        return net(img, sigma)

    return denoise, sum(p.numel() for p in net.parameters())


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    denoise, n_params = build_denoiser(args.denoiser, device)
    method_name = f"{args.denoiser}_pnp_admm"
    print(f"Denoiser: {args.denoiser} ({n_params:,} params)")

    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    images = [load_image(str(p)) for p in img_paths]
    print(f"Loaded {len(images)} images")

    sigma_blurs = [float(x) for x in args.sigma_blurs.split(",")]
    sigma_noises = [float(x) for x in args.sigma_noises.split(",")]

    logger = ResultsLogger(args.csv_path)
    key_fields = ("method", "blur_type", "sigma_blur", "sigma_noise", "nfe", "zeta", "image_id")
    already_done = logger.done_keys(key_fields)
    print(f"Resume-skip: {len(already_done)} rows already in {args.csv_path}")

    lpips_metric = LPIPSMetric(device=device)

    total = len(sigma_blurs) * len(sigma_noises) * len(images)
    done = 0
    t_start = time.time()

    for sb, sn in itertools.product(sigma_blurs, sigma_noises):
        # Noise-adaptive sigma schedule. Tuned on a 4-img sweep.
        smax = max(0.10, sn * 4 + 0.05)
        smin = max(0.02, sn)
        for img_idx, x in enumerate(images):
            key = (
                method_name, "gaussian",
                f"{sb}", f"{sn}",
                f"{args.num_iters}", f"{args.rho}", f"{img_idx}",
            )
            if key in already_done:
                done += 1
                continue

            x_dev = x.unsqueeze(0).to(device)
            y = degrade(
                x_dev, blur_type="gaussian",
                blur_sigma=sb, noise_sigma=sn, seed=args.seed + img_idx,
            )

            with CUDATimer() as timer:
                x_hat = pnp_admm_deblur(
                    y, sigma_blur=sb, sigma_noise=sn,
                    denoiser=denoise,
                    num_iters=args.num_iters, rho=args.rho,
                    sigma_schedule=(smax, smin),
                )

            p = psnr(x_hat[0], x_dev[0])
            s = ssim(x_hat[0], x_dev[0])
            (l,) = lpips_metric(x_hat, x_dev)

            logger.log({
                "method": method_name,
                "blur_type": "gaussian",
                "sigma_blur": sb,
                "sigma_noise": sn,
                "motion_length": "",
                "motion_angle": "",
                "nfe": args.num_iters,  # PnP-ADMM iters
                "zeta": args.rho,
                "seed": args.seed + img_idx,
                "image_id": img_idx,
                "psnr": round(p, 4),
                "ssim": round(s, 4),
                "lpips": round(float(l), 4),
                "time_s": round(timer.elapsed_s, 3),
                "resolution": 256,
                "notes": f"PnP-ADMM iters={args.num_iters} rho={args.rho} sigma=[{args.sigma_max},{args.sigma_min}]",
            })

            done += 1
            if done % 25 == 0 or done == total:
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done) if done else float("inf")
                print(
                    f"[{done:>4}/{total}] {method_name} sb={sb} sn={sn} img={img_idx} "
                    f"PSNR={p:.2f} SSIM={s:.3f} LPIPS={l:.3f} "
                    f"dt={timer.elapsed_s:.2f}s "
                    f"(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)",
                    flush=True,
                )

    logger.close()
    print(f"\n{method_name} done. {done} rows in {args.csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser", choices=["dncnn", "drunet"], default="dncnn")
    p.add_argument("--sigma_blurs", default="1.5,3.0,5.0")
    p.add_argument("--sigma_noises", default="0.0,0.05")
    p.add_argument("--num_iters", type=int, default=24)
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--sigma_max", type=float, default=0.10)
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--csv_path", default="outputs/results/baselines.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
