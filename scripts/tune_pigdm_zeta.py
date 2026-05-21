"""Tiny ζ sweep for the post-fix Tier-C samplers.

After switching the Π-GDM path to L2-squared + unnormalized W, the
effective ζ scale moves by orders of magnitude. Sweep a couple
decades on 2 imgs × 1 representative cell to find a working scalar.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forward import degrade
from src.forward.degradation import gaussian_blur
from src.forward.ops_fft import gaussian_otf
from src.metrics.image_metrics import psnr


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    images = [load_image(str(p)).unsqueeze(0).to(device) for p in img_paths]
    print(f"Loaded {len(images)} images")

    sb, sn, nfe = args.sigma_blur, args.sigma_noise, args.nfe
    H_otf = gaussian_otf(sb, (256, 256), device=device)

    if args.method == "pixel_dps":
        from diffusers import DDPMPipeline
        from src.samplers.pixel_dps import PixelDPS
        pipe = DDPMPipeline.from_pretrained(
            "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache"
        )
        sampler = PixelDPS(pipe.unet, scheduler_config=pipe.scheduler.config, device=device)
        is_pixel = True
    else:
        from src.models.rf_celebahq import load_rf_model
        from src.samplers.flowdps_rf import FlowDPSRF
        rf = load_rf_model(device=device, use_non_ema=False)
        sampler = FlowDPSRF(rf, device=device)
        is_pixel = False

    zetas = [float(z) for z in args.zetas.split(",")]
    print(f"\nSweep: method={args.method}  cell sb={sb}, sn={sn}, NFE={nfe}")
    print(f"{'zeta':>12}  " + "  ".join(f"img{i}_PSNR" for i in range(len(images))) + f"  {'mean':>6}  {'time':>5}")

    for zeta in zetas:
        psnrs = []
        t0 = time.time()
        for img_idx, x_dev in enumerate(images):
            y = degrade(x_dev, blur_type="gaussian", blur_sigma=sb, noise_sigma=sn, seed=args.seed + img_idx)

            def forward_op(z, sb_=sb):
                return gaussian_blur(z, sigma=sb_)

            if is_pixel:
                res = sampler.sample(
                    y=y, forward_op=forward_op, num_steps=nfe, zeta=zeta,
                    sigma_y=max(sn, 1e-3), seed=args.seed + img_idx, verbose=False,
                    pigdm_otf=H_otf, pigdm_sigma_n=max(sn, 1e-3),
                )
            else:
                res = sampler.sample(
                    y=y, forward_op=forward_op, num_steps=nfe, zeta=zeta,
                    seed=args.seed + img_idx, verbose=False,
                    pigdm_otf=H_otf, pigdm_sigma_n=max(sn, 1e-3),
                )
            p = psnr(res.x_hat[0], x_dev[0])
            psnrs.append(p)
        mean_p = sum(psnrs) / len(psnrs)
        dt = time.time() - t0
        print(f"  {zeta:>10.4g}  " + "  ".join(f"{p:>9.2f}" for p in psnrs) + f"  {mean_p:>6.2f}  {dt:>5.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["pixel_dps", "flowdps_rf"], default="pixel_dps")
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--num_images", type=int, default=2)
    p.add_argument("--sigma_blur", type=float, default=3.0)
    p.add_argument("--sigma_noise", type=float, default=0.05)
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--zetas", default="100,10,1,0.1,0.01,0.001,0.0001,0.00001")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
