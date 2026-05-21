"""Validation sweep for a Pixel-DPS time-varying ζ schedule.

`pixel_dps_v2` (scalar ζ=30) failed the gate by over-correcting easy
cells. Hypothesis: a step-dependent ζ — moderate guidance that tapers
toward the data-side steps so the prior finishes the reconstruction —
beats a scalar. We sweep `zeta_power` (decay toward data) and
`zeta_ramp` (rise toward data) shapes here.

Tuned on HELD-OUT images (default offset 50, i.e. images 50..54) so
nothing leaks from the 50-image evaluation set (images 0..49).
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
from src.metrics.image_metrics import psnr
from src.samplers.schedules import zeta_constant, zeta_power, zeta_ramp


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_paths = sorted(Path(args.test_dir).glob("*.png"))
    val_paths = all_paths[args.val_offset : args.val_offset + args.num_images]
    images = [load_image(str(p)).unsqueeze(0).to(device) for p in val_paths]
    print(f"Held-out validation images: {[p.name for p in val_paths]}")

    from diffusers import DDPMPipeline
    from src.samplers.pixel_dps import PixelDPS
    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache"
    )
    sampler = PixelDPS(pipe.unet, scheduler_config=pipe.scheduler.config, device=device)

    # (label, zeta-schedule) candidates.
    candidates = [
        ("const(10)  [v1 ref]", zeta_constant(10.0)),
        ("const(30)  [v2 ref]", zeta_constant(30.0)),
        ("power(20,1.0)", zeta_power(20.0, 1.0)),
        ("power(40,1.0)", zeta_power(40.0, 1.0)),
        ("power(40,0.5)", zeta_power(40.0, 0.5)),
        ("power(80,0.5)", zeta_power(80.0, 0.5)),
        ("ramp(40,0.5)", zeta_ramp(40.0, 0.5)),
        ("ramp(80,0.5)", zeta_ramp(80.0, 0.5)),
    ]
    cells = [(float(c.split(",")[0]), float(c.split(",")[1]))
             for c in args.cells.split(";")]
    nfe = args.nfe

    print(f"\nNFE={nfe}, cells={cells}, {len(images)} val imgs each")
    header = f"{'schedule':<22}" + "".join(f"  sb={sb}/sn={sn}" for sb, sn in cells) + f"  {'overall':>8}"
    print(header)
    print("-" * len(header))

    for label, zeta in candidates:
        cell_means = []
        for sb, sn in cells:
            psnrs = []
            for img_idx, x_dev in enumerate(images):
                y = degrade(x_dev, blur_type="gaussian", blur_sigma=sb,
                            noise_sigma=sn, seed=args.seed + img_idx)

                def forward_op(z, sb_=sb):
                    return gaussian_blur(z, sigma=sb_)

                res = sampler.sample(
                    y=y, forward_op=forward_op, num_steps=nfe, zeta=zeta,
                    sigma_y=max(sn, 1e-3), seed=args.seed + img_idx, verbose=False,
                )
                psnrs.append(psnr(res.x_hat[0], x_dev[0]))
            cell_means.append(sum(psnrs) / len(psnrs))
        overall = sum(cell_means) / len(cell_means)
        cells_str = "".join(f"  {m:>10.2f}" for m in cell_means)
        print(f"{label:<22}{cells_str}  {overall:>8.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--val_offset", type=int, default=50,
                   help="Start index for held-out validation imgs (eval set is 0..49).")
    p.add_argument("--num_images", type=int, default=5)
    p.add_argument("--cells", default="1.5,0.0;3.0,0.05;5.0,0.05",
                   help="Semicolon-separated sb,sn cells.")
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
