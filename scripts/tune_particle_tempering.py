"""Held-out sweep for the SMC tempering schedule on `ParticleDPS`.

Round-3 measured the un-tempered baseline at ~11.8 dB across all cells —
classic particle collapse (peaky weights at σ_y=0.05 → all mass collapses
onto one particle on the first resample → unguided DDIM thereafter).
Fix-#9 adds an annealed `σ_y_eff(i) = σ_y · (1 + (T_max−1)(1−i/N)^α)`.
Sweep T_max on held-out images 50..52 at one representative cell, pick
the best, then run the full grid via run_grid.py --tempering_max=BEST.

Quick sanity: T_max=1 should reproduce the 11.8 dB un-tempered baseline.
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


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paths = sorted(Path(args.test_dir).glob("*.png"))
    val_paths = all_paths[args.val_offset : args.val_offset + args.num_images]
    print(f"Held-out validation images (offset {args.val_offset}): "
          f"{[p.name for p in val_paths]}")
    images = [load_image(str(p)).unsqueeze(0).to(device) for p in val_paths]

    from diffusers import DDPMPipeline
    from src.samplers.particle_dps import ParticleDPS
    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache"
    )
    sampler = ParticleDPS(
        pipe.unet, scheduler_config=pipe.scheduler.config,
        device=device, num_particles=args.num_particles,
    )

    T_max_values = [float(v) for v in args.tempering_max.split(",")]
    sb, sn, nfe, alpha = args.sigma_blur, args.sigma_noise, args.nfe, args.tempering_alpha
    print(f"\nCell: sb={sb}, sn={sn}, NFE={nfe}, P={args.num_particles}, α={alpha}")
    print(f"{'T_max':>8}  " + "  ".join(f"img{i:>2}_PSNR" for i in range(len(images)))
          + f"  {'mean':>7}  {'time':>6}")

    for T_max in T_max_values:
        t0 = time.time()
        psnrs = []
        for k, x_dev in enumerate(images):
            y = degrade(x_dev, blur_type="gaussian", blur_sigma=sb,
                        noise_sigma=sn, seed=args.seed + k)
            res = sampler.sample(
                y=y, forward_op=lambda z: gaussian_blur(z, sigma=sb),
                num_steps=nfe, sigma_y=max(sn, 1e-3),
                tempering_max=T_max, tempering_alpha=alpha,
                force_resample=args.force_resample,
                seed=args.seed + k, verbose=False,
            )
            psnrs.append(psnr(res.x_hat[0], x_dev[0]))
        mean = sum(psnrs) / len(psnrs)
        dt = time.time() - t0
        print(f"{T_max:>8.2f}  " + "  ".join(f"{p:>9.2f}" for p in psnrs)
              + f"  {mean:>7.2f}  {dt:>5.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--val_offset", type=int, default=50,
                   help="Held-out image start index; eval set is 0..49.")
    p.add_argument("--num_images", type=int, default=3)
    p.add_argument("--num_particles", type=int, default=8)
    p.add_argument("--sigma_blur", type=float, default=3.0)
    p.add_argument("--sigma_noise", type=float, default=0.05)
    p.add_argument("--nfe", type=int, default=100)
    p.add_argument("--tempering_max", default="1,3,10,30,100",
                   help="Comma-separated T_max values to sweep.")
    p.add_argument("--tempering_alpha", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=100,
                   help="Base seed (offset 100 to match the diversity study).")
    p.add_argument("--force_resample", action="store_true",
                   help="Force multinomial resampling at every step regardless of ESS.")
    args = p.parse_args()
    main(args)
