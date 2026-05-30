"""Unconditional DDPM baseline — the lower bound the proposal asked for.

Generates N samples from the unconditional DDPM (no measurement guidance)
and reports PSNR/SSIM/LPIPS against the test-set clean images. Since
unconditional samples have no relation to any specific clean image, the
expected outcome is very low PSNR — this quantifies what the prior gives
when measurement information is discarded entirely.

Output CSV: outputs/results/unconditional_ddpm.csv with one row per
(image_id, seed) — seed corresponds to the noise initialization, so
results are deterministic and reproducible.

Usage:
    python scripts/run_unconditional_baseline.py --num_images 50 --nfe 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.metrics.image_metrics import psnr, ssim, LPIPSMetric
from src.metrics.results_logger import ResultsLogger


UNCONDITIONAL_FIELDS = (
    "method", "blur_type", "sigma_blur", "sigma_noise",
    "motion_length", "motion_angle", "nfe", "zeta",
    "seed", "image_id", "psnr", "ssim", "lpips",
    "time_s", "resolution", "notes",
)


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--nfe", type=int, default=100,
                   help="DDIM sampling steps for the unconditional draw.")
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--csv_path", default="outputs/results/unconditional_ddpm.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build the unguided DDPM pipeline.
    print("[1] Loading google/ddpm-ema-celebahq-256...")
    from diffusers import DDPMPipeline, DDIMScheduler
    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache",
    )
    unet = pipe.unet.to(device)
    unet.eval()
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(args.nfe, device=device)

    # Load reference clean images.
    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    print(f"[2] Loading {len(img_paths)} reference images")
    clean = [load_image(str(p)).to(device).unsqueeze(0) for p in img_paths]

    logger = ResultsLogger(args.csv_path, fields=UNCONDITIONAL_FIELDS)
    lpips = LPIPSMetric(device=device)

    print(f"[3] Sampling {len(clean)} unconditional images at NFE={args.nfe}")
    t_start = time.time()

    for i in range(len(clean)):
        seed = args.seed + i
        gen = torch.Generator(device=device).manual_seed(int(seed))
        # Initialize from pure Gaussian noise in [-1, 1] space.
        x = torch.randn(
            (1, 3, 256, 256), generator=gen, device=device, dtype=torch.float32,
        )

        t0 = time.time()
        with torch.no_grad():
            for t in scheduler.timesteps:
                eps = unet(x, t).sample
                x = scheduler.step(eps, t, x).prev_sample
        dt = time.time() - t0

        # Convert [-1, 1] to [0, 1] for metric comparison.
        x_pix = ((x + 1) / 2).clamp(0, 1)
        ref = clean[i]

        p_val = psnr(x_pix[0], ref[0])
        s_val = ssim(x_pix[0], ref[0])
        (l_val,) = lpips(x_pix, ref)

        logger.log({
            "method": "unconditional_ddpm",
            "blur_type": "none",
            "sigma_blur": "",
            "sigma_noise": "",
            "motion_length": "",
            "motion_angle": "",
            "nfe": args.nfe,
            "zeta": "0",
            "seed": seed,
            "image_id": i,
            "psnr": round(p_val, 4),
            "ssim": round(s_val, 4),
            "lpips": round(float(l_val), 4),
            "time_s": round(dt, 3),
            "resolution": 256,
            "notes": "unconditional baseline",
        })
        elapsed = time.time() - t_start
        print(f"[{i+1:3d}/{len(clean)}] PSNR={p_val:.2f} SSIM={s_val:.3f} LPIPS={l_val:.3f} dt={dt:.1f}s elapsed={elapsed/60:.1f}m")

    logger.close()
    print(f"\nDone. {len(clean)} rows written to {args.csv_path}")


if __name__ == "__main__":
    main()
