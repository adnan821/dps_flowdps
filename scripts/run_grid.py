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


def make_flowdps_rf_sampler(device, use_ema: bool = False):
    """Build a FlowDPS-on-RF sampler. `use_ema=False` matches the published
    v1 baseline (non-EMA `model` state_dict). `use_ema=True` loads the
    properly-aligned EMA shadow_params and is used for `flowdps_rf_v2`."""
    from src.models.rf_celebahq import load_rf_model
    from src.samplers.flowdps_rf import FlowDPSRF
    rf = load_rf_model(device=device, use_non_ema=not use_ema)
    return FlowDPSRF(rf, device=device)


# Method registry: method-name → (sampler_factory_kwargs, zeta_value_or_schedule)
# zeta entries that are callables get their __name__ stringified into the CSV
# `zeta` column via src.samplers.schedules.stringify so resume stays idempotent.
from src.samplers.schedules import stringify as _zeta_stringify, zeta_ramp


def _resolve_method_config(method: str, args):
    """Return (sampler_factory_callable, zeta_value_or_schedule,
    extra_kwargs_dict) for `method`. The extra_kwargs are passed through
    to sampler.sample() — used for Tier-B (`spectral_weight`) and
    Tier-C (`pigdm_otf`).

    Adding a new method here is the only change needed to extend the grid
    runner with a new sampler variant.
    """
    if method == "pixel_dps":
        return ("pixel_dps", args.zeta_pixel_dps, {})
    if method == "pixel_dps_v2":
        # Re-tuned scalar ζ on a 2-img validation; 30 in the middle of the
        # 25-40 plateau.
        return ("pixel_dps", args.zeta_pixel_dps_v2, {})
    if method == "pixel_dps_pigdm":
        # Tier-C: Tweedie-corrected likelihood, per-step weight depending
        # on r_t = (1 - alpha_bar_t) / alpha_bar_t. The actual `pigdm_otf`
        # tensor is computed per condition inside main() because it depends
        # on sigma_b. ζ=100 picked from 2-img sweep on (σ_b=3.0, σ_n=0.05).
        return ("pixel_dps", args.zeta_pigdm,
                {"_pigdm": True, "pigdm_sigma_n": None})  # sigma_n filled in per-cell
    if method == "flowdps_rf":
        return ("flowdps_rf", args.zeta_flowdps_rf, {})
    if method == "flowdps_rf_v2":
        # EMA weights + ramp(200, α=0.5) ramps guidance up toward t=1
        # (data side). Won the 2-img validation by ~0.8 dB over EMA+scalar.
        return ("flowdps_rf_ema", zeta_ramp(args.zeta_flowdps_rf_v2, 0.5), {})
    if method == "flowdps_rf_pigdm":
        # Tier-C on FlowDPS-RF: scalar ζ=100 from 2-img sweep (the v2 ramp
        # was tuned for the non-pigdm path and over-corrects here).
        return ("flowdps_rf_ema", args.zeta_pigdm,
                {"_pigdm": True, "pigdm_sigma_n": None})
    raise ValueError(f"Unknown method: {method}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve grid axes.
    methods = args.methods.split(",")
    sigma_blurs = [float(x) for x in args.sigma_blurs.split(",")]
    sigma_noises = [float(x) for x in args.sigma_noises.split(",")]
    nfes = [int(x) for x in args.nfes.split(",")]

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

    # Cache `gaussian_otf` per sigma_b so we don't rebuild it 50 times.
    from src.forward.ops_fft import gaussian_otf
    otf_cache: dict[float, torch.Tensor] = {}

    for method, sb, sn, nfe in itertools.product(methods, sigma_blurs, sigma_noises, nfes):
        sampler_kind, zeta, extra_kwargs = _resolve_method_config(method, args)
        zeta_str = _zeta_stringify(zeta)
        # If this method is Tier-C (pigdm), resolve the per-cell OTF + sigma_n.
        use_pigdm = extra_kwargs.pop("_pigdm", False)
        sample_kwargs: dict = {}
        if use_pigdm:
            if sb not in otf_cache:
                otf_cache[sb] = gaussian_otf(sb, (256, 256), device=device)
            sample_kwargs["pigdm_otf"] = otf_cache[sb]
            sample_kwargs["pigdm_sigma_n"] = max(sn, 1e-3)
        # Build sampler once per method.
        if method not in samplers:
            print(f"\n=== Building sampler: {method} (kind={sampler_kind}) ===")
            if sampler_kind == "pixel_dps":
                samplers[method] = make_pixel_dps_sampler(device)
            elif sampler_kind == "flowdps_rf":
                samplers[method] = make_flowdps_rf_sampler(device, use_ema=False)
            elif sampler_kind == "flowdps_rf_ema":
                samplers[method] = make_flowdps_rf_sampler(device, use_ema=True)
            else:
                raise ValueError(f"Unknown sampler kind: {sampler_kind}")
        sampler = samplers[method]
        is_pixel_dps_family = sampler_kind == "pixel_dps"

        for img_idx, x in enumerate(images):
            key = (method, "gaussian", f"{sb}", f"{sn}", f"{nfe}", zeta_str, f"{img_idx}")
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
                if is_pixel_dps_family:
                    res = sampler.sample(
                        y=y, forward_op=forward_op,
                        num_steps=nfe, zeta=zeta, sigma_y=max(sn, 1e-3),
                        seed=args.seed + img_idx, verbose=False,
                        **sample_kwargs,
                    )
                else:
                    res = sampler.sample(
                        y=y, forward_op=forward_op,
                        num_steps=nfe, zeta=zeta,
                        seed=args.seed + img_idx, verbose=False,
                        **sample_kwargs,
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
                "zeta": zeta_str,
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
    p.add_argument("--zeta_pixel_dps", type=float, default=10.0)
    p.add_argument("--zeta_flowdps_rf", type=float, default=100.0)
    # Tier-A v2 method scalars. Tuned on 2-image validation at
    # sigma_b=3.0, sigma_n=0.05, NFE=50:
    # - pixel_dps_v2 ζ=30: middle of broad peak (ζ=25,30 both at 29.49 dB,
    #   ζ=40 peak at 29.74 dB), v1 ζ=10 was 28.14 → +1.35 dB gain.
    # - flowdps_rf_v2 ramp(200, α=0.5) + EMA: 24.43 dB vs v1's 23.45 → +0.98 dB.
    p.add_argument("--zeta_pixel_dps_v2", type=float, default=30.0,
                   help="Re-tuned scalar zeta for pixel_dps_v2 (default 30).")
    p.add_argument("--zeta_flowdps_rf_v2", type=float, default=200.0,
                   help="Base zeta for flowdps_rf_v2's ramp(zeta_0, alpha=0.5) schedule.")
    p.add_argument("--zeta_pigdm", type=float, default=100.0,
                   help="Scalar zeta for Tier-C pigdm methods. Picked from 2-img "
                        "sweep on (sb=3.0, sn=0.05, NFE=50): pixel→20.48 dB, RF→21.34 dB.")
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--csv_path", default="outputs/results/main_grid.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
