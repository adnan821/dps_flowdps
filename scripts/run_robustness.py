"""Operator-mismatch robustness study.

The TRUE forward operator is motion blur (with varying length × angle):
    y = motion_blur(x; L, theta) + noise
But the SAMPLER's likelihood assumes a Gaussian blur with a fixed sigma:
    forward_op(x) = gaussian_blur(x; sigma=ASSUMED_SIGMA)

This deliberate mismatch tests how each posterior sampler degrades when the
assumed forward operator doesn't match the true one — a standard real-world
failure mode for inverse-problem solvers.

Grid: 3 motion_length × 2 motion_angle × 2 sigma_noise = 12 conditions × 2
methods × N images, all at NFE=50 per the proposal.
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

from src.forward.degradation import gaussian_blur, motion_blur, add_noise
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
    return FlowDPSRF(load_rf_model(device=device), device=device)


def _resolve_method_config(method: str, args):
    """Return (sampler_factory_kind, zeta, use_spectral_weight, use_pigdm)."""
    if method == "pixel_dps":
        return ("pixel_dps", args.zeta_pixel_dps, False, False)
    if method == "pixel_dps_spectral":
        return ("pixel_dps", args.zeta_pixel_dps, True, False)
    if method == "pixel_dps_pigdm":
        # Tier-C Pi-GDM on Pixel-DPS for the robustness study.
        return ("pixel_dps", args.zeta_pixel_dps, False, True)
    if method == "flowdps_rf":
        return ("flowdps_rf", args.zeta_flowdps_rf, False, False)
    if method == "flowdps_rf_spectral":
        return ("flowdps_rf", args.zeta_flowdps_rf, True, False)
    if method == "flowdps_rf_pigdm":
        return ("flowdps_rf", args.zeta_flowdps_rf, False, True)
    raise ValueError(f"Unknown method: {method}")


def main(args):
    from src.forward.ops_fft import gaussian_otf
    from src.samplers.spectral_weight import noise_floor_weight
    from src.samplers.schedules import stringify as _zeta_stringify

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    methods = args.methods.split(",")
    motion_lengths = [int(x) for x in args.motion_lengths.split(",")]
    motion_angles = [float(x) for x in args.motion_angles.split(",")]
    sigma_noises = [float(x) for x in args.sigma_noises.split(",")]
    assumed_sigma = args.assumed_sigma  # what the sampler's forward_op uses

    # Precompute the spectral weight once (depends only on assumed_sigma + shape).
    H_assumed = gaussian_otf(assumed_sigma, (256, 256), device=device)
    spectral_W = noise_floor_weight(
        H_assumed, eps=args.spectral_eps, alpha=args.spectral_alpha
    )

    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    print(f"Loading {len(img_paths)} test images from {args.test_dir}")
    images = [load_image(str(p)) for p in img_paths]

    logger = ResultsLogger(args.csv_path)
    key_fields = (
        "method", "blur_type", "motion_length", "motion_angle",
        "sigma_noise", "nfe", "zeta", "image_id"
    )
    already_done = logger.done_keys(key_fields)
    print(f"Resume-skip: {len(already_done)} rows already in {args.csv_path}")

    lpips_metric = LPIPSMetric(device=device)
    samplers = {}

    total = len(methods) * len(motion_lengths) * len(motion_angles) * len(sigma_noises) * len(images)
    done = 0
    t_start = time.time()

    for method, L, theta, sn in itertools.product(methods, motion_lengths, motion_angles, sigma_noises):
        sampler_kind, zeta, use_spectral, use_pigdm = _resolve_method_config(method, args)
        zeta_str = _zeta_stringify(zeta)
        W_for_this_method = spectral_W if use_spectral else None
        # Tier-C: pigdm_otf is the assumed Gaussian OTF (same as Tier-B's H_assumed).
        pigdm_otf_arg = H_assumed if use_pigdm else None

        if method not in samplers:
            print(f"\n=== Building sampler: {method} (kind={sampler_kind}, spectral={use_spectral}) ===")
            samplers[method] = (
                make_pixel_dps_sampler(device) if sampler_kind == "pixel_dps"
                else make_flowdps_rf_sampler(device)
            )
        sampler = samplers[method]

        # Sampler's assumed forward operator: Gaussian blur at a FIXED sigma.
        # This is the "wrong" model that the inverse solver is calibrated to.
        def forward_op(z, _s=assumed_sigma):
            return gaussian_blur(z, sigma=_s)

        for img_idx, x in enumerate(images):
            key = (
                method, "motion",
                f"{L}", f"{theta}",
                f"{sn}", f"{args.nfe}", zeta_str,
                f"{img_idx}",
            )
            if key in already_done:
                done += 1
                continue

            x_dev = x.unsqueeze(0).to(device)

            # TRUE measurement: motion blur (not what the sampler assumes).
            y = motion_blur(x_dev, length=L, angle_deg=theta)
            gen = torch.Generator(device=device).manual_seed(args.seed + img_idx)
            y = add_noise(y, sigma=sn, generator=gen)

            with CUDATimer() as timer:
                if sampler_kind == "pixel_dps":
                    res = sampler.sample(
                        y=y, forward_op=forward_op,
                        num_steps=args.nfe, zeta=zeta, sigma_y=max(sn, 1e-3),
                        seed=args.seed + img_idx, verbose=False,
                        spectral_weight=W_for_this_method,
                        pigdm_otf=pigdm_otf_arg,
                        pigdm_sigma_n=max(sn, 1e-3),
                    )
                else:
                    res = sampler.sample(
                        y=y, forward_op=forward_op,
                        num_steps=args.nfe, zeta=zeta,
                        seed=args.seed + img_idx, verbose=False,
                        spectral_weight=W_for_this_method,
                        pigdm_otf=pigdm_otf_arg,
                        pigdm_sigma_n=max(sn, 1e-3),
                    )
            x_hat = res.x_hat
            p = psnr(x_hat[0], x_dev[0])
            s = ssim(x_hat[0], x_dev[0])
            (l,) = lpips_metric(x_hat, x_dev)

            logger.log({
                "method": method,
                "blur_type": "motion",
                "sigma_blur": assumed_sigma,
                "sigma_noise": sn,
                "motion_length": L,
                "motion_angle": theta,
                "nfe": args.nfe,
                "zeta": zeta_str,
                "seed": args.seed + img_idx,
                "image_id": img_idx,
                "psnr": round(p, 4),
                "ssim": round(s, 4),
                "lpips": round(float(l), 4),
                "time_s": round(timer.elapsed_s, 3),
                "resolution": 256,
                "notes": f"true=motion(L={L},theta={theta}),assumed=gauss(s={assumed_sigma})",
            })

            done += 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done) if done else float("inf")
            print(
                f"[{done:>5}/{total}] {method} L={L} a={theta} sn={sn} img={img_idx} "
                f"PSNR={p:.2f} SSIM={s:.3f} LPIPS={l:.3f} "
                f"dt={timer.elapsed_s:.1f}s "
                f"(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)",
                flush=True,
            )

    logger.close()
    print(f"\nRobustness study done. {done} rows in {args.csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="pixel_dps,flowdps_rf")
    p.add_argument("--motion_lengths", default="15,25,35")
    p.add_argument("--motion_angles", default="0,45")
    p.add_argument("--sigma_noises", default="0.0,0.05")
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--zeta_pixel_dps", type=float, default=10.0)
    p.add_argument("--zeta_flowdps_rf", type=float, default=100.0)
    p.add_argument("--spectral_eps", type=float, default=0.10,
                   help="Tier-B spectral-weight noise-floor threshold (|H(f)| below this is "
                        "downweighted). Used only when method is pixel_dps_spectral or "
                        "flowdps_rf_spectral. Tuned on 4-img validation.")
    p.add_argument("--spectral_alpha", type=float, default=10.0,
                   help="Tier-B spectral-weight steepness. Higher = more aggressive downweighting.")
    p.add_argument("--assumed_sigma", type=float, default=3.0,
                   help="Sigma the sampler's likelihood uses (Gaussian) — the 'wrong' model")
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--csv_path", default="outputs/results/robustness.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
