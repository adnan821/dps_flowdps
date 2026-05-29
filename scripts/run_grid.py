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
from src.forward.degradation import gaussian_blur, motion_blur, sr_bicubic
from src.forward.ops_fft import gaussian_otf, motion_blur_otf
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


def make_particle_dps_sampler(device, num_particles: int = 8):
    """Build a gradient-free SMC-style sampler on the same DDPM. P particles
    of x_t are propagated through DDIM with no autograd; each step reweights
    by measurement likelihood and resamples when ESS drops below P/2."""
    from diffusers import DDPMPipeline
    from src.samplers.particle_dps import ParticleDPS
    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache"
    )
    return ParticleDPS(
        pipe.unet, scheduler_config=pipe.scheduler.config,
        device=device, num_particles=num_particles,
    )


def make_flowdps_rf_pigdm_pure_sampler(device):
    """Build the analytical Π-GDM-pure FlowDPS-on-RF sampler.
    Loads the same EMA-corrected RF weights as flowdps_rf_v2 — for a
    fair pigdm-vs-v2 comparison the model should be identical (the
    sampling algorithm is the only knob)."""
    from src.models.rf_celebahq import load_rf_model
    from src.samplers.flowdps_rf_pigdm_pure import FlowDPSRFPiGDMPure
    rf = load_rf_model(device=device, use_non_ema=True)
    return FlowDPSRFPiGDMPure(rf, device=device, sigma_n_floor=0.01)


# Method registry: method-name → (sampler_factory_kwargs, zeta_value_or_schedule)
# zeta entries that are callables get their __name__ stringified into the CSV
# `zeta` column via src.samplers.schedules.stringify so resume stays idempotent.
from src.samplers.schedules import (
    stringify as _zeta_stringify,
    zeta_ramp, zeta_power, zeta_linear_warmup_then_decay,
)


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
    if method == "pixel_dps_sched":
        # Time-varying ζ schedule for Pixel-DPS — the genuine analog of
        # what made flowdps_rf_v2 win (a schedule, not a scalar). Shape /
        # zeta0 / alpha are chosen by scripts/tune_pixel_schedule.py on a
        # held-out validation set and passed via CLI. (Schedule builders
        # are imported at module scope — a function-local `from import`
        # would shadow `zeta_ramp` for the whole function.)
        if args.sched_kind == "power":
            sched = zeta_power(args.sched_zeta0, args.sched_alpha)
        elif args.sched_kind == "ramp":
            sched = zeta_ramp(args.sched_zeta0, args.sched_alpha)
        elif args.sched_kind == "warmup":
            sched = zeta_linear_warmup_then_decay(args.sched_zeta0, 0.1, args.sched_alpha)
        else:
            raise ValueError(f"Unknown sched_kind: {args.sched_kind}")
        return ("pixel_dps", sched, {})
    if method == "pixel_dps_sched_v2":
        # σ_n-adaptive variant: use the original pixel_dps_sched winning
        # config (ramp(80, 0.5)) when σ_n > 0, fall back to v1 scalar
        # ζ=10 when σ_n = 0. The σ_n-dependent dispatch is done in main()
        # because _resolve_method_config doesn't see (sb, sn). The
        # placeholder zeta below is overwritten there per-cell.
        return ("pixel_dps", "sn_adaptive_placeholder", {"_sn_adaptive_sched": True})
    if method == "flowdps_rf_heun":
        # FlowDPS-on-RF v2 config (EMA + ramp(200, 0.5)) with a 2nd-order
        # Heun integrator. Same guidance as v2 — isolates the integrator.
        return ("flowdps_rf_ema", zeta_ramp(args.zeta_flowdps_rf_v2, 0.5),
                {"_integrator": "heun"})
    if method == "particle_dps":
        # Gradient-free SMC-style sampler on the same DDPM as pixel_dps —
        # P particles of x_t, no autograd, multinomial resample when ESS
        # drops below P/2. zeta is unused (no gradient term). The CSV
        # `zeta` column gets the particle count for resume-key uniqueness.
        return ("particle_dps", f"P={args.num_particles}", {})
    if method == "pixel_dps_spectral":
        # Tier-B on matched-Gaussian main grid: spectral noise-floor weight
        # built from the SAME Gaussian OTF the sampler assumes (matched
        # condition). The weight resolves per cell in main() because the
        # OTF depends on σ_b.
        return ("pixel_dps", args.zeta_pixel_dps, {"_spectral": True})
    if method == "flowdps_rf_spectral":
        # Same on FlowDPS-RF.
        return ("flowdps_rf", args.zeta_flowdps_rf, {"_spectral": True})
    if method == "flowdps_rf_pigdm_pure":
        # Fix #1: analytical Π-GDM (Song 2023) on RF. No free ζ — the
        # gradient magnitude is derived from the Tweedie covariance
        # r_t = (1-t)². σ_y floored at 0.01 to keep the σ_n=0 cells
        # finite. The H_otf and sigma_y are injected per-cell in main()
        # (same pattern as the old `*_pigdm` Tier-C path).
        return ("flowdps_rf_pigdm_pure", "eta=1.0",
                {"_pigdm_pure": True, "pigdm_sigma_n": None})
    if method == "particle_dps_tempered":
        # Fix #9 (revised after held-out sweep): SMC with annealed
        # likelihood AND force-resampling at every step. Tempering alone
        # was insufficient because broad weights kept ESS high → the
        # ESS-gated resampler never fired → particles drifted as
        # independent prior samples and the final weighted-mean readout
        # averaged unconditional draws. Forcing resample at every step,
        # paired with tempering for diverse offspring, biases the
        # particle ensemble toward the posterior every step.
        zeta_str = f"P={args.num_particles}/T={args.tempering_max}/force"
        return ("particle_dps", zeta_str,
                {"_tempering_max": args.tempering_max,
                 "_tempering_alpha": args.tempering_alpha,
                 "_force_resample": True})
    raise ValueError(f"Unknown method: {method}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve grid axes.
    methods = args.methods.split(",")
    sigma_noises = [float(x) for x in args.sigma_noises.split(",")]
    nfes = [int(x) for x in args.nfes.split(",")]
    blur_type = args.blur_type
    if blur_type == "gaussian":
        sigma_blurs = [float(x) for x in args.sigma_blurs.split(",")]
        blur_axes = [("gaussian", sb) for sb in sigma_blurs]
    elif blur_type == "motion":
        motion_lengths = [int(x) for x in args.motion_lengths.split(",")]
        motion_angles = [float(x) for x in args.motion_angles.split(",")]
        blur_axes = [("motion", L, theta) for L in motion_lengths for theta in motion_angles]
    elif blur_type == "sr":
        sr_factors = [int(x) for x in args.sr_factors.split(",")]
        blur_axes = [("sr", r) for r in sr_factors]
    else:
        raise ValueError(f"Unknown blur_type: {blur_type}")

    # Load test images.
    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    print(f"Loading {len(img_paths)} test images from {args.test_dir}")
    images = [load_image(str(p)) for p in img_paths]

    # CSV logger. Dedup key depends on blur_type so motion and Gaussian
    # rows don't collide.
    logger = ResultsLogger(args.csv_path)
    if blur_type == "gaussian":
        key_fields = ("method", "blur_type", "sigma_blur", "sigma_noise", "nfe", "zeta", "image_id")
    elif blur_type == "sr":
        key_fields = ("method", "blur_type", "sigma_blur", "sigma_noise", "nfe", "zeta", "image_id")
    else:
        key_fields = ("method", "blur_type", "motion_length", "motion_angle", "sigma_noise", "nfe", "zeta", "image_id")
    already_done = logger.done_keys(key_fields)
    print(f"Resume-skip: {len(already_done)} rows already in {args.csv_path}")

    lpips = LPIPSMetric(device=device)
    samplers = {}

    total_rows = len(methods) * len(blur_axes) * len(sigma_noises) * len(nfes) * len(images)
    done = 0
    t_start = time.time()

    # OTF cache keyed by blur-axis tuple so Gaussian and motion don't collide.
    otf_cache: dict[tuple, torch.Tensor] = {}

    for method, blur_params, sn, nfe in itertools.product(methods, blur_axes, sigma_noises, nfes):
        # Unpack blur axis.
        if blur_params[0] == "gaussian":
            _, sb = blur_params
            L = theta = sr_factor = None
        elif blur_params[0] == "sr":
            _, sr_factor = blur_params
            sb = L = theta = None
        else:
            _, L, theta = blur_params
            sb = sr_factor = None
        sampler_kind, zeta, extra_kwargs = _resolve_method_config(method, args)
        # Per-cell ζ dispatch for σ_n-adaptive methods.
        if extra_kwargs.pop("_sn_adaptive_sched", False):
            if sn == 0.0:
                zeta = args.zeta_pixel_dps   # v1 scalar = 10
            else:
                zeta = zeta_ramp(args.sched_zeta0, args.sched_alpha)
        zeta_str = _zeta_stringify(zeta)
        # If this method is Tier-C (pigdm), resolve the per-cell OTF + sigma_n.
        use_pigdm = extra_kwargs.pop("_pigdm", False)
        use_pigdm_pure = extra_kwargs.pop("_pigdm_pure", False)
        use_spectral = extra_kwargs.pop("_spectral", False)
        integrator = extra_kwargs.pop("_integrator", None)
        tempering_max = extra_kwargs.pop("_tempering_max", None)
        tempering_alpha = extra_kwargs.pop("_tempering_alpha", None)
        force_resample = extra_kwargs.pop("_force_resample", None)
        sample_kwargs: dict = {}

        # Build the per-cell OTF if any spectral/pigdm method needs it.
        # Matched config: assumed operator IS the true operator, so the
        # OTF is built from the same params used in the measurement.
        def _build_otf():
            if blur_params[0] == "gaussian":
                return gaussian_otf(sb, (256, 256), device=device)
            else:
                return motion_blur_otf(L, theta, (256, 256), device=device)

        if use_spectral:
            from src.samplers.spectral_weight import noise_floor_weight
            if blur_params not in otf_cache:
                otf_cache[blur_params] = _build_otf()
            sample_kwargs["spectral_weight"] = noise_floor_weight(
                otf_cache[blur_params], eps=args.spectral_eps, alpha=args.spectral_alpha,
            )
        if use_pigdm:
            if blur_params not in otf_cache:
                otf_cache[blur_params] = _build_otf()
            sample_kwargs["pigdm_otf"] = otf_cache[blur_params]
            sample_kwargs["pigdm_sigma_n"] = max(sn, 1e-3)
        if use_pigdm_pure:
            if blur_params not in otf_cache:
                otf_cache[blur_params] = _build_otf()
            sample_kwargs["H_otf"] = otf_cache[blur_params]
            sample_kwargs["sigma_n"] = sn   # the sampler floors internally at 0.01
        if integrator is not None:
            sample_kwargs["integrator"] = integrator
        if tempering_max is not None:
            sample_kwargs["tempering_max"] = tempering_max
            sample_kwargs["tempering_alpha"] = tempering_alpha
        if force_resample is not None:
            sample_kwargs["force_resample"] = force_resample
        # Build sampler once per method.
        if method not in samplers:
            print(f"\n=== Building sampler: {method} (kind={sampler_kind}) ===")
            if sampler_kind == "pixel_dps":
                samplers[method] = make_pixel_dps_sampler(device)
            elif sampler_kind == "flowdps_rf":
                samplers[method] = make_flowdps_rf_sampler(device, use_ema=False)
            elif sampler_kind == "flowdps_rf_ema":
                samplers[method] = make_flowdps_rf_sampler(device, use_ema=True)
            elif sampler_kind == "particle_dps":
                samplers[method] = make_particle_dps_sampler(device, num_particles=args.num_particles)
            elif sampler_kind == "flowdps_rf_pigdm_pure":
                samplers[method] = make_flowdps_rf_pigdm_pure_sampler(device)
            else:
                raise ValueError(f"Unknown sampler kind: {sampler_kind}")
        sampler = samplers[method]
        # Both pixel_dps and particle_dps expose a `sigma_y` kwarg and don't
        # take an `integrator`; flowdps_rf families don't take sigma_y at the
        # public sample() entry. pigdm_pure samplers take sigma_n explicitly
        # via sample_kwargs (set above) so they don't need the sigma_y kwarg.
        is_pixel_dps_family = sampler_kind in ("pixel_dps", "particle_dps")

        for img_idx, x in enumerate(images):
            if blur_type == "gaussian":
                key = (method, "gaussian", f"{sb}", f"{sn}", f"{nfe}", zeta_str, f"{img_idx}")
            elif blur_type == "sr":
                key = (method, "sr", f"{sr_factor}", f"{sn}", f"{nfe}", zeta_str, f"{img_idx}")
            else:
                key = (method, "motion", f"{L}", f"{theta}", f"{sn}", f"{nfe}", zeta_str, f"{img_idx}")
            if key in already_done:
                done += 1
                continue

            x_dev = x.unsqueeze(0).to(device)

            # Forward op closure (matched to blur_type).
            if blur_type == "gaussian":
                def forward_op(z, sb_=sb):
                    return gaussian_blur(z, sigma=sb_)
                y = degrade(
                    x_dev, blur_type="gaussian",
                    blur_sigma=sb, noise_sigma=sn, seed=args.seed + img_idx,
                )
            elif blur_type == "sr":
                def forward_op(z, r=sr_factor):
                    return sr_bicubic(z, factor=r)
                y = degrade(
                    x_dev, blur_type="sr",
                    sr_factor=sr_factor, noise_sigma=sn,
                    seed=args.seed + img_idx,
                )
            else:
                def forward_op(z, L_=L, theta_=theta):
                    return motion_blur(z, length=L_, angle_deg=theta_)
                y = degrade(
                    x_dev, blur_type="motion",
                    motion_length=L, motion_angle_deg=theta,
                    noise_sigma=sn, seed=args.seed + img_idx,
                )

            # For SR, the sampler must initialize x at the model's native
            # spatial shape (= the clean image shape), not at y's shape.
            if blur_type == "sr":
                sample_kwargs["out_shape"] = tuple(x_dev.shape)

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
                "blur_type": blur_type,
                "sigma_blur": sb if blur_type == "gaussian" else (sr_factor if blur_type == "sr" else ""),
                "sigma_noise": sn,
                "motion_length": L if blur_type == "motion" else "",
                "motion_angle": theta if blur_type == "motion" else "",
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
            cell_label = (
                f"sb={sb}" if blur_type == "gaussian" else
                f"r={sr_factor}" if blur_type == "sr" else
                f"L={L},θ={theta}"
            )
            print(
                f"[{done:>5}/{total_rows}] {method} {cell_label} sn={sn} nfe={nfe} img={img_idx} "
                f"PSNR={p:.2f} SSIM={s:.3f} LPIPS={l:.3f} "
                f"dt={timer.elapsed_s:.1f}s "
                f"(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)"
            )

    logger.close()
    print(f"\nGrid done. {done} rows written to {args.csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="pixel_dps,flowdps_rf")
    p.add_argument("--blur_type", default="gaussian", choices=["gaussian", "motion", "sr"],
                   help="Forward-operator family. 'motion' enables matched motion deblurring "
                        "(sampler's forward_op IS the motion-blur operator, not Gaussian). "
                        "'sr' enables matched super-resolution (bicubic downsample by --sr_factors).")
    p.add_argument("--sigma_blurs", default="1.5,3.0,5.0",
                   help="Used when --blur_type=gaussian.")
    p.add_argument("--motion_lengths", default="15,25,35",
                   help="Used when --blur_type=motion. Length in pixels of the motion kernel.")
    p.add_argument("--motion_angles", default="0,45",
                   help="Used when --blur_type=motion. Angle in degrees of the motion kernel.")
    p.add_argument("--sr_factors", default="2,4,8",
                   help="Used when --blur_type=sr. Downsampling factors.")
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
    # pixel_dps_sched: time-varying ζ schedule chosen by tune_pixel_schedule.py.
    p.add_argument("--sched_kind", default="power", choices=["power", "ramp", "warmup"],
                   help="ζ-schedule shape for the pixel_dps_sched method.")
    p.add_argument("--sched_zeta0", type=float, default=40.0,
                   help="Base ζ for the pixel_dps_sched schedule.")
    p.add_argument("--sched_alpha", type=float, default=1.0,
                   help="Exponent for the pixel_dps_sched schedule.")
    p.add_argument("--num_particles", type=int, default=8,
                   help="Particle count for the gradient-free particle_dps sampler.")
    p.add_argument("--spectral_eps", type=float, default=0.10,
                   help="Tier-B noise-floor threshold (matches run_robustness.py default).")
    p.add_argument("--spectral_alpha", type=float, default=10.0,
                   help="Tier-B noise-floor steepness (matches run_robustness.py default).")
    p.add_argument("--tempering_max", type=float, default=10.0,
                   help="SMC tempering: σ_y_eff starts at σ_y·T_max at step 0 and "
                        "decays to σ_y at the final step. T_max=1 disables tempering.")
    p.add_argument("--tempering_alpha", type=float, default=2.0,
                   help="Exponent of the tempering schedule (1=linear, 2=quadratic).")
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--csv_path", default="outputs/results/main_grid.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
