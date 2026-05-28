"""Deblur a single user-supplied image with Pixel-DPS and/or FlowDPS-on-RF.

Two modes:

  --mode synthesize   The input is a CLEAN image. We synthetically blur it
                      using the chosen blur kernel + noise, run the sampler
                      with the *matching* forward operator, and save:
                          clean.png, measurement.png, reconstruction.png
                      Useful as a sanity check (we know the ground truth, so
                      a reasonable reconstruction is expected).

  --mode direct       The input is ALREADY a blurred measurement. We feed it
                      directly to the sampler with the forward operator the
                      user specifies (best-guess σ_b or motion params). Saves:
                          measurement.png (copy of input), reconstruction.png
                      No PSNR is reported (no clean reference). Pass
                      --reference <path> to also compute PSNR/SSIM/LPIPS.

Caveats:
  - The DDPM (Pixel-DPS) and Rectified-Flow (FlowDPS-on-RF) priors are both
    trained on CelebA-HQ-256 face crops. Non-face inputs will be reconstructed
    as the closest plausible face under the assumed blur. For best results:
    a centered, cropped, front-facing portrait at 256x256.
  - The sampler assumes the forward operator is KNOWN. If you pass an
    incorrect σ_b (Gaussian) or (length, angle) (motion), expect the §5.2
    operator-mismatch degradation pattern in the project's robustness study.

Examples:

  # Synthetic Gaussian blur on a clean face you supply.
  python scripts/deblur_image.py \\
      --input my_face.png --mode synthesize \\
      --blur_type gaussian --sigma_b 3.0 --sigma_n 0.05 \\
      --method pixel_dps,flowdps_rf --nfe 100

  # Direct deblur of an already-blurry image with a best-guess σ_b.
  python scripts/deblur_image.py \\
      --input blurry.jpg --mode direct \\
      --blur_type gaussian --sigma_b 3.0 --sigma_n 0.02 \\
      --method pixel_dps --nfe 100

  # Synthetic motion blur (length=25 pixels, angle=45°).
  python scripts/deblur_image.py \\
      --input my_face.png --mode synthesize \\
      --blur_type motion --motion_length 25 --motion_angle 45 \\
      --sigma_n 0.05 --method flowdps_rf --nfe 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Resolve repo-root imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.forward.degradation import gaussian_blur, motion_blur, degrade


def load_image(path: str, size: int = 256) -> torch.Tensor:
    """Load + resize an image to [0,1] float32, shape (C, H, W)."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        print(f"  resize {img.size} -> ({size}, {size}) bicubic", flush=True)
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save a tensor (any shape with last 3 dims = C, H, W) as PNG in [0, 255]."""
    while tensor.dim() > 3:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255 + 0.5).astype(np.uint8)).save(path)


def metrics(x_hat: torch.Tensor, x_ref: torch.Tensor, lpips=None):
    """Return PSNR (dB), SSIM, LPIPS for x_hat vs x_ref (both in [0,1])."""
    from src.metrics.image_metrics import psnr, ssim
    p = psnr(x_hat[0], x_ref[0])
    s = ssim(x_hat[0], x_ref[0])
    if lpips is not None:
        (l,) = lpips(x_hat, x_ref)
        return p, s, float(l)
    return p, s, None


def build_sampler(method: str, device: torch.device):
    """Reuse the factory functions in run_grid.py."""
    from scripts.run_grid import make_pixel_dps_sampler, make_flowdps_rf_sampler
    if method == "pixel_dps":
        return make_pixel_dps_sampler(device), "pixel_dps"
    if method == "flowdps_rf":
        return make_flowdps_rf_sampler(device, use_ema=False), "flowdps_rf"
    if method == "flowdps_rf_v2":
        return make_flowdps_rf_sampler(device, use_ema=True), "flowdps_rf"
    raise ValueError(f"Unknown method: {method}")


def make_forward_op(blur_type: str, sigma_b: float, motion_length: int, motion_angle: float):
    """Return the differentiable A(x) used by the sampler.

    Must match how `y` was generated (or how the user believes `y` was
    generated, for --mode direct)."""
    if blur_type == "gaussian":
        return lambda z: gaussian_blur(z, sigma=sigma_b)
    if blur_type == "motion":
        return lambda z: motion_blur(z, length=motion_length, angle_deg=motion_angle)
    raise ValueError(f"Unknown blur_type: {blur_type}")


def default_zeta(method: str) -> float:
    """v1 zeta defaults (matching scripts/run_grid.py)."""
    if method.startswith("pixel_dps"):
        return 10.0
    if method.startswith("flowdps_rf"):
        return 100.0
    return 10.0


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--input", required=True, help="Path to the input image (clean or blurred).")
    p.add_argument("--mode", default="synthesize", choices=["synthesize", "direct"],
                   help="synthesize: blur the input yourself, then deblur (sanity check). "
                        "direct: the input is already blurred; just deblur.")
    p.add_argument("--reference", default=None,
                   help="Optional clean reference path for --mode direct, to enable PSNR/SSIM/LPIPS.")
    p.add_argument("--blur_type", default="gaussian", choices=["gaussian", "motion"])
    p.add_argument("--sigma_b", type=float, default=3.0,
                   help="Gaussian blur std (in pixels). Used when --blur_type=gaussian.")
    p.add_argument("--motion_length", type=int, default=25,
                   help="Motion blur kernel length in pixels. Used when --blur_type=motion.")
    p.add_argument("--motion_angle", type=float, default=45.0,
                   help="Motion blur angle in degrees. Used when --blur_type=motion.")
    p.add_argument("--sigma_n", type=float, default=0.05,
                   help="Measurement noise std (in [0,1] image units).")
    p.add_argument("--method", default="pixel_dps",
                   help="Sampler(s) to run, comma-separated. "
                        "Options: pixel_dps, flowdps_rf, flowdps_rf_v2.")
    p.add_argument("--nfe", type=int, default=100, help="Number of reverse steps.")
    p.add_argument("--zeta", type=float, default=None,
                   help="Likelihood-gradient guidance scale. Defaults: pixel_dps=10, "
                        "flowdps_rf=100, flowdps_rf_v2 uses its tuned ramp.")
    p.add_argument("--output_dir", default="outputs/user_deblur",
                   help="Where to write {clean,measurement,reconstruction_<method>}.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load input.
    print(f"\n[1] Loading input: {args.input}")
    x_input = load_image(args.input).to(device).unsqueeze(0)  # (1, C, H, W)

    # 2. Build the measurement `y`.
    if args.mode == "synthesize":
        print(f"[2] Synthesizing measurement with {args.blur_type} blur "
              f"+ noise (sigma_n={args.sigma_n})")
        if args.blur_type == "gaussian":
            y = degrade(x_input, blur_type="gaussian",
                        blur_sigma=args.sigma_b, noise_sigma=args.sigma_n,
                        seed=args.seed)
        else:
            y = degrade(x_input, blur_type="motion",
                        motion_length=args.motion_length,
                        motion_angle_deg=args.motion_angle,
                        noise_sigma=args.sigma_n, seed=args.seed)
        x_ref = x_input  # clean is known
        save_image(x_input, out_dir / "clean.png")
        save_image(y, out_dir / "measurement.png")
        print(f"    wrote clean.png and measurement.png to {out_dir}")
    else:  # direct
        print(f"[2] Treating input as the measurement y (no synthetic blur applied)")
        y = x_input
        if args.reference:
            x_ref = load_image(args.reference).to(device).unsqueeze(0)
        else:
            x_ref = None
        save_image(y, out_dir / "measurement.png")

    # 3. Build forward operator (matched to the assumed blur).
    forward_op = make_forward_op(
        args.blur_type, args.sigma_b, args.motion_length, args.motion_angle,
    )

    # 4. Build LPIPS once if we have a reference.
    lpips = None
    if x_ref is not None:
        from src.metrics.image_metrics import LPIPSMetric
        lpips = LPIPSMetric(device=device)

    # 5. Run each requested sampler.
    methods = [m.strip() for m in args.method.split(",") if m.strip()]
    print(f"\n[3] Running sampler(s): {methods}")
    for method in methods:
        print(f"\n--- {method} (NFE={args.nfe}) ---")
        sampler, family = build_sampler(method, device)
        zeta = args.zeta if args.zeta is not None else default_zeta(method)

        t0 = time.time()
        if family == "pixel_dps":
            res = sampler.sample(
                y=y, forward_op=forward_op,
                num_steps=args.nfe, zeta=zeta, sigma_y=max(args.sigma_n, 1e-3),
                seed=args.seed, verbose=False,
            )
        else:  # flowdps_rf
            res = sampler.sample(
                y=y, forward_op=forward_op,
                num_steps=args.nfe, zeta=zeta,
                seed=args.seed, verbose=False,
            )
        dt = time.time() - t0
        x_hat = res.x_hat

        out_path = out_dir / f"reconstruction_{method}.png"
        save_image(x_hat, out_path)

        msg = f"    {method}: wrote {out_path.name}, {dt:.1f}s"
        if x_ref is not None:
            p_, s_, l_ = metrics(x_hat, x_ref, lpips=lpips)
            msg += f", PSNR={p_:.2f} dB, SSIM={s_:.3f}"
            if l_ is not None:
                msg += f", LPIPS={l_:.3f}"
        print(msg)

        del sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
