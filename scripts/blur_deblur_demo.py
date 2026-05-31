"""All-operators demo: blur one image three ways and reconstruct each.

Takes a single clean face image, applies all three forward-operator
families with sensible defaults (Gaussian / motion / super-resolution),
then reconstructs each measurement with Pixel-DPS (and optionally
FlowDPS-on-RF). Useful for a quick visual sanity check of the
end-to-end pipeline across operator families.

Usage:
    python scripts/blur_deblur_demo.py \\
        --input data/celeba_hq_256/test/00000.png \\
        --output_dir outputs/demo/

With both samplers (slower, ~2 min total):
    python scripts/blur_deblur_demo.py \\
        --input my_face.png \\
        --output_dir outputs/demo/ \\
        --methods pixel_dps,flowdps_rf

Output layout under --output_dir:
    clean.png                                       (the resized 256x256 input)
    gaussian/
        measurement.png                             (the Gaussian-blurred y)
        reconstruction_pixel_dps.png                (sampler reconstruction)
    motion/
        measurement.png
        reconstruction_pixel_dps.png
    sr/
        measurement.png   (low-res: 256/factor)
        reconstruction_pixel_dps.png
    summary.txt                                     (PSNR/SSIM/LPIPS per op)

Defaults (matches the project's main-grid headline cells):
    Gaussian:  sigma_b=3.0, sigma_n=0.05
    Motion:    length=25 pixels, angle=45 degrees, sigma_n=0.05
    SR:        factor=4, sigma_n=0.05
    Sampler:   pixel_dps at NFE=100 (~20s/image on RTX 3060)
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

from src.forward.degradation import gaussian_blur, motion_blur, sr_bicubic, degrade
from src.metrics.image_metrics import psnr, ssim, LPIPSMetric


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        print(f"  resize {img.size} -> ({size}, {size}) bicubic", flush=True)
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    while tensor.dim() > 3:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255 + 0.5).astype(np.uint8)).save(path)


def build_sampler(method: str, device: torch.device):
    """Returns (sampler, family_name)."""
    from scripts.run_grid import make_pixel_dps_sampler, make_flowdps_rf_sampler
    if method == "pixel_dps":
        return make_pixel_dps_sampler(device), "pixel_dps"
    if method == "flowdps_rf":
        return make_flowdps_rf_sampler(device, use_ema=False), "flowdps_rf"
    if method == "flowdps_rf_v2":
        return make_flowdps_rf_sampler(device, use_ema=True), "flowdps_rf"
    raise ValueError(f"Unknown method: {method}")


def default_zeta(method: str) -> float:
    return 10.0 if method.startswith("pixel_dps") else 100.0


def run_one_operator(
    x: torch.Tensor,
    blur_type: str,
    blur_params: dict,
    sigma_n: float,
    methods: list[str],
    nfe: int,
    out_dir: Path,
    device: torch.device,
    lpips,
):
    """Generate measurement for one operator + reconstruct with each method.

    Returns a list of (method, psnr, ssim, lpips, time_s) tuples.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build measurement.
    if blur_type == "gaussian":
        y = degrade(x, blur_type="gaussian", blur_sigma=blur_params["sigma_b"],
                    noise_sigma=sigma_n, seed=0)

        def fop(z, sb=blur_params["sigma_b"]):
            return gaussian_blur(z, sigma=sb)

        out_shape = None
        cell_label = f"sigma_b={blur_params['sigma_b']}, sigma_n={sigma_n}"
    elif blur_type == "motion":
        y = degrade(x, blur_type="motion",
                    motion_length=blur_params["length"],
                    motion_angle_deg=blur_params["angle"],
                    noise_sigma=sigma_n, seed=0)

        def fop(z, L=blur_params["length"], th=blur_params["angle"]):
            return motion_blur(z, length=L, angle_deg=th)

        out_shape = None
        cell_label = f"L={blur_params['length']}, theta={blur_params['angle']}, sigma_n={sigma_n}"
    elif blur_type == "sr":
        y = degrade(x, blur_type="sr", sr_factor=blur_params["factor"],
                    noise_sigma=sigma_n, seed=0)

        def fop(z, r=blur_params["factor"]):
            return sr_bicubic(z, factor=r)

        out_shape = (1, 3, 256, 256)
        cell_label = f"factor={blur_params['factor']}, sigma_n={sigma_n}"
    else:
        raise ValueError(f"Unknown blur_type: {blur_type}")

    save_image(y, out_dir / "measurement.png")
    print(f"  measurement saved: {out_dir / 'measurement.png'}  shape={tuple(y.shape)}")

    results = []
    for method in methods:
        sampler, family = build_sampler(method, device)
        zeta = default_zeta(method)
        sample_kwargs = {"out_shape": out_shape} if out_shape is not None else {}

        t0 = time.time()
        if family == "pixel_dps":
            res = sampler.sample(
                y=y, forward_op=fop,
                num_steps=nfe, zeta=zeta, sigma_y=max(sigma_n, 1e-3),
                seed=0, verbose=False, **sample_kwargs,
            )
        else:  # flowdps_rf
            res = sampler.sample(
                y=y, forward_op=fop,
                num_steps=nfe, zeta=zeta,
                seed=0, verbose=False, **sample_kwargs,
            )
        dt = time.time() - t0
        x_hat = res.x_hat

        save_image(x_hat, out_dir / f"reconstruction_{method}.png")
        p_ = psnr(x_hat[0], x[0])
        s_ = ssim(x_hat[0], x[0])
        (l_,) = lpips(x_hat, x.to(device).unsqueeze(0) if x.dim() == 3 else x)
        l_ = float(l_)
        print(f"  {method:14s}  PSNR={p_:.2f}  SSIM={s_:.3f}  LPIPS={l_:.3f}  dt={dt:.1f}s")
        results.append((method, p_, s_, l_, dt))

        # Free CUDA memory before next sampler.
        del sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cell_label, results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to the CLEAN input image.")
    p.add_argument("--output_dir", default="outputs/demo",
                   help="Where to write clean.png + per-op subfolders + summary.txt")
    p.add_argument("--methods", default="pixel_dps",
                   help="Comma-separated samplers. 'pixel_dps' default; "
                        "'pixel_dps,flowdps_rf' runs both (~2x slower).")
    p.add_argument("--nfe", type=int, default=100, help="NFE budget per reconstruction.")
    # Operator parameters
    p.add_argument("--sigma_b", type=float, default=3.0,
                   help="Gaussian blur sigma (default 3.0).")
    p.add_argument("--motion_length", type=int, default=25,
                   help="Motion-blur kernel length in pixels (default 25).")
    p.add_argument("--motion_angle", type=float, default=45.0,
                   help="Motion-blur angle in degrees (default 45).")
    p.add_argument("--sr_factor", type=int, default=4,
                   help="Super-resolution downsample factor (default 4).")
    p.add_argument("--sigma_n", type=float, default=0.05,
                   help="Measurement noise std applied to all three ops (default 0.05).")
    # Optional operator skip
    p.add_argument("--skip", default="",
                   help="Comma-separated operators to skip from the demo. "
                        "E.g., 'sr' to only run Gaussian + motion.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load clean.
    print(f"\n[1] Loading: {args.input}")
    x_cpu = load_image(args.input)  # (C, H, W)
    save_image(x_cpu, out_dir / "clean.png")
    x = x_cpu.to(device).unsqueeze(0)  # (1, C, H, W)

    # 2. Build LPIPS once.
    print("[2] Loading LPIPS network")
    lpips = LPIPSMetric(device=device)

    # 3. Run each operator family.
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    operators_to_run = []
    if "gaussian" not in skip:
        operators_to_run.append(("gaussian", {"sigma_b": args.sigma_b}))
    if "motion" not in skip:
        operators_to_run.append(("motion", {"length": args.motion_length,
                                            "angle": args.motion_angle}))
    if "sr" not in skip:
        operators_to_run.append(("sr", {"factor": args.sr_factor}))

    print(f"\n[3] Running {len(operators_to_run)} operator(s) x {len(methods)} method(s)")
    t_total = time.time()

    summary_lines = [
        f"Demo run on {args.input}",
        f"NFE={args.nfe}, sigma_n={args.sigma_n}, methods={methods}",
        "",
        f"{'operator':<10}{'cell':<40}{'method':<14}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'time':>8}",
        "-" * 100,
    ]
    for blur_type, blur_params in operators_to_run:
        print(f"\n--- {blur_type} ---")
        op_out = out_dir / blur_type
        cell_label, results = run_one_operator(
            x, blur_type, blur_params, args.sigma_n,
            methods, args.nfe, op_out, device, lpips,
        )
        for method, p_, s_, l_, dt in results:
            summary_lines.append(
                f"{blur_type:<10}{cell_label:<40}{method:<14}{p_:>8.2f}{s_:>8.3f}{l_:>8.3f}{dt:>7.1f}s"
            )

    summary_lines.append("")
    summary_lines.append(f"Total wall-clock: {(time.time() - t_total)/60:.1f} min")

    summary_text = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_text)

    print("\n" + "=" * 80)
    print(summary_text)
    print("\nOutputs saved under:", out_dir)


if __name__ == "__main__":
    main()
