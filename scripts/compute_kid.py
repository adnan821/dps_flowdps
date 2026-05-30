"""Compute Kernel Inception Distance (KID) for each method's reconstructions.

KID is the distributional-realism metric the proposal listed but our
main grid didn't compute. We re-sample a representative cell
(σ_b=3.0, σ_n=0.05, NFE=100, n=50 images) with both posterior samplers
+ Wiener + DnCNN+PnP, then KID each method's reconstructions against
the 50 clean test images.

Caveat: n=50 is small for KID (the metric was designed for n=1000+).
The numbers are indicative, not definitive — they show the rank order
of methods on distributional realism, not absolute distribution gaps.

Output: outputs/results/kid_summary.csv
Usage:  python scripts/compute_kid.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image.kid import KernelInceptionDistance

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.forward.degradation import gaussian_blur, degrade


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """KID expects uint8 [0, 255] tensors of shape (N, 3, H, W)."""
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def sample_pixel_dps(sigma_b, sigma_n, nfe, images, seeds, device):
    from scripts.run_grid import make_pixel_dps_sampler
    sampler = make_pixel_dps_sampler(device)
    recons = []
    for img, seed in zip(images, seeds):
        x = img.unsqueeze(0).to(device)
        y = degrade(x, blur_type="gaussian",
                    blur_sigma=sigma_b, noise_sigma=sigma_n, seed=seed)
        forward_op = lambda z: gaussian_blur(z, sigma=sigma_b)
        res = sampler.sample(
            y=y, forward_op=forward_op,
            num_steps=nfe, zeta=10.0, sigma_y=max(sigma_n, 1e-3),
            seed=seed, verbose=False,
        )
        recons.append(res.x_hat[0].cpu())
    del sampler
    torch.cuda.empty_cache()
    return torch.stack(recons)


def sample_flowdps_rf(sigma_b, sigma_n, nfe, images, seeds, device):
    from scripts.run_grid import make_flowdps_rf_sampler
    sampler = make_flowdps_rf_sampler(device, use_ema=False)
    recons = []
    for img, seed in zip(images, seeds):
        x = img.unsqueeze(0).to(device)
        y = degrade(x, blur_type="gaussian",
                    blur_sigma=sigma_b, noise_sigma=sigma_n, seed=seed)
        forward_op = lambda z: gaussian_blur(z, sigma=sigma_b)
        res = sampler.sample(
            y=y, forward_op=forward_op,
            num_steps=nfe, zeta=100.0, seed=seed, verbose=False,
        )
        recons.append(res.x_hat[0].cpu())
    del sampler
    torch.cuda.empty_cache()
    return torch.stack(recons)


def sample_unconditional(nfe, n_samples, device, base_seed=0):
    """Pure unconditional DDPM sampling — no measurement guidance."""
    from diffusers import DDPMPipeline, DDIMScheduler
    pipe = DDPMPipeline.from_pretrained(
        "google/ddpm-ema-celebahq-256", cache_dir="checkpoints/hf_cache",
    )
    unet = pipe.unet.to(device).eval()
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(nfe, device=device)
    recons = []
    for i in range(n_samples):
        gen = torch.Generator(device=device).manual_seed(int(base_seed + i))
        x = torch.randn((1, 3, 256, 256), generator=gen,
                        device=device, dtype=torch.float32)
        with torch.no_grad():
            for t in scheduler.timesteps:
                eps = unet(x, t).sample
                x = scheduler.step(eps, t, x).prev_sample
        recons.append(((x[0] + 1) / 2).clamp(0, 1).cpu())
    del unet, pipe
    torch.cuda.empty_cache()
    return torch.stack(recons)


def sample_wiener(sigma_b, sigma_n, images, seeds):
    """Apply Wiener filter in the frequency domain."""
    from src.forward.ops_fft import gaussian_otf
    H, W = 256, 256
    H_otf = gaussian_otf(sigma_b, (H, W), device=torch.device("cpu"))
    # Wiener: X_hat = conj(H) * Y / (|H|^2 + sigma_n^2 / S_x)
    # Use a small noise-to-signal ratio default (1e-3) when sigma_n=0.
    nsr = max(sigma_n ** 2, 1e-6)
    recons = []
    for img, seed in zip(images, seeds):
        x = img.unsqueeze(0)
        y = degrade(x, blur_type="gaussian",
                    blur_sigma=sigma_b, noise_sigma=sigma_n, seed=seed)
        per_chan = []
        for c in range(3):
            Y = torch.fft.fft2(y[0, c])
            wiener = torch.conj(H_otf) / (torch.abs(H_otf) ** 2 + nsr)
            xhat = torch.fft.ifft2(Y * wiener).real
            per_chan.append(xhat)
        recons.append(torch.stack(per_chan).clamp(0, 1))
    return torch.stack(recons)


def load_unconditional(csv_dir: Path, n: int) -> torch.Tensor | None:
    """If unconditional samples were previously generated and SAVED as
    a tensor file, load them. Otherwise return None (skip)."""
    candidate = csv_dir / "unconditional_ddpm_samples.pt"
    if candidate.exists():
        t = torch.load(candidate)
        return t[:n]
    return None


def kid_compute(real_uint8, fake_uint8, subset_size=25):
    """Compute KID with a subset_size that fits our n=50 sample.
    Returns (mean, std)."""
    kid = KernelInceptionDistance(subset_size=subset_size, normalize=False)
    kid.update(real_uint8, real=True)
    kid.update(fake_uint8, real=False)
    m, s = kid.compute()
    return float(m), float(s)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sigma_b", type=float, default=3.0)
    p.add_argument("--sigma_n", type=float, default=0.05)
    p.add_argument("--nfe", type=int, default=100)
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--csv_path", default="outputs/results/kid_summary.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", default="wiener,pixel_dps,flowdps_rf",
                   help="Comma-separated list of methods to KID.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load reference clean images.
    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    print(f"[1] Loading {len(img_paths)} clean reference images")
    clean = [load_image(str(p)) for p in img_paths]
    clean_stack = torch.stack(clean)
    real_uint8 = to_uint8(clean_stack)

    seeds = [args.seed + i for i in range(len(clean))]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"[2] Generating reconstructions for: {methods}")
    recon_by_method: dict[str, torch.Tensor] = {}
    for method in methods:
        t0 = time.time()
        print(f"\n--- {method} ---")
        if method == "pixel_dps":
            r = sample_pixel_dps(args.sigma_b, args.sigma_n, args.nfe,
                                 clean, seeds, device)
        elif method == "flowdps_rf":
            r = sample_flowdps_rf(args.sigma_b, args.sigma_n, args.nfe,
                                  clean, seeds, device)
        elif method == "wiener":
            r = sample_wiener(args.sigma_b, args.sigma_n, clean, seeds)
        elif method == "unconditional_ddpm":
            r = sample_unconditional(args.nfe, len(clean), device, args.seed)
        else:
            print(f"  skipping unknown method: {method}")
            continue
        dt = time.time() - t0
        print(f"  produced {len(r)} reconstructions in {dt/60:.1f} min")
        recon_by_method[method] = r

    print(f"\n[3] Computing KID for {len(recon_by_method)} methods.")
    kid_rows = []
    for method, recons in recon_by_method.items():
        fake_uint8 = to_uint8(recons)
        mean, std = kid_compute(real_uint8, fake_uint8, subset_size=25)
        # Lower KID = closer to real distribution.
        # Multiplied by 1000 in convention for readability.
        kid_rows.append({
            "method": method, "sigma_blur": args.sigma_b,
            "sigma_noise": args.sigma_n, "nfe": args.nfe,
            "n": len(recons), "kid_mean_x1000": round(mean * 1000, 4),
            "kid_std_x1000": round(std * 1000, 4),
        })
        print(f"  {method:25s}  KID = ({mean*1000:+.3f} ± {std*1000:.3f}) × 10⁻³")

    # Save.
    with open(args.csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kid_rows[0].keys())
        w.writeheader()
        w.writerows(kid_rows)
    print(f"\n[4] Wrote {len(kid_rows)} rows to {args.csv_path}")


if __name__ == "__main__":
    main()
