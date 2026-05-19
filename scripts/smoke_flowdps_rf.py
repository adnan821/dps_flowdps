"""FlowDPS-on-RF smoke test, mirroring scripts/smoke_pixel_dps.py."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forward import degrade
from src.forward.degradation import gaussian_blur
from src.metrics.image_metrics import psnr, ssim, LPIPSMetric
from src.metrics.timing import CUDATimer
from src.models.rf_celebahq import load_rf_model
from src.samplers.flowdps_rf import FlowDPSRF


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_grid(tensors, titles, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = tensors[0].shape[0]
    cols = len(tensors)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows + 0.5))
    if rows == 1:
        axes = axes[None, :]
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            img = tensors[c][r].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img); ax.set_axis_off()
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved grid: {path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading RF CelebA-HQ-256 model...")
    rf = load_rf_model(device=device)
    print(f"RF param count: {sum(p.numel() for p in rf.model.parameters()):,}")
    sampler = FlowDPSRF(rf, device=device)

    img_paths = sorted(Path(args.test_dir).glob("*.png"))[: args.num_images]
    print(f"Test images: {len(img_paths)} from {args.test_dir}")
    x_clean = torch.stack([load_image(str(p)) for p in img_paths]).to(device)

    def forward_op(x):
        return gaussian_blur(x, sigma=args.sigma_blur)

    y = degrade(
        x_clean,
        blur_type="gaussian",
        blur_sigma=args.sigma_blur,
        noise_sigma=args.sigma_noise,
        seed=args.seed,
    ).to(device)

    print(f"\nRunning FlowDPS-on-RF @ NFE={args.nfe}, zeta={args.zeta}, "
          f"sigma_blur={args.sigma_blur}, sigma_noise={args.sigma_noise}")
    with CUDATimer() as timer:
        result = sampler.sample(
            y=y, forward_op=forward_op,
            num_steps=args.nfe, zeta=args.zeta,
            seed=args.seed, verbose=True,
        )
    x_hat = result.x_hat
    print(f"FlowDPS-on-RF finished in {timer.elapsed_s:.1f}s ({timer.elapsed_s/len(img_paths):.2f}s/img)")

    lpips = LPIPSMetric(device=device)
    psnr_y = [psnr(y[i], x_clean[i]) for i in range(len(img_paths))]
    psnr_r = [psnr(x_hat[i], x_clean[i]) for i in range(len(img_paths))]
    ssim_r = [ssim(x_hat[i], x_clean[i]) for i in range(len(img_paths))]
    lpips_r = lpips(x_hat, x_clean)

    print(f"\nPer-image metrics:")
    print(f"  {'idx':>4} {'PSNR(y)':>9} {'PSNR(x_hat)':>13} {'SSIM(x_hat)':>13} {'LPIPS(x_hat)':>14}")
    for i in range(len(img_paths)):
        print(f"  {i:>4} {psnr_y[i]:>9.2f} {psnr_r[i]:>13.2f} {ssim_r[i]:>13.4f} {lpips_r[i]:>14.4f}")
    print(f"\nMean PSNR(y)={np.mean(psnr_y):.2f}  PSNR(x_hat)={np.mean(psnr_r):.2f}  "
          f"SSIM(x_hat)={np.mean(ssim_r):.4f}  LPIPS(x_hat)={np.mean(lpips_r):.4f}")

    save_grid(
        [x_clean.cpu(), y.cpu(), x_hat.cpu()],
        ["clean", f"y (sigma_b={args.sigma_blur}, sigma_n={args.sigma_noise})", "x_hat (FlowDPS-on-RF)"],
        str(out_dir / f"flowdps_rf_nfe{args.nfe}_zeta{args.zeta}_sb{args.sigma_blur}_sn{args.sigma_noise}.png"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="data/celeba_hq_256/test")
    p.add_argument("--out_dir", default="outputs/smoke")
    p.add_argument("--num_images", type=int, default=2)
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--sigma_blur", type=float, default=3.0)
    p.add_argument("--sigma_noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
