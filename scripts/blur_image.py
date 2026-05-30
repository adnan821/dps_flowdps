"""Blur a single user-supplied image and report the parameters used.

Useful as the first step of a two-step pipeline:
  1. scripts/blur_image.py   — pick (or randomize) blur params, produce blurry.png
  2. scripts/deblur_image.py — read the sidecar JSON, deblur with the same params

Three ways to pick the blur:

  (a) Explicit (most reproducible):
      python scripts/blur_image.py --input clean.png \\
          --blur_type gaussian --sigma_b 3.0 --sigma_n 0.05

  (b) Random within sensible ranges (matches the project's main grid):
      python scripts/blur_image.py --input clean.png --randomize
      # picks blur_type uniformly; for gaussian draws sigma_b from
      # {1.5, 3.0, 5.0} and sigma_n from {0.0, 0.05}; for motion draws
      # length from {15, 25, 35} and angle from {0, 45} degrees.

  (c) Defaults (gaussian, sigma_b=3.0, sigma_n=0.05):
      python scripts/blur_image.py --input clean.png

Outputs (in --output_dir, default outputs/user_deblur/):
  - clean.png        a 256x256 resize of the input
  - blurry.png       the measurement y = A(x) + noise
  - blurry.json      sidecar with every parameter used, so the deblur
                     script can read it back: see --blur_metadata in
                     scripts/deblur_image.py

The printed params at the end of the run are the same ones written to
the JSON. Either copy-paste them into the deblur script's CLI flags,
or just point --blur_metadata at blurry.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.forward.degradation import degrade


# Sensible-range pools for --randomize. These are exactly the project's
# main-grid axes, so randomly-blurred test cases map directly onto regimes
# where we know the samplers' behavior.
RANDOM_POOLS = {
    "gaussian": {
        "sigma_b": [1.5, 3.0, 5.0],
        "sigma_n": [0.0, 0.05],
    },
    "motion": {
        "motion_length": [15, 25, 35],
        "motion_angle": [0.0, 45.0],
        "sigma_n": [0.0, 0.05],
    },
    "sr": {
        "sr_factor": [2, 4, 8],
        "sigma_n": [0.0, 0.05],
    },
}


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


def randomize_params(rng: random.Random, blur_type: str | None) -> dict:
    """Draw blur params uniformly from the project's main-grid pools."""
    if blur_type is None:
        blur_type = rng.choice(["gaussian", "motion", "sr"])
    pool = RANDOM_POOLS[blur_type]
    if blur_type == "gaussian":
        return {
            "blur_type": "gaussian",
            "sigma_b": rng.choice(pool["sigma_b"]),
            "motion_length": None,
            "motion_angle": None,
            "sr_factor": None,
            "sigma_n": rng.choice(pool["sigma_n"]),
        }
    if blur_type == "motion":
        return {
            "blur_type": "motion",
            "sigma_b": None,
            "motion_length": rng.choice(pool["motion_length"]),
            "motion_angle": rng.choice(pool["motion_angle"]),
            "sr_factor": None,
            "sigma_n": rng.choice(pool["sigma_n"]),
        }
    return {
        "blur_type": "sr",
        "sigma_b": None,
        "motion_length": None,
        "motion_angle": None,
        "sr_factor": rng.choice(pool["sr_factor"]),
        "sigma_n": rng.choice(pool["sigma_n"]),
    }


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--input", required=True, help="Path to the CLEAN input image.")
    p.add_argument("--output_dir", default="outputs/user_deblur",
                   help="Where to write clean.png, blurry.png, blurry.json.")
    p.add_argument("--randomize", action="store_true",
                   help="Pick blur params uniformly from the project's main-grid "
                        "pools instead of using the explicit flags below.")
    p.add_argument("--blur_type", default=None, choices=["gaussian", "motion", "sr"],
                   help="If --randomize is set, restrict the random draw to this "
                        "operator family. Otherwise selects the operator family.")
    p.add_argument("--sigma_b", type=float, default=3.0,
                   help="Gaussian blur std in pixels (used when --blur_type=gaussian).")
    p.add_argument("--motion_length", type=int, default=25,
                   help="Motion-blur kernel length in pixels.")
    p.add_argument("--motion_angle", type=float, default=45.0,
                   help="Motion-blur angle in degrees.")
    p.add_argument("--sr_factor", type=int, default=4,
                   help="Super-resolution downsampling factor (used when --blur_type=sr).")
    p.add_argument("--sigma_n", type=float, default=0.05,
                   help="Measurement noise std in [0,1] image units.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the noise (and for --randomize draws).")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load clean image.
    print(f"[1] Loading clean image: {args.input}")
    clean = load_image(args.input)  # (C, H, W), [0,1]
    clean_b = clean.unsqueeze(0)    # (1, C, H, W)
    save_image(clean, out_dir / "clean.png")

    # 2. Pick params.
    if args.randomize:
        rng = random.Random(args.seed)
        chosen = randomize_params(rng, blur_type=args.blur_type)
        print(f"[2] Randomized params: {chosen}")
    else:
        blur_type = args.blur_type or "gaussian"
        chosen = {
            "blur_type": blur_type,
            "sigma_b": args.sigma_b if blur_type == "gaussian" else None,
            "motion_length": args.motion_length if blur_type == "motion" else None,
            "motion_angle": args.motion_angle if blur_type == "motion" else None,
            "sr_factor": args.sr_factor if blur_type == "sr" else None,
            "sigma_n": args.sigma_n,
        }
        print(f"[2] Explicit params: {chosen}")

    # 3. Apply degradation.
    print(f"[3] Applying degradation (seed={args.seed})")
    if chosen["blur_type"] == "gaussian":
        y = degrade(
            clean_b, blur_type="gaussian",
            blur_sigma=chosen["sigma_b"],
            noise_sigma=chosen["sigma_n"], seed=args.seed,
        )
    elif chosen["blur_type"] == "motion":
        y = degrade(
            clean_b, blur_type="motion",
            motion_length=chosen["motion_length"],
            motion_angle_deg=chosen["motion_angle"],
            noise_sigma=chosen["sigma_n"], seed=args.seed,
        )
    else:  # sr
        y = degrade(
            clean_b, blur_type="sr",
            sr_factor=chosen["sr_factor"],
            noise_sigma=chosen["sigma_n"], seed=args.seed,
        )
    save_image(y, out_dir / "blurry.png")

    # 4. Sidecar JSON for deblur consumption.
    sidecar = {
        "clean_input": str(Path(args.input).resolve()),
        "clean_image": "clean.png",
        "blurry_image": "blurry.png",
        "blur_type": chosen["blur_type"],
        "sigma_b": chosen["sigma_b"],
        "motion_length": chosen["motion_length"],
        "motion_angle": chosen["motion_angle"],
        "sr_factor": chosen.get("sr_factor"),
        "sigma_n": chosen["sigma_n"],
        "seed": args.seed,
        "resolution": 256,
    }
    sidecar_path = out_dir / "blurry.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    # 5. Report.
    print(f"\nDone. Wrote:")
    print(f"  {out_dir/'clean.png'}")
    print(f"  {out_dir/'blurry.png'}")
    print(f"  {sidecar_path}")
    print(f"\nTo deblur with the same parameters:")
    print(f"  python scripts/deblur_image.py \\")
    print(f"      --input {out_dir/'blurry.png'} \\")
    print(f"      --blur_metadata {sidecar_path} \\")
    print(f"      --method pixel_dps,flowdps_rf --nfe 100")


if __name__ == "__main__":
    main()
