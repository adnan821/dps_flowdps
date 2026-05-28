"""Estimate blur-kernel parameters from a single blurry image (calibrated).

Reads a blurry image and writes a `blurry.json` sidecar that
`scripts/deblur_image.py --blur_metadata` consumes.

Two-stage algorithm:

  1. CALIBRATION (done once per run): re-degrade a small set of clean
     CelebA-HQ test images at every (blur_type, params, sigma_n) combo in
     the project's main-grid pools. For each combo, average the radial
     profile of |FFT(y)|. This produces a reference signature per cell.

  2. INFERENCE: compute the same radial profile on the user's blurry
     input, then pick the pool cell whose reference signature has the
     smallest L2 distance on log-radial-profile.

This is robust by construction: we're matching the spectral signature
to one we KNOW how to generate from the same pipeline. The estimate is
guaranteed to be a valid pool cell.

Caveats:
  - Faces from a similar distribution to CelebA-HQ are needed for the
    calibration images. Default uses 5 test images, which is enough.
  - The estimator can only return values inside the pool. If the user's
    blur is outside {1.5, 3.0, 5.0} for Gaussian sigma or {15, 25, 35}
    for motion length, the nearest pool value is returned.
  - For real-world photos with JPEG artefacts, defocus, or complex
    content, the estimate degrades — fall back to manual --sigma_b /
    --motion_* flags or trust the printed confidence indicator.

Usage:

  python scripts/estimate_blur.py --input blurry.jpg
  # -> writes outputs/user_deblur/blurry.json and prints the
  #    deblur_image.py command line that consumes it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.forward.degradation import degrade


# Main-grid pools.
SIGMA_B_POOL = [1.5, 3.0, 5.0]
MOTION_LENGTH_POOL = [15, 25, 35]
MOTION_ANGLE_POOL = [0.0, 45.0]
SIGMA_N_POOL = [0.0, 0.05]


def load_image_tensor(path: str, size: int = 256) -> torch.Tensor:
    """Load image, resize, return as (1, C, H, W) tensor in [0,1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def spectral_feature(img_chw: torch.Tensor, n_sectors: int = 4) -> np.ndarray:
    """Compute a polar-binned log-magnitude FFT feature.

    The 2D spectrum is binned by (radius, angular sector). Returns a
    concatenation of (n_sectors) angular-sector radial profiles. This
    captures BOTH radial decay (which discriminates blur strength) AND
    angular anisotropy (which discriminates Gaussian-isotropic from
    motion-anisotropic blur).
    """
    if img_chw.dim() == 4:
        img_chw = img_chw.squeeze(0)
    img = img_chw.mean(0).cpu().numpy()
    H, W = img.shape
    assert H == W
    N = H

    Y = np.fft.fft2(img - img.mean())
    mag = np.abs(np.fft.fftshift(Y))

    cy, cx = N // 2, N // 2
    yy, xx = np.indices((N, N))
    dy = yy - cy
    dx = xx - cx
    r = np.round(np.sqrt(dy ** 2 + dx ** 2)).astype(int)
    # Angular sector index in [0, n_sectors). Motion blur is symmetric
    # under 180-deg rotation, so we wrap angles into [0, 180) and bin.
    theta = (np.degrees(np.arctan2(dy, dx)) % 180.0)
    sector = np.minimum((theta * n_sectors / 180.0).astype(int), n_sectors - 1)

    rmax = N // 2
    profiles = []
    for s in range(n_sectors):
        mask = (sector == s)
        rs = r[mask].ravel()
        ms = mag[mask].ravel()
        prof = np.bincount(rs, weights=ms, minlength=rmax + 1)
        cnts = np.bincount(rs, minlength=rmax + 1)
        cnts = np.maximum(cnts, 1)
        profiles.append(np.log(prof[: rmax + 1] / cnts[: rmax + 1] + 1e-12))
    return np.concatenate(profiles)


# Backward-compat alias (kept in case anything imports the old name).
def radial_profile(img_chw: torch.Tensor) -> np.ndarray:
    return spectral_feature(img_chw, n_sectors=1)


def _feature(img: torch.Tensor, n_sectors: int, band: tuple[int, int]) -> np.ndarray:
    rmax = 256 // 2 + 1
    full = spectral_feature(img, n_sectors=n_sectors)
    # full has length n_sectors * rmax; slice each sector to [lo, hi]
    lo, hi = band
    sectors = [full[s * rmax: (s + 1) * rmax][lo:hi] for s in range(n_sectors)]
    return np.concatenate(sectors)


def build_reference_signatures(
    calibration_paths: list[str],
    feature_band: tuple[int, int],
    n_sectors: int,
    verbose: bool = True,
) -> dict[tuple, np.ndarray]:
    """For every pool cell, average the polar-binned spectral feature
    across the calibration images."""
    print(f"[calibrate] Loading {len(calibration_paths)} reference clean images")
    cleans = [load_image_tensor(str(p)) for p in calibration_paths]

    refs: dict[tuple, np.ndarray] = {}

    g_cells = list(itertools.product(SIGMA_B_POOL, SIGMA_N_POOL))
    print(f"[calibrate] Building references for {len(g_cells)} Gaussian cells")
    for sb, sn in g_cells:
        profs = []
        for x in cleans:
            y = degrade(x, blur_type="gaussian", blur_sigma=sb, noise_sigma=sn, seed=0)
            profs.append(_feature(y, n_sectors, feature_band))
        refs[("gaussian", sb, sn)] = np.stack(profs).mean(0)

    m_cells = list(itertools.product(MOTION_LENGTH_POOL, MOTION_ANGLE_POOL, SIGMA_N_POOL))
    print(f"[calibrate] Building references for {len(m_cells)} motion cells")
    for L, theta, sn in m_cells:
        profs = []
        for x in cleans:
            y = degrade(x, blur_type="motion",
                        motion_length=L, motion_angle_deg=theta,
                        noise_sigma=sn, seed=0)
            profs.append(_feature(y, n_sectors, feature_band))
        refs[("motion", L, theta, sn)] = np.stack(profs).mean(0)

    if verbose:
        print(f"[calibrate] Done — {len(refs)} reference signatures built")
    return refs


def best_match(input_profile: np.ndarray, refs: dict[tuple, np.ndarray]) -> tuple[tuple, float, dict]:
    """Find the closest reference signature in L2.
    Returns (best_key, best_distance, dict[key -> distance]).
    """
    distances = {}
    for key, ref in refs.items():
        d = float(np.linalg.norm(input_profile - ref))
        distances[key] = d
    best_key = min(distances, key=distances.get)
    return best_key, distances[best_key], distances


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--input", required=True, help="Path to the BLURRY input image.")
    p.add_argument("--output_dir", default="outputs/user_deblur",
                   help="Where to write the blurry.json sidecar.")
    p.add_argument("--calibration_glob",
                   default="data/celeba_hq_256/test/0000[0-4].png",
                   help="Glob pattern for clean calibration images. The pattern's "
                        "default matches the first five test images; pass a different "
                        "glob if your faces are in another folder.")
    p.add_argument("--verbose", action="store_true",
                   help="Print the full distance table to all reference cells.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load input.
    print(f"[1] Loading: {args.input}")
    x_input = load_image_tensor(args.input)
    feature_band = (10, 80)   # mid-frequency band where blur OTF dominates
    n_sectors = 8             # angular sectors; captures motion anisotropy

    input_profile = _feature(x_input, n_sectors, feature_band)

    # 2. Calibration — re-blur clean images at every pool cell.
    calibration_paths = sorted(Path().glob(args.calibration_glob))
    if not calibration_paths:
        raise FileNotFoundError(f"No calibration images matched {args.calibration_glob!r}. "
                                f"Pass --calibration_glob to override.")
    refs = build_reference_signatures(calibration_paths, feature_band, n_sectors)

    # 3. Nearest-neighbor on log-radial-profile L2.
    best_key, best_dist, all_dist = best_match(input_profile, refs)
    runner_up = sorted(all_dist.items(), key=lambda kv: kv[1])[1]
    margin = runner_up[1] - best_dist  # how confidently the winner beat the runner-up
    confidence_ratio = margin / max(best_dist, 1e-6)

    print(f"\n[2] Best match: {best_key}  (L2={best_dist:.3f})")
    print(f"    Runner-up:  {runner_up[0]}  (L2={runner_up[1]:.3f})")
    print(f"    Confidence (margin / best_dist): {confidence_ratio:.2f}")
    if args.verbose:
        print("\nAll cell distances:")
        for key, d in sorted(all_dist.items(), key=lambda kv: kv[1]):
            print(f"    {key} -> {d:.3f}")

    # 4. Unpack into sidecar fields.
    if best_key[0] == "gaussian":
        _, sigma_b, sigma_n = best_key
        blur_type = "gaussian"
        motion_length = motion_angle = None
    else:
        _, motion_length, motion_angle, sigma_n = best_key
        blur_type = "motion"
        sigma_b = None

    if blur_type == "gaussian":
        print(f"\n[3] Estimate: gaussian, sigma_b={sigma_b}, sigma_n={sigma_n}")
    else:
        print(f"\n[3] Estimate: motion, length={motion_length}, "
              f"angle={motion_angle} deg, sigma_n={sigma_n}")

    if confidence_ratio < 0.05:
        print(f"    WARNING: low confidence (margin tiny). The estimate "
              f"may be unreliable — consider passing manual flags to deblur.")

    sidecar = {
        "clean_input": None,
        "clean_image": None,
        "blurry_image": str(Path(args.input).resolve()),
        "blur_type": blur_type,
        "sigma_b": sigma_b,
        "motion_length": motion_length,
        "motion_angle": motion_angle,
        "sigma_n": float(sigma_n),
        "seed": 0,
        "resolution": 256,
        "estimation": {
            "method": "calibrated_nearest_neighbor",
            "best_l2": float(best_dist),
            "runner_up_l2": float(runner_up[1]),
            "confidence_ratio": float(confidence_ratio),
            "calibration_images": [str(p) for p in calibration_paths],
            "feature_band": list(feature_band),
        },
    }
    sidecar_path = out_dir / "blurry.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    print(f"\n[4] Wrote {sidecar_path}")
    print("\nTo deblur with the estimated parameters:")
    print(f"  python scripts/deblur_image.py \\")
    print(f"      --input {args.input} \\")
    print(f"      --blur_metadata {sidecar_path} \\")
    print(f"      --method pixel_dps,flowdps_rf --nfe 100")


if __name__ == "__main__":
    main()
