"""Paired statistical-significance tests for the main-grid + baselines.

For each pair (Pixel-DPS, other_method) and each metric (PSNR, SSIM,
LPIPS), aligns the per-image vectors across matched (sigma_b, sigma_n,
NFE) cells and runs:

- Paired t-test (parametric)
- Wilcoxon signed-rank test (non-parametric, robust to non-normality)
- Cohen's d effect size

Bonferroni-corrected p-values are reported with respect to the number
of (method, metric) comparisons.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_psnr_by_image(csv_path: str, methods: set[str]) -> dict:
    """Returns {(method, sb, sn, nfe): {image_id: psnr}}."""
    out = defaultdict(dict)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["method"] not in methods:
                continue
            key = (r["method"], float(r["sigma_blur"]), float(r["sigma_noise"]),
                   int(r["nfe"]) if r["nfe"] else 0)
            out[key][int(r["image_id"])] = {
                "psnr": float(r["psnr"]),
                "ssim": float(r["ssim"]),
                "lpips": float(r["lpips"]),
            }
    return out


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples (uses std of differences)."""
    diff = x - y
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def main(args):
    # Load both CSVs.
    methods = {"pixel_dps", "flowdps_rf", "wiener", "dncnn_pnp_admm"}
    rows = []
    for csv_path in [args.main_csv, args.baselines_csv]:
        if Path(csv_path).exists():
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    if r["method"] in methods:
                        rows.append(r)
    print(f"Loaded {len(rows)} rows across all methods")

    # For each NFE setting + sigma cell, collect per-image vectors per method.
    cells = defaultdict(lambda: defaultdict(dict))  # (sb, sn, nfe) -> method -> img -> psnr
    for r in rows:
        sb = float(r["sigma_blur"]); sn = float(r["sigma_noise"])
        method = r["method"]
        nfe_raw = r["nfe"]
        # Wiener / DnCNN+PnP record nfe=0 or iter count. We compare them
        # only at the NFE=100 cells of the diffusion methods (apples to
        # the diffusion-best comparison). Use 'aggregate over their grid'.
        nfe = int(nfe_raw) if nfe_raw and nfe_raw != "0" else 0
        if method in ("pixel_dps", "flowdps_rf") and nfe not in (25, 50, 100):
            continue
        img = int(r["image_id"])
        cells[(sb, sn, nfe if method in ("pixel_dps", "flowdps_rf") else "any")][method][img] = float(r["psnr"])

    # Compare Pixel-DPS @ NFE=100 against each other method at the same (sb, sn).
    print(f"\n{'Comparison':<35} {'n':>4} {'mean diff':>9} {'t':>7} {'p (raw)':>10} {'p (bonf)':>10} {'d':>6}")
    print("-" * 90)

    other_methods = ["flowdps_rf", "wiener", "dncnn_pnp_admm"]
    sbs = [1.5, 3.0, 5.0]
    sns = [0.0, 0.05]
    # We have (3 sb) * (2 sn) * (3 methods) = 18 tests at NFE=100 (vs Pixel-DPS).
    bonf_n = len(sbs) * len(sns) * len(other_methods)

    out_rows = []
    for other in other_methods:
        for sb in sbs:
            for sn in sns:
                # Pixel-DPS at NFE=100
                cell_p = cells.get((sb, sn, 100))
                if not cell_p or "pixel_dps" not in cell_p:
                    continue
                # Other method: use NFE=100 for flowdps_rf, "any" for classical
                if other == "flowdps_rf":
                    cell_o = cells.get((sb, sn, 100))
                else:
                    cell_o = cells.get((sb, sn, "any"))
                if not cell_o or other not in cell_o:
                    continue
                pix = cell_p["pixel_dps"]
                oth = cell_o[other]
                shared = sorted(set(pix.keys()) & set(oth.keys()))
                if len(shared) < 5:
                    continue
                x = np.array([pix[i] for i in shared])
                y = np.array([oth[i] for i in shared])
                diff = x - y
                t, p = stats.ttest_rel(x, y)
                w, p_w = stats.wilcoxon(x, y)
                d = cohen_d(x, y)
                p_bonf = min(1.0, p * bonf_n)
                label = f"pixel_dps - {other} @ sb={sb}, sn={sn}"
                print(f"{label:<35} {len(shared):>4} {diff.mean():>9.3f} {t:>7.2f} {p:>10.2e} {p_bonf:>10.2e} {d:>6.2f}")
                out_rows.append({
                    "comparison": label,
                    "n": len(shared),
                    "mean_diff": round(float(diff.mean()), 4),
                    "std_diff": round(float(diff.std(ddof=1)), 4),
                    "t_stat": round(float(t), 4),
                    "p_raw": float(f"{p:.3e}"),
                    "p_bonferroni": float(f"{p_bonf:.3e}"),
                    "wilcoxon_p": float(f"{p_w:.3e}"),
                    "cohen_d": round(float(d), 4),
                    "sigma_blur": sb,
                    "sigma_noise": sn,
                })

    # Write CSV.
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys()); w.writeheader(); w.writerows(out_rows)
    print(f"\nWrote {out_path}")

    # Summary one-liners.
    print("\n=== Summary ===")
    for other in other_methods:
        rows = [r for r in out_rows if other in r["comparison"]]
        if not rows: continue
        avg_diff = statistics.mean(r["mean_diff"] for r in rows)
        all_p_bonf_lt_001 = all(r["p_bonferroni"] < 0.001 for r in rows)
        max_p_bonf = max(r["p_bonferroni"] for r in rows)
        print(f"Pixel-DPS vs {other:<16}: mean diff = {avg_diff:+.2f} dB across {len(rows)} cells, "
              f"max Bonferroni p = {max_p_bonf:.2e} {'(all p < 0.001)' if all_p_bonf_lt_001 else ''}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--main_csv", default="outputs/results/main_grid.csv")
    p.add_argument("--baselines_csv", default="outputs/results/baselines.csv")
    p.add_argument("--out_csv", default="outputs/results/significance.csv")
    args = p.parse_args()
    main(args)
