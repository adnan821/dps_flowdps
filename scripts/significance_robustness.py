"""Paired statistical-significance tests for the motion-blur robustness study.

Mirrors `scripts/significance_tests.py` but groups cells by
(motion_length, motion_angle, sigma_noise) instead of (sigma_blur,
sigma_noise). Used to evaluate Tier-B spectral-weight variants against
their v1 baselines under operator-mismatch (true motion blur, sampler
assumes Gaussian).

Compares:
- pixel_dps_spectral vs pixel_dps
- flowdps_rf_spectral vs flowdps_rf
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def main(args):
    methods = {"pixel_dps", "pixel_dps_spectral", "flowdps_rf", "flowdps_rf_spectral"}
    cells: dict[tuple, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    n_loaded = 0
    with open(args.robustness_csv) as f:
        for r in csv.DictReader(f):
            m = r["method"]
            if m not in methods:
                continue
            try:
                L = int(float(r["motion_length"])); theta = float(r["motion_angle"])
                sn = float(r["sigma_noise"]); img = int(r["image_id"])
                psnr = float(r["psnr"])
            except (ValueError, KeyError):
                continue
            cells[(L, theta, sn)][m][img] = psnr
            n_loaded += 1
    print(f"Loaded {n_loaded} rows from {args.robustness_csv}")
    print(f"Cells: {len(cells)}")

    tier_pairs = [
        ("pixel_dps_spectral", "pixel_dps"),
        ("flowdps_rf_spectral", "flowdps_rf"),
    ]
    Ls = sorted({k[0] for k in cells})
    thetas = sorted({k[1] for k in cells})
    sns = sorted({k[2] for k in cells})
    bonf_n = len(Ls) * len(thetas) * len(sns) * len(tier_pairs)

    print(f"\n{'Comparison':<60} {'n':>4} {'mean diff':>9} {'t':>7} {'p (raw)':>10} {'p (bonf)':>10} {'d':>6}")
    print("-" * 116)

    out_rows = []
    for m_new, m_base in tier_pairs:
        for L in Ls:
            for theta in thetas:
                for sn in sns:
                    cell = cells.get((L, theta, sn))
                    if not cell or m_new not in cell or m_base not in cell:
                        continue
                    a = cell[m_new]; b = cell[m_base]
                    shared = sorted(set(a.keys()) & set(b.keys()))
                    if len(shared) < 5:
                        continue
                    x = np.array([a[i] for i in shared])
                    y = np.array([b[i] for i in shared])
                    diff = x - y
                    t, p = stats.ttest_rel(x, y)
                    try:
                        _, p_w = stats.wilcoxon(x, y)
                    except ValueError:
                        p_w = 1.0
                    d = cohen_d(x, y)
                    p_bonf = min(1.0, p * bonf_n)
                    label = f"{m_new} - {m_base} @ L={L}, theta={theta}, sn={sn}"
                    print(f"{label:<60} {len(shared):>4} {diff.mean():>9.3f} {t:>7.2f} {p:>10.2e} {p_bonf:>10.2e} {d:>6.2f}")
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
                        "motion_length": L,
                        "motion_angle": theta,
                        "sigma_noise": sn,
                    })

    if not out_rows:
        print("\nNo comparisons computed.")
        return
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys()); w.writeheader(); w.writerows(out_rows)
    print(f"\nWrote {out_path}")

    print("\n=== Summary by comparison family ===")
    for prefix in ("pixel_dps_spectral - pixel_dps", "flowdps_rf_spectral - flowdps_rf"):
        rows = [r for r in out_rows if r["comparison"].startswith(prefix + " @")]
        if not rows: continue
        avg_diff = statistics.mean(r["mean_diff"] for r in rows)
        max_p_bonf = max(r["p_bonferroni"] for r in rows)
        sig_count = sum(1 for r in rows if r["p_bonferroni"] < 0.05)
        sig_positive = sum(1 for r in rows if r["p_bonferroni"] < 0.05 and r["mean_diff"] > 0)
        sig_negative = sum(1 for r in rows if r["p_bonferroni"] < 0.05 and r["mean_diff"] < 0)
        print(f"{prefix:<45}: mean diff = {avg_diff:+.2f} dB across {len(rows)} cells, "
              f"{sig_count}/{len(rows)} cells Bonferroni p<0.05 "
              f"({sig_positive} positive, {sig_negative} negative), max p={max_p_bonf:.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--robustness_csv", default="outputs/results/robustness.csv")
    p.add_argument("--out_csv", default="outputs/results/significance_robustness.csv")
    args = p.parse_args()
    main(args)
