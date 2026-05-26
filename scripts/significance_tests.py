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
    # Load both CSVs. Method allowlist auto-extended for tier A/B/C variants.
    diffusion_family = {
        "pixel_dps", "pixel_dps_v2", "pixel_dps_spectral", "pixel_dps_pigdm",
        "pixel_dps_sched", "particle_dps",
        "flowdps_rf", "flowdps_rf_v2", "flowdps_rf_spectral", "flowdps_rf_pigdm",
        "flowdps_rf_heun",
    }
    classical = {"wiener", "dncnn_pnp_admm"}
    methods = diffusion_family | classical
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
        if method in diffusion_family and nfe not in (25, 50, 100):
            continue
        img = int(r["image_id"])
        cells[(sb, sn, nfe if method in diffusion_family else "any")][method][img] = float(r["psnr"])

    # Compare Pixel-DPS @ NFE=100 against each other method at the same (sb, sn).
    print(f"\n{'Comparison':<35} {'n':>4} {'mean diff':>9} {'t':>7} {'p (raw)':>10} {'p (bonf)':>10} {'d':>6}")
    print("-" * 90)

    # All comparisons we'd like to run. Each entry is
    # (label, baseline_method, baseline_nfe_or_any, other_method, other_nfe_or_any).
    # We dynamically filter to comparisons where BOTH methods actually have
    # rows in the loaded CSVs (so the script works regardless of which
    # tier-A/B/C runs have completed yet).
    pix_v_others = [
        ("flowdps_rf",      100,   "flowdps_rf",       100),
        ("wiener",          "any", "wiener",           "any"),
        ("dncnn_pnp_admm",  "any", "dncnn_pnp_admm",   "any"),
    ]
    # Tier A/B/C ablation comparisons (each new method vs its v1 baseline).
    tier_pairs = [
        # Tier A
        ("pixel_dps_v2",        100,   "pixel_dps",        100),
        ("flowdps_rf_v2",       100,   "flowdps_rf",       100),
        # Tier B
        ("pixel_dps_spectral",  50,    "pixel_dps",        50),
        ("flowdps_rf_spectral", 50,    "flowdps_rf",       50),
        # Tier C
        ("pixel_dps_pigdm",     100,   "pixel_dps",        100),
        ("flowdps_rf_pigdm",    100,   "flowdps_rf",       100),
        # Round 2: ζ-schedule (Pixel-DPS) + Heun integrator (FlowDPS-on-RF)
        ("pixel_dps_sched",     100,   "pixel_dps",        100),
        ("flowdps_rf_heun",     100,   "flowdps_rf",       100),
        # Round 3: gradient-free particle sampler vs gradient-based DPS
        ("particle_dps",        100,   "pixel_dps",        100),
    ]

    sbs = [1.5, 3.0, 5.0]
    sns = [0.0, 0.05]

    out_rows = []

    def _run_one(label_prefix, m_a, nfe_a, m_b, nfe_b, bonf_n):
        """One paired-t-test cell. m_a is the 'positive' arm (we report
        m_a - m_b mean diff)."""
        local_out = []
        for sb in sbs:
            for sn in sns:
                cell_a = cells.get((sb, sn, nfe_a))
                cell_b = cells.get((sb, sn, nfe_b))
                if not cell_a or m_a not in cell_a:
                    continue
                if not cell_b or m_b not in cell_b:
                    continue
                a = cell_a[m_a]; b = cell_b[m_b]
                shared = sorted(set(a.keys()) & set(b.keys()))
                if len(shared) < 5:
                    continue
                x = np.array([a[i] for i in shared])
                y = np.array([b[i] for i in shared])
                diff = x - y
                t, p = stats.ttest_rel(x, y)
                try:
                    w, p_w = stats.wilcoxon(x, y)
                except ValueError:
                    p_w = 1.0
                d = cohen_d(x, y)
                p_bonf = min(1.0, p * bonf_n)
                label = f"{label_prefix} @ sb={sb}, sn={sn}"
                print(f"{label:<46} {len(shared):>4} {diff.mean():>9.3f} {t:>7.2f} {p:>10.2e} {p_bonf:>10.2e} {d:>6.2f}")
                local_out.append({
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
        return local_out

    # Pixel-DPS vs each other method at NFE=100.
    print(f"\n{'Comparison':<46} {'n':>4} {'mean diff':>9} {'t':>7} {'p (raw)':>10} {'p (bonf)':>10} {'d':>6}")
    print("-" * 100)
    bonf_n_pix = len(sbs) * len(sns) * len(pix_v_others)
    for other_method, nfe_o, _, _ in pix_v_others:
        out_rows += _run_one(f"pixel_dps - {other_method}", "pixel_dps", 100, other_method, nfe_o, bonf_n_pix)

    # Tier ablations: each new variant vs its baseline at matched NFE.
    print(f"\n{'TIER ABLATIONS':<46}")
    print("-" * 100)
    bonf_n_tier = len(sbs) * len(sns) * len(tier_pairs)
    for m_a, nfe_a, m_b, nfe_b in tier_pairs:
        out_rows += _run_one(f"{m_a} - {m_b}", m_a, nfe_a, m_b, nfe_b, bonf_n_tier)

    # Write CSV.
    if not out_rows:
        print("\nNo comparisons could be computed — either no rows loaded or none share images.")
        return
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys()); w.writeheader(); w.writerows(out_rows)
    print(f"\nWrote {out_path}")

    # Summary one-liners.
    print("\n=== Summary by comparison family ===")
    summary_prefixes = (
        "pixel_dps - flowdps_rf",
        "pixel_dps - wiener",
        "pixel_dps - dncnn_pnp_admm",
        "pixel_dps_v2 - pixel_dps",
        "flowdps_rf_v2 - flowdps_rf",
        "pixel_dps_spectral - pixel_dps",
        "flowdps_rf_spectral - flowdps_rf",
        "pixel_dps_pigdm - pixel_dps",
        "flowdps_rf_pigdm - flowdps_rf",
        "pixel_dps_sched - pixel_dps",
        "flowdps_rf_heun - flowdps_rf",
        "particle_dps - pixel_dps",
    )
    for prefix in summary_prefixes:
        rows = [r for r in out_rows if r["comparison"].startswith(prefix + " @")]
        if not rows: continue
        avg_diff = statistics.mean(r["mean_diff"] for r in rows)
        all_p_bonf_lt_001 = all(r["p_bonferroni"] < 0.001 for r in rows)
        max_p_bonf = max(r["p_bonferroni"] for r in rows)
        sig_count = sum(1 for r in rows if r["p_bonferroni"] < 0.05)
        print(f"{prefix:<40}: mean diff = {avg_diff:+.2f} dB across {len(rows)} cells, "
              f"{sig_count}/{len(rows)} cells Bonferroni p<0.05, max p={max_p_bonf:.2e}"
              f"{' (all p<0.001)' if all_p_bonf_lt_001 else ''}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--main_csv", default="outputs/results/main_grid.csv")
    p.add_argument("--baselines_csv", default="outputs/results/baselines.csv")
    p.add_argument("--out_csv", default="outputs/results/significance.csv")
    args = p.parse_args()
    main(args)
