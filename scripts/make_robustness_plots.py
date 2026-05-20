"""Generate robustness-study figures from outputs/results/robustness.csv,
plus a matched-vs-mismatched comparison against the main grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {"pixel_dps": "Pixel-DPS", "flowdps_rf": "FlowDPS-on-RF"}
METHOD_COLORS = {"pixel_dps": "#1f77b4", "flowdps_rf": "#d62728"}
METHOD_MARKERS = {"pixel_dps": "o", "flowdps_rf": "s"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["psnr", "ssim", "lpips", "time_s", "sigma_blur", "sigma_noise",
                "motion_length", "motion_angle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["nfe"] = pd.to_numeric(df["nfe"], errors="coerce").astype(int)
    return df


def save_fig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png + .pdf")


# -------------------------------------------------------------------------
# 1. PSNR vs motion length, faceted by sigma_n, averaged across angle.
# -------------------------------------------------------------------------
def plot_psnr_vs_length(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, sn in zip(axes, sorted(df.sigma_noise.unique())):
        sub = df[df.sigma_noise == sn]
        for method in METHOD_LABELS:
            m = sub[sub.method == method].groupby("motion_length").psnr.agg(["mean", "std"])
            ax.errorbar(
                m.index, m["mean"], yerr=m["std"],
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method], lw=2, capsize=4,
            )
        ax.set_title(f"σ_n = {sn}")
        ax.set_xlabel("Motion length L (pixels)")
        ax.set_xticks([15, 25, 35])
        ax.set_xlim(10, 40)
        ax.grid(True, ls=":", alpha=0.5)
        if sn == sorted(df.sigma_noise.unique())[0]:
            ax.set_ylabel("PSNR (dB)")
            ax.legend()
    fig.suptitle("Robustness under operator mismatch: PSNR vs motion-blur length")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_dir, "robustness_psnr_vs_length")


# -------------------------------------------------------------------------
# 2. 3-metric panel vs motion length (PSNR, SSIM, LPIPS).
# -------------------------------------------------------------------------
def plot_metrics_vs_length(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric in zip(axes, ["psnr", "ssim", "lpips"]):
        for method in METHOD_LABELS:
            for sn in sorted(df.sigma_noise.unique()):
                sub = df[(df.method == method) & (df.sigma_noise == sn)]
                m = sub.groupby("motion_length")[metric].mean()
                ls = "-" if sn == 0.0 else "--"
                ax.plot(
                    m.index, m.values,
                    color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], ls=ls,
                    label=f"{METHOD_LABELS[method]}, σ_n={sn}", lw=2,
                )
        ax.set_xlabel("Motion length L")
        ax.set_xticks([15, 25, 35])
        ax.set_ylabel(metric.upper() + (" (dB)" if metric == "psnr" else ""))
        ax.grid(True, ls=":", alpha=0.5)
        if metric == "psnr":
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Robustness study — quality vs motion-blur severity (angle averaged)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_dir, "robustness_metrics_vs_length")


# -------------------------------------------------------------------------
# 3. Heatmap of per-cell PSNR.
# -------------------------------------------------------------------------
def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, method in zip(axes, METHOD_LABELS):
        sub = df[df.method == method]
        pivot = sub.pivot_table(
            index="motion_length",
            columns=["sigma_noise", "motion_angle"],
            values="psnr",
            aggfunc="mean",
        )
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=19, vmax=26)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"σ_n={c[0]}\nθ={c[1]:.0f}°" for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"L={v}" for v in pivot.index])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < 22 else "black", fontsize=10)
        ax.set_title(f"{METHOD_LABELS[method]} — PSNR (dB)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-cell mean PSNR under operator mismatch (assumed=gauss(σ=3.0))")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_dir, "robustness_psnr_heatmap")


# -------------------------------------------------------------------------
# 4. Matched (main grid) vs mismatched (robustness): bar chart per method.
# -------------------------------------------------------------------------
def plot_matched_vs_mismatched(
    rob_df: pd.DataFrame, main_df: pd.DataFrame, out_dir: Path
) -> None:
    """Same NFE=50 setting; compare matched-Gaussian vs mismatched-motion."""
    # Matched: main-grid Gaussian at σ_b=3.0, σ_n=0.05, NFE=50 → one number per method.
    matched = (
        main_df[(main_df.sigma_blur == 3.0) & (main_df.sigma_noise == 0.05) & (main_df.nfe == 50)]
        .groupby("method").psnr.mean()
    )
    # Mismatched: robustness at θ=0 (we average angle below for fairness with above
    # which is single-angle). Take same σ_n=0.05.
    mismatched = (
        rob_df[(rob_df.sigma_noise == 0.05) & (rob_df.nfe == 50)]
        .groupby(["method", "motion_length"]).psnr.mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = ["pixel_dps", "flowdps_rf"]
    x = np.arange(4)
    width = 0.38
    for i, method in enumerate(methods):
        vals = [
            matched.get(method, np.nan),
            mismatched.loc[method, 15],
            mismatched.loc[method, 25],
            mismatched.loc[method, 35],
        ]
        ax.bar(x + (i - 0.5) * width, vals, width,
               color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 0.5) * width, v + 0.15, f"{v:.1f}",
                    ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([
        "Matched\n(Gauss σ=3.0)",
        "Mismatched\nMotion L=15",
        "Mismatched\nMotion L=25",
        "Mismatched\nMotion L=35",
    ])
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Operator-mismatch cost (σ_n=0.05, NFE=50, angle averaged for L>=15)")
    ax.grid(True, ls=":", alpha=0.4, axis="y")
    ax.legend(loc="upper right")
    ax.set_ylim(18, 28)
    fig.tight_layout()
    save_fig(fig, out_dir, "matched_vs_mismatched")


# -------------------------------------------------------------------------
# 5. Dump per-cell summary table.
# -------------------------------------------------------------------------
def dump_summary(df: pd.DataFrame, out_dir: Path) -> None:
    grp = df.groupby(["method", "motion_length", "motion_angle", "sigma_noise"]).agg(
        n=("psnr", "count"),
        psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
        ssim_mean=("ssim", "mean"),
        lpips_mean=("lpips", "mean"),
        time_mean=("time_s", "mean"),
    ).reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "robustness_summary.md"
    with md.open("w") as f:
        f.write("# Robustness study summary (mean ± std over 50 images per cell)\n\n")
        f.write("| method | L | θ | σ_n | n | PSNR (dB) | SSIM | LPIPS | s/img |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in grp.iterrows():
            f.write(
                f"| {r['method']} | {int(r['motion_length'])} | {r['motion_angle']:.0f}° | "
                f"{r['sigma_noise']} | {int(r['n'])} | "
                f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f} | "
                f"{r['ssim_mean']:.3f} | {r['lpips_mean']:.3f} | {r['time_mean']:.2f} |\n"
            )
    print(f"  saved {md}")
    grp.to_csv(out_dir / "robustness_summary.csv", index=False, float_format="%.4f")
    print(f"  saved {out_dir / 'robustness_summary.csv'}")


def main(args):
    rob = load(args.robustness_csv)
    main_df = load(args.main_csv)
    print(f"Robustness rows: {len(rob)}; main-grid rows: {len(main_df)}")
    out_dir = Path(args.out_dir)
    print(f"Writing figures to {out_dir}/")
    plot_psnr_vs_length(rob, out_dir)
    plot_metrics_vs_length(rob, out_dir)
    plot_heatmap(rob, out_dir)
    plot_matched_vs_mismatched(rob, main_df, out_dir)
    dump_summary(rob, out_dir)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--robustness_csv", default="outputs/results/robustness.csv")
    p.add_argument("--main_csv", default="outputs/results/main_grid.csv")
    p.add_argument("--out_dir", default="outputs/figures")
    args = p.parse_args()
    main(args)
