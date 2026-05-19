"""Generate all main-grid analysis figures from outputs/results/main_grid.csv.

Each plot is in its own function so they can be regenerated individually
later. Writes PNG + PDF (PDF for the report LaTeX).
"""
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


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["psnr"] = pd.to_numeric(df["psnr"], errors="coerce")
    df["ssim"] = pd.to_numeric(df["ssim"], errors="coerce")
    df["lpips"] = pd.to_numeric(df["lpips"], errors="coerce")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["nfe"] = pd.to_numeric(df["nfe"], errors="coerce").astype(int)
    df["sigma_blur"] = pd.to_numeric(df["sigma_blur"], errors="coerce")
    df["sigma_noise"] = pd.to_numeric(df["sigma_noise"], errors="coerce")
    return df


def save_fig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {png} + {pdf}")


# -------------------------------------------------------------------------
# 1. PSNR / SSIM / LPIPS vs NFE, faceted by (sigma_b, sigma_n).
# -------------------------------------------------------------------------
def plot_metric_vs_nfe(df: pd.DataFrame, out_dir: Path, metric: str = "psnr") -> None:
    sigma_bs = sorted(df["sigma_blur"].unique())
    sigma_ns = sorted(df["sigma_noise"].unique())
    fig, axes = plt.subplots(
        len(sigma_ns), len(sigma_bs),
        figsize=(3.2 * len(sigma_bs), 3 * len(sigma_ns)),
        sharey=True, squeeze=False,
    )
    for i, sn in enumerate(sigma_ns):
        for j, sb in enumerate(sigma_bs):
            ax = axes[i, j]
            sub = df[(df.sigma_blur == sb) & (df.sigma_noise == sn)]
            for method in METHOD_LABELS:
                m = sub[sub.method == method].groupby("nfe")[metric].agg(["mean", "std"])
                if m.empty:
                    continue
                ax.errorbar(
                    m.index, m["mean"], yerr=m["std"],
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    capsize=3, lw=2,
                )
            # linear is clearer than log for 3 discrete NFE budgets
            ax.set_xticks([25, 50, 100])
            ax.set_xticklabels(["25", "50", "100"])
            ax.set_xlim(15, 110)
            ax.set_xlabel("NFE")
            if j == 0:
                ax.set_ylabel(metric.upper() + (" (dB)" if metric == "psnr" else ""))
            ax.set_title(f"σ_b = {sb}, σ_n = {sn}", fontsize=10)
            ax.grid(True, ls=":", alpha=0.5)
            if i == 0 and j == 0:
                ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{metric.upper()} vs NFE across degradation conditions", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, out_dir, f"{metric}_vs_nfe")


# -------------------------------------------------------------------------
# 2. PSNR vs wall-clock per image — efficiency frontier.
# -------------------------------------------------------------------------
def plot_efficiency_frontier(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method in METHOD_LABELS:
        sub = df[df.method == method]
        agg = sub.groupby("nfe").agg(psnr=("psnr", "mean"), time_s=("time_s", "mean"))
        ax.plot(
            agg["time_s"], agg["psnr"],
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            label=METHOD_LABELS[method], lw=2, markersize=8,
        )
        for nfe, row in agg.iterrows():
            ax.annotate(f"N={nfe}", (row["time_s"], row["psnr"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Wall-clock time per image (s)")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("Quality–efficiency frontier (avg across all degradation conditions)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "efficiency_frontier")


# -------------------------------------------------------------------------
# 3. LPIPS-PSNR scatter — perceptual vs distortion.
# -------------------------------------------------------------------------
def plot_lpips_vs_psnr(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for method in METHOD_LABELS:
        sub = df[df.method == method]
        ax.scatter(
            sub["psnr"], sub["lpips"],
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            alpha=0.18, s=14, label=None,
        )
        agg = sub.groupby(["sigma_blur", "sigma_noise", "nfe"]).agg(
            psnr=("psnr", "mean"), lpips=("lpips", "mean"),
        )
        ax.scatter(
            agg["psnr"], agg["lpips"],
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            edgecolor="black", s=70, lw=0.7,
            label=f"{METHOD_LABELS[method]} (cell mean)",
        )
    ax.set_xlabel("PSNR (dB) ↑")
    ax.set_ylabel("LPIPS ↓")
    ax.set_title("Perceptual vs distortion tradeoff\nlight: per-image, dark+outlined: cell mean")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, out_dir, "lpips_vs_psnr")


# -------------------------------------------------------------------------
# 4. Quality vs σ_blur (each method, NFE=100 only).
# -------------------------------------------------------------------------
def plot_quality_vs_sigma_b(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["psnr", "ssim", "lpips"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric in zip(axes, metrics):
        for sn in sorted(df.sigma_noise.unique()):
            for method in METHOD_LABELS:
                sub = df[(df.method == method) & (df.sigma_noise == sn) & (df.nfe == 100)]
                m = sub.groupby("sigma_blur")[metric].mean()
                ls = "-" if sn == 0.0 else "--"
                ax.plot(
                    m.index, m.values,
                    color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                    ls=ls,
                    label=f"{METHOD_LABELS[method]}, σ_n={sn}",
                    lw=2,
                )
        ax.set_xlabel("σ_blur (Gaussian kernel std)")
        ax.set_ylabel(metric.upper() + (" (dB)" if metric == "psnr" else ""))
        ax.grid(True, ls=":", alpha=0.5)
        if metric == "psnr":
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Quality vs blur severity at NFE=100")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_dir, "quality_vs_sigma_blur")


# -------------------------------------------------------------------------
# 5. PSNR delta heatmap: Pixel-DPS minus FlowDPS-on-RF, per (σ_b, σ_n, NFE).
# -------------------------------------------------------------------------
def plot_psnr_delta_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    pivot_p = df[df.method == "pixel_dps"].groupby(
        ["sigma_blur", "sigma_noise", "nfe"])["psnr"].mean().reset_index()
    pivot_f = df[df.method == "flowdps_rf"].groupby(
        ["sigma_blur", "sigma_noise", "nfe"])["psnr"].mean().reset_index()
    merged = pivot_p.merge(
        pivot_f, on=["sigma_blur", "sigma_noise", "nfe"],
        suffixes=("_pixel", "_flow"),
    )
    merged["delta"] = merged["psnr_pixel"] - merged["psnr_flow"]
    pivot = merged.pivot_table(
        index="sigma_blur", columns=["sigma_noise", "nfe"], values="delta",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="RdYlBu_r", aspect="auto",
                   vmin=0, vmax=max(4.0, pivot.values.max()))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"σ_n={c[0]}\nNFE={c[1]}" for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"σ_b={v}" for v in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="black" if v < pivot.values.max() * 0.65 else "white", fontsize=9)
    fig.colorbar(im, ax=ax, label="ΔPSNR (Pixel-DPS − FlowDPS-on-RF) [dB]")
    ax.set_title("Pixel-DPS advantage over FlowDPS-on-RF in PSNR")
    fig.tight_layout()
    save_fig(fig, out_dir, "psnr_delta_heatmap")


# -------------------------------------------------------------------------
# 6. Per-condition mean table → markdown + LaTeX.
# -------------------------------------------------------------------------
def dump_summary_table(df: pd.DataFrame, out_dir: Path) -> None:
    grp = df.groupby(["method", "sigma_blur", "sigma_noise", "nfe"]).agg(
        n=("psnr", "count"),
        psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
        ssim_mean=("ssim", "mean"),
        lpips_mean=("lpips", "mean"),
        time_mean=("time_s", "mean"),
    ).reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "main_grid_summary.md"
    with md.open("w") as f:
        f.write("# Main-grid summary (mean ± std over 50 images per cell)\n\n")
        f.write("| method | σ_b | σ_n | NFE | n | PSNR (dB) | SSIM | LPIPS | s/img |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in grp.iterrows():
            f.write(
                f"| {r['method']} | {r['sigma_blur']} | {r['sigma_noise']} | {r['nfe']} | "
                f"{int(r['n'])} | {r['psnr_mean']:.2f} ± {r['psnr_std']:.2f} | "
                f"{r['ssim_mean']:.3f} | {r['lpips_mean']:.3f} | {r['time_mean']:.2f} |\n"
            )
    print(f"  saved {md}")
    # Also save a long-form CSV for LaTeX / further analysis.
    csv = out_dir / "main_grid_summary.csv"
    grp.to_csv(csv, index=False, float_format="%.4f")
    print(f"  saved {csv}")


def main(args):
    df = load_df(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}")
    print(f"Methods: {df.method.unique().tolist()}")
    print(f"σ_blurs: {sorted(df.sigma_blur.unique().tolist())}")
    print(f"σ_noises: {sorted(df.sigma_noise.unique().tolist())}")
    print(f"NFEs: {sorted(df.nfe.unique().tolist())}")

    out_dir = Path(args.out_dir)
    print(f"\nWriting figures to {out_dir}/")
    plot_metric_vs_nfe(df, out_dir, "psnr")
    plot_metric_vs_nfe(df, out_dir, "ssim")
    plot_metric_vs_nfe(df, out_dir, "lpips")
    plot_efficiency_frontier(df, out_dir)
    plot_lpips_vs_psnr(df, out_dir)
    plot_quality_vs_sigma_b(df, out_dir)
    plot_psnr_delta_heatmap(df, out_dir)
    dump_summary_table(df, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", default="outputs/results/main_grid.csv")
    p.add_argument("--out_dir", default="outputs/figures")
    args = p.parse_args()
    main(args)
