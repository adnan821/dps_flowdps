"""Generate tier-ablation artifacts (figure + markdown table + LaTeX table)
from outputs/results/{main_grid,baselines}.csv.

Produces, for both Pixel-DPS and FlowDPS-on-RF families:
- A 1-figure bar chart of mean PSNR per cell, with v1 + v2 + spectral
  + pigdm bars side-by-side.
- A markdown table.
- A LaTeX table ready to \\input{} into docs/group13_v2.tex.

Resilient to partial data: methods/cells that haven't been measured
yet are simply omitted. Re-run after each tier's grid completes.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Method display labels + canonical ordering for the plot legend.
LABEL = {
    "pixel_dps": "Pixel-DPS (v1)",
    "pixel_dps_v2": "Pixel-DPS v2 (EMA-ish baseline + ζ=30)",
    "pixel_dps_spectral": "Pixel-DPS + spectral-W",
    "pixel_dps_pigdm": "Pixel-DPS + Π-GDM",
    "flowdps_rf": "FlowDPS-on-RF (v1)",
    "flowdps_rf_v2": "FlowDPS v2 (EMA + ramp ζ)",
    "flowdps_rf_spectral": "FlowDPS + spectral-W",
    "flowdps_rf_pigdm": "FlowDPS + Π-GDM",
}
COLOR = {
    "pixel_dps":           "#1f77b4",
    "pixel_dps_v2":        "#73c2fb",
    "pixel_dps_spectral":  "#1f77b4",
    "pixel_dps_pigdm":     "#0050a0",
    "flowdps_rf":          "#d62728",
    "flowdps_rf_v2":       "#ff7f7f",
    "flowdps_rf_spectral": "#d62728",
    "flowdps_rf_pigdm":    "#a02020",
}
PIXEL_FAMILY = ["pixel_dps", "pixel_dps_v2", "pixel_dps_spectral", "pixel_dps_pigdm"]
FLOW_FAMILY = ["flowdps_rf", "flowdps_rf_v2", "flowdps_rf_spectral", "flowdps_rf_pigdm"]


def load_diffusion_rows(csv_paths: Iterable[str], nfe_filter: int | None = 100) -> dict:
    """Return: {(method, sigma_b, sigma_n): [psnr per image, ...]}."""
    g: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    for csv_path in csv_paths:
        p = Path(csv_path)
        if not p.exists():
            continue
        with p.open() as f:
            for r in csv.DictReader(f):
                method = r["method"]
                if method not in LABEL:
                    continue
                nfe_raw = r["nfe"]
                nfe = int(nfe_raw) if nfe_raw and nfe_raw != "0" else 0
                if nfe_filter is not None and nfe != nfe_filter:
                    continue
                try:
                    sb = float(r["sigma_blur"]); sn = float(r["sigma_noise"])
                    psnr = float(r["psnr"])
                except (ValueError, KeyError):
                    continue
                g[(method, sb, sn)].append(psnr)
    return g


def make_table_md(g: dict, family: list[str], out_md: Path) -> str:
    """Markdown table: rows = (σ_b, σ_n), cols = methods in family."""
    sbs = sorted({k[1] for k in g.keys()})
    sns = sorted({k[2] for k in g.keys()})
    present_methods = [m for m in family if any((m, sb, sn) in g for sb in sbs for sn in sns)]
    if not present_methods:
        return ""
    lines = []
    header = "| σ_b | σ_n | " + " | ".join(LABEL[m] for m in present_methods) + " |"
    sep = "|" + "---|" * (2 + len(present_methods))
    lines.append(header)
    lines.append(sep)
    for sb in sbs:
        for sn in sns:
            cells = []
            for m in present_methods:
                vals = g.get((m, sb, sn), [])
                if not vals:
                    cells.append("—")
                else:
                    cells.append(f"{statistics.mean(vals):.2f} (n={len(vals)})")
            lines.append(f"| {sb} | {sn} | " + " | ".join(cells) + " |")
    content = "\n".join(lines) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(content)
    return content


def make_table_tex(g: dict, family: list[str], caption: str, label: str, out_tex: Path) -> str:
    """LaTeX table ready to \\input{}."""
    sbs = sorted({k[1] for k in g.keys()})
    sns = sorted({k[2] for k in g.keys()})
    present_methods = [m for m in family if any((m, sb, sn) in g for sb in sbs for sn in sns)]
    if not present_methods:
        return ""
    n_cols = 2 + len(present_methods)
    col_spec = "ll" + "r" * len(present_methods)
    short_label = {
        "pixel_dps": "v1",
        "pixel_dps_v2": "v2",
        "pixel_dps_spectral": "spectral",
        "pixel_dps_pigdm": "Π-GDM",
        "flowdps_rf": "v1",
        "flowdps_rf_v2": "v2",
        "flowdps_rf_spectral": "spectral",
        "flowdps_rf_pigdm": "Π-GDM",
    }
    head = "$\\sigma_b$ & $\\sigma_n$ & " + " & ".join(short_label[m] for m in present_methods)
    lines = [
        "\\begin{table}[h]", "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    " + head + " \\\\",
        "    \\midrule",
    ]
    for sb in sbs:
        for sn in sns:
            cells = []
            for m in present_methods:
                vals = g.get((m, sb, sn), [])
                if not vals:
                    cells.append("---")
                else:
                    cells.append(f"{statistics.mean(vals):.2f}")
            lines.append(f"    {sb} & {sn} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    content = "\n".join(lines) + "\n"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(content)
    return content


def plot_family(g: dict, family: list[str], title: str, out_path: Path) -> None:
    sbs = sorted({k[1] for k in g.keys()})
    sns = sorted({k[2] for k in g.keys()})
    present_methods = [m for m in family if any((m, sb, sn) in g for sb in sbs for sn in sns)]
    if not present_methods:
        return

    cells = [(sb, sn) for sb in sbs for sn in sns]
    cell_labels = [f"σ_b={sb}\nσ_n={sn}" for sb, sn in cells]
    x = np.arange(len(cells))
    n_methods = len(present_methods)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(2.0 + 1.4 * len(cells), 4.5))
    for i, m in enumerate(present_methods):
        means = []
        for sb, sn in cells:
            vals = g.get((m, sb, sn), [])
            means.append(statistics.mean(vals) if vals else np.nan)
        offsets = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offsets, means, width, color=COLOR[m],
                      label=LABEL[m], edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, means):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}",
                        ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_labels)
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4, axis="y")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def main(args):
    g = load_diffusion_rows(args.csv_paths, nfe_filter=args.nfe)
    print(f"Loaded data for {len(g)} (method, sigma_b, sigma_n) cells at NFE={args.nfe}")
    methods_seen = sorted({k[0] for k in g})
    print(f"Methods present: {methods_seen}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pixel-DPS family
    plot_family(g, PIXEL_FAMILY,
                f"Pixel-DPS family — tier ablation at NFE={args.nfe}",
                out_dir / "tier_ablation_pixel_dps")
    make_table_md(g, PIXEL_FAMILY, out_dir / "tier_ablation_pixel_dps.md")
    make_table_tex(g, PIXEL_FAMILY,
                   f"Pixel-DPS tier ablation (mean PSNR dB, NFE={args.nfe}, 50 imgs/cell).",
                   "tab:tier_pixel", out_dir / "tier_ablation_pixel_dps.tex")
    print(f"  pixel_dps family → {out_dir/'tier_ablation_pixel_dps.pdf'}")

    # FlowDPS-on-RF family
    plot_family(g, FLOW_FAMILY,
                f"FlowDPS-on-RF family — tier ablation at NFE={args.nfe}",
                out_dir / "tier_ablation_flowdps_rf")
    make_table_md(g, FLOW_FAMILY, out_dir / "tier_ablation_flowdps_rf.md")
    make_table_tex(g, FLOW_FAMILY,
                   f"FlowDPS-on-RF tier ablation (mean PSNR dB, NFE={args.nfe}, 50 imgs/cell).",
                   "tab:tier_flowdps", out_dir / "tier_ablation_flowdps_rf.tex")
    print(f"  flowdps_rf family → {out_dir/'tier_ablation_flowdps_rf.pdf'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_paths", nargs="+",
                   default=["outputs/results/main_grid.csv", "outputs/results/baselines.csv"])
    p.add_argument("--nfe", type=int, default=100,
                   help="Filter to one NFE budget (default 100). Use 50 for robustness comparisons.")
    p.add_argument("--out_dir", default="outputs/figures")
    args = p.parse_args()
    main(args)
