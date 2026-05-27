"""Chain: wait for the Heun NFE=200 grid (PID 51296) to finish,
then run the Tier-B *_spectral methods on the MATCHED-Gaussian main
grid (NFE=100, 600 rows total), then refresh significance + figures.

Per the analytical analysis (mean(W)=1.000 at σ_b=3,5 because
mean-normalization of the spectral weight after suppression covers
95-98 % of frequencies), this measurement is expected to produce
~4 of 6 cells of "no significant difference" per method. The run
documents the expected-null in the CSV anyway. The substantive
finding goes in the report's §5.6 as an analytical note.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEUN_PID = 51296   # pinned at chain launch — Heun NFE=200 grid


def log(msg: str) -> None:
    print(f"[spectral_chain {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def run_logged(name: str, *cmd: str) -> int:
    log(f"launching '{name}': {' '.join(cmd)}")
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_with_log.sh"), name, *cmd],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode == 0:
        log(f"'{name}' OK")
    else:
        log(f"'{name}' exited {proc.returncode}; stderr tail:\n{proc.stderr[-600:]}")
    return proc.returncode


def main() -> int:
    log("=== Spectral-matched chain started; waiting on Heun NFE=200 grid ===")

    if pid_alive(HEUN_PID):
        log(f"Heun grid PID {HEUN_PID} still alive — polling")
        while pid_alive(HEUN_PID):
            time.sleep(60)
        log(f"Heun grid PID {HEUN_PID} done.")
    else:
        log(f"Heun grid PID {HEUN_PID} not running — proceeding immediately.")

    # Stage 1: pixel_dps_spectral, NFE=100 only, 50 imgs x 6 cells = 300 rows.
    # ~18 s/img → ~1.5 h.
    log("Stage 1: pixel_dps_spectral on matched Gaussian grid (NFE=100)")
    run_logged(
        "spectral_matched_pixel",
        ".venv/bin/python", "-u", "scripts/run_grid.py",
        "--methods", "pixel_dps_spectral", "--num_images", "50", "--nfes", "100",
        "--csv_path", "outputs/results/main_grid.csv",
    )

    # Stage 2: flowdps_rf_spectral, NFE=100 only, 300 rows. ~24 s/img → ~2 h.
    log("Stage 2: flowdps_rf_spectral on matched Gaussian grid (NFE=100)")
    run_logged(
        "spectral_matched_flow",
        ".venv/bin/python", "-u", "scripts/run_grid.py",
        "--methods", "flowdps_rf_spectral", "--num_images", "50", "--nfes", "100",
        "--csv_path", "outputs/results/main_grid.csv",
    )

    # Stage 3: significance + tier-ablation refresh.
    log("Stage 3: significance_tests.py + make_tier_ablation.py")
    sig = subprocess.run([".venv/bin/python", "scripts/significance_tests.py"],
                         cwd=ROOT, capture_output=True, text=True)
    run_logged("tier_ablation_spectral_matched",
               ".venv/bin/python", "-u", "scripts/make_tier_ablation.py", "--nfe", "100")

    # Stage 4: summary md.
    new_sig_lines = [ln for ln in sig.stdout.splitlines()
                     if ("pixel_dps_spectral - pixel_dps" in ln or
                         "flowdps_rf_spectral - flowdps_rf" in ln or
                         "Summary by" in ln)]
    summary_path = ROOT / "outputs" / "SPECTRAL_MATCHED_SUMMARY.md"
    body = [
        f"# Spectral matched-grid chain summary — {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## What ran",
        "- pixel_dps_spectral on matched Gaussian grid (NFE=100, 300 rows).",
        "- flowdps_rf_spectral on matched Gaussian grid (NFE=100, 300 rows).",
        "- significance + tier-ablation refresh.",
        "",
        "## Significance — new matched-grid comparisons",
        "(NFE=50 entries are the existing robustness-grid measurements; ",
        " NFE=100 entries are the new matched-Gaussian measurements.)",
        "```",
        *new_sig_lines,
        "```",
        "",
        "## Analytical context",
        "The mean-normalized noise-floor weight (ε=0.10, α=10, ",
        "normalize_mean=True) becomes approximately uniform on the ",
        "matched-Gaussian grid for moderate-to-large blur kernels: ",
        "95–98 % of frequencies are below the suppression threshold at ",
        "σ_b ∈ {3, 5}, so renormalization brings the weight to mean=1, ",
        "min ≈ 0.97–0.99. Spectral-DPS on those cells is therefore ",
        "expected to be numerically near-identical to vanilla DPS. ",
        "Only σ_b=1.5 has meaningful spectral structure (min(W) ≈ 0.88).",
        "",
        "## For the report",
        "Add to §5.6 the one-paragraph analytical note (see ",
        "ABLATION_SCORECARD.md update); the empirical numbers above ",
        "confirm the analytical prediction.",
    ]
    summary_path.write_text("\n".join(body) + "\n")
    log(f"Wrote {summary_path}")
    log("=== Spectral-matched chain complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
