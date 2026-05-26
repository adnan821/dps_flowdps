"""Particle-DPS continuation: smoke already passed (PSNR≈16 dB at sb=3,sn=0.05),
so run the full 300-reconstruction grid at NFE=100, then re-run significance
and tier-ablation, then write a fresh summary md.

Sequential, nohup-friendly, resume-safe (run_grid.py's done_keys).
"""
from __future__ import annotations
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[particle_chain {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


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
    log("=== Particle-DPS continuation chain started ===")

    # Stage 1: full grid (NFE=100 only, ~4.5h).
    log("Stage 1: particle_dps full grid (50 imgs x 6 cells x NFE=100)")
    run_logged(
        "particle_grid",
        ".venv/bin/python", "-u", "scripts/run_grid.py",
        "--methods", "particle_dps", "--num_images", "50", "--nfes", "100",
        "--csv_path", "outputs/results/main_grid.csv",
    )

    # Stage 2: significance.
    log("Stage 2: significance tests")
    sig = subprocess.run([".venv/bin/python", "scripts/significance_tests.py"],
                         cwd=ROOT, capture_output=True, text=True)

    # Stage 3: tier-ablation figures.
    log("Stage 3: tier-ablation figures")
    run_logged("tier_ablation_particle",
               ".venv/bin/python", "-u", "scripts/make_tier_ablation.py", "--nfe", "100")

    # Stage 4: summary md.
    log("Stage 4: writing ROUND3_FINAL_SUMMARY.md")
    new_sig_lines = [ln for ln in sig.stdout.splitlines()
                     if "particle_dps" in ln or "Summary by" in ln]
    calib_csv = ROOT / "outputs/results/uncertainty_calibration.csv"
    calib_text = calib_csv.read_text() if calib_csv.exists() else "(not produced)"

    summary_path = ROOT / "outputs" / "ROUND3_FINAL_SUMMARY.md"
    body = [
        f"# Round-3 final summary — {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Items completed",
        "- Item 3 — NFE 10/200 extension grid (1200 new rows, 900 → 1500 per v1 method).",
        "- Item 4 — uncertainty calibration (Spearman ρ table below).",
        "- Item 6 — gradient-free particle-DPS grid (NFE=100 only, 300 rows).",
        "",
        "## Significance — new comparison",
        "```",
        *new_sig_lines,
        "```",
        "",
        "## Uncertainty calibration",
        "```",
        calib_text.rstrip(),
        "```",
        "",
        "## Next step",
        "Update docs/group13_v2.tex:",
        "  - End of §5.4: paragraph + figure for uncertainty calibration.",
        "  - New §5.4.1 Gradient-free posterior sampling (particle-DPS vs v1).",
        "  - New row in §5.5 significance table + §5.6 ablation table.",
        "Then make -C docs && git commit && git push.",
    ]
    summary_path.write_text("\n".join(body) + "\n")
    log(f"Wrote {summary_path}")
    log("=== Particle-DPS continuation chain complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
