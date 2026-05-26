"""Pure Π-GDM (Fix #1) overnight chain: full flowdps_rf_pigdm_pure grid
at all NFE budgets, then significance + tier-ablation, then summary md.

Same pattern as run_particle_chain.py. Resume-safe via ResultsLogger.
"""
from __future__ import annotations
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[pigdm_chain {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run_logged(name: str, *cmd: str) -> int:
    log(f"launching '{name}': {' '.join(cmd)}")
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_with_log.sh"), name, *cmd],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode == 0:
        log(f"'{name}' OK")
    else:
        log(f"'{name}' exited {proc.returncode}; stderr tail:\n{proc.stderr[-800:]}")
    return proc.returncode


def main() -> int:
    log("=== Pure Π-GDM chain started ===")

    # Stage 1: full flowdps_rf_pigdm_pure grid at NFE in {25, 50, 100}.
    # 50 imgs x 6 cells x 3 NFEs = 900 rows. ~13 s/img at NFE=100,
    # ~6 s at NFE=50, ~3 s at NFE=25 → ~3.5 h total.
    log("Stage 1: flowdps_rf_pigdm_pure full grid (NFE=25,50,100)")
    run_logged(
        "pigdm_pure_grid",
        ".venv/bin/python", "-u", "scripts/run_grid.py",
        "--methods", "flowdps_rf_pigdm_pure", "--num_images", "50",
        "--nfes", "25,50,100",
        "--csv_path", "outputs/results/main_grid.csv",
    )

    # Stage 2: significance tests.
    log("Stage 2: significance_tests.py")
    sig = subprocess.run([".venv/bin/python", "scripts/significance_tests.py"],
                         cwd=ROOT, capture_output=True, text=True)

    # Stage 3: tier-ablation figures.
    log("Stage 3: tier-ablation regen")
    run_logged("tier_ablation_pigdm_pure",
               ".venv/bin/python", "-u", "scripts/make_tier_ablation.py", "--nfe", "100")

    # Stage 4: summary md.
    log("Stage 4: writing ROUND4_PIGDM_PURE_SUMMARY.md")
    new_sig_lines = [ln for ln in sig.stdout.splitlines()
                     if "flowdps_rf_pigdm_pure" in ln or "Summary by" in ln]
    summary_path = ROOT / "outputs" / "ROUND4_PIGDM_PURE_SUMMARY.md"
    body = [
        f"# Pure Π-GDM (Fix #1) chain summary — {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## What ran",
        "- flowdps_rf_pigdm_pure full grid (50 imgs × 6 cells × 3 NFEs = 900 rows).",
        "- significance vs flowdps_rf (v1) and flowdps_rf_v2.",
        "- Tier-ablation figures regenerated.",
        "",
        "## Significance — new comparisons",
        "```",
        *new_sig_lines,
        "```",
        "",
        "## Next steps for the user",
        "- Read this file, eyeball whether `flowdps_rf_pigdm_pure - flowdps_rf` is positive or negative.",
        "- If positive in ≥ 4/6 cells AND Bonferroni p<0.05: WIN — promote to report.",
        "- If negative: 10th documented failed ablation with literature-matching magnitude.",
        "- Either way: add EXP-021 entry to WORKING_NOTES, then commit.",
    ]
    summary_path.write_text("\n".join(body) + "\n")
    log(f"Wrote {summary_path}")
    log("=== Pure Π-GDM chain complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
