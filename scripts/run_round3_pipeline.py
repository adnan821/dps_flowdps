"""Round-3 orchestrator (items 3, 4, 6 of the audited Bucket 3 plan).

Stages (each teed via scripts/run_with_log.sh, resume-safe):
  1. Wait for the running NFE-extend grid (PID passed via $1) to exit.
  2. Stage A: uncertainty_calibration.py (~17 min on GPU).
  3. Stage B: particle_dps smoke (1 img × 1 cell × NFE=100).
  4. Stage C: full particle_dps grid (NFE=100 only, ~2.5 h).
  5. Stage D: re-run significance_tests.py + make_tier_ablation.py.
  6. Write outputs/ROUND3_SUMMARY.md.

Launched once via nohup; survives logout.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[round3 {ts}] {msg}", flush=True)


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
    log("=== Round-3 pipeline started ===")
    nfe_pid_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        nfe_pid = int(nfe_pid_arg)
    except ValueError:
        nfe_pid = 0

    if nfe_pid and pid_alive(nfe_pid):
        log(f"Stage 1: waiting for NFE-extend grid PID {nfe_pid} to finish")
        while pid_alive(nfe_pid):
            time.sleep(30)
        log(f"NFE-extend grid PID {nfe_pid} done.")
    else:
        log(f"Stage 1 skipped — no alive NFE-extend PID (got '{nfe_pid_arg}').")

    # --- Stage A: uncertainty calibration ---
    log("Stage A: uncertainty_calibration.py")
    run_logged("uncertainty_calib",
               ".venv/bin/python", "-u", "scripts/uncertainty_calibration.py")

    # --- Stage B: particle_dps smoke ---
    log("Stage B: particle_dps smoke (1 img × 1 cell × NFE=100)")
    smoke_rc = run_logged(
        "particle_smoke",
        ".venv/bin/python", "-u", "scripts/run_grid.py",
        "--methods", "particle_dps", "--num_images", "1", "--nfes", "100",
        "--sigma_blurs", "3.0", "--sigma_noises", "0.05",
        "--csv_path", "outputs/results/main_grid.csv",
    )

    # --- Stage C: particle_dps full grid ---
    if smoke_rc == 0:
        log("Stage C: particle_dps full grid (NFE=100 only)")
        run_logged(
            "particle_grid",
            ".venv/bin/python", "-u", "scripts/run_grid.py",
            "--methods", "particle_dps", "--num_images", "50", "--nfes", "100",
            "--csv_path", "outputs/results/main_grid.csv",
        )
    else:
        log("Stage C SKIPPED — smoke failed")

    # --- Stage D: significance + tier-ablation regen ---
    log("Stage D: significance + tier-ablation")
    subprocess.run([".venv/bin/python", "scripts/significance_tests.py"],
                   cwd=ROOT, capture_output=True, text=True)
    run_logged("tier_ablation_round3",
               ".venv/bin/python", "-u", "scripts/make_tier_ablation.py", "--nfe", "100")

    # --- Stage E: summary ---
    summary_path = ROOT / "outputs" / "ROUND3_SUMMARY.md"
    sig_out = subprocess.run(
        [".venv/bin/python", "scripts/significance_tests.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    new_sig_lines = [ln for ln in sig_out.stdout.splitlines()
                     if "particle_dps" in ln or "Summary by" in ln]
    calib_csv = ROOT / "outputs/results/uncertainty_calibration.csv"
    calib_text = calib_csv.read_text() if calib_csv.exists() else "(not produced)"

    body = [
        f"# Round-3 pipeline summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## What ran",
        "- Item 3: NFE 10/200 extension grid (`outputs/logs/nfe_extend_*.log`).",
        "- Item 4: Uncertainty calibration (`outputs/logs/uncertainty_calib_*.log`).",
        "- Item 6: Gradient-free particle-DPS grid (`outputs/logs/particle_grid_*.log`).",
        "",
        "## New significance comparisons",
        "```",
        *new_sig_lines,
        "```",
        "",
        "## Uncertainty calibration (Spearman ρ: edge-mag vs sample-std)",
        "```",
        calib_text.rstrip(),
        "```",
        "",
        "## Next step",
        "Update `docs/group13_v2.tex` §5.4 (uncertainty calibration paragraph + figure)",
        "and add §5.4.1 *Gradient-free posterior sampling* with the particle-DPS table.",
    ]
    summary_path.write_text("\n".join(body) + "\n")
    log(f"Wrote {summary_path}")
    log("=== Round-3 pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
