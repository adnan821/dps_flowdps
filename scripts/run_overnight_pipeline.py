"""Overnight orchestrator: runs the Round-2 experiments end-to-end.

Stages (each grid teed through run_with_log.sh, resume-safe):
  1. Wait for the running pixel_dps ζ-schedule validation sweep.
  2. Parse the best (non-reference) schedule from the sweep log.
  3. Run the full pixel_dps_sched 18-cell grid (900 rows).
  4. Smoke-test the FlowDPS-RF Heun integrator (2 imgs); skip stage 5
     if it is broken.
  5. Run the full flowdps_rf_heun 18-cell grid (900 rows).
  6. Re-run significance tests + tier-ablation artifacts.
  7. Write outputs/OVERNIGHT_SUMMARY.md.

No decision points need a human — stage 2 picks argmax(overall PSNR).
Designed to be launched once via nohup and left overnight.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"
RESULTS = ROOT / "outputs" / "results"
MAIN_CSV = RESULTS / "main_grid.csv"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[orchestrator {ts}] {msg}", flush=True)


def newest(glob_pat: str) -> Path | None:
    hits = sorted(LOGS.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def wait_for_log_done(glob_pat: str, timeout_s: int = 3600) -> Path | None:
    """Poll until a run_with_log.sh log matching `glob_pat` has its `ts_end`
    footer. Returns the log path (or None on timeout)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        lg = newest(glob_pat)
        if lg and "ts_end" in lg.read_text(errors="ignore"):
            return lg
        time.sleep(20)
    return None


def run_logged(name: str, *cmd: str, retries: int = 1) -> int:
    """Run a command teed through run_with_log.sh. Retry once on failure
    (run_grid.py is resume-safe, so a retry just continues)."""
    for attempt in range(retries + 1):
        tag = name if attempt == 0 else f"{name}_retry{attempt}"
        log(f"launching '{tag}': {' '.join(cmd)}")
        proc = subprocess.run(
            ["bash", str(ROOT / "scripts" / "run_with_log.sh"), tag, *cmd],
            cwd=ROOT, capture_output=True, text=True,
        )
        if proc.returncode == 0:
            log(f"'{tag}' finished OK")
            return 0
        log(f"'{tag}' exited {proc.returncode}; stderr tail:\n{proc.stderr[-800:]}")
    return 1


def parse_best_schedule(sweep_log: Path) -> tuple[str, float, float]:
    """Pick the non-reference schedule with the highest 'overall' PSNR.
    Falls back to power(40,1.0) if parsing fails or everything collapsed."""
    best = None  # (overall, kind, zeta0, alpha)
    for line in sweep_log.read_text(errors="ignore").splitlines():
        m = re.search(r"\b(power|ramp|warmup)\(([0-9.]+),\s*([0-9.]+)\)", line)
        if not m:
            continue
        floats = re.findall(r"(-?\d+\.\d+)", line)
        # the schedule descriptor itself contributes 2 floats (zeta0, alpha);
        # the trailing one is 'overall'.
        if len(floats) < 3:
            continue
        overall = float(floats[-1])
        kind, z0, alpha = m.group(1), float(m.group(2)), float(m.group(3))
        if best is None or overall > best[0]:
            best = (overall, kind, z0, alpha)
    if best is None or best[0] < 5.0:
        log("WARNING: could not parse a sane schedule; falling back to power(40,1.0)")
        return ("power", 40.0, 1.0)
    log(f"best schedule: {best[1]}({best[2]},{best[3]}) at overall PSNR {best[0]:.2f}")
    return (best[1], best[2], best[3])


def heun_smoke() -> bool:
    """2-image smoke of the Heun integrator. Returns True if outputs look
    sane (finite PSNR > 12 dB)."""
    log("Heun smoke test (2 imgs, NFE=50)")
    rc = run_logged(
        "heun_smoke",
        ".venv/bin/python", "-u", "scripts/run_grid.py",
        "--methods", "flowdps_rf_heun", "--num_images", "2",
        "--nfes", "50", "--csv_path", str(MAIN_CSV),
        retries=0,
    )
    if rc != 0:
        log("Heun smoke crashed")
        return False
    psnrs = []
    for line in MAIN_CSV.read_text(errors="ignore").splitlines():
        f = line.split(",")
        if f and f[0] == "flowdps_rf_heun" and f[6] == "50":
            try:
                psnrs.append(float(f[10]))
            except (ValueError, IndexError):
                pass
    ok = bool(psnrs) and all(p == p and p > 12.0 for p in psnrs)  # p==p rejects NaN
    log(f"Heun smoke PSNRs={[round(p, 2) for p in psnrs]} -> {'OK' if ok else 'BROKEN'}")
    return ok


def summarize() -> None:
    """Run significance tests, regenerate artifacts, write the summary md."""
    log("running significance tests")
    sig = subprocess.run(
        [".venv/bin/python", "scripts/significance_tests.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    log("regenerating tier-ablation artifacts")
    run_logged("tier_ablation_round2", ".venv/bin/python", "-u",
               "scripts/make_tier_ablation.py", "--nfe", "100", retries=0)

    sig_summary = [ln for ln in sig.stdout.splitlines()
                   if ("pixel_dps_sched" in ln or "flowdps_rf_heun" in ln
                       or "Summary by" in ln)]
    out = ROOT / "outputs" / "OVERNIGHT_SUMMARY.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = [
        f"# Overnight pipeline summary — {ts}",
        "",
        "Round-2 experiments: Pixel-DPS ζ-schedule + FlowDPS-RF Heun integrator.",
        "",
        "## Significance-test lines (new methods)",
        "```",
        *sig_summary,
        "```",
        "",
        "## Full significance stdout",
        "See `outputs/results/significance.csv` and the significance_* log.",
        "",
        "## Decision gate",
        "A method is a WIN iff mean ΔPSNR > 0 AND Bonferroni p<0.05 in ≥4/6 cells.",
        "Cross-check the lines above before promoting anything to the report.",
    ]
    out.write_text("\n".join(body) + "\n")
    log(f"wrote {out}")


def main() -> int:
    log("=== overnight pipeline started ===")

    # Stage 1: wait for the ζ-schedule sweep already running.
    log("Stage 1: waiting for pixel_dps schedule sweep")
    sweep_log = wait_for_log_done("tune_pixel_sched_2*.log", timeout_s=3600)
    if sweep_log is None:
        log("FATAL: schedule sweep did not finish within 1 h; aborting")
        return 1
    log(f"sweep done: {sweep_log}")

    # Stage 2: pick the best schedule.
    kind, z0, alpha = parse_best_schedule(sweep_log)

    # Stage 3: full pixel_dps_sched grid.
    log("Stage 3: pixel_dps_sched full grid (900 rows)")
    run_logged(
        "tierR2_pixel_sched",
        ".venv/bin/python", "-u", "scripts/run_grid.py",
        "--methods", "pixel_dps_sched", "--num_images", "50",
        "--sched_kind", kind, "--sched_zeta0", str(z0), "--sched_alpha", str(alpha),
        "--csv_path", str(MAIN_CSV),
    )

    # Stage 4 + 5: Heun smoke, then full grid.
    if heun_smoke():
        log("Stage 5: flowdps_rf_heun full grid (900 rows)")
        run_logged(
            "tierR2_flow_heun",
            ".venv/bin/python", "-u", "scripts/run_grid.py",
            "--methods", "flowdps_rf_heun", "--num_images", "50",
            "--csv_path", str(MAIN_CSV),
        )
    else:
        log("Stage 5 SKIPPED: Heun smoke failed — flowdps_rf_heun grid not run")

    # Stage 6 + 7: analysis + summary.
    log("Stage 6/7: significance + artifacts + summary")
    summarize()
    log("=== overnight pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
