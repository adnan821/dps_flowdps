#!/usr/bin/env bash
# Round-3 orchestrator: runs uncertainty calibration → particle smoke →
# particle full grid → significance + tier-ablation regen.
#
# Designed to be nohup'd after the NFE-extend grid (PID passed via $1)
# finishes. Each stage is teed via run_with_log.sh; each is resume-safe
# (uncertainty re-samples deterministically; particle grid uses
# ResultsLogger.done_keys).
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=outputs/logs/round3_chain_stdout.log
exec > >(tee -a "$LOG") 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[round3 $(ts)] $*"; }

NFE_PID="${1:-}"
if [[ -n "$NFE_PID" ]] && kill -0 "$NFE_PID" 2>/dev/null; then
  log "Waiting for NFE-extend grid (PID $NFE_PID) to finish..."
  while kill -0 "$NFE_PID" 2>/dev/null; do sleep 30; done
  log "NFE-extend grid PID $NFE_PID exited."
else
  log "No alive NFE-extend PID given (got '$NFE_PID'); proceeding directly."
fi

# --- Stage A: uncertainty calibration (~17 min) ---
log "Stage A: uncertainty_calibration.py"
bash scripts/run_with_log.sh uncertainty_calib \
  .venv/bin/python -u scripts/uncertainty_calibration.py \
  || log "  Stage A non-zero exit (continuing)"

# --- Stage B: particle-DPS smoke (1 img x 1 cell x NFE=100, ~1 min) ---
log "Stage B: particle_dps smoke (1 image)"
bash scripts/run_with_log.sh particle_smoke \
  .venv/bin/python -u scripts/run_grid.py \
    --methods particle_dps --num_images 1 --nfes 100 \
    --sigma_blurs 3.0 --sigma_noises 0.05 \
    --csv_path outputs/results/main_grid.csv \
  || { log "  Stage B FAILED — aborting particle grid"; STAGE_B_OK=no; }
STAGE_B_OK=${STAGE_B_OK:-yes}

# --- Stage C: particle-DPS full grid at NFE=100 (~2.5h) ---
if [[ "$STAGE_B_OK" == "yes" ]]; then
  log "Stage C: particle_dps full grid (NFE=100 only)"
  bash scripts/run_with_log.sh particle_grid \
    .venv/bin/python -u scripts/run_grid.py \
      --methods particle_dps --num_images 50 --nfes 100 \
      --csv_path outputs/results/main_grid.csv \
    || log "  Stage C non-zero exit (continuing; ResultsLogger is append-safe)"
else
  log "Stage C SKIPPED — smoke failed"
fi

# --- Stage D: significance + tier-ablation refresh ---
log "Stage D: significance + tier-ablation"
.venv/bin/python scripts/significance_tests.py 2>&1 | tail -40
.venv/bin/python scripts/make_tier_ablation.py --nfe 100 2>&1 | tail -5

# --- Stage E: summary ---
SUM=outputs/ROUND3_SUMMARY.md
{
  echo "# Round-3 chain summary — $(ts)"
  echo ""
  echo "## Items"
  echo "- 3 (NFE 10/200 grid): see outputs/logs/nfe_extend_*.log"
  echo "- 4 (uncertainty calibration): see outputs/logs/uncertainty_calib_*.log"
  echo "- 6 (gradient-free particle-DPS): see outputs/logs/particle_grid_*.log"
  echo ""
  echo "## Significance — new comparisons"
  .venv/bin/python scripts/significance_tests.py 2>&1 | grep -E "particle_dps|Summary by"
  echo ""
  echo "## Uncertainty calibration"
  if [[ -f outputs/results/uncertainty_calibration.csv ]]; then
    head -1 outputs/results/uncertainty_calibration.csv
    tail -n +2 outputs/results/uncertainty_calibration.csv
  fi
} > "$SUM"
log "Wrote $SUM"
log "=== Round-3 chain complete ==="
