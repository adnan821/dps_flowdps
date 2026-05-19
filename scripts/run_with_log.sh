#!/usr/bin/env bash
# Wrapper that runs a command with stdout+stderr teed to a timestamped log
# in outputs/logs/. Usage:
#   ./scripts/run_with_log.sh <log_basename> <command...>
# Example:
#   ./scripts/run_with_log.sh exp_002_flowdps_smoke .venv/bin/python scripts/smoke_flowdps.py
#
# The final log path is printed at the end so it can be referenced from
# docs/WORKING_NOTES.md.

set -u

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <log_basename> <command...>" >&2
    exit 64
fi

BASENAME="$1"; shift
TS=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${BASENAME}_${TS}.log"

# Capture environment + start metadata.
{
    echo "==== run_with_log ===="
    echo "ts_start    : $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "cwd         : $(pwd)"
    echo "git rev     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
    echo "git status  : $(git status --porcelain 2>/dev/null | wc -l) modified files"
    echo "cmd         : $*"
    echo "log path    : $LOG"
    echo "----"
} | tee "$LOG"

START_NS=$(date +%s)

# Tee both stdout AND stderr to the log, plus terminal.
# `set -o pipefail` ensures we capture the actual command's exit code.
set -o pipefail
"$@" 2>&1 | tee -a "$LOG"
STATUS=$?

END_NS=$(date +%s)
ELAPSED=$(( END_NS - START_NS ))

{
    echo "----"
    echo "ts_end      : $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "elapsed     : ${ELAPSED}s"
    echo "exit_code   : $STATUS"
} | tee -a "$LOG"

echo ""
echo ">>> log saved to: $LOG"
exit $STATUS
