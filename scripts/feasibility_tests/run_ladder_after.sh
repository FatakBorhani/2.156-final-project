#!/bin/bash
# Test 13 (simplicity ladder, incl. its extension rungs), then Test 14 (compactness ladder),
# one after the other. If a pid is given, waits for that process to exit first so runs don't
# compete for cores. Both are resumable: rerunning continues where they stopped (Test 13's
# finished rungs are skipped).
#   nohup caffeinate -i bash scripts/feasibility_tests/run_ladder_after.sh [pid] > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")/../.." || exit 1
PY=/Users/leifakerley/miniforge3/envs/stellarator/bin/python
LOG=data/feasibility_tests
if [ -n "$1" ]; then
  echo "=== $(date): waiting for process $1 to finish ===" >> $LOG/ladder_run.log
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
echo "=== $(date): Test 13 start ===" >> $LOG/ladder_run.log
HF_HUB_OFFLINE=1 $PY scripts/feasibility_tests/ladder_test.py >> $LOG/ladder_run.log 2>&1
echo "=== $(date): Test 14 start ===" >> $LOG/compact_run.log
HF_HUB_OFFLINE=1 $PY scripts/feasibility_tests/compact_test.py >> $LOG/compact_run.log 2>&1
