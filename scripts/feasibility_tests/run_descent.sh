#!/bin/bash
# Test 16 (adaptive compactness descent), detached and kept awake. Resumable: rerunning
# replays finished rungs from the ledger and continues an interrupted one.
#   nohup caffeinate -i bash scripts/feasibility_tests/run_descent.sh > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")/../.." || exit 1
PY=/Users/leifakerley/miniforge3/envs/stellarator/bin/python
LOG=data/feasibility_tests/descent_run.log
echo "=== $(date): Test 16 start ===" >> $LOG
HF_HUB_OFFLINE=1 $PY scripts/feasibility_tests/descent_test.py 2>&1 \
  | grep --line-buffered -vE 'RuntimeWarning|fsolve|improvement from the last' >> $LOG
echo "=== $(date): Test 16 end ===" >> $LOG
