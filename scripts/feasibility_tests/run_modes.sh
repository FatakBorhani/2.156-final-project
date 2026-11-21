#!/bin/bash
# Test 18 (descent with Fourier modes up to 5 x 5), detached and kept awake. Resumable:
# rerunning replays the saved rungs and continues an interrupted one.
#   nohup caffeinate -i bash scripts/feasibility_tests/run_modes.sh > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")/../.." || exit 1
PY=/Users/leifakerley/miniforge3/envs/stellarator/bin/python
LOG=data/feasibility_tests/modes5_run.log
echo "=== $(date): Test 18 start ===" >> $LOG
HF_HUB_OFFLINE=1 $PY scripts/feasibility_tests/modes_test.py 2>&1 \
  | grep --line-buffered -vE 'RuntimeWarning|fsolve|improvement from the last' >> $LOG
echo "=== $(date): Test 18 end ===" >> $LOG
