#!/bin/bash
# Test 17 (retrained diffusion seeds below AR 10.2, then descent), detached and kept awake.
# Resumable: training is skipped once the candidates file exists, screened evaluations are
# reused, finished targets and rungs are skipped.
#   nohup caffeinate -i bash scripts/feasibility_tests/run_diffusion2.sh > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")/../.." || exit 1
PY=/Users/leifakerley/miniforge3/envs/stellarator/bin/python
LOG=data/feasibility_tests/diffusion2_run.log
echo "=== $(date): Test 17 start ===" >> $LOG
HF_HUB_OFFLINE=1 $PY scripts/feasibility_tests/diffusion2_test.py 2>&1 \
  | grep --line-buffered -vE 'RuntimeWarning|fsolve|improvement from the last' >> $LOG
echo "=== $(date): Test 17 end ===" >> $LOG
