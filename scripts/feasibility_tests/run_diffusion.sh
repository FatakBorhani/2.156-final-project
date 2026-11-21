#!/bin/bash
# Test 15 (diffusion + surrogate refinement), detached and kept awake. Resumable: rerunning
# skips training if the candidates file exists, reuses screened evaluations, and skips
# targets the finisher already started.
#   nohup caffeinate -i bash scripts/feasibility_tests/run_diffusion.sh > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")/../.." || exit 1
PY=/Users/leifakerley/miniforge3/envs/stellarator/bin/python
LOG=data/feasibility_tests/diffusion_run.log
echo "=== $(date): Test 15 start ===" >> $LOG
HF_HUB_OFFLINE=1 $PY scripts/feasibility_tests/diffusion_test.py 2>&1 \
  | grep --line-buffered -vE 'RuntimeWarning|fsolve|improvement from the last' >> $LOG
echo "=== $(date): Test 15 end ===" >> $LOG
