#!/bin/bash
# Test 17 --raw (finisher from unrefined diffusion samples, then descent), queued after the
# main Test 17 run: waits for the given pid to exit first so runs don't compete for cores.
#   nohup caffeinate -i bash scripts/feasibility_tests/run_diffusion2_raw.sh [pid] > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")/../.." || exit 1
PY=/Users/leifakerley/miniforge3/envs/stellarator/bin/python
LOG=data/feasibility_tests/diffusion2_run.log
if [ -n "$1" ]; then
  echo "=== $(date): --raw queued, waiting for process $1 ===" >> $LOG
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
echo "=== $(date): Test 17 --raw start ===" >> $LOG
HF_HUB_OFFLINE=1 $PY scripts/feasibility_tests/diffusion2_test.py --raw 2>&1 \
  | grep --line-buffered -vE 'RuntimeWarning|fsolve|improvement from the last' >> $LOG
echo "=== $(date): Test 17 --raw end ===" >> $LOG
