#!/bin/bash
# Test 21 (stretch and fatten our own designs, then finish), detached and kept awake. Resumable:
# rerunning replays saved rungs and targets. With a pid, waits for that
# process (e.g. another test) to exit first so runs don't compete for cores.
#   nohup caffeinate -i bash scripts/feasibility_tests/run_stretch.sh [pid] > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")/../.." || exit 1
PY=/Users/leifakerley/miniforge3/envs/stellarator/bin/python
LOG=data/feasibility_tests/stretch_run.log
if [ -n "$1" ]; then
  echo "=== $(date): Test 21 queued, waiting for process $1 ===" >> $LOG
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
echo "=== $(date): Test 21 start ===" >> $LOG
HF_HUB_OFFLINE=1 $PY scripts/feasibility_tests/stretch_test.py 2>&1 \
  | grep --line-buffered -vE 'RuntimeWarning|fsolve|improvement from the last' >> $LOG
echo "=== $(date): Test 21 end ===" >> $LOG
