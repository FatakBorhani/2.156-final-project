#!/bin/bash
# Test 19 (4 field periods: approach, push, then descent), detached and kept awake. Resumable:
# rerunning skips finished rounds and replays the saved rungs. With a pid, waits for that
# process (e.g. Test 18) to exit first so runs don't compete for cores.
#   nohup caffeinate -i bash scripts/feasibility_tests/run_nfp4.sh [pid] > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")/../.." || exit 1
PY=/Users/leifakerley/miniforge3/envs/stellarator/bin/python
LOG=data/feasibility_tests/nfp4_run.log
if [ -n "$1" ]; then
  echo "=== $(date): Test 19 queued, waiting for process $1 ===" >> $LOG
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
echo "=== $(date): Test 19 start ===" >> $LOG
HF_HUB_OFFLINE=1 $PY scripts/feasibility_tests/nfp4_test.py 2>&1 \
  | grep --line-buffered -vE 'RuntimeWarning|fsolve|improvement from the last' >> $LOG
echo "=== $(date): Test 19 end ===" >> $LOG
