#!/bin/bash
# Test 11 (compactness chain), then Test 12 (generator screen + finish), one after the other.
# Both are resumable: rerunning this script continues where they stopped.
#   nohup caffeinate -i bash scripts/feasibility_tests/run_overnight.sh > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")/../.." || exit 1
PY=/Users/leifakerley/miniforge3/envs/stellarator/bin/python
LOG=data/feasibility_tests
$PY scripts/feasibility_tests/chain_test.py >> $LOG/chain_run.log 2>&1
$PY scripts/feasibility_tests/gen_test.py >> $LOG/gen_run.log 2>&1
