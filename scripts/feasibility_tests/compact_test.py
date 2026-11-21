"""Test 14: compactness ladder (Test 13's ladder, walking aspect ratio down).

Test 11 walked the aspect-ratio limit down and stalled below 11.5: each step started from
the design closest to the new limit, which was never quite feasible, and the leftover
violation grew step by step (0.013 at AR 11, 0.033 at 10.5, 0.04-0.05 at 10 and 9.5).
Test 13 climbed 6 rungs in a row by always starting from a Benchmark-3-feasible design and
making L_gradB an explicit floor. Now that the front is tall (L_gradB up to ~14), moving it
left is worth more hypervolume than moving it up, so this runs the same ladder downward:

- smaller steps than Test 11 (AR -0.25 a rung, about 2%), from our most compact feasible
  design (AR 11.58, L_gradB 9.30, Test 12);
- a sliding L_gradB floor (about -1.6 per unit of AR, the public front's slope) so the
  optimizer can trade simplicity for compactness, but only as much as the front allows;
- every rung starts from the closest feasible design (Test 13's rule), not the closest
  design overall (Test 11's).

Same finisher, budgets and tolerances as Test 13 (ladder_test.py), separate ledger.

    python scripts/feasibility_tests/compact_test.py            # ~6-10 h
    python scripts/feasibility_tests/compact_test.py --dry      # plumbing check
    python scripts/feasibility_tests/compact_test.py --report   # re-plot only
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import ladder_test as lt  # noqa: E402

lt.TAG = "compact"
lt.RUNGS = [(11.25, 8.75), (11.0, 8.4), (10.75, 8.0), (10.5, 7.6), (10.25, 7.2),
            (10.0, 6.8), (9.75, 6.4), (9.5, 6.0), (9.25, 5.6), (9.0, 5.2)]
lt.SOURCES = {**lt.SOURCES, "test13": "ladder_ledger.jsonl"}

if __name__ == "__main__":
    lt.main()
