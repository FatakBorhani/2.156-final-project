"""Test 18: Test 16's descent with more shape freedom (Fourier modes up to 5 x 5 instead of 4 x 4).

Below AR ~10.2 every method has stalled at 0.014-0.04 from feasible, always on QI and mirror
ratio: Test 16-17's descents (from feasible designs), Test 15's and Test 17's generated
starts, Test 14's ladder. All of them let the finisher vary only m <= 4, |n| <= 4 (~80
coefficients, the dataset's resolution). QI and mirror ratio depend on fine detail of |B|
along field lines, so the wall may be our parametrization rather than physics. This reruns
the same descent with m, |n| <= 5 (~120 coefficients; the new ones start at 0, so every
start design is unchanged and scores the same):

- same finisher, budgets, tolerances and step rules as Test 16 (descent_test.py), except a
  smaller first step (0.125: Tests 16-17 gained only ~0.04 AR a rung here) and one more
  halving (to 0.0625) before stopping;
- so the first rung, AR <= 10.05 (L_gradB >= 7.45) from the AR 10.18 feasible design, is the rung Test 17
  failed at 4 x 4 (AR <= 10.05, L_gradB >= 7.4; best 0.0083 at AR 10.17), a direct comparison;
- Benchmark 3 does not limit the mode count (problems.py checks only the requirements and
  objectives); VMEC's resolution follows the boundary's largest nonzero mode, so runs are slower.

    python scripts/feasibility_tests/modes_test.py            # ~3-6 h
    python scripts/feasibility_tests/modes_test.py --dry      # plumbing check
    python scripts/feasibility_tests/modes_test.py --report   # re-plot only
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import descent_test as ds  # noqa: E402
import ladder_test as lt  # noqa: E402
import multifid_test as mf  # noqa: E402

mf.MAX_MODE = 5
lt.TAG = "modes5"
lt.SOURCES = {**lt.SOURCES, "test16": "descent_ledger.jsonl", "test17": "diffusion2_ledger.jsonl",
              "test17d": "descent2_ledger.jsonl"}
ds.STEP0, ds.STEP_MIN = 0.125, 0.06

if __name__ == "__main__":
    ds.main()
