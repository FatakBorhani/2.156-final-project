"""Test 16: adaptive compactness descent from Test 15's most compact feasible design.

Test 14 walked aspect ratio down from AR 11.26 and failed every rung: after a rung failed,
each lower rung started from the same design and failed further away. Test 15's generators
then gave starting designs in the compact region, and the finisher reached feasibility at
AR 10.51 (L_gradB 8.03), but its AR 9-10 targets stalled at 0.025-0.045 from those
generated starts. This test walks down from the new feasible design with Test 13's rule
(start each rung from a Benchmark-3-feasible design), fixing Test 14's flaw:

- the next target is always set from the most compact feasible design found so far
  (target AR = its AR - step, rounded down to 0.05; L_gradB floor slides down at 1.6 per
  unit of AR from its L_gradB, the public front's slope);
- each rung gets two attempts: MAXIT iterations from the closest feasible design, then
  EXT_IT iterations from the closest design of any kind (often the first attempt's best
  near miss, or a Test 14/15 near miss), with a fresh search box and penalties;
- after a failed rung the step is halved (0.25 -> 0.125); below STEP_MIN the descent stops,
  instead of trying lower rungs from the same design.

Resumable: rerunning replays the same decisions from the ledgers (finished rungs return at
once) and continues an interrupted rung from its best design.

    python scripts/feasibility_tests/descent_test.py            # ~2-10 h, depending on how far it gets
    python scripts/feasibility_tests/descent_test.py --dry      # plumbing check
    python scripts/feasibility_tests/descent_test.py --report   # re-plot only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import ladder_test as lt  # noqa: E402
import multifid_test as mf  # noqa: E402
from front_test import read  # noqa: E402

lt.TAG = "descent"
lt.SOURCES = {**lt.SOURCES, "test13": "ladder_ledger.jsonl", "test14": "compact_ledger.jsonl",
              "test15": "diffusion_ledger.jsonl"}
lt.TOL = 0.005  # a rung counts only at AR <= limit * 1.005 (Test 13/14 used 1%)
STEP0, STEP_MIN, SLOPE = 0.25, 0.1, 1.6
AR_STOP, MAX_RUNGS = 8.5, 12
MAXIT, EXT_IT, W_OBJ = 15, 10, 0.05


def down(v: float, q: float = 0.05) -> float:
    return round(math.floor(v / q + 1e-9) * q, 2)


def most_compact(ledger: Path) -> dict:
    feas = [r for r in lt.hi_rows(ledger) if r.get("feasible")]
    return min(feas, key=lambda r: (round(r["aspect_ratio"], 3), -r["L_gradB"]))


def run_rung(ar: float, lgb: float, dry: bool, ledger: Path) -> bool:
    """Two attempts at AR <= ar, L_gradB >= lgb; True if a design meets the rung."""
    name = lt.rung_name(ar, lgb)
    mf.AR_BOUND, mf.W_OBJ, mf.STOP_ON_FEASIBLE, mf.AR_FEAS_TOL = ar, W_OBJ, True, lt.TOL * ar
    mf.LGB_MIN, mf.LGB_FEAS_TOL = lgb, lt.TOL * lgb
    attempts = [(name, MAXIT, True)] if dry else [(name, MAXIT, True), (name + "_x", EXT_IT, False)]
    for attempt, maxit, from_feasible in attempts:
        if any(lt.feasible_at(r, ar, lgb) for r in lt.hi_rows(ledger)):
            return True
        tries = [r for r in read(ledger) if lt.belongs(r["start"], attempt)
                 and not (attempt == name and r["start"].startswith(name + "_x"))]
        its = sum(r["kind"] == "iteration" for r in tries)
        if tries and its >= (2 if dry else maxit):
            continue  # this attempt ran to the end without success
        cands = lt.hi_rows(ledger)
        if tries:  # interrupted: the iterations left, from this attempt's best design
            mine = [r for r in cands if lt.belongs(r["start"], attempt)] or cands
            start = min(mine, key=lambda r: lt.gap_at(r, ar, lgb))
            attempt = f"{attempt}_r{len({r['start'] for r in tries})}"
            maxit = max(maxit - its, 3)
        else:
            pool = [r for r in cands if r.get("feasible")] if from_feasible else cands
            start = min(pool, key=lambda r: lt.gap_at(r, ar, lgb))
        mf.MAXIT = 2 if dry else maxit
        print(f"[{attempt}] AR <= {ar}, L_gradB >= {lgb}: start gap {lt.gap_at(start, ar, lgb):.4f} "
              f"(worst {start['worst']:.4f}, AR {start['aspect_ratio']:.3f}, "
              f"L_gradB {start['L_gradB']:.2f}, from {start['start']}), {mf.MAXIT} iterations", flush=True)
        mf.run_start(attempt, start["boundary"], dry, ledger)
    return any(lt.feasible_at(r, ar, lgb) for r in lt.hi_rows(ledger))


def descend(dry: bool, ledger: Path, run: bool) -> list[tuple[float, float]]:
    """Walk the rungs; with run=False only replay the ones already run.

    The rung list is saved next to the ledger as it grows, so a resumed run or a report
    replays the same rungs even though the most compact design changes along the way."""
    state = ledger.with_name(ledger.stem + "_rungs.json")
    saved = json.loads(state.read_text()) if state.exists() else []
    rungs, step = [], STEP0
    for i in range(1 if dry else MAX_RUNGS):
        best = most_compact(ledger)
        if i < len(saved):
            ar, lgb, step = saved[i]
        else:
            ar = down(best["aspect_ratio"] - step)
            lgb = down(best["L_gradB"] - SLOPE * (best["aspect_ratio"] - ar))
            if ar < AR_STOP or not run:
                break
            saved.append([ar, lgb, step])
            state.write_text(json.dumps(saved))
        rungs.append((ar, lgb))
        print(f"=== rung {len(rungs)}: AR <= {ar}, L_gradB >= {lgb} (step {step}; most compact "
              f"feasible so far AR {best['aspect_ratio']:.3f}, L_gradB {best['L_gradB']:.2f}) ===",
              flush=True)
        if run_rung(ar, lgb, dry, ledger) if run else any(
                lt.feasible_at(r, ar, lgb) for r in lt.hi_rows(ledger)):
            step = STEP0
        else:
            step /= 2
            if step < STEP_MIN:
                print(f"=== rung failed and the step would drop below {STEP_MIN}: stopping ===", flush=True)
                break
    return rungs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    lt.DATA.mkdir(parents=True, exist_ok=True)
    ledger = lt.DATA / f"{lt.TAG}_ledger{'_dry' if a.dry else ''}.jsonl"
    lt.RUNGS = descend(a.dry, ledger, run=not a.report)
    lt.report(ledger, a.dry)


if __name__ == "__main__":
    main()
