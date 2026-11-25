"""Test 22: a second stretch-and-fatten pass (Test 21's method) with targets chosen as it goes.

Test 21 raised hypervolume 147.2 -> 157.9 in 1.6 h. All 242 stretched copies were feasible, and all five
fattened starts were finished in 1-5 iterations. Its front still has two jumps (AR 10.05 and
12.9) where the stretch from below sits under what fattening from above predicts, and its
compact end (AR 9.23) did not move, because Test 21's targets started at AR 11.

Here the target ARs form a grid (8.5 to 16 in steps of 0.5). Each round re-reads the front,
picks the untried grid AR where the best fattened copy (sources up to 25% larger AR) is
predicted furthest above the front, and finishes it as Test 21 did, then stretches again.
Below the most compact design, "the front" is that design's L_gradB less 1.6 per unit AR
(the descent's slope), so compact targets get floors too.

Only our own designs are used (all Track A ledgers, including Test 21's).

    python scripts/feasibility_tests/fatten2_test.py            # ~4-10 h
    python scripts/feasibility_tests/fatten2_test.py --dry      # plumbing check
    python scripts/feasibility_tests/fatten2_test.py --report   # re-plot only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import ladder_test as lt  # noqa: E402
import stretch_test as st  # noqa: E402

st.BASE_SOURCES = {**st.BASE_SOURCES, "stretch": "stretch_ledger.jsonl", "fatten": "fatten_ledger.jsonl"}
st.TAG_A, st.TAG_B = "stretch2", "fatten2"
GRID = [round(x, 2) for x in np.arange(8.5, 16.01, 0.5)]
MIN_MARGIN, SLOPE = 0.2, 1.6

_front_at = st.front_at


def front_at(front: list[dict], ar: float) -> float:
    lo = front[0]
    if ar < lo["aspect_ratio"]:
        return lo["L_gradB"] - SLOPE * (lo["aspect_ratio"] - ar)
    return _front_at(front, ar)


st.front_at = front_at


def state_path(dry: bool) -> Path:
    ledger = st.ledger_of(st.TAG_B, dry)
    return ledger.with_name(ledger.stem + "_targets.json")


def next_target(dry: bool, tried: set[float]) -> list | None:
    """Pick, seed and save the next target; None when no grid AR is left worth trying."""
    while True:
        front = st.pareto(st.feasible_rows(dry))
        best = None
        for t in GRID:
            if t in tried:
                continue
            preds = sorted(((p["L_gradB"] * t / p["aspect_ratio"], p) for p in front
                            if t < p["aspect_ratio"] <= t * st.FATTEN_MAX), key=lambda x: -x[0])
            if preds and preds[0][0] - front_at(front, t) >= MIN_MARGIN:
                m = preds[0][0] - front_at(front, t)
                if best is None or m > best[0]:
                    best = (m, t, preds[:1 if dry else st.N_SEEDS])
        if best is None:
            return None
        m, t, preds = best
        tried.add(t)
        seeds = [(f"seed{t:g}_from{p['aspect_ratio']:.3f}_{p['L_gradB']:.3f}",
                  st.scaled(p["boundary"], p["aspect_ratio"] / t)) for _, p in preds]
        print(f"=== next target AR <= {t}: predicted {m:.2f} above the front ===", flush=True)
        rows = [r for r in st.evaluate_copies(seeds, st.ledger_of(st.TAG_B, dry)) if r["ok"]]
        f0 = front_at(front, t)
        good = [r for r in rows if r["L_gradB"] >= f0 + 0.3] or rows
        target = None
        if good:
            s = min(good, key=lambda r: r["worst"])
            floor = round(st.down(f0 + st.FLOOR_FRAC * (s["L_gradB"] - f0)), 2)
            if floor > f0:
                target = [t, floor, s["start"], round(s["L_gradB"] - f0, 3)]
                print(f"target AR <= {t}: front {f0:.2f}, start L_gradB {s['L_gradB']:.2f} "
                      f"(worst {s['worst']:.4f}), floor {floor}", flush=True)
        saved = json.loads(state_path(dry).read_text()) if state_path(dry).exists() else []
        saved.append(target or [t, None, None, None])
        state_path(dry).write_text(json.dumps(saved))
        if target:
            return target


def part_b(dry: bool, run: bool) -> list[list]:
    saved = json.loads(state_path(dry).read_text()) if state_path(dry).exists() else []
    if run:
        for t in saved:  # finish (or confirm) the targets already planned
            if t[1] is not None:
                print(f"=== fatten: AR <= {t[0]}, L_gradB >= {t[1]} (planned earlier) ===", flush=True)
                st.run_target(t[0], t[1], t[2], dry, st.ledger_of(st.TAG_B, dry))
        tried = {t[0] for t in saved}
        for _ in range(1 if dry else len(GRID)):
            t = next_target(dry, tried)
            if t is None:
                break
            print(f"=== fatten: AR <= {t[0]}, L_gradB >= {t[1]} (start {t[2]}, {t[3]:.2f} above the front) ===",
                  flush=True)
            ok = st.run_target(t[0], t[1], t[2], dry, st.ledger_of(st.TAG_B, dry))
            print(f"=== fatten: AR <= {t[0]}, L_gradB >= {t[1]}: {'reached' if ok else 'not reached'} ===",
                  flush=True)
            st.stretch_round(dry)
    saved = json.loads(state_path(dry).read_text()) if state_path(dry).exists() else []
    return [t for t in saved if t[1] is not None]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    lt.DATA.mkdir(parents=True, exist_ok=True)
    if not a.report:
        st.stretch_round(a.dry)
    targets = part_b(a.dry, run=not a.report)
    st.reports(a.dry, targets)


if __name__ == "__main__":
    main()
