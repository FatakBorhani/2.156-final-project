"""Test 20: 5 x 5 modes that actually move, then how much room there is above our front.

A. Fine modes. Test 18 allowed Fourier modes up to 5 x 5, but the finisher scales its search
   space by exp(-1.5 * max(|m|, |n|)), so mode-5 steps were ~400x smaller than mode-1 steps
   and every feasible design kept |mode-5 coefficients| < 1.3e-5: Test 18's gains (AR 10.18
   -> 9.53) came from its smaller descent step, not from finer shapes. Here modes 4-5 step
   like mode 3 (multifid_test.SCALE_FLOOR_MODE = 3). The descent restarts at the target
   Test 18 failed twice (AR <= 9.45, L_gradB >= 6.95, best feasible AR 9.533), then continues
   for up to MAX_RUNGS_A targets. Reaching it is evidence for the finer modes (not proof: it is
   a third attempt with a fresh box).

B. Room above the front. Our front sits below Proxima's baseline at middle AR (L_gradB 7.6 vs
   8.45 at AR 10, ~9.7 vs 11.1 at AR 12). At AR 10, 11 and 12, one target each: the front's
   L_gradB there + RAISE, two attempts from the closest feasible design. This is a diagnostic:
   raising one point adds little hypervolume (~RAISE^2), but if +0.5 comes easily, a broad
   raise of the whole front (+~10 per unit of L_gradB) is worth a campaign.

Resumable: part A replays its saved rungs; part B saves its targets.

    python scripts/feasibility_tests/finemodes_test.py            # ~8-12 h
    python scripts/feasibility_tests/finemodes_test.py --dry      # plumbing check
    python scripts/feasibility_tests/finemodes_test.py --report   # re-plot only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import descent_test as ds  # noqa: E402
import ladder_test as lt  # noqa: E402
import multifid_test as mf  # noqa: E402
from front_test import read  # noqa: E402

NFP3_LEDGERS = ["multifid_ledger.jsonl", "front_ledger.jsonl", "chain_ledger.jsonl", "gen_ledger.jsonl",
                "ladder_ledger.jsonl", "compact_ledger.jsonl", "diffusion_ledger.jsonl",
                "descent_ledger.jsonl", "diffusion2_ledger.jsonl", "descent2_ledger.jsonl",
                "modes5_ledger.jsonl"]
TAG_A, TAG_B = "finemodes", "raise"
FIRST_RUNG = [9.45, 6.95, 0.125]  # Test 18's last failed target
MAX_RUNGS_A = 3
RAISE_AR, RAISE = [10.0, 11.0, 12.0], 0.5

mf.MAX_MODE, mf.SCALE_FLOOR_MODE = 5, 3
ds.STEP0, ds.STEP_MIN, ds.MAX_RUNGS = 0.125, 0.06, MAX_RUNGS_A
BASE_SOURCES = {s.removesuffix("_ledger.jsonl"): s for s in NFP3_LEDGERS}


def ledger_of(tag: str, dry: bool) -> Path:
    return lt.DATA / f"{tag}_ledger{'_dry' if dry else ''}.jsonl"


def part_a(dry: bool, run: bool) -> None:
    lt.TAG, lt.SOURCES = TAG_A, dict(BASE_SOURCES)
    ledger = ledger_of(TAG_A, dry)
    state = ledger.with_name(ledger.stem + "_rungs.json")
    if not state.exists():
        state.write_text(json.dumps([FIRST_RUNG]))
    lt.RUNGS = ds.descend(dry, ledger, run=run)
    lt.report(ledger, dry)


def front_at(ar: float) -> float:
    return max(r["L_gradB"] for r in lt.hi_rows(ledger_of(TAG_B, False))
               if r.get("feasible") and r["aspect_ratio"] <= ar * (1 + lt.TOL))


def part_b(dry: bool, run: bool) -> None:
    lt.TAG = TAG_B
    lt.SOURCES = {**BASE_SOURCES, TAG_A: ledger_of(TAG_A, False).name}
    ledger = ledger_of(TAG_B, dry)
    state = ledger.with_name(ledger.stem + "_targets.json")
    if state.exists():
        targets = json.loads(state.read_text())
    else:
        targets = [[ar, round(math.ceil((front_at(ar) + RAISE) / 0.05 - 1e-9) * 0.05, 2)] for ar in RAISE_AR]
        state.write_text(json.dumps(targets))
    targets = targets[:1] if dry else targets
    for ar, lgb in targets:
        print(f"=== raise: AR <= {ar}, L_gradB >= {lgb} (front there: {front_at(ar):.2f}) ===", flush=True)
        if run:
            ds.run_rung(ar, lgb, dry, ledger)
    lt.RUNGS = [tuple(t) for t in targets]
    lt.report(ledger, dry)


def combined(dry: bool) -> None:
    import numpy as np
    from constellaration import problems

    def feas(paths):
        out = {}
        for p in paths:
            for r in read(p):
                if r["kind"] == "hi" and r.get("feasible"):
                    out[(round(r["aspect_ratio"], 6), round(r["L_gradB"], 6))] = r
        return list(out.values())

    def hv(f):
        return float(problems._hypervolume(np.array([[-p["L_gradB"], p["aspect_ratio"]] for p in f]),
                                           np.array([1.0, 20.0]))) if f else 0.0

    before = feas([lt.DATA / s for s in NFP3_LEDGERS])
    a = feas([ledger_of(TAG_A, dry)])
    b = feas([ledger_of(TAG_B, dry)])
    s = {"hypervolume_before": hv(before), "hypervolume_with_A": hv(before + a),
         "hypervolume_with_A_and_B": hv(before + a + b), "n_feasible_A": len(a), "n_feasible_B": len(b),
         "most_compact_A": min(([round(r["aspect_ratio"], 3), round(r["L_gradB"], 3)] for r in a), default=None)}
    (lt.OUT / f"{TAG_A}_combined{'_dry' if dry else ''}.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2), flush=True)


def fine_mode_check(dry: bool) -> None:
    """Largest |coefficient| at each mode level among this test's high-fidelity designs."""
    import numpy as np

    big = {}
    rows = [r for t in (TAG_A, TAG_B) for r in read(ledger_of(t, dry)) if r["kind"] == "hi" and r["ok"]]
    for r in rows:
        b = json.loads(r["boundary"])
        for a in (np.array(b["r_cos"]), np.array(b["z_sin"])):
            n = (a.shape[1] - 1) // 2
            lev = np.maximum(np.arange(a.shape[0])[:, None], np.abs(np.arange(-n, n + 1))[None, :])
            for k in (3, 4, 5):
                if (lev == k).any():
                    big[k] = max(big.get(k, 0.0), float(np.abs(a[lev == k]).max()))
    print("largest |coefficient| by mode level (Test 18: mode 4 ~2e-3, mode 5 < 1.3e-5):",
          {k: f"{v:.2e}" for k, v in sorted(big.items())}, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    lt.DATA.mkdir(parents=True, exist_ok=True)
    part_a(a.dry, run=not a.report)
    part_b(a.dry, run=not a.report)
    fine_mode_check(a.dry)
    combined(a.dry)


if __name__ == "__main__":
    main()
