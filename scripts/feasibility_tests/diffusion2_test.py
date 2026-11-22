"""Test 17: retrained diffusion seeds below AR 10.2, then Test 16's descent from wherever they land.

Test 15's generators (trained before we had any feasible design below AR 11.26) gave the
starts that got the finisher past AR 11.26 to 10.51; Test 16's descent then walked down to
AR 10.22 and stalled there at ~0.014 (QI and mirror ratio). This repeats Test 15 with what
we have learned since, aimed only at the compact end:

- retrain the surrogate, diffusion model and cVAE with Tests 15-16's ledgers (~150 feasible
  designs at AR 10.2-11.2, none of which Test 15 saw) and Test 15's ~2,500 screened
  generator samples (most of them points where its surrogate was too optimistic);
- diffusion edits now start from the closest feasible designs, i.e. the AR 10.2-10.3 ones;
- targets only below the front: AR 10.0 to 9.0 on the 1.6-per-AR line through our most
  compact design (AR 10.22, L_gradB 7.70); no upper targets;
- same screening and finisher (12 iterations per target, skipped above a 0.12 start gap);
- then Test 16's adaptive descent (descent_test.py) from the most compact feasible design
  found by then, in its own ledger ("descent2").

Added after the first run (--raw): the refined starts were all 0.19-0.26 from feasible at
high fidelity (the surrogate, though accurate on real designs, was exploited by its own
gradient refinement), so the finisher skipped every target; the unrefined diffusion samples
were much closer (best 0.04-0.14 at very low fidelity, at or near each target's AR). --raw
checks the RAW_TOP best unrefined samples per target at high fidelity, runs the finisher
from each target's best one (closest first, start gap <= 0.12), then reruns the descent
from the most compact feasible design found by then.

    python scripts/feasibility_tests/diffusion2_test.py            # ~7-9 h
    python scripts/feasibility_tests/diffusion2_test.py --dry      # plumbing check
    python scripts/feasibility_tests/diffusion2_test.py --raw      # finisher from unrefined samples, then descent
    python scripts/feasibility_tests/diffusion2_test.py --report   # re-plot only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import descent_test as ds  # noqa: E402
import diffusion_test as dt  # noqa: E402
import ladder_test as lt  # noqa: E402

dt.TAG = "diffusion2"
dt.OUR_LEDGERS = [*dt.OUR_LEDGERS, "diffusion_ledger.jsonl", "descent_ledger.jsonl"]
dt.FEAS_LEDGERS = [*dt.FEAS_LEDGERS, "diffusion_ledger.jsonl", "descent_ledger.jsonl"]
dt.EVAL_SETS = [*dt.EVAL_SETS, ("diffusion_candidates.jsonl", "diffusion_eval_ledger.jsonl")]
dt.COMPACT = [(10.0, 7.35), (9.75, 6.95), (9.5, 6.55), (9.25, 6.15), (9.0, 5.75)]
dt.UPPER_AR = []
dt.tname = lambda ar, lgb: f"t17_ar{ar:g}_lgb{lgb:g}".replace(".", "p")

DESCENT_TAG = "descent2"
RAW_TOP = 3  # unrefined samples per target checked at high fidelity


def raw_finish(dry: bool) -> None:
    import multifid_test as mf
    from evaluate import evaluate_designs
    from front_test import read

    tag = "_dry" if dry else ""
    by_id = {c["id"]: c for c in read(dt.DATA / f"{dt.TAG}_candidates{tag}.jsonl")}
    evals = dt.DATA / f"{dt.TAG}_eval_ledger{tag}.jsonl"
    ledger = dt.DATA / f"{dt.TAG}_ledger{tag}.jsonl"

    def gap(r):
        c = by_id[r["id"]]
        return dt.gap_at(r["max_violation"], r["aspect_ratio"], r["L_gradB"], c["ar_t"], c["lg_t"])

    vlo = [r for r in read(evals) if r["fidelity"] == "very_low_fidelity" and r["ok"]
           and r["id"] in by_id and by_id[r["id"]]["arm"] == "diff_raw"]
    tlist = list(dict.fromkeys(c["target"] for c in by_id.values()))
    picks = []
    for t in tlist:
        rs = sorted((r for r in vlo if by_id[r["id"]]["target"] == t), key=gap)
        picks += [by_id[r["id"]] for r in rs[:(1 if dry else RAW_TOP)]]
    hi = [r for r in evaluate_designs(picks, evals, fidelity="high_fidelity") if r["ok"]]
    plan = []
    for t in tlist:
        rs = [r for r in hi if by_id[r["id"]]["target"] == t]
        if rs:
            best = min(rs, key=gap)
            plan.append((gap(best), t, best, by_id[best["id"]]))
    plan.sort(key=lambda p: p[0])
    print("raw finisher plan:", [(t, round(g, 4)) for g, t, _, _ in plan], flush=True)
    done = {r["start"] for r in read(ledger)}
    mf.W_OBJ, mf.STOP_ON_FEASIBLE, mf.LOG_ORACLE_BOUNDARY = dt.W_OBJ, True, True
    for gp, t, best, c in (plan[:1] if dry else plan):
        name = f"{t}_raw"
        if name in done:
            print(f"[{name}] already in ledger; skipping", flush=True)
            continue
        if gp > dt.FINISH_MAX_GAP:
            print(f"[{name}] best start {gp:.4f} > {dt.FINISH_MAX_GAP}; skipping", flush=True)
            continue
        mf.AR_BOUND, mf.AR_FEAS_TOL = c["ar_t"], dt.TOL * c["ar_t"]
        mf.LGB_MIN, mf.LGB_FEAS_TOL = c["lg_t"], dt.TOL * c["lg_t"]
        mf.MAXIT = 2 if dry else dt.MAXIT
        print(f"[{name}] AR <= {c['ar_t']}, L_gradB >= {c['lg_t']}: start gap {gp:.4f} ({best['id']}: worst "
              f"{best['max_violation']:.4f}, AR {best['aspect_ratio']:.2f}, L_gradB {best['L_gradB']:.2f})",
              flush=True)
        mf.run_start(name, c["boundary"], dry, ledger)


def run_descent(dry: bool, report_only: bool) -> None:
    lt.TAG = DESCENT_TAG
    lt.SOURCES = {**lt.SOURCES, "test16": "descent_ledger.jsonl", "test17": "diffusion2_ledger.jsonl"}
    ledger = lt.DATA / f"{lt.TAG}_ledger{'_dry' if dry else ''}.jsonl"
    lt.RUNGS = ds.descend(dry, ledger, run=not report_only)
    lt.report(ledger, dry)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--raw", action="store_true")
    a = p.parse_args()
    tag = "_dry" if a.dry else ""
    dt.DATA.mkdir(parents=True, exist_ok=True)
    if a.raw:
        raw_finish(a.dry)
    elif not a.report:
        if not (dt.DATA / f"{dt.TAG}_candidates{tag}.jsonl").exists():
            dt.train_and_generate(a.dry)
        dt.screen_and_finish(a.dry)
    dt.report(a.dry)
    run_descent(a.dry, a.report)


if __name__ == "__main__":
    main()
