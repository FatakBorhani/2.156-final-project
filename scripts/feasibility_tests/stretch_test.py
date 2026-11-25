"""Test 21: stretch and fatten our own designs, then finish them.

Two scratch probes (Nov 22) changed the plan. Scaling a boundary's cross-section (every m >= 1
Fourier coefficient times s; the m = 0 axis terms untouched) moves a design along a line of
roughly constant L_gradB / AR:
  - Stretch (s < 1): our top design (AR 17.16, L_gradB 16.37) became feasible designs at AR
    18.0 / 19.0 / 19.8 with L_gradB 17.25 / 18.27 / 19.09, with no optimization: +3.0 hypervolume
    (147.25 -> 150.22). The 4 x 4 ladder (Test 13) had stalled at that design on QI.
  - Fatten (s > 1): high-L_gradB designs taken down to AR 11-15 landed 0.7-1.5 L_gradB above our
    front, but infeasible (worst 0.02-0.06, mostly QI and mirror). Good finisher starts: Test 20's
    raises started from feasible designs far below their targets.
Raising the middle of the front by itself is worth little (all of AR 10-13 by +0.5: about +1.5).

A. Stretch. For every point on our front and an AR grid (step 0.1, up to 19.95), the stretched
   copy is predicted to keep L_gradB / AR. Grid ARs where a prediction beats the front by >= 0.05
   get the best two sources evaluated at high fidelity. Costs minutes, not hours.
B. Fatten and finish. At each target AR, fatten the three front designs with the best predicted
   L_gradB there (sources up to 25% larger AR), start from the closest to feasible, and run the
   Test 20 finisher (5 x 5, modes 4-5 step like mode 3) with AR <= target and L_gradB >= a floor
   halfway between our front and that start. Targets run in order of that margin. After each
   target, stretch again so new designs also fill the gaps above them.

Only our own designs are used (all Track A ledgers); no leaderboard or public designs.
Resumable: evaluated copies are skipped by name, targets are saved, interrupted finisher
attempts continue from their best design.

    python scripts/feasibility_tests/stretch_test.py            # ~8-12 h
    python scripts/feasibility_tests/stretch_test.py --dry      # plumbing check (~20 min)
    python scripts/feasibility_tests/stretch_test.py --report   # re-plot only
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import multiprocessing
import sys
from concurrent import futures
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import ladder_test as lt  # noqa: E402
import multifid_test as mf  # noqa: E402
import polish_test as pt  # noqa: E402
from front_test import read  # noqa: E402

NFP3_LEDGERS = ["multifid_ledger.jsonl", "front_ledger.jsonl", "chain_ledger.jsonl", "gen_ledger.jsonl",
                "ladder_ledger.jsonl", "compact_ledger.jsonl", "diffusion_ledger.jsonl",
                "descent_ledger.jsonl", "diffusion2_ledger.jsonl", "descent2_ledger.jsonl",
                "modes5_ledger.jsonl", "finemodes_ledger.jsonl", "raise_ledger.jsonl"]
BASE_SOURCES = {s.removesuffix("_ledger.jsonl"): s for s in NFP3_LEDGERS}
TAG_A, TAG_B = "stretch", "fatten"

GRID_STEP, AR_MAX, STRETCH_MAX = 0.1, 19.95, 1.2  # grid up to AR 19.95; at most 20% longer
BEAT, PER_GRID = 0.05, 2  # predicted L_gradB must beat the front by this; sources per grid AR
TARGETS_B = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
FATTEN_MAX, N_SEEDS, FLOOR_FRAC = 1.25, 3, 0.5
MAXIT, EXT_IT, W_OBJ = 15, 10, 0.05  # as Test 20 (descent_test)

lt.TOL = 0.005  # as Test 20: a target counts at AR <= limit * 1.005, L_gradB >= floor * 0.995
mf.MAX_MODE, mf.SCALE_FLOOR_MODE = 5, 3


def ledger_of(tag: str, dry: bool) -> Path:
    return lt.DATA / f"{tag}_ledger{'_dry' if dry else ''}.jsonl"


def down(v: float, q: float = 0.05) -> float:
    return math.floor(v / q + 1e-9) * q


def feasible_rows(dry: bool, own: bool = True) -> list[dict]:
    rows = [r for s in BASE_SOURCES.values() for r in read(lt.DATA / s)]
    if own:
        rows += read(ledger_of(TAG_A, dry)) + read(ledger_of(TAG_B, dry))
    return [r for r in rows if r["kind"] == "hi" and r.get("ok") and r.get("feasible") and r.get("boundary")]


def pareto(rows: list[dict]) -> list[dict]:
    out, best = [], -1.0
    for r in sorted(rows, key=lambda r: (r["aspect_ratio"], -r["L_gradB"])):
        if r["L_gradB"] > best:
            out.append(r)
            best = r["L_gradB"]
    return out


def front_at(front: list[dict], ar: float) -> float:
    return max((r["L_gradB"] for r in front if r["aspect_ratio"] <= ar), default=1.0)


def scaled(bjson: str, s: float) -> str:
    """Cross-section times s (s < 1 stretches to higher AR, s > 1 fattens); axis unchanged."""
    b = json.loads(bjson)
    for k in ("r_cos", "z_sin", "r_sin", "z_cos"):
        if b.get(k) is not None:
            a = np.array(b[k], float)
            a[1:] *= s
            b[k] = a.tolist()
    return json.dumps(b)


def _fixed(bjson: str, _x):
    import constellaration.geometry.surface_rz_fourier as rz
    return rz.SurfaceRZFourier.model_validate_json(bjson)


def evaluate_copies(items: list[tuple[str, str]], ledger: Path) -> list[dict]:
    """High-fidelity rows (ledger format of the finisher) for (name, boundary) pairs not yet done."""
    from constellaration import forward_model

    from evaluate import default_workers

    done = {r["start"] for r in read(ledger)}
    todo = [(n, b) for n, b in items if n not in done]
    if todo:
        hi = forward_model.ConstellarationSettings.default_high_fidelity()
        ctx = multiprocessing.get_context("forkserver")
        workers = min(default_workers(), len(todo))
        print(f"evaluating {len(todo)} copies with {workers} workers", flush=True)
        with ledger.open("a") as f, futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            jobs = {ex.submit(pt.evaluate, np.zeros(1), 1.0, functools.partial(_fixed, b), hi): (n, b)
                    for n, b in todo}
            for fu in futures.as_completed(jobs):
                n, b = jobs[fu]
                row = {k: v for k, v in fu.result().items() if k != "boundary"}
                row.update(boundary=b, start=n, k=0, n=0, n_hi=1, kind="hi", wall=row["seconds"])
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  {n}: " + (f"{'FEASIBLE' if row.get('feasible') else 'worst %.4f' % row['worst']}, "
                                    f"AR {row['aspect_ratio']:.3f}, L_gradB {row['L_gradB']:.2f}"
                                    if row["ok"] else row.get("error", "failed")), flush=True)
    names = {n for n, _ in items}
    return [r for r in read(ledger) if r["start"] in names]


def stretch_round(dry: bool) -> None:
    front = pareto(feasible_rows(dry))
    items = []
    for g in np.arange(math.ceil(front[0]["aspect_ratio"] / GRID_STEP) * GRID_STEP, AR_MAX + 1e-9, GRID_STEP):
        g = round(min(g, AR_MAX), 2)
        preds = sorted(((p["L_gradB"] * g / p["aspect_ratio"], p) for p in front
                        if p["aspect_ratio"] < g <= p["aspect_ratio"] * STRETCH_MAX), key=lambda t: -t[0])
        for pred, p in preds[:PER_GRID]:
            if pred >= front_at(front, g) + BEAT:
                items.append((f"st{g:.2f}_from{p['aspect_ratio']:.3f}_{p['L_gradB']:.3f}",
                              scaled(p["boundary"], p["aspect_ratio"] / g)))
    items = items[:2] if dry else items
    print(f"=== stretch: {len(items)} copies predicted to beat the front ===", flush=True)
    rows = evaluate_copies(items, ledger_of(TAG_A, dry))
    print(f"=== stretch: {sum(bool(r.get('feasible')) for r in rows)} of {len(rows)} feasible ===", flush=True)


def plan_targets(dry: bool) -> list[list]:
    """[target AR, L_gradB floor, start name] per target, largest margin first; saved."""
    ledger = ledger_of(TAG_B, dry)
    state = ledger.with_name(ledger.stem + "_targets.json")
    if state.exists():
        return json.loads(state.read_text())
    front = pareto(feasible_rows(dry))
    seeds = []
    for t in (TARGETS_B[2:3] if dry else TARGETS_B):
        preds = sorted(((p["L_gradB"] * t / p["aspect_ratio"], p) for p in front
                        if t < p["aspect_ratio"] <= t * FATTEN_MAX), key=lambda x: -x[0])
        seeds += [(f"seed{t:g}_from{p['aspect_ratio']:.3f}_{p['L_gradB']:.3f}", scaled(p["boundary"], p["aspect_ratio"] / t))
                  for _, p in preds[:1 if dry else N_SEEDS]]
    rows = [r for r in evaluate_copies(seeds, ledger) if r["ok"]]
    targets = []
    for t in (TARGETS_B[2:3] if dry else TARGETS_B):
        mine = [r for r in rows if r["start"].startswith(f"seed{t:g}_")]
        f0 = front_at(front, t)
        good = [r for r in mine if r["L_gradB"] >= f0 + 0.3] or mine
        if not good:
            continue
        s = min(good, key=lambda r: r["worst"])
        floor = round(down(f0 + FLOOR_FRAC * (s["L_gradB"] - f0)), 2)
        if floor <= f0:
            continue
        targets.append([t, floor, s["start"], round(s["L_gradB"] - f0, 3)])
        print(f"target AR <= {t}: front {f0:.2f}, start L_gradB {s['L_gradB']:.2f} (worst {s['worst']:.4f}), "
              f"floor {floor}", flush=True)
    targets.sort(key=lambda x: -x[3])
    state.write_text(json.dumps(targets))
    return targets


def run_target(ar: float, lgb: float, seed: str, dry: bool, ledger: Path) -> bool:
    """Two attempts at AR <= ar, L_gradB >= lgb: from the fattened start, then from the best so far."""
    name = lt.rung_name(ar, lgb)
    mf.AR_BOUND, mf.W_OBJ, mf.STOP_ON_FEASIBLE, mf.AR_FEAS_TOL = ar, W_OBJ, True, lt.TOL * ar
    mf.LGB_MIN, mf.LGB_FEAS_TOL = lgb, lt.TOL * lgb
    attempts = [(name, MAXIT)] if dry else [(name, MAXIT), (name + "_x", EXT_IT)]
    for attempt, maxit in attempts:
        rows = read(ledger)
        mine = [r for r in rows if lt.belongs(r["start"], name) or r["start"] == seed]
        his = [r for r in mine if r["kind"] == "hi" and r["ok"] and r.get("boundary")]
        if any(lt.feasible_at(r, ar, lgb) for r in his):
            return True
        tries = [r for r in rows if lt.belongs(r["start"], attempt)
                 and not (attempt == name and r["start"].startswith(name + "_x"))]
        its = sum(r["kind"] == "iteration" for r in tries)
        if tries and its >= (2 if dry else maxit):
            continue  # this attempt ran to the end without success
        if tries:  # interrupted: the iterations left, from this attempt's best design
            pool = [r for r in his if lt.belongs(r["start"], attempt)] or his
            attempt, maxit = f"{attempt}_r{len({r['start'] for r in tries})}", max(maxit - its, 3)
        else:
            pool = [r for r in his if r["start"] == seed] if attempt == name else his
        start = min(pool, key=lambda r: lt.gap_at(r, ar, lgb))
        mf.MAXIT = 2 if dry else maxit
        print(f"[{attempt}] AR <= {ar}, L_gradB >= {lgb}: start gap {lt.gap_at(start, ar, lgb):.4f} "
              f"(worst {start['worst']:.4f}, AR {start['aspect_ratio']:.3f}, L_gradB {start['L_gradB']:.2f}, "
              f"from {start['start']}), {mf.MAXIT} iterations", flush=True)
        mf.run_start(attempt, start["boundary"], dry, ledger)
    return any(lt.feasible_at(r, ar, lgb) for r in read(ledger)
               if r["kind"] == "hi" and r["ok"] and lt.belongs(r["start"], name))


def part_b(dry: bool, run: bool) -> list[list]:
    state = ledger_of(TAG_B, dry).with_name(ledger_of(TAG_B, dry).stem + "_targets.json")
    if not run:
        return json.loads(state.read_text()) if state.exists() else []
    targets = plan_targets(dry)
    for ar, lgb, seed, margin in targets:
        print(f"=== fatten: AR <= {ar}, L_gradB >= {lgb} (start {seed}, {margin:.2f} above the front) ===",
              flush=True)
        ok = run_target(ar, lgb, seed, dry, ledger_of(TAG_B, dry))
        print(f"=== fatten: AR <= {ar}, L_gradB >= {lgb}: {'reached' if ok else 'not reached'} ===", flush=True)
        stretch_round(dry)
    return targets


def reports(dry: bool, targets: list[list]) -> None:
    lt.TAG, lt.SOURCES, lt.RUNGS = TAG_A, dict(BASE_SOURCES), []
    lt.report(ledger_of(TAG_A, dry), dry)
    lt.TAG, lt.SOURCES = TAG_B, {**BASE_SOURCES, TAG_A: ledger_of(TAG_A, dry).name}
    lt.RUNGS = [(t[0], t[1]) for t in targets]
    lt.report(ledger_of(TAG_B, dry), dry)

    from constellaration import problems

    def hv(rows):
        f = pareto(rows)
        return float(problems._hypervolume(np.array([[-p["L_gradB"], p["aspect_ratio"]] for p in f]),
                                           np.array([1.0, 20.0]))) if f else 0.0

    base = feasible_rows(dry, own=False)
    a = read(ledger_of(TAG_A, dry))
    b = read(ledger_of(TAG_B, dry))
    fa = [r for r in a if r["kind"] == "hi" and r.get("feasible")]
    fb = [r for r in b if r["kind"] == "hi" and r.get("feasible")]
    s = {"hypervolume_before": hv(base), "hypervolume_with_A": hv(base + fa),
         "hypervolume_with_A_and_B": hv(base + fa + fb), "n_feasible_A": len(fa), "n_feasible_B": len(fb),
         "n_copies_A": sum(r["kind"] == "hi" for r in a),
         "top_A": max(([round(r["aspect_ratio"], 3), round(r["L_gradB"], 3)] for r in fa),
                      key=lambda x: x[1], default=None)}
    (lt.OUT / f"{TAG_A}_combined{'_dry' if dry else ''}.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    lt.DATA.mkdir(parents=True, exist_ok=True)
    if not a.report:
        stretch_round(a.dry)
    targets = part_b(a.dry, run=not a.report)
    reports(a.dry, targets)


if __name__ == "__main__":
    main()
