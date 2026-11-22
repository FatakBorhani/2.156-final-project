"""Test 13: simplicity ladder (push L_gradB up from designs that are already feasible).

Tests 8-11 treated L_gradB only as a tie-breaker (weight 0.05), so every feasible design
we have sits near AR 11.6, L_gradB 8.6, while the public leaderboard front reaches
L_gradB ~14 at that aspect ratio. Hypervolume gains more from simpler coils at moderate
aspect ratio than from the compactness Test 11 couldn't reach. Here L_gradB becomes a real
constraint: each rung asks for AR <= limit AND L_gradB >= floor, both a step beyond the
last rung, so each rung starts from a design that is feasible (or nearly) one rung down.

Each rung:
- starts from the Benchmark-3-feasible design closest to the rung (counting AR above the
  limit and L_gradB below the floor as violations), from every ledger so far;
- runs Test 9's multi-fidelity finisher with the L_gradB floor as a 7th constraint (its
  fidelity offset is measured like the others), and stops at the first design that passes
  Benchmark 3 with AR <= limit * (1 + TOL) and L_gradB >= floor * (1 - TOL);
- if it fails in MAXIT iterations, gets one extension of EXT_IT iterations, then the
  ladder moves on regardless.

Resumable: finished rungs are skipped; an interrupted rung restarts from its best design.
The first run (Nov 18) climbed the first 7 rungs to AR 15.5, L_gradB 13.9; the last 3
rungs (to AR 19) were added afterwards and run as an extension from the same ledger.

Test 14 (compact_test.py) reuses this ladder downward; the module-level TAG, RUNGS and
SOURCES are what it overrides.

    python scripts/feasibility_tests/ladder_test.py            # ~7-10 h
    python scripts/feasibility_tests/ladder_test.py --dry      # plumbing check
    python scripts/feasibility_tests/ladder_test.py --report   # re-plot only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import constellaration.optimization.augmented_lagrangian_runner  # noqa: F401, E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import multifid_test as mf  # noqa: E402
from common import ROOT  # noqa: E402
from front_test import pareto, read  # noqa: E402

# (aspect-ratio limit, L_gradB floor): from our front point (11.55, 8.64) along a diagonal
# below the public front (L_gradB ~ 1.6 AR - 4), about +0.5-1 AR and +0.75 L_gradB a rung
RUNGS = [(12.0, 9.25), (12.5, 10.0), (13.0, 10.75), (13.5, 11.5), (14.25, 12.25),
         (15.0, 13.0), (16.0, 14.0),
         (17.0, 15.0), (18.0, 16.0), (19.0, 17.0)]  # extension, added after the first 7 succeeded
MAXIT, EXT_IT, W_OBJ = 20, 10, 0.05
TOL = 0.01  # fraction of the limit / floor
TAG = "ladder"  # ledger, results and plot names; this run's label in the report
OUT = ROOT / "results" / "feasibility_tests"
DATA = ROOT / "data" / "feasibility_tests"
# every ledger whose high-fidelity rows kept their boundary (Test 12's only if it ran)
SOURCES = {"test9": "multifid_ledger.jsonl", "test10": "front_ledger.jsonl",
           "test11": "chain_ledger.jsonl", "test12": "gen_ledger.jsonl"}


def rung_name(ar: float, lgb: float) -> str:
    return f"ar{ar:g}_lgb{lgb:g}".replace(".", "p")


def belongs(start: str, name: str) -> bool:
    return start == name or start.startswith(name + "_")


def hi_rows(ledger: Path) -> list[dict]:
    rows = [r for s in SOURCES.values() for r in read(DATA / s)] + read(ledger)
    return [r for r in rows if r["kind"] == "hi" and r["ok"] and r.get("boundary")]


def gap_at(r: dict, ar: float, lgb: float) -> float:
    """Distance to success at this rung, as the finisher sees it."""
    return max(r["worst"], (r["aspect_ratio"] - ar) / ar, (lgb - r["L_gradB"]) / lgb)


def feasible_at(r: dict, ar: float, lgb: float) -> bool:
    return (bool(r.get("feasible")) and r["aspect_ratio"] <= ar * (1 + TOL)
            and r["L_gradB"] >= lgb * (1 - TOL))


def run_rung(ar: float, lgb: float, dry: bool, ledger: Path) -> None:
    name = rung_name(ar, lgb)
    mf.AR_BOUND, mf.W_OBJ, mf.STOP_ON_FEASIBLE, mf.AR_FEAS_TOL = ar, W_OBJ, True, TOL * ar
    mf.LGB_MIN, mf.LGB_FEAS_TOL = lgb, TOL * lgb
    attempts = [(name, MAXIT)] if dry else [(name, MAXIT), (name + "_x", EXT_IT)]
    for attempt, maxit in attempts:
        rows = read(ledger)
        if any(feasible_at(r, ar, lgb) for r in hi_rows(ledger)):
            print(f"[{name}] a design already meets this rung; done", flush=True)
            return
        mine = [r for r in rows if belongs(r["start"], name)]
        tries = [r for r in mine if belongs(r["start"], attempt)]
        its = sum(r["kind"] == "iteration" for r in tries)
        if tries and its >= (2 if dry else maxit):
            continue  # this attempt ran to the end without success
        # prefer Benchmark-3-feasible starts: only the new AR/L_gradB targets are then missing
        cands = hi_rows(ledger)
        start = min([r for r in cands if r.get("feasible")] or cands, key=lambda r: gap_at(r, ar, lgb))
        if tries:  # interrupted: keep the iterations left, from the best design
            attempt = f"{attempt}_r{sum(1 for s in {r['start'] for r in tries} if '_r' in s) + 1}"
            maxit = max(maxit - its, 3)
        mf.MAXIT = maxit
        print(f"[{attempt}] AR <= {ar}, L_gradB >= {lgb}: start gap {gap_at(start, ar, lgb):.4f} "
              f"(worst {start['worst']:.4f}, AR {start['aspect_ratio']:.2f}, "
              f"L_gradB {start['L_gradB']:.2f}, from {start['start']}), {maxit} iterations", flush=True)
        mf.run_start(attempt, start["boundary"], dry, ledger)
        if any(feasible_at(r, ar, lgb) for r in read(ledger)
               if r["start"] == attempt and r["kind"] == "hi" and r["ok"]):
            return


def report(ledger: Path, dry: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from constellaration import problems

    tag = "_dry" if dry else ""
    rows = read(ledger)
    feas, seen = [], set()
    for src, rs in [*[(k, read(DATA / v)) for k, v in SOURCES.items()], (TAG, rows)]:
        for r in rs:
            if r["kind"] == "hi" and r.get("feasible"):
                key = (round(r["aspect_ratio"], 6), round(r["L_gradB"], 6))
                if key not in seen:
                    seen.add(key)
                    feas.append({**r, "src": src})

    def hv(points):
        f = pareto(points)
        return (float(problems._hypervolume(np.array([[-p["L_gradB"], p["aspect_ratio"]] for p in f]),
                                            np.array([1.0, 20.0]))) if f else 0.0), f

    hv_before, _ = hv([p for p in feas if p["src"] != TAG])
    hv_after, front = hv(feas)
    summary = {"hypervolume_before": hv_before, "hypervolume_after": hv_after,
               "n_feasible_total": len(feas), f"n_feasible_{TAG}": sum(p["src"] == TAG for p in feas),
               "rungs": {}}
    for ar, lgb in RUNGS:
        name = rung_name(ar, lgb)
        rs = [r for r in rows if belongs(r["start"], name)]
        if not rs:
            continue
        his = [r for r in rs if r["kind"] == "hi"]
        ok = [r for r in his if r["ok"]]
        hit = [i for i, r in enumerate(his) if r["ok"] and feasible_at(r, ar, lgb)]
        best = min(ok, key=lambda r: gap_at(r, ar, lgb)) if ok else None
        summary["rungs"][name] = {
            "ar_limit": ar, "lgb_floor": lgb, "attempts": list(dict.fromkeys(r["start"] for r in rs)),
            "iterations": sum(r["kind"] == "iteration" for r in rs),
            "very_low_runs": sum(r["kind"] == "oracle" for r in rs), "high_runs": len(his),
            "reached": bool(hit), "high_runs_to_first_success": hit[0] + 1 if hit else None,
            "feasible_designs": sum(1 for r in his if r.get("feasible")),
            "best_gap": gap_at(best, ar, lgb) if best else None,
            "best_worst": best["worst"] if best else None,
            "best_aspect_ratio": best["aspect_ratio"] if best else None,
            "best_L_gradB": best["L_gradB"] if best else None,
            "hours": round(sum(max(r["wall"] for r in rs if r["start"] == a)
                               for a in {r["start"] for r in rs}) / 3600, 2),
        }
    summary["front"] = [{"src": p["src"], "start": p["start"], "aspect_ratio": p["aspect_ratio"],
                         "L_gradB": p["L_gradB"], "worst": p["worst"]} for p in front]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {"test9": "#1f6fd1", "test10": "#9a9a9a", "test11": "#7fb3e6", "test12": "#7b4fc9",
              "test13": "#2a9d6f", "test14": "#e0a030", "test15": "#c03030", "test16": "#8c564b",
              "test17": "#d62728", TAG: "#d9622b"}
    for src, color in colors.items():
        ps = [p for p in feas if p["src"] == src]
        if ps:
            ax.scatter([p["aspect_ratio"] for p in ps], [p["L_gradB"] for p in ps], s=16, color=color,
                       alpha=0.8, label=f"{src} ({len(ps)})")
    ax.scatter([a for a, _ in RUNGS], [b for _, b in RUNGS], marker="+", s=50, color="k", lw=0.8,
               label="rung targets")
    xs = [p["aspect_ratio"] for p in front] + [20.0]
    ys = [p["L_gradB"] for p in front]
    ax.step(xs, ys + [ys[-1]], where="post", color="k", lw=1)
    ax.set_xlabel("aspect ratio (lower = more compact)")
    ax.set_ylabel("L∇B (higher = simpler coils)")
    ax.set_title(f"Feasible designs; front hypervolume {hv_before:.1f} → {hv_after:.1f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{TAG}{tag}.png", dpi=150)
    with open(OUT / f"{TAG}_feasible_designs{tag}.jsonl", "w") as f:
        for p in feas:
            if p["src"] == TAG:
                f.write(json.dumps({k: p.get(k) for k in ("start", "k", "worst", "L_gradB", "aspect_ratio",
                                                         "violations", "boundary")}) + "\n")
    (OUT / f"{TAG}{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "front"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    ledger = DATA / f"{TAG}_ledger{'_dry' if a.dry else ''}.jsonl"
    if not a.report:
        for ar, lgb in (RUNGS[:1] if a.dry else RUNGS):
            run_rung(ar, lgb, a.dry, ledger)
    report(ledger, a.dry)


if __name__ == "__main__":
    main()
