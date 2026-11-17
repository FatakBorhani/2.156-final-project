"""Test 11: compactness chain (the no-ML control for spreading the front).

Test 10 showed the finisher makes many feasible copies of one design but no trade-off,
and a single jump to aspect ratio 11 or 10 didn't reach feasible in 15 iterations. Here
the aspect-ratio limit drops in small steps (11.5, 11, 10.5, 10, 9.5). Each step:

- starts from the known design closest to feasible *at that limit* (worst violation,
  counting AR above the limit as a violation too), from every earlier ledger;
- runs Test 9's multi-fidelity finisher (feasibility first, L_gradB only a tie-breaker)
  and stops at the first design that passes Benchmark 3 with AR <= limit + AR_TOL;
- if it fails in MAXIT iterations, gets one extension of EXT_IT iterations from its best
  design, then the chain moves on regardless.

Resumable: finished steps are skipped; an interrupted step restarts from its best design.

    python scripts/feasibility_tests/chain_test.py            # ~5-8 h
    python scripts/feasibility_tests/chain_test.py --dry      # plumbing check
    python scripts/feasibility_tests/chain_test.py --report   # re-plot only
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

STEPS = [11.5, 11.0, 10.5, 10.0, 9.5]
MAXIT, EXT_IT, AR_TOL, W_OBJ = 20, 15, 0.05, 0.05
OUT = ROOT / "results" / "feasibility_tests"
DATA = ROOT / "data" / "feasibility_tests"
SOURCES = ["multifid_ledger.jsonl", "front_ledger.jsonl"]  # + this test's own ledger


def step_name(ar: float) -> str:
    return f"ar{ar:g}".replace(".", "p")


def hi_rows(ledger: Path) -> list[dict]:
    rows = [r for s in SOURCES for r in read(DATA / s)] + read(ledger)
    return [r for r in rows if r["kind"] == "hi" and r["ok"]]


def gap_at(r: dict, ar: float) -> float:
    """Distance to feasible at this aspect-ratio limit, as the finisher sees it."""
    return max(r["worst"], (r["aspect_ratio"] - ar) / ar)


def belongs(start: str, name: str) -> bool:
    """Exact run name or one of its suffixed attempts ('ar11' must not match 'ar11p5')."""
    return start == name or start.startswith(name + "_")


def feasible_at(r: dict, ar: float) -> bool:
    return bool(r.get("feasible")) and r["aspect_ratio"] <= ar + AR_TOL


def run_step(ar: float, dry: bool, ledger: Path) -> None:
    name = step_name(ar)
    mf.AR_BOUND, mf.W_OBJ, mf.STOP_ON_FEASIBLE, mf.AR_FEAS_TOL = ar, W_OBJ, True, AR_TOL
    attempts = [(name, MAXIT)] if dry else [(name, MAXIT), (name + "_x", EXT_IT)]
    for attempt, maxit in attempts:
        rows = read(ledger)
        mine = [r for r in rows if belongs(r["start"], name)]
        if any(r["kind"] == "hi" and r["ok"] and feasible_at(r, ar) for r in mine):
            print(f"[{name}] feasible design already found; step done", flush=True)
            return
        tries = [r for r in mine if belongs(r["start"], attempt)]
        its = sum(r["kind"] == "iteration" for r in tries)
        if tries and its >= (2 if dry else maxit):
            continue  # this attempt ran to the end without success
        # start: the closest design at this limit, from everything evaluated so far
        cands = hi_rows(ledger)
        start = min(cands, key=lambda r: gap_at(r, ar))
        if tries:  # interrupted: keep the iterations left, from the best design
            attempt = f"{attempt}_r{sum(1 for s in {r['start'] for r in tries} if '_r' in s) + 1}"
            maxit = max(maxit - its, 3)
        mf.MAXIT = maxit
        print(f"[{attempt}] AR <= {ar}: start {gap_at(start, ar):.4f} (worst {start['worst']:.4f}, "
              f"AR {start['aspect_ratio']:.2f}, from {start['start']}), {maxit} iterations", flush=True)
        mf.run_start(attempt, start["boundary"], dry, ledger)
        if any(feasible_at(r, ar) for r in read(ledger)
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
    for src, rs in [("test9", read(DATA / "multifid_ledger.jsonl")), ("test10", read(DATA / "front_ledger.jsonl")),
                    ("chain", rows)]:
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

    hv_before, _ = hv([p for p in feas if p["src"] != "chain"])
    hv_after, front = hv(feas)
    summary = {"hypervolume_before": hv_before, "hypervolume_after": hv_after,
               "n_feasible_total": len(feas), "n_feasible_chain": sum(p["src"] == "chain" for p in feas),
               "steps": {}}
    for ar in STEPS:
        name = step_name(ar)
        rs = [r for r in rows if belongs(r["start"], name)]
        if not rs:
            continue
        his = [r for r in rs if r["kind"] == "hi"]
        ok = [r for r in his if r["ok"]]
        hit = [i for i, r in enumerate(his) if r["ok"] and feasible_at(r, ar)]
        best = min(ok, key=lambda r: gap_at(r, ar)) if ok else None
        summary["steps"][name] = {
            "ar_limit": ar, "attempts": list(dict.fromkeys(r["start"] for r in rs)),
            "iterations": sum(r["kind"] == "iteration" for r in rs),
            "very_low_runs": sum(r["kind"] == "oracle" for r in rs), "high_runs": len(his),
            "reached": bool(hit),
            "high_runs_to_first_feasible": hit[0] + 1 if hit else None,
            "very_low_runs_to_first_feasible": (sum(r["kind"] == "oracle" for r in rs[:rs.index(his[hit[0]])])
                                                if hit else None),
            "best_gap": gap_at(best, ar) if best else None,
            "best_worst": best["worst"] if best else None,
            "best_aspect_ratio": best["aspect_ratio"] if best else None,
            "best_L_gradB": best["L_gradB"] if best else None,
            "hours": round(sum(max(r["wall"] for r in rs if r["start"] == a)
                               for a in {r["start"] for r in rs}) / 3600, 2),
        }
    summary["front"] = [{"src": p["src"], "start": p["start"], "aspect_ratio": p["aspect_ratio"],
                         "L_gradB": p["L_gradB"], "worst": p["worst"]} for p in front]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for src, color in [("test9", "#1f6fd1"), ("test10", "#9a9a9a"), ("chain", "#d9622b")]:
        ps = [p for p in feas if p["src"] == src]
        if ps:
            ax.scatter([p["aspect_ratio"] for p in ps], [p["L_gradB"] for p in ps], s=16, color=color,
                       alpha=0.8, label=f"{src} ({len(ps)})")
    xs = [p["aspect_ratio"] for p in front] + [20.0]
    ys = [p["L_gradB"] for p in front]
    ax.step(xs, ys + [ys[-1]], where="post", color="k", lw=1)
    ax.set_xlabel("aspect ratio (lower = more compact)")
    ax.set_ylabel("L∇B (higher = simpler coils)")
    ax.set_title(f"Feasible designs; front hypervolume {hv_before:.1f} → {hv_after:.1f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"chain{tag}.png", dpi=150)
    with open(OUT / f"chain_feasible_designs{tag}.jsonl", "w") as f:
        for p in feas:
            if p["src"] == "chain":
                f.write(json.dumps({k: p.get(k) for k in ("start", "k", "worst", "L_gradB", "aspect_ratio",
                                                         "violations", "boundary")}) + "\n")
    (OUT / f"chain{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "front"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    ledger = DATA / f"chain_ledger{'_dry' if a.dry else ''}.jsonl"
    if not a.report:
        for ar in (STEPS[:1] if a.dry else STEPS):
            run_step(ar, a.dry, ledger)
    report(ledger, a.dry)


if __name__ == "__main__":
    main()
