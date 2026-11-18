"""Test 10: build our own front from the first feasible design (+ a second seed).

A. Front: Test 9's multi-fidelity ALM, restarted from the feasible design with the goal
   flipped: L_gradB is now a real objective (W_OBJ 1.0, not a tie-breaker) and the run
   keeps going after feasible designs, at several aspect-ratio limits. Every
   high-fidelity check that passes Benchmark 3 is a feasible design for our front.
B. Second feasible design: resume Test 9's seed-2 start from its best design (0.018)
   with the box reset, stopping at the first feasible design.

    python scripts/feasibility_tests/front_test.py            # overnight
    python scripts/feasibility_tests/front_test.py --dry      # plumbing check
    python scripts/feasibility_tests/front_test.py --report   # re-plot only
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

FRONT_RUNS = [("ar12", 12.0), ("ar14", 14.0), ("ar11", 11.0), ("ar10", 10.0)]
FRONT_ITERS, FRONT_W_OBJ = 15, 1.0
OUT = ROOT / "results" / "feasibility_tests"
DATA = ROOT / "data" / "feasibility_tests"


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def feasible_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["kind"] == "hi" and r.get("feasible")]


def configure(ar: float, w_obj: float, maxit: int, stop: bool) -> None:
    mf.AR_BOUND, mf.W_OBJ, mf.MAXIT, mf.STOP_ON_FEASIBLE = ar, w_obj, maxit, stop


def main_runs(dry: bool, ledger: Path) -> None:
    base = feasible_rows(read(DATA / "multifid_ledger.jsonl"))
    assert base, "Test 9's feasible design not found"
    base_b = base[0]["boundary"]
    done = {r["start"] for r in read(ledger)}
    runs = FRONT_RUNS[:1] if dry else FRONT_RUNS
    for name, ar in runs:
        if name in done:
            print(f"[{name}] already in ledger; skipping", flush=True)
            continue
        start = base_b
        if name == "ar10":  # chain: start from the most compact feasible design so far
            feas = [r for r in feasible_rows(read(ledger)) if r["aspect_ratio"] <= 11.0]
            if feas:
                start = min(feas, key=lambda r: r["aspect_ratio"])["boundary"]
        configure(ar, FRONT_W_OBJ, FRONT_ITERS, stop=False)
        print(f"[{name}] front run: AR <= {ar}, W_OBJ {FRONT_W_OBJ}, {FRONT_ITERS} iterations", flush=True)
        mf.run_start(name, start, dry, ledger)
    if dry:
        return
    name = "seed2_resume"
    if name not in done:
        t9 = read(DATA / "multifid_ledger.jsonl")
        best = min((r for r in t9 if r["start"] == "t6_start2" and r["kind"] == "hi" and r["ok"]),
                   key=lambda r: r["worst"])
        configure(12.0, 0.05, 30, stop=True)
        print(f"[{name}] resuming Test 9 seed 2 from {best['worst']:.4f}, box reset", flush=True)
        mf.run_start(name, best["boundary"], dry, ledger)


def pareto(points: list[dict]) -> list[dict]:
    """Non-dominated designs: lower aspect ratio and higher L_gradB are better."""
    out, best = [], -np.inf
    for p in sorted(points, key=lambda p: (p["aspect_ratio"], -p["L_gradB"])):
        if p["L_gradB"] > best:
            out.append(p)
            best = p["L_gradB"]
    return out


def report(ledger: Path, dry: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from constellaration import problems

    tag = "_dry" if dry else ""
    rows = read(ledger)
    base = feasible_rows(read(DATA / "multifid_ledger.jsonl"))
    feas, seen = [], set()
    for r in [{**r, "run": "test9"} for r in base] + [{**r, "run": r["start"]} for r in feasible_rows(rows)]:
        key = (round(r["aspect_ratio"], 6), round(r["L_gradB"], 6))  # same design re-checked
        if key not in seen:
            seen.add(key)
            feas.append(r)
    front = pareto(feas)
    hv = (float(problems._hypervolume(np.array([[-p["L_gradB"], p["aspect_ratio"]] for p in front]),
                                      np.array([1.0, 20.0]))) if front else 0.0)
    runs = list(dict.fromkeys(r["start"] for r in rows))
    summary = {"hypervolume": hv, "n_feasible": len(feas), "n_front": len(front), "runs": {}}
    for s in runs:
        rs = [r for r in rows if r["start"] == s]
        his = [r for r in rs if r["kind"] == "hi" and r["ok"]]
        fs = [r for r in feas if r["run"] == s]
        summary["runs"][s] = {
            "iterations": sum(r["kind"] == "iteration" for r in rs),
            "very_low_runs": sum(r["kind"] == "oracle" for r in rs), "high_runs": len(his),
            "best_worst_hi": min((r["worst"] for r in his), default=None),
            "n_feasible": len(fs),
            "best_feasible_L_gradB": max((r["L_gradB"] for r in fs), default=None),
            "min_feasible_aspect_ratio": min((r["aspect_ratio"] for r in fs), default=None),
            "hours": round(rs[-1]["wall"] / 3600, 2) if rs else None,
        }
    summary["front"] = [{"run": p["run"], "aspect_ratio": p["aspect_ratio"], "L_gradB": p["L_gradB"],
                         "worst": p["worst"]} for p in front]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s in dict.fromkeys(p["run"] for p in feas):
        ps = [p for p in feas if p["run"] == s]
        ax.scatter([p["aspect_ratio"] for p in ps], [p["L_gradB"] for p in ps], s=14, label=s, alpha=0.7)
    if front:
        xs = [p["aspect_ratio"] for p in front] + [20.0]
        ys = [p["L_gradB"] for p in front]
        ax.step(xs, ys + [ys[-1]], where="post", color="k", lw=1)
    ax.set_xlabel("aspect ratio (lower = more compact)")
    ax.set_ylabel("L∇B (higher = simpler coils)")
    ax.set_title(f"Our feasible designs: {len(feas)}, front hypervolume {hv:.1f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"front{tag}.png", dpi=150)
    with open(OUT / f"front_feasible_designs{tag}.jsonl", "w") as f:
        for p in feas:
            f.write(json.dumps({k: p.get(k) for k in ("run", "k", "worst", "L_gradB", "aspect_ratio",
                                                     "violations", "boundary")}) + "\n")
    (OUT / f"front{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "front"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    ledger = DATA / f"front_ledger{'_dry' if a.dry else ''}.jsonl"
    if not a.report:
        main_runs(a.dry, ledger)
    report(ledger, a.dry)


if __name__ == "__main__":
    main()
