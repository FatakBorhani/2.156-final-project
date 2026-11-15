"""Scorer check: does our evaluator reproduce the leaderboard's numbers on public designs?

Most passing public designs sit right at the benchmark's 1% tolerance (max normalized
violation 0.005-0.01), so a tiny platform difference would flip them between passing and
failing. This draws a reproducible sample weighted toward that edge, and after
scripts/evaluate.py has scored it, compares our values with the leaderboard's.

    python scripts/seeded/parity_check.py sample      # -> data/seeded/parity_sample.jsonl
    python scripts/evaluate.py data/seeded/parity_sample.jsonl data/seeded/runs/parity.jsonl --workers 8
    python scripts/seeded/parity_check.py compare     # -> data/seeded/parity_report.json

Passes if no design flips feasibility and the largest |difference| in max violation is
well below the smallest distance any sampled design has to the 0.01 line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "seeded"
TOL = 0.01
SEED = 0


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _pick(rng, pool: list[dict], n: int, by_aspect: bool = True) -> list[dict]:
    """n designs from pool, spread evenly over aspect-ratio deciles if by_aspect."""
    if n >= len(pool):
        return list(pool)
    if not by_aspect:
        return [pool[i] for i in rng.choice(len(pool), n, replace=False)]
    order = np.argsort([d["aspect_ratio"] for d in pool])
    picked = []
    for k, chunk in enumerate(np.array_split(order, 10)):
        take = n // 10 + (k < n % 10)
        picked += [pool[i] for i in rng.choice(chunk, min(take, len(chunk)), replace=False)]
    return picked


def sample(n_edge: int, n_safe: int, n_other: int) -> None:
    rng = np.random.default_rng(SEED)
    designs = _read(DATA / "public_p3.jsonl")
    target = [d for d in designs if d["nfp"] == 3 and d["shape"] == [5, 9]]
    others = [d for d in designs if not (d["nfp"] == 3 and d["shape"] == [5, 9])]

    edge = [d for d in target if 0.005 < d["max_violation"] <= TOL]
    very_edge = [d for d in edge if d["max_violation"] > 0.0095]
    safe = [d for d in target if d["max_violation"] <= 0.005]
    just_failed = [d for d in designs if TOL < d["max_violation"] <= 0.1]

    chosen = (_pick(rng, very_edge, n_edge // 2) + _pick(rng, edge, n_edge - n_edge // 2)
              + _pick(rng, safe, n_safe) + _pick(rng, others, n_other, by_aspect=False)
              + just_failed)
    # every submitter in the target set appears at least once
    for user in sorted({d["sources"][0]["user"] for d in target}):
        if not any(d["sources"][0]["user"] == user for d in chosen):
            pool = [d for d in target if d["sources"][0]["user"] == user]
            chosen.append(pool[rng.integers(len(pool))])
    unique = list({d["id"]: d for d in chosen}.values())

    out = DATA / "parity_sample.jsonl"
    with out.open("w") as fh:
        for d in unique:
            fh.write(json.dumps(d) + "\n")
    print(f"wrote {len(unique)} designs to {out}: edge {n_edge}, safe {n_safe}, "
          f"other shapes/NFP {n_other}, just-failed {len(just_failed)}, "
          f"plus submitter coverage")


def compare(ledger: Path) -> None:
    """Our ledger vs the leaderboard (which scored at high fidelity). Works for any
    fidelity: on a low-fidelity ledger it measures how well the fast setting predicts
    the official result."""
    ref = {d["id"]: d for d in _read(DATA / "parity_sample.jsonl")}
    ours = [r for r in _read(ledger) if r["id"] in ref]
    failed = [r["id"] for r in ours if not r["ok"]]
    ok = [r for r in ours if r["ok"]]
    rows = []
    for r in ok:
        d = ref[r["id"]]
        rows.append({
            "id": r["id"], "nfp": d["nfp"], "shape": d["shape"],
            "lb_violation": d["max_violation"], "our_violation": r["max_violation"],
            "d_violation": r["max_violation"] - d["max_violation"],
            "d_L_gradB": r["L_gradB"] - d["L_gradB"],
            "d_aspect": r["aspect_ratio"] - d["aspect_ratio"],
            "flip": bool(r["feasible"]) != bool(d["feasible"]),
        })
    dv = np.abs([x["d_violation"] for x in rows])
    margin = [TOL - x["lb_violation"] for x in rows if x["lb_violation"] <= TOL]
    flips = [x for x in rows if x["flip"]]
    lb_feasible = [x for x in rows if x["lb_violation"] <= TOL]
    report = {
        "ledger": str(ledger), "fidelity": ours[0]["fidelity"] if ours else None,
        "median_seconds": float(np.median([r["seconds"] for r in ours])) if ours else None,
        "sampled": len(ref), "scored": len(ours), "failed_to_run": failed,
        "missed_passes": sum(x["flip"] for x in lb_feasible),
        "false_passes": sum(x["flip"] for x in rows if x["lb_violation"] > TOL),
        "feasibility_flips": flips,
        "max_abs_d_violation": float(dv.max()) if rows else None,
        "p99_abs_d_violation": float(np.percentile(dv, 99)) if rows else None,
        "max_abs_d_L_gradB": float(np.abs([x["d_L_gradB"] for x in rows]).max()) if rows else None,
        "max_abs_d_aspect": float(np.abs([x["d_aspect"] for x in rows]).max()) if rows else None,
        "smallest_margin_to_tolerance": float(min(margin)) if margin else None,
        "worst": sorted(rows, key=lambda x: -abs(x["d_violation"]))[:10],
    }
    report["verdict"] = ("PASS" if not flips and not failed and rows
                         and report["max_abs_d_violation"] < 1e-4 else "CHECK")
    out = DATA / f"{ledger.stem}_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "worst"}, indent=2))
    print(f"full report: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--edge", type=int, default=150)
    s.add_argument("--safe", type=int, default=50)
    s.add_argument("--other", type=int, default=30)
    c = sub.add_parser("compare")
    c.add_argument("--ledger", type=Path, default=DATA / "runs" / "parity.jsonl")
    args = p.parse_args()
    sample(args.edge, args.safe, args.other) if args.cmd == "sample" else compare(args.ledger)


if __name__ == "__main__":
    main()
