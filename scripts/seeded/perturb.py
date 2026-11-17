"""Propose designs by randomly perturbing public front designs (no ML).

Parents are the non-dominated public NFP-3 4x4 designs, drawn by region of the front:
the compact end (A < 8), the flat gap (A >= 13.8, where no public design beats
L_gradB 17.8 between A 14 and 18.5) and the middle. Each child is

    parent + step * sigma * N(0, I)        (in standard form, shapes.py)

where sigma is each coefficient's spread across all public designs, so `step` is
dimensionless; step 0.1 moves a design ~0.003 (the novelty threshold) on average.

Also writes a few unperturbed parents in standard form (step 0) as a control: their
results should match the leaderboard, showing the standard form doesn't change physics.

    python scripts/seeded/perturb.py data/seeded/pilot.jsonl --n 2400 --seed 0
    python scripts/evaluate.py data/seeded/pilot.jsonl data/seeded/runs/pilot.jsonl --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from families import load, nearest_public  # noqa: E402
from shapes import from_vector, to_vector  # noqa: E402

REGIONS = {"compact": (0.0, 8.0), "middle": (8.0, 13.8), "gap": (13.8, 99.0)}
SHARE = {"compact": 0.35, "gap": 0.35, "middle": 0.30}
STEPS = (0.05, 0.1, 0.2, 0.4)


def front_mask(A: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Non-dominated designs (minimize A, maximize L)."""
    order, best, mask = np.argsort(A), -np.inf, np.zeros(len(A), bool)
    for i in order:
        if L[i] > best:
            mask[i], best = True, L[i]
    return mask


def design(v: np.ndarray, **meta) -> dict:
    b = json.dumps(from_vector(v))
    return {"id": "pert-" + hashlib.sha1(b.encode()).hexdigest()[:12], "boundary": b, **meta}


def propose(n: int, seed: int, steps=STEPS, n_control: int = 16) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = load()
    X, A, L = idx["X"], idx["A"], idx["L"]
    sigma = X.std(axis=0)
    front = np.flatnonzero(front_mask(A, L))
    out = []
    for region, (lo, hi) in REGIONS.items():
        parents = front[(A[front] >= lo) & (A[front] < hi)]
        k = int(round(n * SHARE[region]))
        for j in range(k):
            p = parents[rng.integers(len(parents))]
            step = steps[j % len(steps)]
            v = X[p] + step * sigma * rng.standard_normal(X.shape[1])
            out.append(design(v, parent=str(idx["id"][p]), family=int(idx["family"][p]),
                              region=region, step=step, seed=seed))
    for p in rng.choice(front, n_control, replace=False):
        out.append(design(X[p], parent=str(idx["id"][p]), family=int(idx["family"][p]),
                          region="control", step=0.0, seed=seed))
    novelty = nearest_public(np.array([to_vector(d["boundary"]) for d in out]))
    for d, dist in zip(out, novelty):
        d["novelty"] = float(dist)
    # controls first, the rest shuffled: a run stopped part-way is still balanced
    controls = [d for d in out if d["region"] == "control"]
    rest = [d for d in out if d["region"] != "control"]
    return controls + [rest[i] for i in rng.permutation(len(rest))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=2400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    designs = propose(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for d in designs:
            fh.write(json.dumps(d) + "\n")
    nov = np.array([d["novelty"] for d in designs])
    steps = np.array([d["step"] for d in designs])
    print(f"wrote {len(designs)} designs to {args.out}")
    for s in sorted(set(steps)):
        m = steps == s
        print(f"  step {s:<4}: n={m.sum():4d}  median distance to nearest public "
              f"{np.median(nov[m]):.4f}  novel (>= 0.003): {np.mean(nov[m] >= 0.003):.0%}")


if __name__ == "__main__":
    main()
