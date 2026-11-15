"""Propose designs by moving *along* the surface the passing designs lie on (no ML model).

The pilot showed random perturbations fail almost at once: passing designs occupy a thin
surface in the 80-coefficient space, and a random direction leaves it. Three ways to
stay on it, all built from public front designs in standard form (shapes.py):

  pca      step along the directions in which the K nearest public designs vary
           (local PCA), sized to land a target distance d from the parent
  blend    mix two front designs from different families with similar aspect ratio:
           (1 - t) * a + t * b, t in [0.3, 0.7]
  extend   continue the front past a parent: p + t * (p - q), q a nearby front design
           on the opposite side in aspect ratio; aimed at the two openings
           (compact end, and the flat gap A 14-18.5)

In the first surface batch, blends were the only method that produced novel passing
designs, some beyond the public front. `blend` is the blend-only proposer loop.py uses.

    python scripts/seeded/surface.py data/seeded/surface.jsonl --n 960 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from families import load, nearest_public  # noqa: E402
from perturb import REGIONS, SHARE, design, front_mask  # noqa: E402
from shapes import aspect_ratio, to_vector  # noqa: E402

K = 20
PCA_RANK = 5
PCA_DIST = (0.003, 0.006, 0.012)
EXTEND_T = (0.5, 1.0, 2.0)


def _pick_region(rng, front, A) -> tuple[str, np.ndarray]:
    region = rng.choice(list(SHARE), p=list(SHARE.values()))
    lo, hi = REGIONS[region]
    return region, front[(A[front] >= lo) & (A[front] < hi)]


def propose(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = load()
    X, A, L, fam, ids = idx["X"], idx["A"], idx["L"], idx["family"], idx["id"]
    front = np.flatnonzero(front_mask(A, L))
    knn = NearestNeighbors(n_neighbors=K + 1).fit(X)
    out = []
    n_pca, n_blend = n // 2, n // 4
    n_extend = n - n_pca - n_blend

    for j in range(n_pca):
        region, parents = _pick_region(rng, front, A)
        p = parents[rng.integers(len(parents))]
        nb = knn.kneighbors(X[p:p + 1])[1][0][1:]
        D = X[nb] - X[nb].mean(axis=0)
        _, s, Vt = np.linalg.svd(D, full_matrices=False)
        coef = rng.standard_normal(PCA_RANK) * s[:PCA_RANK]
        step = coef @ Vt[:PCA_RANK]
        d = PCA_DIST[j % len(PCA_DIST)]
        v = X[p] + d * step / np.linalg.norm(step)
        out.append(design(v, method="pca", parent=str(ids[p]), family=int(fam[p]),
                          region=region, step=d, seed=seed))

    out += _blends(rng, X, A, fam, ids, front, n_blend, seed)

    order = front[np.argsort(A[front])]
    for j in range(n_extend):
        region, parents = _pick_region(rng, front, A)
        p = parents[rng.integers(len(parents))]
        k = int(np.flatnonzero(order == p)[0])
        # compact end extends toward lower A, the gap toward higher L; middle either way
        side = -1 if region == "compact" else (1 if region == "gap" else rng.choice([-1, 1]))
        q_pos = int(np.clip(k - side * rng.integers(1, 4), 0, len(order) - 1))
        q = order[q_pos] if q_pos != k else order[int(np.clip(k - side, 0, len(order) - 1))]
        t = EXTEND_T[j % len(EXTEND_T)]
        out.append(design(X[p] + t * (X[p] - X[q]), method="extend", parent=str(ids[p]),
                          parent_b=str(ids[q]), family=int(fam[p]), region=region,
                          step=t, seed=seed))
    return _finish(out, rng)


def _blends(rng, X, A, fam, ids, front, n, seed) -> list[dict]:
    out = []
    for _ in range(n):
        region, parents = _pick_region(rng, front, A)
        a = parents[rng.integers(len(parents))]
        others = front[(fam[front] != fam[a]) & (np.abs(A[front] - A[a]) < 1.0)]
        if len(others) == 0:
            others = front[fam[front] != fam[a]]
        b = others[np.argmin(np.abs(A[others] - A[a]))] if len(others) < 3 else others[rng.integers(len(others))]
        t = rng.uniform(0.3, 0.7)
        out.append(design((1 - t) * X[a] + t * X[b], method="blend", parent=str(ids[a]),
                          parent_b=str(ids[b]), family=int(fam[a]), region=region,
                          step=round(float(t), 3), seed=seed))
    return out


def _finish(out: list[dict], rng) -> list[dict]:
    V = np.array([to_vector(d["boundary"]) for d in out])
    # aspect ratio is exact geometry, not a guess: past the reference A = 20 nothing scores
    for d, dist, a in zip(out, nearest_public(V), aspect_ratio(V)):
        d["novelty"], d["A_geom"] = float(dist), float(a)
    out = [d for d in out if d["A_geom"] < 19.9]
    return [out[i] for i in rng.permutation(len(out))]


def blend(n: int, seed: int, ours: dict | None = None) -> list[dict]:
    """Blends only, from the *current* front: public designs plus `ours` (our novel passing
    designs: arrays X, A, L, id). Our designs join as one-member families, so a new design
    can be blended again with a public one: crossover that compounds across rounds."""
    rng = np.random.default_rng(seed)
    idx = load()
    X, A, L, fam, ids = idx["X"], idx["A"], idx["L"], idx["family"], idx["id"].astype(str)
    if ours is not None and len(ours["X"]):
        k = len(ours["X"])
        X, A, L = np.r_[X, ours["X"]], np.r_[A, ours["A"]], np.r_[L, ours["L"]]
        fam, ids = np.r_[fam, -1 - np.arange(k)], np.r_[ids, ours["id"].astype(str)]
    front = np.flatnonzero(front_mask(A, L))
    return _finish(_blends(rng, X, A, fam, ids, front, n, seed), rng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=960)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    designs = list({d["id"]: d for d in propose(args.n, args.seed)}.values())
    with args.out.open("w") as fh:
        for d in designs:
            fh.write(json.dumps(d) + "\n")
    print(f"wrote {len(designs)} designs to {args.out}")
    for m in ("pca", "blend", "extend"):
        ds = [d for d in designs if d["method"] == m]
        nov = np.array([d["novelty"] for d in ds])
        print(f"  {m:6s}: n={len(ds):4d}  median distance to nearest public {np.median(nov):.4f}  "
              f"novel (>= 0.003): {np.mean(nov >= 0.003):.0%}")


if __name__ == "__main__":
    main()
