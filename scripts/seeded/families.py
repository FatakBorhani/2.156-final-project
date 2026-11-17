"""Group the public NFP-3 4x4 passing designs into families, and measure novelty.

Distances are RMS distances between surfaces in standard form (see shapes.py), in units
of the major radius. Two thresholds, both 0.003 (~3% of the plasma's thickness):

  FAMILY_LINK  designs closer than this are linked; a family is a connected group
               (single linkage), so two families are always > FAMILY_LINK apart.
  NOVELTY      a design of ours counts as novel only if it is at least this far from
               every public design. Public designs sit a median 0.0005 from their nearest
               neighbour and 99% within 0.0019, so this is beyond the public spacing.
  STRICT       0.01 (~10% of thickness), reported as a sensitivity check.

Using one value for both means "accuracy on held-out families" measures accuracy on
novel designs.

    python scripts/seeded/families.py      # -> data/seeded/public_index.npz
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from shapes import to_vector  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "seeded"
INDEX = DATA / "public_index.npz"
FAMILY_LINK = NOVELTY = 0.003
STRICT = 0.01


def build() -> None:
    rows = [json.loads(line) for line in (DATA / "public_p3.jsonl").open()]
    t = [r for r in rows if r["nfp"] == 3 and r["shape"] == [5, 9] and r["feasible"]]
    X = np.array([to_vector(r["boundary"]) for r in t])
    _, keep = np.unique(np.round(X, 9), axis=0, return_index=True)   # drop rescaled copies
    keep = np.sort(keep)
    t, X = [t[i] for i in keep], X[keep]

    graph = NearestNeighbors().fit(X).radius_neighbors_graph(X, radius=FAMILY_LINK)
    n_fam, family = connected_components(graph, directed=False)
    np.savez_compressed(
        INDEX, X=X, family=family,
        id=np.array([r["id"] for r in t]),
        user=np.array([r["sources"][0]["user"] for r in t]),
        A=np.array([r["aspect_ratio"] for r in t]),
        L=np.array([r["L_gradB"] for r in t]),
        violation=np.array([r["max_violation"] for r in t]),
    )
    sizes = np.bincount(family)
    print(f"{len(t)} unique passing NFP-3 4x4 designs, {n_fam} families "
          f"(largest {sizes.max()}, singletons {(sizes == 1).sum()}) -> {INDEX}")


def load() -> dict:
    with np.load(INDEX) as z:
        return {k: z[k] for k in z.files}


_NN = None


def nearest_public(X: np.ndarray) -> np.ndarray:
    """Distance from each row of X (standard-form vectors) to the nearest public design."""
    global _NN
    if _NN is None:
        _NN = NearestNeighbors(n_neighbors=1).fit(load()["X"])
    return _NN.kneighbors(np.atleast_2d(X))[0][:, 0]


if __name__ == "__main__":
    build()
    idx = load()
    print("families per submitter:",
          {u: len(set(idx["family"][idx["user"] == u])) for u, _ in Counter(idx["user"]).most_common()})
