"""Test 2b: the same surrogate-guided search as Test 2, but only along real-design directions.

Test 2 let the search move all 81 coefficients independently. Most resulting shapes
were too wrinkly for VMEC++ to solve, and the surrogate was badly over-optimistic on
the rest. Here the only change is the search space: steps are restricted to the top 16
PCA directions of the real designs (raw coefficient units, where 16 directions explain
99.2% of the variation), scaled by each direction's spread. Each seed keeps its own
fine detail; only the step is constrained. Same seeds, same surrogate (cached), same
objective.

    python scripts/feasibility_tests/search_pca_test.py
    -> results/feasibility_tests/search_pca.json, .png, _candidates.csv, _ledger.jsonl
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import load_designs  # noqa: E402
from search_test import finish, get_ensemble, run_search  # noqa: E402

K = 16
RADII = {"small": 0.5, "large": 1.5}  # in units of each PCA direction's std


def main() -> None:
    X, Y, viol, meta = load_designs(3)
    ens = get_ensemble(X, Y)
    pca = PCA(n_components=K, random_state=0).fit(X.values - X.values.mean(axis=0))
    basis = pca.components_ * np.sqrt(pca.explained_variance_)[:, None]  # (K, 81)
    print(f"PCA-{K} explains {pca.explained_variance_ratio_.sum():.4f} of raw variance")
    res, candidates = run_search(ens, X, viol, meta, basis, RADII)
    finish(res, candidates, RADII, "search_pca",
           f"Test 2b: surrogate-guided search along the top {K} PCA directions, checked with VMEC++")


if __name__ == "__main__":
    main()
