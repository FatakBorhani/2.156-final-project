"""Diagnostic for Test 2b: where along seed -> search answer does VMEC++ start failing?

For 6 seeds, re-runs the Test 2b (PCA-direction, small radius) search, then evaluates
x0 + t * (x - x0) for t in T at very low fidelity. t = 0 checks the seed survives the
conversion; failures at small t mean the step direction itself breaks the shape.

    python scripts/feasibility_tests/line_scan.py
    -> results/feasibility_tests/line_scan.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ROOT, load_designs  # noqa: E402
from evaluate import evaluate_designs  # noqa: E402
from search_test import get_ensemble, pick_seeds, search, to_boundary_json  # noqa: E402

T = [0.0, 0.1, 0.25, 0.5, 1.0]
N_SEEDS = 6
OUT = ROOT / "results" / "feasibility_tests"


def main() -> None:
    X, Y, viol, meta = load_designs(3)
    ens = get_ensemble(X, Y)
    pca = PCA(n_components=16, random_state=0).fit(X.values - X.values.mean(axis=0))
    basis = pca.components_ * np.sqrt(pca.explained_variance_)[:, None]
    sigma = X.std().values
    cols = list(X.columns)
    seeds = pick_seeds(viol)
    seeds = pd.concat([seeds[seeds.group == g].head(2) for g in seeds.group.unique()])[:N_SEEDS]

    designs, info = [], {}
    for sid, grp in seeds.group.items():
        x0 = X.loc[sid].values
        x = search(ens, x0, basis, 0.5)
        dz = (x - x0) / sigma
        # how much of the step is in each poloidal mode m
        by_m = {m: float(np.linalg.norm(dz[[f"_m{m}_" in c for c in cols]])) for m in range(5)}
        print(sid, grp, "step by m (std units):", {k: round(v, 2) for k, v in by_m.items()}, flush=True)
        for t in T:
            did = f"{sid}__t{t}"
            designs.append({"id": did, "boundary": to_boundary_json(x0 + t * (x - x0), cols,
                                                                    meta.loc[sid, "boundary_json"])})
            info[did] = {"seed": sid, "group": grp, "t": t, **{f"step_m{m}": v for m, v in by_m.items()}}

    rows = evaluate_designs(designs, OUT / "line_scan_ledger.jsonl", fidelity="very_low_fidelity")
    df = pd.DataFrame([{**info[r["id"]], "ok": r["ok"], "worst": r.get("max_violation"),
                        "error": (r.get("error") or "")[:60]} for r in rows])
    df.to_csv(OUT / "line_scan.csv", index=False)
    print(df.pivot_table(index="seed", columns="t", values="ok", aggfunc="first"))


if __name__ == "__main__":
    main()
