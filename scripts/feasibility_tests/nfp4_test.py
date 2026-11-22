"""Test 19: a second design family, 4 field periods, from Proxima's dataset only.

All 494 feasible designs from Tests 9-18 have 3 field periods (every test started from the
dataset's nfp-3 designs). By field period, the dataset's nfp-4 designs are farther from
feasible (closest 0.277 vs 0.203 for nfp 3), but its near-feasible ones sit above our front
at the compact end (AR 9.5-9.8 with L_gradB 8.6-9.1, where our front has 7.3 at AR 9.7).
This test bets on that family, with the tools that work now (multi-fidelity finisher,
Fourier modes up to 5 x 5):

1. screen: the K_SCREEN nfp-4 dataset designs with AR <= AR_BOUND and L_gradB >= LGB_POOL
   closest to feasible (dataset metrics), re-checked at high fidelity;
2. approach: the finisher from each of the N_STARTS closest, START_IT iterations each, with a
   wider first search box (BOX_FAR) since they start ~0.3 from feasible;
3. push: up to N_PUSH more rounds of PUSH_IT iterations, each from the closest design found
   so far (any start), with a fresh box and penalties, until one is feasible;
4. descent: Test 18's descent (5 x 5, step 0.125) from the most compact nfp-4 feasible
   design, on nfp-4 designs only.

Stops at the first feasible design in 2-3; a resumed run skips rounds already in the ledger.
Leaderboard designs are not used anywhere.

    python scripts/feasibility_tests/nfp4_test.py            # ~8 h to feasible or give up, then descent
    python scripts/feasibility_tests/nfp4_test.py --dry      # plumbing check
    python scripts/feasibility_tests/nfp4_test.py --report   # re-plot only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import descent_test as ds  # noqa: E402
import ladder_test as lt  # noqa: E402
import multifid_test as mf  # noqa: E402
from front_test import read  # noqa: E402

TAG, NFP = "nfp4", 4
AR_BOUND, LGB_POOL = 10.5, 6.5  # the region where nfp 4 could beat our nfp-3 front
K_SCREEN, N_STARTS, START_IT = 24, 3, 12
N_PUSH, PUSH_IT = 4, 15
BOX_FAR, FAR_GAP = 0.05, 0.10  # first box when the start is more than FAR_GAP from feasible
NFP3_LEDGERS = ["multifid_ledger.jsonl", "front_ledger.jsonl", "chain_ledger.jsonl", "gen_ledger.jsonl",
                "ladder_ledger.jsonl", "compact_ledger.jsonl", "diffusion_ledger.jsonl",
                "descent_ledger.jsonl", "diffusion2_ledger.jsonl", "descent2_ledger.jsonl",
                "modes5_ledger.jsonl"]
DATA, OUT = lt.DATA, lt.OUT

mf.MAX_MODE = 5
lt.TAG, lt.SOURCES = TAG, {}  # descent and report see nfp-4 designs only
ds.STEP0, ds.STEP_MIN = 0.125, 0.06


def candidates(dry: bool) -> list[dict]:
    """Closest-to-feasible nfp-4 dataset designs in the compact, high-L_gradB region."""
    path = DATA / f"{TAG}_candidates{'_dry' if dry else ''}.jsonl"
    if path.exists():
        return read(path)
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    m = "metrics."
    cols = ["plasma_config_id", "boundary.json", "boundary.n_field_periods", m + "aspect_ratio",
            m + "edge_rotational_transform_over_n_field_periods", m + "qi", m + "edge_magnetic_mirror_ratio",
            m + "flux_compression_in_regions_of_bad_curvature", m + "vacuum_well",
            m + "minimum_normalized_magnetic_gradient_scale_length"]
    files = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--proxima-fusion--constellaration/snapshots/*/data/train-*.parquet")))
    df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    df = df.dropna(subset=cols)
    df = df[(df["boundary.n_field_periods"] == NFP) & (df[m + "qi"] > 0)]
    v = np.column_stack([
        (0.25 - df[m + "edge_rotational_transform_over_n_field_periods"].abs()) / 0.25,
        (np.log10(df[m + "qi"]) + 3.5) / 3.5,
        (df[m + "edge_magnetic_mirror_ratio"] - 0.25) / 0.25,
        (df[m + "flux_compression_in_regions_of_bad_curvature"] - 0.9) / 0.9,
        -df[m + "vacuum_well"] / 0.1,
    ])
    df = df.assign(worst=np.clip(v, 0, None).max(axis=1))
    lg = m + "minimum_normalized_magnetic_gradient_scale_length"
    pool = df[(df[m + "aspect_ratio"] <= AR_BOUND) & (df[lg] >= LGB_POOL)].nsmallest(2 if dry else K_SCREEN, "worst")
    out = [{"id": r["plasma_config_id"], "boundary": r["boundary.json"], "worst_dataset": float(r["worst"]),
            "aspect_ratio_dataset": float(r[m + "aspect_ratio"]), "L_gradB_dataset": float(r[lg])}
           for _, r in pool.iterrows()]
    print(f"nfp {NFP}: {len(df)} designs, {len(out)} picked (dataset worst "
          f"{out[0]['worst_dataset']:.3f}-{out[-1]['worst_dataset']:.3f})", flush=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in out))
    return out


def gap(r: dict) -> float:
    return max(r["worst"] if "worst" in r else r["max_violation"], (r["aspect_ratio"] - AR_BOUND) / AR_BOUND)


def finish(name: str, bjson: str, start_gap: float, maxit: int, dry: bool, ledger: Path) -> None:
    mf.B0 = BOX_FAR if start_gap > FAR_GAP else 0.02
    mf.MAXIT = 2 if dry else maxit
    print(f"[{name}] start gap {start_gap:.4f}, box {mf.B0}, {mf.MAXIT} iterations", flush=True)
    mf.run_start(name, bjson, dry, ledger)


def reach_feasible(dry: bool, ledger: Path) -> bool:
    from evaluate import evaluate_designs

    tag = "_dry" if dry else ""
    cands = candidates(dry)
    by_id = {c["id"]: c for c in cands}
    hi = [r for r in evaluate_designs(cands, DATA / f"{TAG}_eval_ledger{tag}.jsonl") if r["ok"]]
    hi.sort(key=gap)
    print("screen (high fidelity):", [(round(gap(r), 3), round(r["aspect_ratio"], 2), round(r["L_gradB"], 2))
                                      for r in hi[:8]], flush=True)
    mf.AR_BOUND, mf.AR_FEAS_TOL, mf.LGB_MIN = AR_BOUND, None, None
    mf.W_OBJ, mf.STOP_ON_FEASIBLE = 0.05, True

    def feasible() -> bool:
        return any(r.get("feasible") for r in lt.hi_rows(ledger))

    done = {r["start"] for r in read(ledger)}
    for i, r in enumerate(hi[:1 if dry else N_STARTS]):
        if feasible():
            return True
        name = f"{TAG}_s{i}"
        if name in done:
            print(f"[{name}] already in ledger; skipping", flush=True)
            continue
        finish(name, by_id[r["id"]]["boundary"], gap(r), START_IT, dry, ledger)
    for k in range(0 if dry else N_PUSH):
        if feasible():
            return True
        name = f"{TAG}_push{k}"
        if name in done:
            print(f"[{name}] already in ledger; skipping", flush=True)
            continue
        best = min(lt.hi_rows(ledger), key=gap)
        print(f"[{name}] from {best['start']} (worst {best['worst']:.4f}, AR {best['aspect_ratio']:.2f}, "
              f"L_gradB {best['L_gradB']:.2f})", flush=True)
        finish(name, best["boundary"], gap(best), PUSH_IT, dry, ledger)
    return feasible()


def combined(dry: bool, ledger: Path) -> None:
    """Hypervolume of all our designs, nfp 3 alone and with nfp 4."""
    import numpy as np
    from constellaration import problems

    def feas(paths):
        out = {}
        for p in paths:
            for r in read(p):
                if r["kind"] == "hi" and r.get("feasible"):
                    out[(round(r["aspect_ratio"], 6), round(r["L_gradB"], 6))] = r
        return list(out.values())

    def hv(f):
        return float(problems._hypervolume(np.array([[-p["L_gradB"], p["aspect_ratio"]] for p in f]),
                                           np.array([1.0, 20.0]))) if f else 0.0

    n3, n4 = feas([DATA / s for s in NFP3_LEDGERS]), feas([ledger])
    s = {"hypervolume_nfp3": hv(n3), "hypervolume_nfp3_plus_nfp4": hv(n3 + n4), "hypervolume_nfp4_alone": hv(n4),
         "n_feasible_nfp3": len(n3), "n_feasible_nfp4": len(n4),
         "nfp4_most_compact": min(([round(r["aspect_ratio"], 3), round(r["L_gradB"], 3)] for r in n4), default=None),
         "nfp4_best_L_gradB": max(([round(r["L_gradB"], 3), round(r["aspect_ratio"], 3)] for r in n4), default=None)}
    (OUT / f"{TAG}_combined{'_dry' if dry else ''}.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    ledger = DATA / f"{TAG}_ledger{'_dry' if a.dry else ''}.jsonl"
    ok = any(r.get("feasible") for r in lt.hi_rows(ledger)) if a.report else reach_feasible(a.dry, ledger)
    if ok:
        mf.B0 = 0.02
        lt.RUNGS = ds.descend(a.dry, ledger, run=not a.report)
        lt.report(ledger, a.dry)
    else:
        print(f"=== no feasible nfp-{NFP} design; best gap "
              f"{min(map(gap, lt.hi_rows(ledger)), default=float('nan')):.4f} ===", flush=True)
    combined(a.dry, ledger)


if __name__ == "__main__":
    main()
