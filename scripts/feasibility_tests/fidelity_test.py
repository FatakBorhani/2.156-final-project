"""Test 4: can low-fidelity VMEC++ stand in for high fidelity?

Picks 50 three-field-period designs (the 25 closest to feasible + 25 random valid ones),
evaluates them at very_low_fidelity and low_fidelity, and compares with the dataset's
stored high-fidelity metrics. (We've checked that those match a fresh high-fidelity run
to ~1e-7: results/evaluator_check_4designs.jsonl.)

    python scripts/feasibility_tests/fidelity_test.py
    -> results/feasibility_tests/fidelity.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import CONSTRAINTS, ROOT, TARGETS, load_designs, violations_from_targets  # noqa: E402
from evaluate import evaluate_designs  # noqa: E402

N_CLOSE, N_RANDOM, SEED = 25, 25, 0
FIDELITIES = ["very_low_fidelity", "low_fidelity"]


def targets_from_metrics(m: dict) -> dict:
    return {
        "log10_qi": float(np.log10(m["qi"])),
        "abs_iota": abs(m["edge_rotational_transform_over_n_field_periods"]),
        "mirror": m["edge_magnetic_mirror_ratio"],
        "flux_compression": m["flux_compression_in_regions_of_bad_curvature"],
        "vacuum_well": m["vacuum_well"],
        "L_gradB": m["minimum_normalized_magnetic_gradient_scale_length"],
        "aspect_ratio": m["aspect_ratio"],
    }


def main() -> None:
    _, targets, viol, meta = load_designs(3)
    worst = viol.clip(lower=0).max(axis=1)
    close = worst.nsmallest(N_CLOSE).index
    rest = worst.index.difference(close)
    rand = pd.Index(np.random.default_rng(SEED).choice(rest, N_RANDOM, replace=False))
    ids = close.append(rand)
    designs = [{"id": i, "boundary": meta.loc[i, "boundary_json"]} for i in ids]

    ledger = ROOT / "data" / "feasibility_tests" / "fidelity_ledger.jsonl"
    out = {"designs": {"closest_to_feasible": N_CLOSE, "random": N_RANDOM}}
    for fid in FIDELITIES:
        rows = evaluate_designs(designs, ledger, fidelity=fid)
        ok = [r for r in rows if r["ok"]]
        lo = pd.DataFrame(
            [targets_from_metrics(r["metrics"]) for r in ok], index=[r["id"] for r in ok]
        )
        hi = targets.loc[lo.index]
        v_lo, v_hi = violations_from_targets(lo), viol.loc[lo.index]
        per_metric = {}
        for k in TARGETS:
            err = lo[k] - hi[k]
            per_metric[k] = {
                "spearman_vs_high": round(float(stats.spearmanr(lo[k], hi[k])[0]), 3),
                "mean_abs_error": round(float(err.abs().mean()), 4),
                "max_abs_error": round(float(err.abs().max()), 4),
                "high_fidelity_std_in_sample": round(float(hi[k].std()), 4),
            }
        pass_agree = {
            k: round(float(((v_lo[k] <= 0.01) == (v_hi[k] <= 0.01)).mean()), 3)
            for k in CONSTRAINTS
        }
        w_lo, w_hi = v_lo.clip(lower=0).max(axis=1), v_hi.clip(lower=0).max(axis=1)
        close_ok = [i for i in close if i in lo.index]
        out[fid] = {
            "evaluated": len(rows),
            "failed": len(rows) - len(ok),
            "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 2),
            "median_seconds": round(float(np.median([r["seconds"] for r in rows])), 2),
            "speedup_vs_high_fidelity_~130s": round(
                130 / float(np.mean([r["seconds"] for r in rows])), 1
            ),
            "per_metric": per_metric,
            "pass_fail_agreement_per_constraint": pass_agree,
            "worst_violation": {
                "spearman_all": round(float(stats.spearmanr(w_lo, w_hi)[0]), 3),
                "spearman_25_closest": round(
                    float(stats.spearmanr(w_lo[close_ok], w_hi[close_ok])[0]), 3
                ),
                "mean_abs_error_25_closest": round(
                    float((w_lo[close_ok] - w_hi[close_ok]).abs().mean()), 4
                ),
                "max_abs_error_25_closest": round(
                    float((w_lo[close_ok] - w_hi[close_ok]).abs().max()), 4
                ),
            },
        }
    dest = ROOT / "results" / "feasibility_tests" / "fidelity.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
