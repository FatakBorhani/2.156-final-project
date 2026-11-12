"""Count how many ConStellaration dataset designs pass Benchmark 3 (MHD-stable QI).

Constraint math copied from constellaration v0.2.6 problems.MHDStableQIStellarator:
normalized violation = (value - bound) / |target|, feasible if every one <= 0.01.

    python scripts/count_feasible.py            # writes results/b3_feasibility.json
                                                # and results/sample_designs.json

Downloads the dataset's default config (~600 MB) into the Hugging Face cache.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from pymoo.indicators import hv

REPO = "proxima-fusion/constellaration"
RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "b3_feasibility.json"
TOL = 1e-2

COLS = [
    "metrics.aspect_ratio",
    "metrics.edge_rotational_transform_over_n_field_periods",
    "metrics.qi",
    "metrics.edge_magnetic_mirror_ratio",
    "metrics.flux_compression_in_regions_of_bad_curvature",
    "metrics.vacuum_well",
    "metrics.max_elongation",
    "metrics.minimum_normalized_magnetic_gradient_scale_length",
    "boundary.n_field_periods",
    "misc.has_neurips_2025_forward_model_error",
    "boundary.json",
]

files = [
    f
    for f in HfApi().list_repo_files(REPO, repo_type="dataset")
    if f.endswith(".parquet") and f.startswith("data/")
]
print("parquet files:", files)
frames = []
for f in files:
    p = hf_hub_download(REPO, f, repo_type="dataset")
    have = set(pq.read_schema(p).names)
    frames.append(pq.read_table(p, columns=[c for c in COLS if c in have]).to_pandas())
df = pd.concat(frames, ignore_index=True)
n_rows = len(df)
m = df.dropna(subset=[c for c in COLS if c.startswith("metrics.")])
m = m[m["metrics.qi"] > 0]
print(f"rows: {n_rows}, rows with all Benchmark 3 metrics: {len(m)}")

iota = np.abs(m["metrics.edge_rotational_transform_over_n_field_periods"])
viol = pd.DataFrame(
    {
        "edge_iota >= 0.25": (0.25 - iota) / 0.25,
        "log10(qi) <= -3.5": (np.log10(m["metrics.qi"]) - (-3.5)) / 3.5,
        "edge_mirror <= 0.25": (m["metrics.edge_magnetic_mirror_ratio"] - 0.25) / 0.25,
        "flux_compression <= 0.9": (
            m["metrics.flux_compression_in_regions_of_bad_curvature"] - 0.9
        )
        / 0.9,
        "vacuum_well >= 0": (0.0 - m["metrics.vacuum_well"]) / 0.1,
    }
)
worst = viol.clip(lower=0).max(axis=1)
feasible = (viol <= TOL).all(axis=1)

summary = {
    "rows_total": n_rows,
    "rows_with_metrics": int(len(m)),
    "feasible": int(feasible.sum()),
    "feasible_pct": round(100 * feasible.mean(), 3),
    "near_feasible_worst_violation_le_0.05": int((worst <= 0.05).sum()),
    "near_feasible_worst_violation_le_0.10": int((worst <= 0.10).sum()),
    "near_feasible_worst_violation_le_0.25": int((worst <= 0.25).sum()),
    "pass_rate_per_constraint_pct": {
        k: round(100 * (v <= TOL).mean(), 2) for k, v in viol.items()
    },
    "fails_only_this_constraint": {
        k: int(((v > TOL) & (viol.drop(columns=k) <= TOL).all(axis=1)).sum())
        for k, v in viol.items()
    },
}

f = m[feasible]
summary["feasible_by_n_field_periods"] = {
    str(k): int(v) for k, v in f["boundary.n_field_periods"].value_counts().items()
}
if len(f):
    X = np.column_stack(
        [
            -f["metrics.minimum_normalized_magnetic_gradient_scale_length"],
            f["metrics.aspect_ratio"],
        ]
    )
    summary["dataset_feasible_set_hypervolume_ref_(1,20)"] = float(
        hv.Hypervolume(ref_point=np.array([1.0, 20.0]))(X)
    )
    summary["feasible_objective_ranges"] = {
        "L_gradB (maximize)": [
            float(f["metrics.minimum_normalized_magnetic_gradient_scale_length"].min()),
            float(f["metrics.minimum_normalized_magnetic_gradient_scale_length"].max()),
        ],
        "aspect_ratio (minimize)": [
            float(f["metrics.aspect_ratio"].min()),
            float(f["metrics.aspect_ratio"].max()),
        ],
    }

print(json.dumps(summary, indent=2))
OUT.write_text(json.dumps(summary, indent=2))

# Save a few designs to re-score with the real evaluator: the best feasible one
# (highest L_gradB) and the most nearly-feasible infeasible one.
picks = {}
if len(f):
    i = f["metrics.minimum_normalized_magnetic_gradient_scale_length"].idxmax()
    picks["feasible_best"] = i
near = worst[~feasible]
if len(near):
    picks["near_feasible"] = near.idxmin()
samples = {
    name: {
        "boundary_json": m.loc[i, "boundary.json"],
        "dataset_metrics": {
            c: float(m.loc[i, c]) for c in COLS if c.startswith("metrics.")
        },
        "dataset_worst_violation": float(worst.loc[i]),
    }
    for name, i in picks.items()
}
OUT.with_name("sample_designs.json").write_text(json.dumps(samples, indent=2))
