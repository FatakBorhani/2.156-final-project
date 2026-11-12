"""Cross-check count_feasible.py with the package's own scoring code.

Runs problems.<Problem>.compute_feasibility on every dataset row's stored metrics for
all three benchmarks (no physics simulation; takes a minute or two), and reports the
distribution of log10(qi), the constraint that most designs fail.

    python scripts/check_feasibility.py        # writes results/official_feasibility.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from constellaration import forward_model, problems

REPO = "proxima-fusion/constellaration"
RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "official_feasibility.json"

files = [
    f
    for f in HfApi().list_repo_files(REPO, repo_type="dataset")
    if f.endswith(".parquet") and f.startswith("data/")
]
df = pd.concat(
    [
        pq.read_table(
            hf_hub_download(REPO, f, repo_type="dataset"),
            columns=["metrics.json", "metrics.qi"],
        ).to_pandas()
        for f in files
    ],
    ignore_index=True,
).dropna(subset=["metrics.json"])

benchmarks = {
    "b1_geometrical": problems.GeometricalProblem(),
    "b2_simple_to_build_qi": problems.SimpleToBuildQIStellarator(),
    "b3_mhd_stable_qi": problems.MHDStableQIStellarator(),
}
violations = {name: [] for name in benchmarks}
for s in df["metrics.json"]:
    m = forward_model.ConstellarationMetrics.model_validate_json(s)
    for name, problem in benchmarks.items():
        try:
            violations[name].append(problem.compute_feasibility(m))
        except (AssertionError, ValueError):  # a metric the benchmark needs is missing
            violations[name].append(np.nan)

out = {}
for name, v in violations.items():
    v = np.array(v)
    out[name] = {
        "evaluated": int(np.isfinite(v).sum()),
        "feasible (max violation <= 0.01)": int((v <= 0.01).sum()),
        "max violation <= 0.1": int((v <= 0.1).sum()),
        "max violation <= 0.25": int((v <= 0.25).sum()),
        "max violation <= 0.5": int((v <= 0.5).sum()),
        "smallest max violation": float(np.nanmin(v)),
    }
log10_qi = np.log10(df["metrics.qi"].dropna().clip(lower=1e-30))
out["log10_qi_quantiles"] = {
    str(q): round(float(log10_qi.quantile(q)), 2) for q in (0.001, 0.01, 0.05, 0.25, 0.5)
}
out["n_with_log10_qi_le_-3.5 (b3 bound)"] = int((log10_qi <= -3.5).sum())
out["n_with_log10_qi_le_-4 (b2 bound)"] = int((log10_qi <= -4).sum())

print(json.dumps(out, indent=2))
OUT.write_text(json.dumps(out, indent=2))
