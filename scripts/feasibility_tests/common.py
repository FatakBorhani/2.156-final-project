"""Shared data loading for the plan feasibility tests.

load_designs(nfp=3) returns one row per *valid* dataset design with that number of field
periods:
  - X: the 80 boundary Fourier coefficients the design is made of (inputs)
  - targets: the 7 Benchmark 3 metrics (outputs), with QI on a log10 scale
  - violations: the 5 normalized constraint violations (<= 0.01 passes), same formulas
    as constellaration v0.2.6 problems.MHDStableQIStellarator
Cached to data/designs_nfp{nfp}.parquet after the first call.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
REPO = "proxima-fusion/constellaration"
M = "metrics."

TARGETS = {
    "log10_qi": "QI residual, log10 (limit <= -3.5)",
    "abs_iota": "|edge twist| per field period (limit >= 0.25)",
    "mirror": "edge mirror ratio (limit <= 0.25)",
    "flux_compression": "flux compression in bad curvature (limit <= 0.9)",
    "vacuum_well": "vacuum well (limit >= 0)",
    "L_gradB": "L_gradB, coil simplicity (goal: maximize)",
    "aspect_ratio": "aspect ratio (goal: minimize)",
}
CONSTRAINTS = ["log10_qi", "abs_iota", "mirror", "flux_compression", "vacuum_well"]


def violations_from_targets(t: pd.DataFrame) -> pd.DataFrame:
    """Normalized violation per constraint; <= 0.01 passes (problems.py tolerance)."""
    return pd.DataFrame(
        {
            "log10_qi": (t["log10_qi"] + 3.5) / 3.5,
            "abs_iota": (0.25 - t["abs_iota"]) / 0.25,
            "mirror": (t["mirror"] - 0.25) / 0.25,
            "flux_compression": (t["flux_compression"] - 0.9) / 0.9,
            "vacuum_well": (0.0 - t["vacuum_well"]) / 0.1,
        },
        index=t.index,
    )


def _download() -> pd.DataFrame:
    files = [
        f
        for f in HfApi().list_repo_files(REPO, repo_type="dataset")
        if f.endswith(".parquet") and f.startswith("data/")
    ]
    cols = [
        "plasma_config_id",
        "boundary.json",
        "boundary.r_cos",
        "boundary.z_sin",
        "boundary.n_field_periods",
        M + "qi",
        M + "edge_rotational_transform_over_n_field_periods",
        M + "edge_magnetic_mirror_ratio",
        M + "flux_compression_in_regions_of_bad_curvature",
        M + "vacuum_well",
        M + "minimum_normalized_magnetic_gradient_scale_length",
        M + "aspect_ratio",
    ]
    return pd.concat(
        [
            pq.read_table(hf_hub_download(REPO, f, repo_type="dataset"), columns=cols)
            .to_pandas()
            for f in files
        ],
        ignore_index=True,
    )


def load_designs(nfp: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (X, targets, violations, meta), row-aligned, index = plasma_config_id."""
    cache = ROOT / "data" / f"designs_nfp{nfp}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        raw = _download()
        raw = raw[(raw["boundary.n_field_periods"] == nfp) & (raw[M + "qi"] > 0)]
        raw = raw.dropna(subset=[M + "vacuum_well", M + "aspect_ratio"])
        r = np.stack([np.stack(list(a)) for a in raw["boundary.r_cos"]])  # (n, 5, 9)
        z = np.stack([np.stack(list(a)) for a in raw["boundary.z_sin"]])
        coeffs = np.concatenate([r.reshape(len(r), -1), z.reshape(len(z), -1)], axis=1)
        names = [f"r_cos_m{m}_n{n - 4}" for m in range(5) for n in range(9)] + [
            f"z_sin_m{m}_n{n - 4}" for m in range(5) for n in range(9)
        ]
        df = pd.DataFrame(coeffs, columns=names, index=raw["plasma_config_id"].values)
        df = df.loc[:, df.std() > 0]  # drop the entries that are always 0 (or R00 = 1)
        df["t_log10_qi"] = np.log10(raw[M + "qi"].values)
        df["t_abs_iota"] = np.abs(
            raw[M + "edge_rotational_transform_over_n_field_periods"].values
        )
        df["t_mirror"] = raw[M + "edge_magnetic_mirror_ratio"].values
        df["t_flux_compression"] = raw[
            M + "flux_compression_in_regions_of_bad_curvature"
        ].values
        df["t_vacuum_well"] = raw[M + "vacuum_well"].values
        df["t_L_gradB"] = raw[
            M + "minimum_normalized_magnetic_gradient_scale_length"
        ].values
        df["t_aspect_ratio"] = raw[M + "aspect_ratio"].values
        df["boundary_json"] = raw["boundary.json"].values
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)

    X = df[[c for c in df.columns if c.startswith(("r_cos", "z_sin"))]]
    targets = df[[f"t_{k}" for k in TARGETS]].rename(columns=lambda c: c[2:])
    violations = violations_from_targets(targets)
    meta = df[["boundary_json"]]
    return X, targets, violations, meta
