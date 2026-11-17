"""Build the screening models' training table from three sources, all NFP 3 at 4x4 modes,
in standard form (shapes.py):

  dataset  ConStellaration simulations: each constraint's normalized violation, L_gradB,
           and whether VMEC++ crashed. Far from the passing region (0 pass).
  public   leaderboard designs: L_gradB and the max violation only (no per-constraint
           values), all passing. The only labels near the front.
  ours     every high-fidelity result in data/seeded/runs/*.jsonl: full labels,
           including crashes and failures near the front. Timeouts are left out (they
           reflect machine load, not the shape); a later file's row for the same id wins.

The dataset part downloads ~0.6 GB into the Hugging Face cache on first use and is
cached in data/seeded/dataset_nfp3.npz; `ours` is re-read every time.

    python scripts/seeded/training_data.py      # -> data/seeded/train.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from families import load as load_public  # noqa: E402
from shapes import N_FREE, N_POL, N_TOR, to_vector  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "seeded"
REPO = "proxima-fusion/constellaration"
CONSTRAINTS = ["edge_iota_ge_0.25", "log10_qi_le_-3.5", "edge_mirror_le_0.25",
               "flux_compression_le_0.9", "vacuum_well_ge_0"]   # same order as evaluate.py


def _is_target(b: dict) -> bool:
    return (b.get("n_field_periods") == 3 and np.shape(b["r_cos"]) == (N_POL, N_TOR)
            and b.get("is_stellarator_symmetric", True))


def dataset_part() -> dict:
    cache = DATA / "dataset_nfp3.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from constellaration import forward_model, problems

    problem = problems.MHDStableQIStellarator()
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.endswith(".parquet") and f.startswith("data/"))
    X, crash, viol, L, A, ids = [], [], [], [], [], []
    n_rows = 0
    for f in files:
        table = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset"),
                              columns=["plasma_config_id", "boundary.json", "metrics.json"])
        for pid, bj, mj in zip(*(table.column(c).to_pylist() for c in table.column_names)):
            n_rows += 1
            if bj is None:          # a few rows carry no boundary
                continue
            b = json.loads(bj)
            if not _is_target(b):
                continue
            X.append(to_vector(b)); ids.append(pid)
            if mj is None:
                crash.append(True); viol.append([np.nan] * 5); L.append(np.nan); A.append(np.nan)
                continue
            m = forward_model.ConstellarationMetrics.model_validate_json(mj)
            try:
                v = problem._normalized_constraint_violations(m)
            except (AssertionError, ValueError, TypeError):
                v = [np.nan] * 5
            crash.append(False); viol.append(list(map(float, v)))
            L.append(m.minimum_normalized_magnetic_gradient_scale_length); A.append(m.aspect_ratio)
    out = {"X": np.array(X), "crash": np.array(crash), "violations": np.array(viol, float),
           "L": np.array(L, float), "A": np.array(A, float), "id": np.array(ids)}
    np.savez_compressed(cache, **out)
    print(f"dataset: {n_rows} rows, {len(X)} at NFP 3 / 4x4 ({int(out['crash'].sum())} crashed) -> {cache}")
    return out


def ours_part() -> dict:
    boundaries = {}
    for f in DATA.glob("*.jsonl"):
        for line in f.open():
            d = json.loads(line)
            if "boundary" in d:
                b = json.loads(d["boundary"]) if isinstance(d["boundary"], str) else d["boundary"]
                if _is_target(b):
                    boundaries[d["id"]] = b
    rows = []
    for f in sorted((DATA / "runs").glob("*.jsonl")):
        # a timeout says the machine was busy (e.g. Game Mode), not that the shape crashes
        rows += [r for r in map(json.loads, f.open())
                 if r["fidelity"] == "high_fidelity" and r["id"] in boundaries
                 and not r.get("timeout")]
    rows = list({r["id"]: r for r in rows}.values())
    X = np.array([to_vector(boundaries[r["id"]]) for r in rows]).reshape(-1, N_FREE)
    return {
        "X": X, "id": np.array([r["id"] for r in rows]),
        "crash": np.array([not r["ok"] for r in rows]),
        "violations": np.array([[r["violations"][c] for c in CONSTRAINTS] if r["ok"] else [np.nan] * 5
                                for r in rows], float).reshape(-1, 5),
        "L": np.array([r.get("L_gradB", np.nan) for r in rows], float),
        "A": np.array([r.get("aspect_ratio", np.nan) for r in rows], float),
    }


def build() -> dict:
    ds, ours, pub = dataset_part(), ours_part(), load_public()
    parts = {
        "dataset": dict(ds, max_violation=np.nanmax(np.maximum(ds["violations"], 0), axis=1)
                        if len(ds["X"]) else np.zeros(0)),
        "ours": dict(ours, max_violation=np.nanmax(np.maximum(ours["violations"], 0), axis=1)
                     if len(ours["X"]) else np.zeros(0)),
        "public": {"X": pub["X"], "id": pub["id"], "crash": np.zeros(len(pub["X"]), bool),
                   "violations": np.full((len(pub["X"]), 5), np.nan), "L": pub["L"],
                   "A": pub["A"], "max_violation": pub["violation"]},
    }
    keys = ["X", "id", "crash", "violations", "L", "A", "max_violation"]
    table = {k: np.concatenate([parts[p][k] for p in parts]) for k in keys}
    table["source"] = np.concatenate([[p] * len(parts[p]["X"]) for p in parts])
    table["family"] = np.full(len(table["X"]), -1)
    table["family"][table["source"] == "public"] = pub["family"]
    np.savez_compressed(DATA / "train.npz", **table)
    for p in parts:
        n, c = len(parts[p]["X"]), int(np.sum(parts[p]["crash"]))
        ok = ~parts[p]["crash"]
        npass = int(np.sum(parts[p]["max_violation"][ok] <= 0.01))
        print(f"  {p:8s}: {n:6d} designs, {c:5d} crashed, {npass:5d} pass")
    return table


if __name__ == "__main__":
    build()
    print(f"-> {DATA / 'train.npz'}")
