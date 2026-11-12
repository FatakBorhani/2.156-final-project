"""Export dataset designs to the JSONL format scripts/evaluate.py reads.

Default: the N designs closest to Benchmark 3 feasibility (smallest max normalized
constraint violation, using the dataset's stored metrics). Useful as a seed set for
search, and as known-answer inputs for testing the evaluator.

    python scripts/export_designs.py data/near_feasible_100.jsonl --n 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from constellaration import forward_model, problems

REPO = "proxima-fusion/constellaration"


def load_rows() -> pd.DataFrame:
    files = [
        f
        for f in HfApi().list_repo_files(REPO, repo_type="dataset")
        if f.endswith(".parquet") and f.startswith("data/")
    ]
    cols = ["plasma_config_id", "boundary.json", "metrics.json"]
    df = pd.concat(
        [
            pq.read_table(hf_hub_download(REPO, f, repo_type="dataset"), columns=cols)
            .to_pandas()
            for f in files
        ],
        ignore_index=True,
    )
    return df.dropna(subset=["metrics.json"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()

    df = load_rows()
    problem = problems.MHDStableQIStellarator()
    violation = np.array(
        [
            problem.compute_feasibility(
                forward_model.ConstellarationMetrics.model_validate_json(s)
            )
            for s in df["metrics.json"]
        ]
    )
    order = np.argsort(violation)[: args.n]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i in order:
            f.write(
                json.dumps(
                    {
                        "id": df.loc[i, "plasma_config_id"],
                        "boundary": df.loc[i, "boundary.json"],
                        "dataset_max_violation": float(violation[i]),
                    }
                )
                + "\n"
            )
    print(f"wrote {len(order)} designs to {args.out} "
          f"(max violation {violation[order[0]]:.3f} to {violation[order[-1]]:.3f})")


if __name__ == "__main__":
    main()
