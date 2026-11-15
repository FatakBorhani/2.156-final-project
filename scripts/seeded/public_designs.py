"""Build a table of the Benchmark 3 designs other teams posted to the public leaderboard.

Downloads every scored Benchmark 3 ("mhd_stable") submission from the Hugging Face
dataset proxima-fusion/constellaration-bench-results (~113 MB, into the normal Hugging
Face cache), then writes one line per unique boundary in the evaluator's input format
plus the leaderboard's own numbers for it:

    {"id", "boundary", "nfp", "shape", "L_gradB", "aspect_ratio", "max_violation",
     "feasible", "sources": [{"user", "submitted", "file"}, ...]}

`feasible` uses the benchmark's rule: max normalized violation <= 0.01. The same boundary
often appears in several submissions (teams re-submit each other's designs); it is kept
once, with every submission listed under `sources`, so credit stays traceable.

Also writes the paper's ALM-NGOpt baseline submission (scadena-pf, 2025-07-02, score
129.80) on its own, for checking that our evaluator reproduces the leaderboard.

    python scripts/seeded/public_designs.py            # writes data/seeded/*.jsonl

Credit: these designs belong to the leaderboard submitters listed in `sources`.
The results dataset has no license; cite it and the submitters in anything we publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "proxima-fusion/constellaration-bench-results"
BASELINE_FILE = "2025-07-02T07-38-21.817193_mhd_stable_results.json"
FEASIBLE_TOL = 0.01  # problems.py _DEFAULT_RELATIVE_TOLERANCE
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "seeded"


def _loads(x):
    """Some fields are stored as JSON strings, some as JSON values."""
    return json.loads(x) if isinstance(x, str) else x


def design_id(boundary_json: str) -> str:
    return "pub-" + hashlib.sha1(boundary_json.encode()).hexdigest()[:12]


def parse_submission(path: Path) -> list[dict]:
    """One row per boundary in a scored submission; [] if it was never scored."""
    sub = json.loads(path.read_text())
    if not sub.get("evaluated") or sub.get("objectives") is None:
        return []
    boundaries = [b if isinstance(b, str) else json.dumps(b)
                  for b in _loads(sub["boundary_json"])]
    objectives = _loads(sub["objectives"])
    violations = _loads(sub["feasibilities"])
    if not (len(boundaries) == len(objectives) == len(violations)):
        raise ValueError(f"{path.name}: boundary/objective/violation counts differ")

    rows = []
    for b, obj, viol in zip(boundaries, objectives, violations):
        parsed = json.loads(b)
        (l_gradb, _), (aspect, _) = obj
        rows.append({
            "id": design_id(b),
            "boundary": b,
            "nfp": parsed["n_field_periods"],
            "shape": [len(parsed["r_cos"]), len(parsed["r_cos"][0])],
            "L_gradB": float(l_gradb),
            "aspect_ratio": float(aspect),
            "max_violation": float(viol),
            "feasible": float(viol) <= FEASIBLE_TOL,
            "sources": [{"user": sub["user"], "submitted": sub["submission_time"],
                         "file": path.name}],
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()

    root = Path(snapshot_download(REPO, repo_type="dataset",
                                  allow_patterns=["mhd_stable/*.json"]))
    files = sorted((root / "mhd_stable").glob("*.json"))

    unique: dict[str, dict] = {}
    n_rows, n_unscored = 0, 0
    for f in files:
        rows = parse_submission(f)
        n_unscored += not rows
        for r in rows:
            n_rows += 1
            if r["id"] in unique:
                unique[r["id"]]["sources"] += r["sources"]
            else:
                unique[r["id"]] = r
    designs = list(unique.values())

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "public_p3.jsonl").open("w") as fh:
        for d in designs:
            fh.write(json.dumps(d) + "\n")
    baseline = parse_submission(root / "mhd_stable" / BASELINE_FILE)
    with (args.out / "paper_baseline.jsonl").open("w") as fh:
        for d in baseline:
            fh.write(json.dumps(d) + "\n")

    feasible = [d for d in designs if d["feasible"]]
    target = [d for d in feasible if d["nfp"] == 3 and d["shape"] == [5, 9]]
    print(f"{len(files)} submission files ({n_unscored} never scored), "
          f"{n_rows} design entries, {len(designs)} unique boundaries")
    print(f"feasible (max violation <= {FEASIBLE_TOL}): {len(feasible)}")
    print("feasible by field periods:", dict(sorted(Counter(d['nfp'] for d in feasible).items())))
    print("feasible by array shape:",
          dict(Counter(tuple(d['shape']) for d in feasible).most_common()))
    print(f"our target set (3 field periods, 4x4 modes = 5x9 arrays): {len(target)}")
    print("target set by submitter (first submission of each design):",
          dict(Counter(d['sources'][0]['user'] for d in target).most_common()))
    print(f"wrote {args.out / 'public_p3.jsonl'} and {args.out / 'paper_baseline.jsonl'} "
          f"({len(baseline)} baseline designs)")


if __name__ == "__main__":
    main()
