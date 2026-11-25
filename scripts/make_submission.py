"""Build a Benchmark 3 (mhd_stable) leaderboard submission from Track A's feasible designs, and
check it the way the leaderboard Space does.

The Space (proxima-fusion/constellaration-bench, constellaration 0.2.6) takes a JSON list of
boundary JSON strings and runs MHDStableQIStellarator().evaluate on it: high-fidelity VMEC++ for
each boundary, one at a time, then the hypervolume of the feasible ones. Only the Pareto front
counts, so we submit the front, after dropping designs whose worst violation is above 0.009 (a
buffer against small numerical differences on their machine; it costs 0.01 of hypervolume).

    python scripts/make_submission.py            # write the file
    python scripts/make_submission.py --verify   # also re-score it locally, as the Space does

Uses only our own Track A ledgers. Writes results/submissions/benchmark3_track_a.json and, with
--verify, results/submissions/benchmark3_track_a_verify.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent import futures
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "feasibility_tests"), str(HERE)]
import resolution_check as rc  # noqa: E402

OUT = rc.ROOT / "results" / "submissions"
MARGIN = 0.009


def front() -> list[dict]:
    out, best = [], -np.inf
    for d in rc.designs():  # sorted by AR, then L_gradB descending
        if d["worst"] <= MARGIN and d["L_gradB"] > best:
            out.append(d)
            best = d["L_gradB"]
    return out


def score_one(boundary_json: str):
    """What the Space's MHDStableQIStellarator.evaluate does for one boundary."""
    from constellaration import forward_model, problems
    from constellaration.geometry import surface_rz_fourier

    b = surface_rz_fourier.SurfaceRZFourier.model_validate_json(boundary_json)
    m, _ = forward_model.forward_model(
        boundary=b, settings=forward_model.ConstellarationSettings.default_high_fidelity())
    p = problems.MHDStableQIStellarator()
    return m, bool(p.is_feasible(m))


def verify(path: Path, ds: list[dict], workers: int) -> None:
    from constellaration import problems

    boundaries = json.loads(path.read_text())
    with futures.ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver")) as ex:
        results = list(ex.map(score_one, boundaries))
    metrics = [m for m, _ in results]
    score = problems.MHDStableQIStellarator()._score(metrics)
    rows = [dict(id=d["id"], feasible=f, L_gradB=m.minimum_normalized_magnetic_gradient_scale_length,
                 aspect_ratio=m.aspect_ratio, ledger_L_gradB=d["L_gradB"])
            for d, (m, f) in zip(ds, results)]
    lost = [r["id"] for r in rows if not r["feasible"]]
    print(f"local score {score:.2f}; {len(rows) - len(lost)}/{len(rows)} feasible; lost {lost}")
    (OUT / f"{path.stem}_verify.json").write_text(json.dumps(dict(score=score, rows=rows), indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    ds = front()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "benchmark3_track_a.json"
    path.write_text(json.dumps([d["boundary"] for d in ds]))
    print(f"wrote {len(ds)} boundaries to {path}", flush=True)
    if a.verify:
        verify(path, ds, a.workers)


if __name__ == "__main__":
    main()
