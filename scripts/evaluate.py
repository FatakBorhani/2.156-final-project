"""Evaluate stellarator boundaries against Benchmark 3 (MHD-stable QI), in parallel.

This is the project's single source of truth for physics evaluations: every VMEC++ run
we count toward "evaluations spent" goes through here, and every result is appended to
a JSONL ledger as soon as it finishes (so a crash or Ctrl-C loses nothing, and re-running
skips designs already in the ledger).

Input: a JSONL file, one design per line: {"id": "...", "boundary": <SurfaceRZFourier JSON>}
(scripts/export_designs.py writes this format from the dataset; a generator should too).

    python scripts/evaluate.py designs.jsonl results/runs/my_run.jsonl
    python scripts/evaluate.py designs.jsonl out.jsonl --workers 8 --fidelity low_fidelity

From Python (e.g. a notebook):

    from evaluate import evaluate_designs, summarize
    rows = evaluate_designs(designs, "results/runs/x.jsonl", workers=8)
    print(summarize(rows))

The official Benchmark 3 score uses high fidelity (the default). Lower fidelities are
only for timing comparisons with the ALM-NGOpt baseline, which optimizes at
very_low_fidelity; their feasibility flags are not leaderboard-comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent import futures
from pathlib import Path

import numpy as np

CONSTRAINTS = [
    "edge_iota_ge_0.25",
    "log10_qi_le_-3.5",
    "edge_mirror_le_0.25",
    "flux_compression_le_0.9",
    "vacuum_well_ge_0",
]
FIDELITIES = ("high_fidelity", "low_fidelity", "very_low_fidelity")
HV_REFERENCE = (1.0, 20.0)  # (-L_gradB, aspect_ratio), from problems.py


def _evaluate_one(design: dict, fidelity: str) -> dict:
    """Run the forward model on one design. Runs in a worker process."""
    from constellaration import forward_model, problems
    from constellaration.geometry import surface_rz_fourier
    from constellaration.mhd import vmec_settings

    row = {"id": design["id"], "fidelity": fidelity}
    t0 = time.perf_counter()
    try:
        boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(
            design["boundary"]
            if isinstance(design["boundary"], str)
            else json.dumps(design["boundary"])
        )
        if fidelity == "high_fidelity":
            settings = forward_model.ConstellarationSettings.default_high_fidelity()
        else:
            settings = forward_model.ConstellarationSettings(
                vmec_preset_settings=vmec_settings.VmecPresetSettings(
                    fidelity=fidelity
                )
            )
        metrics, _ = forward_model.forward_model(boundary, settings=settings)
        problem = problems.MHDStableQIStellarator()
        violations = problem._normalized_constraint_violations(metrics)
        row.update(
            ok=True,
            feasible=bool(problem.is_feasible(metrics)),
            max_violation=float(np.max(np.maximum(violations, 0.0))),
            violations=dict(zip(CONSTRAINTS, map(float, violations))),
            L_gradB=float(metrics.minimum_normalized_magnetic_gradient_scale_length),
            aspect_ratio=float(metrics.aspect_ratio),
            metrics=json.loads(metrics.model_dump_json()),
        )
    except Exception as e:  # VMEC++ not converging on a bad shape is expected
        row.update(ok=False, feasible=False, error=f"{type(e).__name__}: {e}")
        row["traceback"] = traceback.format_exc(limit=3)
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


def default_workers() -> int:
    """Leave 2 cores free and budget ~4 GB of RAM per high-fidelity evaluation."""
    cpus = os.cpu_count() or 2
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        ram_gb = 16
    return max(1, min(cpus - 2, int(ram_gb // 4)))


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate_designs(
    designs: list[dict],
    ledger: str | Path,
    workers: int | None = None,
    fidelity: str = "high_fidelity",
) -> list[dict]:
    """Evaluate designs not already in the ledger; return all ledger rows for them."""
    assert fidelity in FIDELITIES, fidelity
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["fidelity"]) for r in read_jsonl(ledger)}
    todo = [d for d in designs if (d["id"], fidelity) not in done]
    workers = workers or default_workers()
    print(f"{len(designs) - len(todo)} already in {ledger}; evaluating {len(todo)} "
          f"with {workers} workers at {fidelity}", flush=True)

    t0 = time.perf_counter()
    with futures.ProcessPoolExecutor(max_workers=workers) as pool, ledger.open("a") as f:
        jobs = [pool.submit(_evaluate_one, d, fidelity) for d in todo]
        for i, job in enumerate(futures.as_completed(jobs), 1):
            row = job.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            status = (
                "FEASIBLE" if row["feasible"]
                else f"max violation {row['max_violation']:.3f}" if row["ok"]
                else "FAILED"
            )
            print(f"[{i}/{len(todo)}] {row['id']}: {status} ({row['seconds']:.0f} s)",
                  flush=True)
    if todo:
        print(f"wall time {time.perf_counter() - t0:.0f} s", flush=True)

    ids = {d["id"] for d in designs}
    return [r for r in read_jsonl(ledger) if r["id"] in ids and r["fidelity"] == fidelity]


def summarize(rows: list[dict]) -> dict:
    """Counts, compute spent, and the Benchmark 3 hypervolume of the feasible set."""
    from pymoo.indicators import hv

    ok = [r for r in rows if r["ok"]]
    feasible = [r for r in ok if r["feasible"]]
    hypervolume = 0.0
    if feasible:
        X = np.array([(-r["L_gradB"], r["aspect_ratio"]) for r in feasible])
        hypervolume = float(hv.Hypervolume(ref_point=np.array(HV_REFERENCE))(X))
    return {
        "evaluated": len(rows),
        "failed": len(rows) - len(ok),
        "feasible": len(feasible),
        "best_max_violation": min((r["max_violation"] for r in ok), default=None),
        "cpu_hours": round(sum(r["seconds"] for r in rows) / 3600, 3),
        "mean_seconds_per_eval": round(float(np.mean([r["seconds"] for r in rows])), 1)
        if rows else None,
        "hypervolume": hypervolume,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("designs", help="input JSONL: {id, boundary} per line")
    p.add_argument("ledger", help="output JSONL; appended to, never overwritten")
    p.add_argument("--workers", type=int, default=None,
                   help=f"default: {default_workers()} on this machine")
    p.add_argument("--fidelity", choices=FIDELITIES, default="high_fidelity")
    p.add_argument("--limit", type=int, default=None, help="only the first N designs")
    args = p.parse_args()

    designs = read_jsonl(args.designs)[: args.limit]
    rows = evaluate_designs(designs, args.ledger, args.workers, args.fidelity)
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
