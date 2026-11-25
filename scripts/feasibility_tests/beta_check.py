"""Finite-beta check: do Track A's front designs keep their properties with plasma pressure?

The benchmark scores vacuum equilibria (essentially zero pressure, zero current). Here the same
fixed boundaries are solved with VMEC++ at a volume-averaged beta of 1-3% (linear pressure
profile p ~ 1 - s, zero net toroidal current, as is usual for QI), using the benchmark's own
helper to set the pressure and its "high_fidelity" settings. We then recompute the benchmark
metrics (the "vacuum well" becomes the magnetic well with pressure) and add VMEC's Mercier
criterion (DMerc > 0 is stable to local interchange modes), judged on 0.25 <= s <= 0.95
because VMEC's value is numerical noise near the axis.

This does not change the benchmark score; it is a physics check for the report.

    python scripts/feasibility_tests/beta_check.py --ids 3e8736083d,025a48ab59 --workers 6
    python scripts/feasibility_tests/beta_check.py --report

Reads only our own Track A ledgers. Writes data/feasibility_tests/beta_ledger.jsonl (resumable)
and results/feasibility_tests/beta_check.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent import futures
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import resolution_check as rc  # noqa: E402

LEDGER = rc.DATA / "beta_ledger.jsonl"
BETAS = [0.0, 0.01, 0.02, 0.03]


def evaluate(design: dict, beta: float) -> dict:
    """Solve one boundary at one beta and score it. Runs in a worker process."""
    from constellaration import forward_model, problems
    from constellaration.geometry import surface_rz_fourier
    from constellaration.mhd import ideal_mhd_parameters_utils as ipu

    row = {"id": design["id"], "beta_target": beta}
    t0 = time.perf_counter()
    try:
        b = surface_rz_fourier.SurfaceRZFourier.model_validate_json(design["boundary"])
        params = ipu.get_ideal_mhd_parameters_for_volume_averaged_beta(b, beta) if beta else None
        metrics, eq = forward_model.forward_model(
            b, ideal_mhd_parameters=params,
            settings=forward_model.ConstellarationSettings.default_high_fidelity())
        problem = problems.MHDStableQIStellarator()
        viol = problem._normalized_constraint_violations(metrics)
        # VMEC's Mercier output is numerical noise near the axis (its geodesic term grows like 1/s
        # there, even at zero pressure), so judge stability on 0.25 <= s <= 0.95 only. The full
        # profiles are kept so the window can be revisited.
        s_full = np.linspace(0, 1, eq.ns)
        merc = np.asarray(eq.DMerc)[(s_full >= 0.25) & (s_full <= 0.95)]
        row.update(ok=True, beta=float(eq.betatotal), feasible=bool(problem.is_feasible(metrics)),
                   worst=float(np.max(np.maximum(viol, 0.0))),
                   violations=dict(zip(rc.CONSTRAINTS, map(float, viol))),
                   L_gradB=float(metrics.minimum_normalized_magnetic_gradient_scale_length),
                   aspect_ratio=float(metrics.aspect_ratio), well=float(metrics.vacuum_well),
                   mercier_min=float(merc.min()), mercier_unstable_fraction=float(np.mean(merc < 0)),
                   mercier_profiles={k: np.asarray(getattr(eq, k)).round(6).tolist()
                                     for k in ("DMerc", "DShear", "DWell", "DCurr", "DGeod")})
    except Exception as e:  # noqa: BLE001
        row.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=3))
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def run(ds: list[dict], workers: int) -> None:
    done = {(r["id"], r["beta_target"]) for r in rc.read(LEDGER)}
    todo = [(d, b) for d in ds for b in BETAS if (d["id"], b) not in done]
    print(f"{len(todo)} runs ({len(ds)} designs x betas {BETAS}) with {workers} workers", flush=True)
    if not todo:
        return
    ctx = mp.get_context("forkserver")
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex, LEDGER.open("a") as f:
        jobs = {ex.submit(evaluate, d, b): d for d, b in todo}
        for job in futures.as_completed(jobs):
            d, r = jobs[job], job.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            msg = (f"beta {r['beta']:.3f}: worst {r['worst']:.4f} ({'feasible' if r['feasible'] else 'NOT'}), "
                   f"well {r['well']:+.3f}, L_gradB {r['L_gradB']:.2f}, Mercier unstable "
                   f"{r['mercier_unstable_fraction']:.0%}" if r["ok"] else r["error"][:100])
            print(f"{d['id']} AR {d['aspect_ratio']:.2f} target {r['beta_target']}: {msg} ({r['seconds']:.0f} s)",
                  flush=True)


def report(ds: list[dict]) -> None:
    by = {d["id"]: d for d in ds}
    out = {}
    for r in rc.read(LEDGER):
        if r["id"] in by and r.get("ok"):
            d = out.setdefault(r["id"], dict(aspect_ratio=by[r["id"]]["aspect_ratio"],
                                             L_gradB=by[r["id"]]["L_gradB"], worst=by[r["id"]]["worst"],
                                             by_beta={}))
            d["by_beta"][f"{r['beta_target']:.2f}"] = {k: r[k] for k in (
                "beta", "feasible", "worst", "violations", "L_gradB", "well", "mercier_min",
                "mercier_unstable_fraction")}
    for i, d in out.items():
        print(i, f"AR {d['aspect_ratio']:.2f}", {b: (round(v["worst"], 4), v["feasible"],
              round(v["mercier_unstable_fraction"], 2)) for b, v in sorted(d["by_beta"].items())})
    rc.RESULTS.mkdir(parents=True, exist_ok=True)
    (rc.RESULTS / "beta_check.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="", help="comma-separated design ids (from resolution_check)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    ds = rc.designs()
    if not a.report:
        ids = set(a.ids.split(","))
        sel = [d for d in ds if d["id"] in ids]
        missing = ids - {d["id"] for d in sel}
        if missing:
            sys.exit(f"unknown ids: {missing}")
        from evaluate import default_workers
        run(sel, a.workers or default_workers())
    report(ds)


if __name__ == "__main__":
    main()
