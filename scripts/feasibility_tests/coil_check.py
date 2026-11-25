"""Coil check: can simple filament coils reproduce a Track A boundary, and does L_gradB predict how
far away they can sit?

Stage-2 coil optimization with SIMSOPT (as in its stage_two_optimization example): stellarator-
symmetric, modular filament coils are shaped to make the coils' field tangent to the plasma
boundary (vacuum: no plasma current). Every design is first scaled to minor radius a = 1, the
normalization the benchmark's L_gradB uses, so all lengths below are in units of a.

For each design we sweep the minimum coil-plasma distance d and keep the coil "complexity" limits
fixed (in units of a): coils per half period, Fourier order, curvature, mean-squared curvature,
coil-coil spacing and length. The result is the residual field error <|B.n|>/<|B|> on the
boundary. Kappel et al. (2024) predict that a design with larger L_gradB keeps a small error out
to larger d.

    python scripts/feasibility_tests/coil_check.py --ids 0ba599fd66,025a48ab59 --d 1,1.5,2,2.5,3
    python scripts/feasibility_tests/coil_check.py --report

Reads only our own Track A ledgers. Writes data/feasibility_tests/coil_ledger.jsonl (resumable),
data/feasibility_tests/coils/ (coil sets) and results/feasibility_tests/coil_check.json.
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

LEDGER = rc.DATA / "coil_ledger.jsonl"
COILS = rc.DATA / "coils"

# complexity limits, in units of the plasma minor radius a
ORDER = 8                  # Fourier modes per coil
CC_MIN = 0.5               # minimum coil-coil distance
KAPPA_MAX = 1.0            # maximum curvature (radius of curvature >= a)
MSC_MAX = 0.3              # maximum mean-squared curvature
LENGTH_FACTOR = 1.6        # each coil at most 1.6x the length of a circle of radius (1 + d)
MAXITER = 3000


def surface(design: dict):
    """The design's boundary as a SIMSOPT surface on a half-period grid, scaled to a = 1."""
    from constellaration.geometry import surface_rz_fourier as srf

    b = srf.SurfaceRZFourier.model_validate_json(design["boundary"])
    nfp = b.n_field_periods
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    phi = np.linspace(0, np.pi / nfp, 32, endpoint=False)
    tp = np.stack(np.meshgrid(theta, phi, indexing="ij"), axis=-1)
    s = srf.to_simsopt(b, theta_phi=tp)
    a = float(srf.evaluate_minor_radius(b))
    s.x = s.x / a
    return s, a, nfp


def optimize(design: dict, d: float, ncoils: int) -> dict:
    """One stage-2 coil optimization. Runs in a worker process."""
    from scipy.optimize import minimize
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.geo import (CurveCurveDistance, CurveLength, CurveSurfaceDistance, LpCurveCurvature,
                             MeanSquaredCurvature, create_equally_spaced_curves, curves_to_vtk)
    from simsopt.objectives import QuadraticPenalty, SquaredFlux

    row = {"id": design["id"], "d": d, "ncoils": ncoils}
    t0 = time.perf_counter()
    try:
        s, a, nfp = surface(design)
        R0 = float(s.get_rc(0, 0))
        base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=1.0 + 1.5 * d,
                                                   order=ORDER, numquadpoints=128)
        base_currents = [Current(1.0) * 1e5 for _ in range(ncoils)]
        base_currents[0].fix_all()  # otherwise all currents -> 0 is a trivial solution
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
        curves = [c.curve for c in coils]
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))

        Jf = SquaredFlux(s, bs, definition="normalized")
        Jls = [CurveLength(c) for c in base_curves]
        Jcc = CurveCurveDistance(curves, CC_MIN, num_basecurves=ncoils)
        Jcs = CurveSurfaceDistance(curves, s, d)
        Jk = [LpCurveCurvature(c, 2, KAPPA_MAX) for c in base_curves]
        Jmsc = [MeanSquaredCurvature(c) for c in base_curves]
        lmax = LENGTH_FACTOR * 2 * np.pi * (1.0 + d)
        JF = (Jf + 1e3 * Jcc + 1e3 * Jcs + 1e-1 * sum(Jk)
              + 1e-1 * sum(QuadraticPenalty(J, MSC_MAX, "max") for J in Jmsc)
              + 1e-1 * sum(QuadraticPenalty(J, lmax, "max") for J in Jls))

        def fun(x):
            JF.x = x
            return JF.J(), JF.dJ()

        res = minimize(fun, JF.x, jac=True, method="L-BFGS-B", tol=1e-15,
                       options={"maxiter": MAXITER, "maxcor": 300})
        JF.x = res.x

        B = bs.B().reshape(s.gamma().shape)
        n = s.unitnormal()
        bn = np.abs(np.sum(B * n, axis=-1)) / np.linalg.norm(B, axis=-1)
        cs = np.min([np.min(np.linalg.norm(c.gamma()[:, None, :] - s.gamma().reshape(1, -1, 3), axis=-1))
                     for c in base_curves])
        row.update(ok=True, iterations=int(res.nit), minor_radius=a, aspect_ratio=R0,
                   bn_mean=float(bn.mean()), bn_max=float(bn.max()), squared_flux=float(Jf.J()),
                   min_coil_surface=float(cs), min_coil_coil=float(Jcc.shortest_distance()),
                   max_curvature=float(max(np.max(c.kappa()) for c in base_curves)),
                   max_msc=float(max(J.J() for J in Jmsc)), lengths=[float(J.J()) for J in Jls])
        COILS.mkdir(parents=True, exist_ok=True)
        stem = COILS / f"{design['id']}_d{d:g}_n{ncoils}"
        bs.save(str(stem) + ".json")
        curves_to_vtk(curves, str(stem))
    except Exception as e:  # noqa: BLE001
        row.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=4))
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def run(ds: list[dict], dists: list[float], ncoils: list[int], workers: int) -> None:
    done = {(r["id"], r["d"], r["ncoils"]) for r in rc.read(LEDGER) if r.get("ok")}
    todo = [(x, d, n) for x in ds for d in dists for n in ncoils if (x["id"], d, n) not in done]
    print(f"{len(todo)} coil optimizations with {workers} workers", flush=True)
    if not todo:
        return
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("forkserver")) as ex, \
            LEDGER.open("a") as f:
        jobs = {ex.submit(optimize, *t): t for t in todo}
        for job in futures.as_completed(jobs):
            x, d, n = jobs[job]
            r = job.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            msg = (f"<|B.n|/|B|> {r['bn_mean']:.2e} (max {r['bn_max']:.2e}), coil-plasma {r['min_coil_surface']:.2f}, "
                   f"coil-coil {r['min_coil_coil']:.2f}, max curvature {r['max_curvature']:.2f}, "
                   f"{r['iterations']} it" if r["ok"] else r["error"][:150])
            print(f"{x['id']} AR {x['aspect_ratio']:.1f} d {d:g} ncoils {n}: {msg} ({r['seconds']:.0f} s)",
                  flush=True)


def report() -> None:
    out = {}
    for r in rc.read(LEDGER):
        if r.get("ok"):
            out.setdefault(r["id"], {})[f"d{r['d']:g}_n{r['ncoils']}"] = {
                k: r[k] for k in ("d", "ncoils", "bn_mean", "bn_max", "min_coil_surface", "min_coil_coil",
                                  "max_curvature", "max_msc", "lengths", "aspect_ratio")}
    for i, v in out.items():
        print(i, {k: f"{x['bn_mean']:.1e}" for k, x in sorted(v.items(), key=lambda kv: (kv[1]['ncoils'], kv[1]['d']))})
    rc.RESULTS.mkdir(parents=True, exist_ok=True)
    (rc.RESULTS / "coil_check.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="")
    p.add_argument("--d", default="1,1.5,2,2.5,3", help="coil-plasma distances, in minor radii")
    p.add_argument("--ncoils", default="4", help="coils per half period")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if not a.report:
        ids = set(a.ids.split(","))
        sel = [x for x in rc.designs() if x["id"] in ids]
        if ids - {x["id"] for x in sel}:
            sys.exit(f"unknown ids: {ids - {x['id'] for x in sel}}")
        from evaluate import default_workers
        run(sel, [float(x) for x in a.d.split(",")], [int(x) for x in a.ncoils.split(",")],
            a.workers or default_workers())
    report()


if __name__ == "__main__":
    main()
