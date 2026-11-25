"""Resolution check: do Track A's feasible designs stay feasible at higher resolution?

Every design we call feasible was scored with the benchmark's "high_fidelity" preset:
fixed-boundary, vacuum VMEC++ with 10 poloidal x 10 toroidal modes and 25 -> 51 -> 99 flux
surfaces (force tolerance 1e-15), then booz_xform (resolution set from VMEC's: 6*mpol, 2*ntor-1)
and the QI calculation with 75 field lines x 601 points. Feasible means every normalized
violation (violation / |limit|; 0.1 for the vacuum well) is <= 0.01.

This re-scores the same boundaries with finer settings and compares each metric:

    radial  99 -> 199 flux surfaces (adds a 4th multigrid stage)
    modes   10x10 -> 14x14 VMEC modes (Boozer modes follow)
    qi      QI calculation with 2x field lines, points and contours
    all     all three at once

    python scripts/feasibility_tests/resolution_check.py --trial 2 --variant all --workers 2
    python scripts/feasibility_tests/resolution_check.py --sample 20 --variant all  # 20 front designs
    python scripts/feasibility_tests/resolution_check.py --variant all            # all designs
    python scripts/feasibility_tests/resolution_check.py --report

Reads only our own Track A ledgers. Writes data/feasibility_tests/resolution_ledger.jsonl
(resumable) and results/feasibility_tests/resolution_check.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent import futures
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "feasibility_tests"
RESULTS = ROOT / "results" / "feasibility_tests"
LEDGER = DATA / "resolution_ledger.jsonl"
CONSTRAINTS = ["iota", "log10_qi", "mirror", "flux_compression", "vacuum_well"]
VARIANTS = {
    "radial": dict(ns=199, modes=10, qi=1),
    "modes": dict(ns=99, modes=14, qi=1),
    "qi": dict(ns=99, modes=10, qi=2),
    "all": dict(ns=199, modes=14, qi=2),
}


def designs() -> list[dict]:
    """Track A's feasible designs (high-fidelity rows), one per boundary."""
    out = {}
    for p in sorted(DATA.glob("*_ledger.jsonl")):
        if "dry" in p.name or p == LEDGER:
            continue
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "hi" and r.get("ok") and r.get("feasible") and r.get("boundary"):
                out.setdefault(r["boundary"], dict(source=p.name.replace("_ledger.jsonl", ""), **r))
    rows = sorted(out.values(), key=lambda r: (r["aspect_ratio"], -r["L_gradB"]))
    for r in rows:  # stable across runs, unlike a position (new designs keep arriving)
        r["id"] = hashlib.sha1(r["boundary"].encode()).hexdigest()[:10]
    return rows


def front_sample(ds: list[dict], n: int) -> list[dict]:
    """n Pareto-front designs, weighted toward those nearest the 1% tolerance and spread in AR:
    40% with worst > 0.009, 30% with 0.005-0.009, 30% with worst <= 0.005."""
    front, best = [], -np.inf
    for d in ds:  # ds is sorted by AR, then L_gradB descending
        if d["L_gradB"] > best:
            front.append(d)
            best = d["L_gradB"]
    bins = [([d for d in front if d["worst"] > 0.009], round(0.4 * n)),
            ([d for d in front if 0.005 < d["worst"] <= 0.009], round(0.3 * n))]
    bins.append(([d for d in front if d["worst"] <= 0.005], n - sum(k for _, k in bins)))
    out = []
    for group, k in bins:
        k = min(k, len(group))
        out += [group[i] for i in np.unique(np.linspace(0, len(group) - 1, k).round().astype(int))]
    print(f"front has {len(front)} designs; sampling {len(out)}", flush=True)
    return out


def _settings(variant: str, boundary):
    from constellaration import forward_model
    from constellaration.geometry import surface_rz_fourier, surface_utils
    from constellaration.mhd import vmec_settings as vs
    from constellaration.omnigeneity import qi

    v = VARIANTS[variant]
    m, n = surface_rz_fourier.get_largest_non_zero_modes(surface=boundary)
    mpol, ntor = max(v["modes"], m), max(v["modes"], n)
    npts, ntpts = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        n_poloidal_modes=mpol, max_toroidal_mode=ntor)
    steps = [vs.VmecMultiGridStep(n_flux_surfaces=25, force_tolerance=1e-17, n_max_iterations=2000),
             vs.VmecMultiGridStep(n_flux_surfaces=51, force_tolerance=1e-17, n_max_iterations=2000),
             vs.VmecMultiGridStep(n_flux_surfaces=99, force_tolerance=1e-15, n_max_iterations=40000)]
    if v["ns"] > 99:
        steps.append(vs.VmecMultiGridStep(n_flux_surfaces=v["ns"], force_tolerance=1e-15,
                                          n_max_iterations=40000))
    vmec = vs.VmecSettings(n_poloidal_modes=mpol, max_toroidal_mode=ntor, n_poloidal_grid_points=npts,
                           n_toroidal_grid_points=ntpts, multigrid_steps=steps, time_step=0.5)
    vmec.max_threads = 1
    q = qi.QISettings()
    if v["qi"] > 1:
        k = v["qi"]
        q = qi.QISettings(n_field_lines=q.n_field_lines * k, n_toroidal_points=q.n_toroidal_points * k,
                          n_field_contours=q.n_field_contours * k,
                          n_output_toroidal_points=q.n_output_toroidal_points * k)
    return vmec, forward_model.ConstellarationSettings.default_high_fidelity().model_copy(
        update={"qi_settings": q})


def evaluate(design: dict, variant: str) -> dict:
    """Score one boundary at the finer settings. Runs in a worker process."""
    from constellaration import forward_model, problems
    from constellaration.geometry import surface_rz_fourier
    from constellaration.mhd import vmec_settings as vs

    row = {"id": design["id"], "variant": variant}
    t0 = time.perf_counter()
    try:
        b = surface_rz_fourier.SurfaceRZFourier.model_validate_json(design["boundary"])
        vmec, settings = _settings(variant, b)
        vs.create_vmec_settings_from_preset = lambda boundary, settings: vmec  # finer VMEC
        metrics, _ = forward_model.forward_model(b, settings=settings)
        problem = problems.MHDStableQIStellarator()
        viol = problem._normalized_constraint_violations(metrics)
        row.update(ok=True, feasible=bool(problem.is_feasible(metrics)),
                   worst=float(np.max(np.maximum(viol, 0.0))),
                   violations=dict(zip(CONSTRAINTS, map(float, viol))),
                   L_gradB=float(metrics.minimum_normalized_magnetic_gradient_scale_length),
                   aspect_ratio=float(metrics.aspect_ratio))
    except Exception as e:  # noqa: BLE001
        row.update(ok=False, feasible=False, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc(limit=3))
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def run(ds: list[dict], variant: str, workers: int) -> None:
    done = {(r["id"], r["variant"]) for r in read(LEDGER)}
    todo = [d for d in ds if (d["id"], variant) not in done]
    print(f"{len(ds) - len(todo)} done; {len(todo)} to run at '{variant}' with {workers} workers",
          flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    ctx = mp.get_context("forkserver")
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex, LEDGER.open("a") as f:
        jobs = {ex.submit(evaluate, d, variant): d for d in todo}
        for i, job in enumerate(futures.as_completed(jobs), 1):
            d, r = jobs[job], job.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            msg = (f"worst {d['worst']:.4f} -> {r['worst']:.4f}, L_gradB {d['L_gradB']:.3f} -> "
                   f"{r['L_gradB']:.3f}, {'feasible' if r['feasible'] else 'NOT feasible'}"
                   if r["ok"] else r["error"][:80])
            print(f"[{i}/{len(todo)}] {d['id']} AR {d['aspect_ratio']:.2f}: {msg} ({r['seconds']:.0f} s)",
                  flush=True)
    print(f"wall time {time.perf_counter() - t0:.0f} s", flush=True)


def report(ds: list[dict]) -> None:
    by = {d["id"]: d for d in ds}
    out = {}
    for variant in VARIANTS:
        rs = [r for r in read(LEDGER) if r["variant"] == variant and r["id"] in by]
        if not rs:
            continue
        ok = [r for r in rs if r["ok"]]
        def base(i: str) -> dict:  # our ledgers store the 5 violations as a list, in this order
            v = by[i].get("violations")
            return dict(zip(CONSTRAINTS, v)) if isinstance(v, list) else v or {}
        dv = {c: [abs(r["violations"][c] - base(r["id"])[c]) for r in ok if c in base(r["id"])]
              for c in CONSTRAINTS}
        out[variant] = dict(
            n=len(rs), failed=len(rs) - len(ok), still_feasible=sum(r["feasible"] for r in ok),
            lost=[r["id"] for r in ok if not r["feasible"]],
            median_seconds=float(np.median([r["seconds"] for r in rs])),
            max_change_worst=max(abs(r["worst"] - by[r["id"]]["worst"]) for r in ok) if ok else None,
            max_change_L_gradB=max(abs(r["L_gradB"] - by[r["id"]]["L_gradB"]) for r in ok) if ok else None,
            max_change_by_constraint={c: max(v) if v else None for c, v in dv.items()})
        print(variant, json.dumps(out[variant], indent=1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "resolution_check.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), default="all")
    p.add_argument("--trial", type=int, default=0, help="only N designs (most and least margin)")
    p.add_argument("--sample", type=int, default=0, help="only N front designs (see front_sample)")
    p.add_argument("--ids", default="", help="only these comma-separated design ids")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    ds = designs()
    print(f"{len(ds)} feasible Track A designs", flush=True)
    if not a.report:
        sel = ds
        if a.trial:
            by_worst = sorted(ds, key=lambda d: d["worst"])
            sel = (by_worst[-((a.trial + 1) // 2):] + by_worst[:a.trial // 2])[:a.trial]
        if a.sample:
            sel = front_sample(ds, a.sample)
        if a.ids:
            sel = [d for d in ds if d["id"] in set(a.ids.split(","))]
        sys.path.insert(0, str(HERE.parent))
        from evaluate import default_workers
        run(sel, a.variant, a.workers or default_workers())
    report(ds)


if __name__ == "__main__":
    main()
