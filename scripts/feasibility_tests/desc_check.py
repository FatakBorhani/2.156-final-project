"""DESC cross-check: does an independent equilibrium code give the same benchmark metrics?

Every Track A metric comes from VMEC++. Here the same fixed boundaries are solved with DESC
(a spectral code with a different numerical method), in the separate `desc` conda env
(desc_solve.py). DESC's equilibrium is written as a VMEC-format wout file and then scored by the
benchmark's own metric code (constellaration.forward_model, with its VMEC call replaced by that
file), so the only thing that differs from our usual scoring is the equilibrium solver.

    python scripts/feasibility_tests/desc_check.py --ids 3e8736083d,025a48ab59 --M 10
    python scripts/feasibility_tests/desc_check.py --report

Reads only our own Track A ledgers. Writes data/feasibility_tests/desc/ (wout files),
data/feasibility_tests/desc_ledger.jsonl and results/feasibility_tests/desc_check.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import resolution_check as rc  # noqa: E402

DESC_PY = Path.home() / "miniforge3" / "envs" / "desc" / "bin" / "python"
WOUTS = rc.DATA / "desc"
LEDGER = rc.DATA / "desc_ledger.jsonl"


def solve(ds: list[dict], M: int) -> None:
    from constellaration.geometry import surface_rz_fourier
    from constellaration.mhd import ideal_mhd_parameters as imp

    exp = []
    for d in ds:
        b = surface_rz_fourier.SurfaceRZFourier.model_validate_json(d["boundary"])
        exp.append({"id": d["id"], "boundary": json.loads(d["boundary"]),
                    "phiedge": imp.boundary_to_ideal_mhd_parameters(b).boundary_toroidal_flux})
    WOUTS.mkdir(parents=True, exist_ok=True)
    inp = WOUTS / "boundaries.json"
    inp.write_text(json.dumps(exp))
    subprocess.run([str(DESC_PY), str(HERE / "desc_solve.py"), str(inp), str(WOUTS), "--M", str(M)],
                   check=True)


def load_wout(path: Path):
    """Read DESC's wout file like vmecpp.VmecWOut.from_wout_file, filling the VMEC++-only fields
    DESC does not write (solver residuals, input echoes, diagnostics) with NaN or empty arrays.
    None of them enter the benchmark metrics; the Boozer transform uses lmns, which DESC writes."""
    import netCDF4
    import vmecpp

    with netCDF4.Dataset(path, "r") as f:
        f.set_auto_mask(False)
        attrs = {}
        for k, v in f.variables.items():
            if v.dtype is str or v.dtype == "S1":
                attrs[k] = f[k][()].tobytes().decode("ascii").strip("\x00").strip()
            else:
                attrs[k] = np.transpose(f[k][()]) if v.ndim == 2 else f[k][()]
    ns, mnmax = int(attrs["ns"]), int(attrs["mnmax"])
    for k in ("IonLarmor", "fsqr", "fsqz", "fsql", "ftolv"):
        attrs.setdefault(k, np.nan)
    for k in ("equif", "specw", "over_r"):
        attrs.setdefault(k, np.full(ns, np.nan))
    for k in ("am_aux_s", "am_aux_f", "ac_aux_s", "ac_aux_f", "ai_aux_s", "ai_aux_f", "extcur"):
        attrs.setdefault(k, np.array([]))
    attrs.setdefault("niter", 0)
    attrs.setdefault("lmns_full", np.zeros([mnmax, ns]))
    return vmecpp.VmecWOut.model_validate(attrs, by_alias=True)


def score(d: dict, M: int) -> dict:
    """Score DESC's equilibrium with the benchmark's metric code."""
    from constellaration import forward_model, problems
    from constellaration.geometry import surface_rz_fourier
    from constellaration.mhd import vmec_utils

    row = {"id": d["id"], "M": M}
    t0 = time.perf_counter()
    try:
        wout = vmec_utils.vmecppwout_from_wout(load_wout(WOUTS / f"wout_{d['id']}_M{M}.nc"))
        vmec_utils.run_vmec = lambda **_: wout  # use DESC's equilibrium instead of running VMEC++
        b = surface_rz_fourier.SurfaceRZFourier.model_validate_json(d["boundary"])
        metrics, _ = forward_model.forward_model(
            b, settings=forward_model.ConstellarationSettings.default_high_fidelity())
        problem = problems.MHDStableQIStellarator()
        viol = problem._normalized_constraint_violations(metrics)
        row.update(ok=True, feasible=bool(problem.is_feasible(metrics)),
                   worst=float(np.max(np.maximum(viol, 0.0))),
                   violations=dict(zip(rc.CONSTRAINTS, map(float, viol))),
                   L_gradB=float(metrics.minimum_normalized_magnetic_gradient_scale_length),
                   aspect_ratio=float(metrics.aspect_ratio))
    except Exception as e:  # noqa: BLE001
        row.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=4))
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def report(ds: list[dict]) -> None:
    by = {d["id"]: d for d in ds}
    out = {}
    for r in rc.read(LEDGER):
        if r["id"] not in by or not r.get("ok"):
            continue
        v = by[r["id"]]
        base = dict(zip(rc.CONSTRAINTS, v["violations"])) if isinstance(v["violations"], list) else v["violations"]
        out[f"{r['id']}_M{r['M']}"] = dict(
            aspect_ratio=[v["aspect_ratio"], r["aspect_ratio"]], L_gradB=[v["L_gradB"], r["L_gradB"]],
            worst=[v["worst"], r["worst"]], feasible=[True, r["feasible"]],
            violations={c: [base[c], r["violations"][c]] for c in rc.CONSTRAINTS})
        print(r["id"], f"M{r['M']}", "VMEC++ -> DESC:",
              f"AR {v['aspect_ratio']:.3f} -> {r['aspect_ratio']:.3f},",
              f"L_gradB {v['L_gradB']:.3f} -> {r['L_gradB']:.3f},",
              f"worst {v['worst']:.4f} -> {r['worst']:.4f}",
              {c: f"{base[c]:+.4f} -> {r['violations'][c]:+.4f}" for c in rc.CONSTRAINTS})
    rc.RESULTS.mkdir(parents=True, exist_ok=True)
    (rc.RESULTS / "desc_check.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="")
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    ds = rc.designs()
    if not a.report:
        ids = set(a.ids.split(","))
        sel = [d for d in ds if d["id"] in ids]
        if ids - {d["id"] for d in sel}:
            sys.exit(f"unknown ids: {ids - {d['id'] for d in sel}}")
        solve(sel, a.M)
        done = {(r["id"], r["M"]) for r in rc.read(LEDGER) if r.get("ok")}
        with LEDGER.open("a") as f:
            for d in sel:
                if (d["id"], a.M) not in done:
                    r = score(d, a.M)
                    f.write(json.dumps(r) + "\n")
                    print(d["id"], "scored" if r["ok"] else r["error"][:200], f"({r['seconds']} s)", flush=True)
    report(ds)


if __name__ == "__main__":
    main()
