"""DESC half of the DESC cross-check (see desc_check.py). Runs in the separate `desc` conda env.

Reads boundaries exported by desc_check.py, solves each as a fixed-boundary vacuum equilibrium
(zero pressure, zero net toroidal current, same edge flux as the benchmark's VMEC runs) with
DESC, and writes a VMEC-format wout file so the benchmark's own metric code can score it.

    ~/miniforge3/envs/desc/bin/python scripts/feasibility_tests/desc_solve.py IN.json OUT_DIR --M 10
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path


def vmec_input(b: dict, phiedge: float) -> str:
    """A minimal VMEC &INDATA namelist for the boundary (VMEC's cos(m theta - n nfp phi) convention)."""
    ntor = (len(b["r_cos"][0]) - 1) // 2
    lines = ["&INDATA", "  LASYM = F", f"  NFP = {b['n_field_periods']}", f"  PHIEDGE = {phiedge!r}",
             f"  MPOL = {len(b['r_cos'])}", f"  NTOR = {ntor}",
             "  NCURR = 1", "  AM = 0.0", "  AC = 0.0"]
    for m, (rrow, zrow) in enumerate(zip(b["r_cos"], b["z_sin"])):
        for j, (r, z) in enumerate(zip(rrow, zrow)):
            n = j - ntor
            if r:
                lines.append(f"  RBC({n},{m}) = {r!r}")
            if z:
                lines.append(f"  ZBS({n},{m}) = {z!r}")
    return "\n".join(lines + ["/", ""])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--M", type=int, default=10, help="DESC poloidal = toroidal resolution")
    p.add_argument("--surfs", type=int, default=99, help="radial surfaces in the wout file")
    a = p.parse_args()

    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface
    from desc.profiles import PowerSeriesProfile
    from desc.vmec import VMECIO

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in json.loads(Path(a.inp).read_text()):
        wout = out / f"wout_{d['id']}_M{a.M}.nc"
        if wout.exists():
            continue
        t0 = time.perf_counter()
        b = d["boundary"]
        with tempfile.NamedTemporaryFile("w", suffix=".vmec", delete=False) as f:
            f.write(vmec_input(b, d["phiedge"]))
        surface = FourierRZToroidalSurface.from_input_file(f.name)
        eq = Equilibrium(surface=surface, NFP=b["n_field_periods"], M=a.M, N=a.M, L=a.M,
                         Psi=d["phiedge"], pressure=PowerSeriesProfile([0.0]),
                         current=PowerSeriesProfile([0.0]))
        eq.solve(verbose=0, ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=500)
        VMECIO.save(eq, str(wout), surfs=a.surfs, verbose=0)
        eq.save(str(wout.with_suffix(".h5")))
        # DESC's own values, to tell solver differences from wout-conversion differences
        import numpy as np
        from desc.grid import LinearGrid
        edge = LinearGrid(rho=np.array([1.0]), M=4 * a.M, N=4 * a.M, NFP=eq.NFP)
        nat = eq.compute(["iota", "L_grad(B)", "a", "R0/a", "mirror ratio"], grid=edge)
        print(json.dumps({"id": d["id"], "M": a.M, "seconds": round(time.perf_counter() - t0, 1),
                          "wout": str(wout), "native": {
                              "iota_edge_over_nfp": float(nat["iota"][0]) / eq.NFP,
                              "aspect_ratio": float(nat["R0/a"]),
                              "L_gradB": float(np.min(nat["L_grad(B)"]) / nat["a"] * eq.NFP),
                              "edge_mirror": float(np.max(nat["mirror ratio"]))}}), flush=True)


if __name__ == "__main__":
    main()
