"""Effective ripple (epsilon_eff) of DESC equilibria: the standard proxy for neoclassical (1/nu)
transport in the banana regime (Nemov et al. 1999). Stellarators are usually optimized to
epsilon_eff < 1e-2; W7-X is designed for about 1e-2 or below.

Runs in the separate `desc` conda env, on the .h5 equilibria desc_check.py saved (vacuum,
fixed boundary, M = N = 10). Uses DESC's EffectiveRipple with its default settings.

    ~/miniforge3/envs/desc/bin/python scripts/feasibility_tests/desc_ripple.py \
        f268a78916 0ba599fd66 55b35e9f0c 025a48ab59
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WOUTS = ROOT / "data" / "feasibility_tests" / "desc"
OUT = ROOT / "results" / "feasibility_tests" / "desc_ripple.json"
RHO = np.array([0.25, 0.5, 0.75, 1.0])


def main() -> None:
    import desc.io
    from desc.grid import LinearGrid
    from desc.objectives import EffectiveRipple

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for i in sys.argv[1:]:
        t0 = time.perf_counter()
        eq = desc.io.load(str(WOUTS / f"wout_{i}_M10.h5"))
        grid = LinearGrid(rho=RHO, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        obj = EffectiveRipple(eq, grid=grid)
        obj.build(verbose=0)
        eps = np.asarray(obj.compute(eq.params_dict))
        out[i] = {"rho": RHO.tolist(), "epsilon_eff": eps.tolist(),
                  "seconds": round(time.perf_counter() - t0, 1)}
        print(i, " ".join(f"rho {r:.2f}: {e:.2e}" for r, e in zip(RHO, eps)),
              f"({out[i]['seconds']} s)", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
