"""Score designs with the official Benchmark 3 evaluator (VMEC++ high fidelity) and time it.

Reads boundaries saved by count_feasible.py. Each evaluation takes ~110 s on one core
and ~3.6 GB of RAM on an M3 Max.

    python scripts/time_eval.py                 # reads results/sample_designs.json,
                                                # writes results/timing.json
"""

import json
import sys
import time
from pathlib import Path

from constellaration import problems
from constellaration.geometry import surface_rz_fourier

RESULTS = Path(__file__).resolve().parent.parent / "results"
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "sample_designs.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "timing.json"

samples = json.loads(IN.read_text())
problem = problems.MHDStableQIStellarator()
results = {}
for name, s in samples.items():
    boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(
        s["boundary_json"]
    )
    t0 = time.perf_counter()
    ev = problem.evaluate([boundary])
    dt = time.perf_counter() - t0
    results[name] = {
        "seconds": round(dt, 1),
        "feasibility (max normalized violation)": ev.feasibility[0],
        "objectives [(L_gradB, minimize?), (aspect_ratio, minimize?)]": ev.objectives[0],
        "hypervolume_score": ev.score,
        "dataset_worst_violation": s["dataset_worst_violation"],
        "dataset_L_gradB": s["dataset_metrics"][
            "metrics.minimum_normalized_magnetic_gradient_scale_length"
        ],
        "dataset_aspect_ratio": s["dataset_metrics"]["metrics.aspect_ratio"],
    }
    print(name, json.dumps(results[name], indent=2), flush=True)
OUT.write_text(json.dumps(results, indent=2))
