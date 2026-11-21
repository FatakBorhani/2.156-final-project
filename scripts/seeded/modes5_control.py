"""Control for Track A's Test 18: do its designs need the 5th Fourier modes?

Test 18 reran Leif's compactness descent with modes up to 5x5 and broke the "AR ~10.2 wall"
(most compact feasible design AR 9.61, HV 143.8). The review of that code found the mode-5
coefficients those designs ended up with are tiny (~1e-5 vs ~4e-2 in the 4x4 block), so the
gain may come from something else (the higher very-low-fidelity resolution a nonzero mode 5
triggers, a bigger search, the restart). This re-scores each of the 122 Test 18 designs at
high fidelity with every mode-5 coefficient removed (truncated to 4x4), plus 10 untouched
originals as a scorer check.

  still feasible without mode 5  -> the designs are 4x4 designs; the extra modes didn't matter
  infeasible without mode 5      -> the extra modes are doing real work

    python scripts/seeded/modes5_control.py run       # ~132 sims, ~25 min at 8 workers
    python scripts/seeded/modes5_control.py report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop import hypervolume  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "feasibility_tests" / "modes5_feasible_designs.jsonl"
OUT = ROOT / "data" / "track_a"
LEDGER = OUT / "runs" / "modes5_control.jsonl"
DESIGNS = OUT / "modes5_control_designs.jsonl"
N_ORIGINALS = 10


def truncate(b: dict) -> tuple[dict, float, float]:
    """The same boundary without any |m| = 5 or |n| = 5 term; also the largest dropped
    coefficient and the RMS of the dropped block relative to the kept one."""
    rc, zs = np.array(b["r_cos"], float), np.array(b["z_sin"], float)
    assert rc.shape == (6, 11), rc.shape
    keep = np.zeros_like(rc, bool); keep[:5, 1:10] = True
    dropped = np.r_[rc[~keep], zs[~keep]]
    kept = np.r_[rc[keep], zs[keep]]
    out = dict(b, r_cos=rc[:5, 1:10].tolist(), z_sin=zs[:5, 1:10].tolist())
    return out, float(np.abs(dropped).max()), float(np.linalg.norm(dropped) / np.linalg.norm(kept))


def designs() -> list[dict]:
    rows = [json.loads(x) for x in SRC.open()]
    out = []
    for i, r in enumerate(rows):
        b = json.loads(r["boundary"])
        t, biggest, rel = truncate(b)
        src = {"source_index": i, "orig_A": r["aspect_ratio"], "orig_L": r["L_gradB"],
               "orig_worst": r["worst"], "dropped_max": biggest, "dropped_rel": rel}
        bj = json.dumps(t)
        out.append({"id": "m4-" + hashlib.sha1(bj.encode()).hexdigest()[:12], "boundary": bj,
                    "kind": "truncated", **src})
        if i < N_ORIGINALS:
            bj = json.dumps(b)
            out.append({"id": "m5-" + hashlib.sha1(bj.encode()).hexdigest()[:12], "boundary": bj,
                        "kind": "original", **src})
    return out


def run(workers: int) -> None:
    import run_sims
    d = designs()
    OUT.mkdir(parents=True, exist_ok=True)
    with DESIGNS.open("w") as fh:
        for x in d:
            fh.write(json.dumps(x) + "\n")
    run_sims.run(d, LEDGER, workers=workers)


def report() -> None:
    meta = {d["id"]: d for d in map(json.loads, DESIGNS.open())}
    res = {r["id"]: r for r in map(json.loads, LEDGER.open()) if r["id"] in meta}
    for kind in ("original", "truncated"):
        ids = [i for i in meta if meta[i]["kind"] == kind and i in res]
        ok = [i for i in ids if res[i]["ok"]]
        feas = [i for i in ok if res[i]["feasible"]]
        print(f"{kind}: {len(ids)} simulated, {len(ids) - len(ok)} crashed/timed out, "
              f"{len(feas)} feasible ({len(feas) / max(1, len(ids)):.0%})")
        if ok:
            dA = [abs(res[i]["aspect_ratio"] - meta[i]["orig_A"]) for i in ok]
            dL = [abs(res[i]["L_gradB"] - meta[i]["orig_L"]) for i in ok]
            dv = [res[i]["max_violation"] - meta[i]["orig_worst"] for i in ok]
            print(f"  vs Leif's values: max |dA| {max(dA):.2e}, max |dL| {max(dL):.2e}, "
                  f"worst violation change median {np.median(dv):+.4f}, max {max(dv):+.4f}")
        if kind == "truncated" and ids:
            print(f"  dropped mode-5 content: largest coefficient {max(meta[i]['dropped_max'] for i in ids):.1e}, "
                  f"relative RMS median {np.median([meta[i]['dropped_rel'] for i in ids]):.1e}")
            A = np.array([res[i]["aspect_ratio"] for i in feas]); L = np.array([res[i]["L_gradB"] for i in feas])
            oA = np.array([meta[i]["orig_A"] for i in ids]); oL = np.array([meta[i]["orig_L"] for i in ids])
            print(f"  Test 18 designs' own HV: originals {hypervolume(oA, oL):.2f}, "
                  f"truncated and still feasible {hypervolume(A, L):.2f}")
            if len(A):
                print(f"  most compact still feasible: AR {A.min():.3f} (Test 18: {oA.min():.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "report", "list"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.workers)
    elif a.cmd == "report":
        report()
    else:
        d = designs()
        print(len(d), "designs;", sum(x["kind"] == "truncated" for x in d), "truncated")
        print("dropped max", max(x["dropped_max"] for x in d),
              "rel median", np.median([x["dropped_rel"] for x in d]))


if __name__ == "__main__":
    main()
