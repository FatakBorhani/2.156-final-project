"""The seeded track's totals, as quoted in the mid-term report (Track B).

Reads every high-fidelity simulation ledger (data/seeded/runs/*.jsonl and the generator
pilot, data/gen/runs/pilot.jsonl) and the training table (data/seeded/train.npz, from
training_data.py), and reports:

  simulations   count, CPU-hours (sum of per-simulation wall time; one thread each), crashes
  novel passes  our passing designs >= 0.003 (and >= 0.01) from every public design
  hypervolume   public NFP-3 4x4 front; public + ours; ours alone (reference A 20, L_gradB -1)
  front points  which of our novel passes improve on the public front (for Figure 4a)

Writes results/seeded/track_b_totals.json and results/seeded/track_b_feasible_designs.csv.

    python scripts/seeded/report_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import loop  # noqa: E402
from families import NOVELTY, STRICT, load, nearest_public  # noqa: E402
from screen import TOL, load_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "seeded"
LEDGERS = sorted((ROOT / "data" / "seeded" / "runs").glob("*.jsonl")) + [ROOT / "data" / "gen" / "runs" / "pilot.jsonl"]


def simulations() -> dict:
    groups, seconds = {}, []
    for f in LEDGERS:
        rows = [r for r in map(json.loads, f.open()) if r.get("fidelity") == "high_fidelity"]
        name = "gen_pilot" if f.parent.parent.name == "gen" else re.sub(r"_s\d+$", "", f.stem)
        g = groups.setdefault(name, {"sims": 0, "cpu_hours": 0.0, "crashed": 0})
        g["sims"] += len(rows)
        g["cpu_hours"] += sum(r.get("seconds") or 0 for r in rows) / 3600
        g["crashed"] += sum(not r["ok"] for r in rows)
        seconds += [r.get("seconds") or 0 for r in rows]
    return {"total": sum(g["sims"] for g in groups.values()),
            "cpu_hours": sum(g["cpu_hours"] for g in groups.values()),
            "crashed": sum(g["crashed"] for g in groups.values()),
            "median_seconds": float(np.median(seconds)), "by_ledger": groups}


def main() -> None:
    t, pub = load_table(), load()
    ok = (t["source"] == "ours") & ~t["crash"] & (t["max_violation"] <= TOL)
    dist = np.full(len(t["X"]), np.nan)
    dist[ok] = nearest_public(t["X"][ok])
    out = {"scope": "NFP 3, 4x4 Fourier modes", "public_designs": int(len(pub["A"])),
           "public_front_hv": loop.hypervolume(pub["A"], pub["L"]), "simulations": simulations()}
    for thresh in (NOVELTY, STRICT):
        m = ok & (dist >= thresh)
        out[f"novelty_{thresh}"] = {
            "novel_passes": int(m.sum()),
            "hv_ours_alone": loop.hypervolume(t["A"][m], t["L"][m]),
            "hv_public_plus_ours": loop.hypervolume(np.r_[pub["A"], t["A"][m]], np.r_[pub["L"], t["L"][m]])}

    m = ok & (dist >= NOVELTY)
    A, L = t["A"][m], t["L"][m]
    allA, allL = np.r_[pub["A"], A], np.r_[pub["L"], L]
    front = np.array([not np.any((allA <= a) & (allL >= l) & ((allA < a) | (allL > l))) for a, l in zip(A, L)])
    out["improves_public_front"] = int(front.sum())

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "track_b_totals.json").write_text(json.dumps(out, indent=1))
    with (OUT / "track_b_feasible_designs.csv").open("w") as f:
        f.write("aspect_ratio,L_gradB,nearest_public,improves_public_front\n")
        for i in np.argsort(A):
            f.write(f"{A[i]:.5f},{L[i]:.5f},{dist[m][i]:.5f},{int(front[i])}\n")

    s, n = out["simulations"], out[f"novelty_{NOVELTY}"]
    print(f"{s['total']} simulations, {s['cpu_hours']:.0f} CPU-hours (median {s['median_seconds']:.0f} s), "
          f"{s['crashed']} crashed")
    print(f"public front HV {out['public_front_hv']:.2f}; {n['novel_passes']} novel passes: "
          f"public + ours {n['hv_public_plus_ours']:.2f}, ours alone {n['hv_ours_alone']:.2f}; "
          f"{out['improves_public_front']} improve on the public front")
    st = out[f"novelty_{STRICT}"]
    print(f"strict ({STRICT}): {st['novel_passes']} novel passes, public + ours {st['hv_public_plus_ours']:.2f}")


if __name__ == "__main__":
    main()
