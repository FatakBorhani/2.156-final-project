"""Generator pilot: real simulations of the trade study's finalists (gen_study.py).

For each finalist: train it, generate designs, keep the novel ones (>= 0.003 from every
public design and from everything we've simulated) with A < 19.9, and pick 60 at random,
spaced 0.003 apart. No screening, so the pass rate measures the generator itself and is
comparable with blending's random-pick pass rate (42% in the widening pilot).

Each pick records the screening models' predictions (to check whether the surrogate is
calibrated on generated designs) and, for the conditional VAE, the (A, L_gradB) it was
asked for (to check whether it hits targets in real physics, not just in geometry).

Everything goes in data/gen/, which the seeded pipeline never reads, so the pilots can't
leak into any arm.

    python scripts/seeded/gen_pilot.py propose
    python scripts/seeded/gen_pilot.py propose --only diff_w256_s3k --append   # add a finalist
    python scripts/seeded/gen_pilot.py simulate
    python scripts/seeded/gen_pilot.py report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_study as S  # noqa: E402
import generators as G  # noqa: E402
from families import NOVELTY, STRICT, nearest_public  # noqa: E402
from loop import hv_gain, hypervolume  # noqa: E402
from shapes import aspect_ratio, from_vector  # noqa: E402

OUT = S.OUT
DESIGNS, LEDGER = OUT / "pilot_designs.jsonl", OUT / "runs" / "pilot.jsonl"
FINALISTS = G.FINALISTS   # name: (model factory, how to sample); shared with the loop


def propose(n: int, seed: int, only: list[str] | None = None, append: bool = False) -> None:
    J = S.judge()
    d = G.training_set()
    snap = np.load(OUT / "snapshot" / "train.npz")
    ours = NearestNeighbors(n_neighbors=1).fit(snap["X"][snap["source"] == "ours"])
    out, have = [], set()
    if append and DESIGNS.exists():
        have = {json.loads(x)["method"] for x in DESIGNS.open()}
    for name in only or list(FINALISTS):
        if name in have:
            print(f"{name}: already in {DESIGNS.name}; skipped")
            continue
        make, spec = FINALISTS[name]
        strategy, temp, guide = spec.get("strategy", "prior"), spec["temp"], spec.get("guide", 0.0)
        rng = np.random.default_rng(seed)
        model = make().fit(d)
        V, asked = G.generate(model, spec, 4000, rng, d)    # aimed at the public front
        A, dist = aspect_ratio(V), nearest_public(V)
        near = ours.kneighbors(V)[0][:, 0]
        cand = np.flatnonzero((A < S.A_MAX) & (dist >= NOVELTY) & (near >= NOVELTY))
        pick = S.spaced(V, rng.permutation(cand), n)
        p = J["screen"].predict(V[pick])
        for j, i in enumerate(pick):
            b = json.dumps(from_vector(V[i]))
            row = {"id": "gen-" + hashlib.sha1(b.encode()).hexdigest()[:12], "boundary": b,
                   "method": name, "config": model.config(), "strategy": strategy, "temp": temp,
                   "guide": guide, "seed": seed,
                   "novelty": float(dist[i]), "A_geom": float(A[i]),
                   "pred_pass": float(p["p_pass"][j]), "pred_crash": float(p["p_crash"][j]),
                   "pred_L": float(p["L_gradB"][j])}
            if asked is not None:
                row["target_A"], row["target_L"] = float(asked[i, 0]), float(asked[i, 1])
            out.append(row)
        print(f"{name}: {len(cand)} novel candidates of 4000 -> {len(pick)} picked; "
              f"fit {model.fit_info}", flush=True)
    DESIGNS.parent.mkdir(parents=True, exist_ok=True)
    with DESIGNS.open("a" if append else "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} designs -> {DESIGNS}")


def simulate(workers: int) -> None:
    import run_sims
    run_sims.run([json.loads(x) for x in DESIGNS.open()], LEDGER, workers=workers)


def report() -> None:
    designs = {d["id"]: d for d in map(json.loads, DESIGNS.open())}
    rows = {r["id"]: r for r in map(json.loads, LEDGER.open()) if r["id"] in designs}
    J = S.judge()
    pA, pL = J["fA"], J["fL"]
    snap = np.load(OUT / "snapshot" / "train.npz")
    ok_ours = (snap["source"] == "ours") & ~snap["crash"] & (snap["max_violation"] <= 0.01)
    nov_ours = ok_ours.copy(); nov_ours[ok_ours] = nearest_public(snap["X"][ok_ours]) >= NOVELTY
    oA, oL = snap["A"][nov_ours], snap["L"][nov_ours]
    print(f"public 4x4 front HV {hypervolume(pA, pL):.3f}; with all our novel passes so far "
          f"{hypervolume(np.r_[pA, oA], np.r_[pL, oL]):.3f}\n")
    for name in FINALISTS:
        ds = [(designs[i], r) for i, r in rows.items() if designs[i]["method"] == name]
        if not ds:
            continue
        ok = [(d, r) for d, r in ds if r["ok"]]
        ps = [(d, r) for d, r in ok if r["feasible"]]
        pred = np.mean([d["pred_pass"] * (1 - d["pred_crash"]) for d, _ in ds])
        print(f"{name}: {len(ds)} simulated, {len(ds) - len(ok)} crashed, {len(ps)} passed "
              f"({len(ps) / len(ds):.0%}; surrogate predicted {pred:.0%}); strict-novel (>= {STRICT}) "
              f"passes {sum(d['novelty'] >= STRICT for d, _ in ps)}")
        if ps:
            A = np.array([r["aspect_ratio"] for _, r in ps]); L = np.array([r["L_gradB"] for _, r in ps])
            g_pub = hv_gain(A, L, pA, pL); g_all = hv_gain(A, L, np.r_[pA, oA], np.r_[pL, oL])
            print(f"  beyond the public 4x4 front: {int((g_pub > 1e-9).sum())} designs, HV "
                  f"{hypervolume(pA, pL):.3f} -> {hypervolume(np.r_[pA, A], np.r_[pL, L]):.3f}; "
                  f"beyond our best front too: {int((g_all > 1e-9).sum())}")
            print(f"  A range of passes {A.min():.2f}-{A.max():.2f}; L_gradB error of the surrogate "
                  f"{np.median([abs(d['pred_L'] - r['L_gradB']) for d, r in ps]):.2f} (median)")
            if "target_A" in ps[0][0]:
                print(f"  conditional targets, passes: median |A - asked| "
                      f"{np.median([abs(r['aspect_ratio'] - d['target_A']) for d, r in ps]):.2f}, "
                      f"median L_gradB - asked {np.median([r['L_gradB'] - d['target_L'] for d, r in ps]):+.2f}")
        fails = {}
        for d, r in ok:
            if not r["feasible"]:
                k = max(r["violations"], key=r["violations"].get); fails[k] = fails.get(k, 0) + 1
        print(f"  failing designs, worst constraint: {fails}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose"); p.add_argument("--n", type=int, default=60); p.add_argument("--seed", type=int, default=50)
    p.add_argument("--only", help="comma-separated finalist names"); p.add_argument("--append", action="store_true")
    s = sub.add_parser("simulate"); s.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    a = ap.parse_args()
    {"propose": lambda: propose(a.n, a.seed, a.only.split(",") if a.only else None, a.append), "simulate": lambda: simulate(a.workers), "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
