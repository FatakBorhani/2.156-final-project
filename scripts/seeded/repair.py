"""QI repair pilot: snap generated designs back onto the passing surface, then simulate.

The generator pilots (gen_pilot.py) failed on precision: VAE and diffusion designs land a
few percent past the QI limit, because small errors spread over all 80 coefficients. Blends
(convex combinations of real passing designs) pass ~44% of the time. So the repair replaces
each generated design v with the closest convex combination of its K nearest public passing
designs, one per family (non-negative weights summing to 1). The generator still chooses
where to go; the repair only removes what doesn't look like a real design. It costs no
simulations, so the comparison with blending stays at equal budget.

Three groups, `--n` designs each, picked like the generator pilot (novel: >= 0.003 from
every public design and from everything we've simulated; A < 19.9; random, spaced 0.003):

  repair:vae     VAE finalist (vae_z32_b0.1, temp 0.7), then repaired; unrepaired it passed 15%
  repair:diff    diffusion finalist (diff_w256_s3k_t0.5), then repaired; unrepaired 5%
  repair:noise   control, no ML: a random public design plus random noise as large as the
                 VAE's typical distance from real designs, then repaired. Result: the repair
                 removes nearly all of the noise (0% of 4,000 stay novel), so this group is
                 tiny. Random noise points off the passing surface; the VAE's offsets partly
                 run along it.
  repair:local   the working control, no ML: around a random public design, a random convex
                 combination of the same K nearest families (Dirichlet weights, alpha 0.3, so
                 about 2.5 parents count, like the repaired designs), then the same repair. If
                 this passes as often as the repaired generators, the generator's choice of
                 where to go adds nothing.

Result (Nov 18, 122 simulations): repair lifts the VAE from 15% to 67% passing and diffusion
from 5% to 77%, but the no-ML local control passes 90%. Within 0.01 of a public design all
three pass ~90%; the generators' designs beyond 0.01 nearly all fail (1 of 14). The
generators add nothing over local blending.

Everything goes in data/gen/ (repair_designs.jsonl, runs/repair.jsonl), which the seeded
pipeline never reads.

    python scripts/seeded/repair.py propose
    python scripts/seeded/repair.py simulate
    python scripts/seeded/repair.py report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_study as S  # noqa: E402
import generators as G  # noqa: E402
from families import NOVELTY, STRICT, nearest_public  # noqa: E402
from loop import hv_gain, hypervolume  # noqa: E402
from shapes import aspect_ratio, from_vector  # noqa: E402

OUT = S.OUT
DESIGNS, LEDGER = OUT / "repair_designs.jsonl", OUT / "runs" / "repair.jsonl"
GENERATORS = {"repair:vae": "vae_z32_b0.1", "repair:diff": "diff_w256_s3k_t0.5"}
GROUPS = ("repair:vae", "repair:diff", "repair:noise", "repair:local")
RAW_PASS = {"repair:vae": 0.15, "repair:diff": 0.05}       # unrepaired, from gen_pilot.py
K = 8                                                       # parents per repair, one per family


class Repair:
    """Closest convex combination of the K nearest public passing designs, one per family."""

    def __init__(self, d: dict, k: int = K):
        self.X, self.family, self.k = d["X"], d["family"], k
        self.nn = NearestNeighbors(n_neighbors=min(len(self.X), 40 * k)).fit(self.X)

    def __call__(self, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Repaired designs, and how many parents each one effectively uses (1 / sum w^2)."""
        idx = self.nn.kneighbors(V, return_distance=False)
        out, used = np.empty_like(V), np.empty(len(V))
        for r, (v, cand) in enumerate(zip(V, idx)):
            _, first = np.unique(self.family[cand], return_index=True)
            P = self.X[cand[np.sort(first)[: self.k]]]                  # nearest member of each family
            lam = 10.0                                                    # weights sum to 1 (soft, then exact)
            w, _ = nnls(np.vstack([P.T, lam * np.ones(len(P))]), np.r_[v, lam])
            w = w / w.sum() if w.sum() > 0 else np.full(len(P), 1 / len(P))
            out[r], used[r] = w @ P, 1 / np.sum(w ** 2)
        return out, used


def raw_designs(group: str, n: int, rng, d: dict, vae_scale: float | None, fix: Repair | None = None) -> np.ndarray:
    if group == "repair:local":
        base = rng.choice(len(d["X"]), n, p=d["weight"])
        idx = fix.nn.kneighbors(d["X"][base], return_distance=False)
        out = np.empty((n, d["X"].shape[1]))
        for r, cand in enumerate(idx):
            _, first = np.unique(fix.family[cand], return_index=True)
            P = fix.X[cand[np.sort(first)[: fix.k]]]
            out[r] = rng.dirichlet(np.full(len(P), 0.3)) @ P
        return out
    if group == "repair:noise":
        base = d["X"][rng.choice(len(d["X"]), n, p=d["weight"])]
        step = rng.standard_normal(base.shape)
        return base + vae_scale * step / np.linalg.norm(step, axis=1, keepdims=True)
    make, spec = G.FINALISTS[GENERATORS[group]]
    return G.generate(make().fit(d), spec, n, rng, d)[0]          # aimed at the public front


def propose(n: int, seed: int, only: list[str] | None = None, append: bool = False) -> None:
    J = S.judge()
    d = G.training_set()
    fix = Repair(d)
    snap = np.load(OUT / "snapshot" / "train.npz")
    ours = NearestNeighbors(n_neighbors=1).fit(snap["X"][snap["source"] == "ours"])
    have = {json.loads(x)["method"] for x in DESIGNS.open()} if append and DESIGNS.exists() else set()
    out, vae_scale = [], None
    for group in only or GROUPS:
        if group in have:
            print(f"{group}: already in {DESIGNS.name}; skipped")
            continue
        if group == "repair:noise" and vae_scale is None:
            raise SystemExit("repair:noise needs repair:vae in the same run (it sets the noise size)")
        rng = np.random.default_rng(seed)
        V0 = raw_designs(group, 4000, rng, d, vae_scale, fix)
        d0 = nearest_public(V0)
        if group == "repair:vae":
            vae_scale = float(np.median(d0))                     # the control's noise size
        V, used = fix(V0)
        moved = np.linalg.norm(V - V0, axis=1)
        A, dist = aspect_ratio(V), nearest_public(V)
        near = ours.kneighbors(V)[0][:, 0]
        cand = np.flatnonzero((A < S.A_MAX) & (dist >= NOVELTY) & (near >= NOVELTY))
        pick = S.spaced(V, rng.permutation(cand), n)
        p0, p = J["screen"].predict(V0[pick]), J["screen"].predict(V[pick])
        for j, i in enumerate(pick):
            b = json.dumps(from_vector(V[i]))
            out.append({"id": "rep-" + hashlib.sha1(b.encode()).hexdigest()[:12], "boundary": b,
                        "method": group, "seed": seed, "k": K,
                        "raw_novelty": float(d0[i]), "moved": float(moved[i]),
                        "novelty": float(dist[i]), "parents_used": float(used[i]), "A_geom": float(A[i]),
                        "pred_pass_raw": float(p0["p_pass"][j] * (1 - p0["p_crash"][j])),
                        "pred_pass": float(p["p_pass"][j] * (1 - p["p_crash"][j])),
                        "pred_L": float(p["L_gradB"][j])})
        print(f"{group}: raw distance to a public design {np.median(d0):.4f} (median); repair moved "
              f"{np.median(moved):.4f}; after repair {np.median(dist):.4f}, {np.mean(dist >= NOVELTY):.0%} "
              f"novel; {len(cand)} candidates -> {len(pick)} picked, using {np.median(used[pick]):.1f} "
              f"parents (median)", flush=True)
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
    pub = G.training_set()
    pA, pL = pub["A"], pub["L"]
    for group in GROUPS:
        ds = [(designs[i], r) for i, r in rows.items() if designs[i]["method"] == group]
        if not ds:
            continue
        ok = [(d, r) for d, r in ds if r["ok"]]
        ps = [(d, r) for d, r in ok if r["feasible"]]
        k, n = len(ps), len(ds)
        se = np.sqrt(max(k, 1) * max(n - k, 1) / n) / n
        raw = f"; unrepaired {RAW_PASS[group]:.0%}" if group in RAW_PASS else ""
        print(f"{group}: {n} simulated, {n - len(ok)} crashed, {k} passed ({k / n:.0%} ± {se:.0%}{raw}); "
              f"screen predicted {np.mean([d['pred_pass'] for d, _ in ds]):.0%} after repair, "
              f"{np.mean([d['pred_pass_raw'] for d, _ in ds]):.0%} before")
        nov = np.array([d["novelty"] for d, _ in ps])
        if k:
            A = np.array([r["aspect_ratio"] for _, r in ps]); L = np.array([r["L_gradB"] for _, r in ps])
            print(f"  passes: novelty median {np.median(nov):.4f}, {int((nov >= STRICT).sum())} at >= {STRICT}; "
                  f"{int((hv_gain(A, L, pA, pL) > 1e-9).sum())} beyond the public front "
                  f"(HV {hypervolume(pA, pL):.2f} -> {hypervolume(np.r_[pA, A], np.r_[pL, L]):.2f})")
        fails = {}
        for d, r in ok:
            if not r["feasible"]:
                c = max(r["violations"], key=r["violations"].get); fails[c] = fails.get(c, 0) + 1
        print(f"  failing designs, worst constraint: {fails}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose"); p.add_argument("--n", type=int, default=30); p.add_argument("--seed", type=int, default=60)
    p.add_argument("--only", help="comma-separated groups"); p.add_argument("--append", action="store_true")
    s = sub.add_parser("simulate"); s.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    a = ap.parse_args()
    {"propose": lambda: propose(a.n, a.seed, a.only.split(",") if a.only else None, a.append),
     "simulate": lambda: simulate(a.workers), "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
