"""The pipeline: propose -> screen -> simulate, in rounds.

Each round:
  1. propose a large pool of candidates (perturb.py, surface.py, or surface.blend, which
     also blends our own new passing designs; later a VAE)
  2. keep novel ones (>= 0.003 from every public design and everything we've simulated)
     with exact aspect ratio < 19.9
  3. --screen: retrain the screening models on everything simulated so far, then rank by
         P(no crash) * P(pass) * (predicted score gain over the current front + 0.1)
     and pick the top `batch`, skipping any within 0.003 of an earlier pick.
     no --screen: pick `batch` at random (the no-ML arm).
  4. simulate the batch at full accuracy (run_sims.py: evaluate.py's simulator with a
     450 s limit per design), log the round to data/seeded/logs/loop_<name>.jsonl.

Re-running the same command resumes: logged rounds are skipped, and a round that was
interrupted mid-simulation finishes its existing picks (data/seeded/loop_<name>_r<k>.jsonl)
rather than picking again.

"Current front" = the public NFP-3 4x4 front + this arm's novel passing designs so far.
Each arm sees the shared starting data plus only its own simulations (arm_view). Every
simulation counts toward the budget, crashes included. Run arms one after another, not at
once: they share train.npz, and two 8-worker runs don't fit in RAM.

    python scripts/seeded/loop.py --name blend_screen --proposer blend --screen --rounds 4 --batch 120
    python scripts/seeded/loop.py --name blend_random --proposer blend --rounds 4 --batch 120
    python scripts/seeded/loop.py ... --dry-run      # pick the next round only -> logs/dryrun_*
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import training_data  # noqa: E402
from families import NOVELTY, load as load_public, nearest_public  # noqa: E402
from perturb import front_mask  # noqa: E402
from screen import TOL, Screen, load_table  # noqa: E402
from shapes import aspect_ratio, to_vector  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "seeded"
LOGS = DATA / "logs"       # round summaries; not in runs/ (read as ledgers) or DATA (read as picks)
REF_A, REF_L = 20.0, -1.0


def hv_gain(A: np.ndarray, L: np.ndarray, fA: np.ndarray, fL: np.ndarray) -> np.ndarray:
    """Score added to the front (fA, fL) by each single new design (A, L)."""
    keep = front_mask(fA, fL)
    o = np.argsort(fA[keep]); xa, xl = fA[keep][o], fL[keep][o]
    x0 = np.r_[-np.inf, xa]; x1 = np.r_[xa, REF_A]; h = np.r_[REF_L, xl]
    width = np.clip(np.minimum(x1, REF_A)[None] - np.maximum(x0[None], A[:, None]), 0, None)
    return (np.clip(L[:, None] - h[None], 0, None) * width).sum(axis=1)


def hypervolume(A: np.ndarray, L: np.ndarray) -> float:
    if len(A) == 0:
        return 0.0
    keep = front_mask(A, L); o = np.argsort(A[keep]); a, l = A[keep][o], L[keep][o]
    return float(np.sum((np.r_[a[1:], REF_A] - a) * (l - REF_L)))


def arm_view(t: dict, name: str) -> dict:
    """The training table as one arm sees it: shared data (dataset, public, pilot, surface
    batch) plus this arm's own simulations, never another arm's. Keeps arms comparable."""
    own = re.compile(rf"loop_{re.escape(name)}_r\d+\.jsonl")
    mine, other = set(), set()
    for f in DATA.glob("loop_*_r*.jsonl"):
        (mine if own.fullmatch(f.name) else other).update(json.loads(x)["id"] for x in f.open())
    keep = ~np.isin(t["id"].astype(str), list(other - mine))
    return {k: v[keep] for k, v in t.items()}


def ours_status(t: dict) -> dict:
    """Our simulated designs: ids, and the novel passing ones."""
    ours = t["source"] == "ours"
    ok = ours & ~t["crash"] & (t["max_violation"] <= TOL)
    nov = np.zeros(len(t["X"]), bool)
    if ok.any():
        nov[ok] = nearest_public(t["X"][ok]) >= NOVELTY
    return {"ids": set(t["id"][ours].astype(str)), "novel_pass": nov}


def propose(proposer: str, n: int, seed: int, ours: dict) -> list[dict]:
    import perturb
    import surface
    if proposer == "random":
        return perturb.propose(n, seed, steps=(0.1, 0.2), n_control=0)
    if proposer == "surface":
        return surface.propose(n, seed)
    if proposer == "blend":
        return surface.blend(n, seed, ours)
    raise ValueError(proposer)


def pick(name: str, proposer: str, screen: bool, batch: int, pool: int, seed: int, r: int,
         pub: dict) -> list[dict]:
    """One round's batch: propose, keep novel candidates, then screen or pick at random."""
    training_data.build()
    t = arm_view(load_table(), name); st = ours_status(t)
    mine = st["novel_pass"]
    fA = np.r_[pub["A"], t["A"][mine]]; fL = np.r_[pub["L"], t["L"][mine]]
    ours = {k: t[k][mine] for k in ("X", "A", "L", "id")}

    cands = [d for d in propose(proposer, pool, seed * 1000 + r, ours) if d["id"] not in st["ids"]]
    V = np.array([to_vector(d["boundary"]) for d in cands])
    A, nov = aspect_ratio(V), nearest_public(V)
    done = t["X"][t["source"] == "ours"]          # don't re-simulate near-copies of our own
    near = NearestNeighbors(n_neighbors=1).fit(done).kneighbors(V)[0][:, 0] if len(done) else np.inf
    keep = (nov >= NOVELTY) & (near >= NOVELTY) & (A < 19.9)
    cands, V, A = [c for c, k in zip(cands, keep) if k], V[keep], A[keep]
    rng = np.random.default_rng(seed * 1000 + r)
    if screen:
        p = Screen().fit(t).predict(V)
        score = (1 - p["p_crash"]) * p["p_pass"] * (hv_gain(A, p["L_gradB"], fA, fL) + 0.1)
        order = np.argsort(-score)
    else:
        order = rng.permutation(len(cands))
    chosen = []
    for i in order:
        if len(chosen) == batch:
            break
        if all(np.linalg.norm(V[i] - V[j]) >= NOVELTY for j in chosen):
            chosen.append(i)
    picks = [dict(cands[i], A_geom=float(A[i]), arm=name, round=r) for i in chosen]
    if screen:
        for i, d in zip(chosen, picks):
            d.update(pred_pass=float(p["p_pass"][i]), pred_L=float(p["L_gradB"][i]),
                     pred_crash=float(p["p_crash"][i]))
    print(f"[{name} round {r}] {len(cands)} novel candidates -> {len(picks)} picked", flush=True)
    return picks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for d in rows:
            fh.write(json.dumps(d) + "\n")


def run(name: str, proposer: str, screen: bool, rounds: int, batch: int, pool: int,
        seed: int, workers: int, dry_run: bool) -> None:
    import run_sims
    pub = load_public()
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"loop_{name}.jsonl"
    logged = {json.loads(x)["round"] for x in log.open()} if log.exists() else set()
    for r in range(rounds):
        if r in logged:
            continue
        infile = DATA / f"loop_{name}_r{r}.jsonl"
        if dry_run:                               # kept out of DATA so arm_view never sees it
            write_jsonl(LOGS / f"dryrun_{name}_r{r}.jsonl", pick(name, proposer, screen, batch, pool, seed, r, pub))
            return
        if infile.exists():                       # interrupted mid-round: finish the same picks
            picks = [json.loads(x) for x in infile.open()]
            print(f"[{name} round {r}] resuming {len(picks)} picks from {infile.name}", flush=True)
        else:
            picks = pick(name, proposer, screen, batch, pool, seed, r, pub)
            write_jsonl(infile, picks)
        rows = run_sims.run(picks, DATA / "runs" / f"loop_{name}.jsonl", workers=workers)
        npass = sum(x["feasible"] for x in rows if x["id"] in {d["id"] for d in picks})
        training_data.build()
        t = arm_view(load_table(), name); st = ours_status(t)
        mine = st["novel_pass"]
        entry = {"round": r, "simulated": len(picks), "passed": npass,
                 "novel_passing_total": int(mine.sum()),
                 "hv_ours_alone": hypervolume(t["A"][mine], t["L"][mine]),
                 "hv_public_plus_ours": hypervolume(np.r_[pub["A"], t["A"][mine]],
                                                    np.r_[pub["L"], t["L"][mine]])}
        with log.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--proposer", choices=["random", "surface", "blend"], default="blend")
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=120)
    ap.add_argument("--pool", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.name, a.proposer, a.screen, a.rounds, a.batch, a.pool, a.seed, a.workers, a.dry_run)


if __name__ == "__main__":
    main()
