"""Track B's method on Track A's designs: blend + screen, starting from Leif's dataset-only
passing designs, with no public leaderboard design anywhere in the search.

  parents   Leif's NFP-3 4x4 feasible designs (results/feasibility_tests/*_feasible_designs.jsonl,
            Tests 10-17; Test 18's 5x5 designs are left out because this pipeline is 4x4)
            plus this arm's own novel passes, which can be blended again
  screen    the same four boosted-tree models as screen.py, trained on Proxima's dataset +
            Leif's designs + this arm's own simulations (no public designs, no Track B sims)
  novelty   >= 0.003 from every one of Leif's designs and from everything this arm simulated
  front     Leif's 4x4 front (HV 138.82 from the committed files) + this arm's novel passes

Supply: Leif's 366 designs form only 8 families (two hold 282), so blends crowd together.
Round 0 (seed 50) has ~11k novel candidates but only 36 spaced >= 0.003 apart with Track B's
settings (window 2); window 4 gives 69, same-family blends don't help (32). Defaults are
therefore window 4 and batch 40, blend weights t in [0.3, 0.7] as in Track B. Leif's designs
and these candidates sit ~0.064 from the nearest public design (6x the strict threshold).

Distance to the nearest public design is recorded for each pick, for reporting only; it is
never used to choose anything.

Each round: propose `pool` blends of two designs from different families in the first `layers`
non-dominated layers with aspect ratios within `window`; keep novel ones; --screen ranks them
by P(no crash) * P(pass) * (predicted gain over the current front + 0.1), otherwise random;
take `batch` spaced >= 0.003 apart; simulate at high fidelity (run_sims.py). Data lives in
data/track_a/, which training_data.py never reads, so Track B's arms are unaffected.
Re-running resumes (finished rounds are skipped; an interrupted round finishes its picks).

    python scripts/seeded/track_a_loop.py --name ta_screen_s50 --screen --seed 50
    python scripts/seeded/track_a_loop.py --name ta_random_s50 --seed 50
    python scripts/seeded/track_a_loop.py ... --dry-run     # round 0's pick only, nothing simulated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from families import FAMILY_LINK, NOVELTY, nearest_public  # noqa: E402
from loop import hv_gain, hypervolume  # noqa: E402
from perturb import design  # noqa: E402
from screen import TOL, Screen  # noqa: E402
from shapes import N_POL, N_TOR, aspect_ratio, to_vector  # noqa: E402
from surface import layer_mask  # noqa: E402
from training_data import dataset_part  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "results" / "feasibility_tests"
DATA = ROOT / "data" / "track_a"
T_RANGE = (0.3, 0.7)          # blend weight range, as in surface.py


def leif_designs() -> dict:
    """Leif's unique 4x4 feasible designs: X (standard form), A, L, worst violation, family."""
    rows = []
    for f in sorted(TESTS.glob("*_feasible_designs.jsonl")):
        for r in map(json.loads, f.open()):
            b = json.loads(r["boundary"])
            if b["n_field_periods"] == 3 and np.shape(b["r_cos"]) == (N_POL, N_TOR):
                rows.append((to_vector(b), r["aspect_ratio"], r["L_gradB"], r["worst"], f.stem))
    X = np.array([r[0] for r in rows])
    _, keep = np.unique(np.round(X, 9), axis=0, return_index=True)
    keep = np.sort(keep)
    X = X[keep]
    graph = NearestNeighbors().fit(X).radius_neighbors_graph(X, radius=FAMILY_LINK)
    _, family = connected_components(graph, directed=False)
    return {"X": X, "A": np.array([rows[i][1] for i in keep]), "L": np.array([rows[i][2] for i in keep]),
            "worst": np.array([rows[i][3] for i in keep]), "family": family,
            "test": np.array([rows[i][4] for i in keep])}


def own_results(name: str) -> tuple[list[dict], dict]:
    """This arm's picks (all rounds) and its ledger rows by id (timeouts dropped: machine load)."""
    picks = [d for f in sorted(DATA.glob(f"picks/{name}_r*.jsonl")) for d in map(json.loads, f.open())]
    ledger = DATA / "runs" / f"{name}.jsonl"
    rows = {}
    if ledger.exists():
        for r in map(json.loads, ledger.open()):
            if r["fidelity"] == "high_fidelity" and not r.get("timeout"):
                rows[r["id"]] = r
    return picks, rows


def state(name: str, leif: dict) -> dict:
    """What this arm knows: its simulated designs and its novel passes."""
    picks, rows = own_results(name)
    sim = [d for d in picks if d["id"] in rows]
    X = np.array([to_vector(d["boundary"]) for d in sim]).reshape(-1, leif["X"].shape[1])
    r = [rows[d["id"]] for d in sim]
    crash = np.array([not x["ok"] for x in r], bool)
    mv = np.array([x["max_violation"] if x["ok"] else np.nan for x in r], float)
    L = np.array([x.get("L_gradB", np.nan) if x["ok"] else np.nan for x in r], float)
    A = np.array([x.get("aspect_ratio", np.nan) if x["ok"] else np.nan for x in r], float)
    d_leif = (NearestNeighbors(n_neighbors=1).fit(leif["X"]).kneighbors(X)[0][:, 0]
              if len(X) else np.zeros(0))
    novel_pass = ~crash & (np.nan_to_num(mv, nan=1.0) <= TOL) & (d_leif >= NOVELTY)
    return {"X": X, "crash": crash, "max_violation": mv, "L": L, "A": A,
            "ids": [d["id"] for d in sim], "novel_pass": novel_pass}


def table(leif: dict, st: dict) -> dict:
    """Screen training table: dataset + Leif's designs + this arm's simulations."""
    ds = dataset_part()
    ds_mv = np.nanmax(np.maximum(ds["violations"], 0), axis=1)
    parts = [
        ("dataset", ds["X"], ds["crash"], ds_mv, ds["L"]),
        ("leif", leif["X"], np.zeros(len(leif["X"]), bool), leif["worst"], leif["L"]),
        ("ours", st["X"], st["crash"], st["max_violation"], st["L"]),
    ]
    t = {"X": np.concatenate([p[1] for p in parts]),
         "crash": np.concatenate([p[2] for p in parts]),
         "max_violation": np.concatenate([p[3] for p in parts]),
         "L": np.concatenate([p[4] for p in parts]),
         "source": np.concatenate([[p[0]] * len(p[1]) for p in parts])}
    t["weight"] = np.ones(len(t["X"]))
    good = t["crash"] | (np.isfinite(t["max_violation"]) & np.isfinite(t["L"]))
    return {k: v[good] for k, v in t.items()}


def blends(rng, X, A, fam, ids, parents, n, window, same_family=False) -> list[dict]:
    out = []
    for _ in range(n):
        a = parents[rng.integers(len(parents))]
        other_fam = np.ones(len(parents), bool) if same_family else fam[parents] != fam[a]
        others = parents[other_fam & (parents != a) & (np.abs(A[parents] - A[a]) < window)]
        if len(others) == 0:
            others = parents[fam[parents] != fam[a]]
        if len(others) == 0:
            continue
        b = others[rng.integers(len(others))]
        t = rng.uniform(*T_RANGE)
        out.append(design((1 - t) * X[a] + t * X[b], method="blend", parent=str(ids[a]),
                          parent_b=str(ids[b]), step=round(float(t), 3)))
    return out


def pick(name: str, screen: bool, batch: int, pool: int, seed: int, r: int, layers: int,
         window: float, leif: dict, same_family: bool = False) -> list[dict]:
    st = state(name, leif)
    mine = st["novel_pass"]
    X = np.r_[leif["X"], st["X"][mine]]
    A = np.r_[leif["A"], st["A"][mine]]
    L = np.r_[leif["L"], st["L"][mine]]
    fam = np.r_[leif["family"], -1 - np.arange(mine.sum())]
    ids = np.array([f"leif-{i}" for i in range(len(leif["X"]))]
                   + [st["ids"][j] for j in np.flatnonzero(mine)])
    parents = np.flatnonzero(layer_mask(A, L, layers))
    rng = np.random.default_rng(seed * 1000 + r)
    cands = list({d["id"]: d for d in blends(rng, X, A, fam, ids, parents, pool, window, same_family)}.values())
    V = np.array([to_vector(d["boundary"]) for d in cands])
    Ag = aspect_ratio(V)
    d_leif = NearestNeighbors(n_neighbors=1).fit(leif["X"]).kneighbors(V)[0][:, 0]
    near = (NearestNeighbors(n_neighbors=1).fit(st["X"]).kneighbors(V)[0][:, 0]
            if len(st["X"]) else np.full(len(V), np.inf))
    keep = (d_leif >= NOVELTY) & (near >= NOVELTY) & (Ag < 19.9)
    cands, V, Ag, d_leif = [c for c, k in zip(cands, keep) if k], V[keep], Ag[keep], d_leif[keep]
    if screen:
        p = Screen().fit(table(leif, st)).predict(V)
        fA = np.r_[leif["A"], st["A"][mine]]; fL = np.r_[leif["L"], st["L"][mine]]
        score = (1 - p["p_crash"]) * p["p_pass"] * (hv_gain(Ag, p["L_gradB"], fA, fL) + 0.1)
        order = np.argsort(-score)
    else:
        order = rng.permutation(len(cands))
    chosen = []
    for i in order:
        if len(chosen) == batch:
            break
        if all(np.linalg.norm(V[i] - V[j]) >= NOVELTY for j in chosen):
            chosen.append(i)
    d_pub = nearest_public(V[chosen]) if chosen else []
    picks = []
    for k, i in enumerate(chosen):
        d = dict(cands[i], A_geom=float(Ag[i]), d_leif=float(d_leif[i]), d_public=float(d_pub[k]),
                 arm=name, round=r, seed=seed)
        if screen:
            d.update(pred_pass=float(p["p_pass"][i]), pred_L=float(p["L_gradB"][i]),
                     pred_crash=float(p["p_crash"][i]))
        picks.append(d)
    print(f"[{name} round {r}] {len(parents)} parents, {len(cands)} novel candidates -> "
          f"{len(picks)} picked" + (f", mean predicted pass {np.mean([d['pred_pass'] for d in picks]):.2f}"
                                    if screen and picks else ""), flush=True)
    return picks


def run(name: str, screen: bool, rounds: int, batch: int, pool: int, seed: int, layers: int,
        window: float, workers: int, dry_run: bool, same_family: bool = False) -> None:
    import run_sims
    leif = leif_designs()
    for sub in ("picks", "runs", "logs"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    log = DATA / "logs" / f"{name}.jsonl"
    logged = {json.loads(x)["round"] for x in log.open()} if log.exists() else set()
    print(f"[{name}] Leif's 4x4 designs: {len(leif['X'])} unique, {leif['family'].max() + 1} families, "
          f"front HV {hypervolume(leif['A'], leif['L']):.3f}", flush=True)
    for r in range(rounds):
        if r in logged:
            continue
        if dry_run:
            picks = pick(name, screen, batch, pool, seed, r, layers, window, leif, same_family)
            with (DATA / "logs" / f"dryrun_{name}_r{r}.jsonl").open("w") as fh:
                fh.writelines(json.dumps(d) + "\n" for d in picks)
            return
        infile = DATA / "picks" / f"{name}_r{r}.jsonl"
        if infile.exists():
            picks = [json.loads(x) for x in infile.open()]
            print(f"[{name} round {r}] resuming {len(picks)} picks", flush=True)
        else:
            picks = pick(name, screen, batch, pool, seed, r, layers, window, leif, same_family)
            with infile.open("w") as fh:
                fh.writelines(json.dumps(d) + "\n" for d in picks)
        rows = run_sims.run(picks, DATA / "runs" / f"{name}.jsonl", workers=workers)
        ids = {d["id"] for d in picks}
        st = state(name, leif)
        mine = st["novel_pass"]
        entry = {"round": r, "simulated": len(picks),
                 "passed": sum(bool(x["feasible"]) for x in rows if x["id"] in ids),
                 "timeouts": sum(bool(x.get("timeout")) for x in rows if x["id"] in ids),
                 "novel_passing_total": int(mine.sum()),
                 "hv_ours_alone": hypervolume(st["A"][mine], st["L"][mine]),
                 "hv_leif_plus_ours": hypervolume(np.r_[leif["A"], st["A"][mine]],
                                                  np.r_[leif["L"], st["L"][mine]]),
                 "min_A_pass": float(np.min(st["A"][mine])) if mine.any() else None}
        with log.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--same-family", action="store_true",
                    help="also blend two designs from the same family (Leif's 366 designs form only "
                         "8 families; novelty still requires >= 0.003 from all of them)")
    a = ap.parse_args()
    run(a.name, a.screen, a.rounds, a.batch, a.pool, a.seed, a.layers, a.window, a.workers, a.dry_run,
        a.same_family)


if __name__ == "__main__":
    main()
