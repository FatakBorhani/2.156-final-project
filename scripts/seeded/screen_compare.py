"""Screen comparison: which model should decide what to simulate, and what does it look at?

The screen (screen.py) is where ML paid off in the loop, but it was badly fooled by
generated designs: it predicted 30-97% would pass, and 0-15% did. This compares
pass/fail models on the tests that matter for the loop. Each is trained on the training
table (data/seeded/train.npz: dataset, public, ours), never on its test rows:

  gen_unseen    generated designs, none seen in training: the generator pilot (420,
                unscreened) and the generator loop arms (2,700, screen-picked)
  gen_learned   the generator pilot again, now trained with the loop arms' 2,700 generated
                designs: does seeing generated failures fix the screen?
  blends        the wide blending arms (2,880 sims), trained without them
  repair        the QI repair pilot (repair.py), if it has been simulated

Models: boosted trees as used now (hgb); boosted trees plus two "how far off the passing
surface" features (hgb+dist: distance to the nearest public design, and how far the convex
repair in repair.py would move the design); random forest; an ensemble of 3 neural nets
(unweighted: sklearn 1.5's MLP takes no sample weights); 10-nearest-neighbours on the RMS
surface distance; a Gaussian-process classifier on a 3,000-row subsample (1,000 each of
dataset, public and ours; exact GPs don't scale to 75k rows).

Metrics on each test set: ROC AUC (ranking), predicted vs actual pass rate (calibration:
how badly it is fooled), and the pass rate of the quarter it ranks highest (what a screened
round would actually get).

Then Shapley values (sampling estimate, interventional, no extra packages) for boosted trees
predicting the QI violation and L_gradB, on designs near the front: which of the 80 shape
coefficients drive each.

Result (Nov 18): model type barely matters where the loop uses it. On held-out blends every
model ranks well (AUC 0.92-0.96; its top quarter passes 94-98%). Every model is fooled by
generated designs it has never seen (predicted 58-76% vs. 10% actual). The distance features
didn't help, but they are confounded: public rows exclude their own family, so they look
off-surface yet all pass. Training on the loop's 2,700 generated designs is what helps: the
over-prediction drops from 60% to 28%, and the random forest reaches AUC 0.84. QI is driven
by the m = 1 modes that vary around the ring (R m1 n-1, n+1); L_gradB mostly by the
cross-section size (R and Z m1 n0), i.e. by compactness itself.

Writes results/seeded/screen_compare.json.

    python scripts/seeded/screen_compare.py
    python scripts/seeded/screen_compare.py --skip gp,mlp      # faster
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from families import load as load_public  # noqa: E402
from repair import Repair  # noqa: E402
from screen import TOL, load_table  # noqa: E402
from shapes import MID, N_POL, N_TOR, _R_FREE, _Z_FREE, to_vector  # noqa: E402
from training_data import CONSTRAINTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS, GEN = ROOT / "data" / "seeded" / "runs", ROOT / "data" / "gen"
OUT = ROOT / "results" / "seeded" / "screen_compare.json"
HGB = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, random_state=0)


# ----------------------------------------------------------------------------- features
class Dist:
    """Two off-surface features: distance to the nearest public design, and the distance the
    convex repair would move the design. A public design's own family is excluded, so the
    public rows look like what a new design would."""

    def __init__(self):
        self.pub = load_public()
        self.fix = Repair(self.pub)
        self.nn = NearestNeighbors(n_neighbors=64).fit(self.pub["X"])

    def __call__(self, X: np.ndarray, family: np.ndarray | None = None) -> np.ndarray:
        family = np.full(len(X), -1) if family is None else family
        pf, PX, k = self.pub["family"], self.pub["X"], self.fix.k
        idx = self.nn.kneighbors(X, return_distance=False)
        d_pub, out = np.empty(len(X)), np.empty(len(X))
        for r in range(len(X)):           # the repair, without the design's own family
            cand = idx[r][pf[idx[r]] != family[r]]
            if len(np.unique(pf[cand])) < k:          # inside a big family: search all of them
                full = np.argsort(np.linalg.norm(PX - X[r], axis=1))
                cand = full[pf[full] != family[r]][: 40 * k]
            _, first = np.unique(pf[cand], return_index=True)
            P = PX[cand[np.sort(first)[:k]]]
            w = self._weights(P, X[r])
            d_pub[r] = np.linalg.norm(PX[cand[0]] - X[r])
            out[r] = np.linalg.norm(w @ P - X[r])
        return np.c_[np.log10(d_pub + 1e-4), np.log10(out + 1e-4)]

    @staticmethod
    def _weights(P, v):
        from scipy.optimize import nnls
        w, _ = nnls(np.vstack([P.T, 10.0 * np.ones(len(P))]), np.r_[v, 10.0])
        return w / w.sum() if w.sum() > 0 else np.full(len(P), 1 / len(P))


# ----------------------------------------------------------------------------- models
def models(skip: set[str]) -> dict:
    def hgb(X, y, w, _):
        return HistGradientBoostingClassifier(**HGB).fit(X, y, sample_weight=w)

    def rf(X, y, w, _):
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=6,
                                      random_state=0).fit(X, y, sample_weight=w)

    def knn(X, y, w, _):
        return KNeighborsClassifier(n_neighbors=10, weights="distance").fit(X, y)

    def mlp(X, y, w, _):
        sc = StandardScaler().fit(X)
        nets = [MLPClassifier((256, 256), alpha=1e-4, early_stopping=True, max_iter=200,
                              random_state=s).fit(sc.transform(X), y) for s in range(3)]
        return _Avg(nets, sc)

    def gp(X, y, w, src):
        rng = np.random.default_rng(0)
        rows = np.concatenate([rng.choice(np.flatnonzero(src == s), min(1000, (src == s).sum()), replace=False,
                                          p=w[src == s] / w[src == s].sum()) for s in ("dataset", "public", "ours")])
        sc = StandardScaler().fit(X[rows])
        k = ConstantKernel(1.0) * RBF(length_scale=np.sqrt(X.shape[1]))
        return _Avg([GaussianProcessClassifier(k, random_state=0).fit(sc.transform(X[rows]), y[rows])], sc)

    m = {"hgb": (hgb, False), "hgb+dist": (hgb, True), "rf": (rf, False), "mlp": (mlp, False),
         "knn": (knn, False), "gp": (gp, False)}
    return {k: v for k, v in m.items() if k not in skip}


class _Avg:
    def __init__(self, nets, sc):
        self.nets, self.sc = nets, sc

    def predict_proba(self, X):
        return np.mean([n.predict_proba(self.sc.transform(X)) for n in self.nets], axis=0)


# ----------------------------------------------------------------------------- data
def ledger_rows(designs: Path, ledger: Path) -> dict:
    """Vectors and pass labels for designs simulated outside the training table."""
    if not (designs.exists() and ledger.exists()):
        return {}
    ds = {d["id"]: d for d in map(json.loads, designs.open())}
    rows = [r for r in map(json.loads, ledger.open())
            if r["id"] in ds and r["ok"] and r["fidelity"] == "high_fidelity"]
    return {"X": np.array([to_vector(ds[r["id"]]["boundary"]) for r in rows]),
            "y": np.array([r["max_violation"] <= TOL for r in rows]),
            "group": np.array([ds[r["id"]]["method"] for r in rows])}


def ids_in(pattern: str) -> set[str]:
    return {json.loads(x)["id"] for f in RUNS.glob(pattern) for x in f.open()}


def evaluate(p: np.ndarray, y: np.ndarray) -> dict:
    top = np.argsort(-p)[: max(1, len(p) // 4)]
    return {"n": int(len(y)), "actual": float(y.mean()), "predicted": float(p.mean()),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "top_quarter_pass": float(y[top].mean())}


# ----------------------------------------------------------------------------- Shapley values
def labels() -> list[str]:
    out = []
    for name, mask in (("R", _R_FREE), ("Z", _Z_FREE)):
        out += [f"{name} m{m} n{j - MID:+d}" for m in range(N_POL) for j in range(N_TOR) if mask[m, j]]
    return out


def shapley(f, X: np.ndarray, background: np.ndarray, perms: int, rng) -> np.ndarray:
    """Sampling estimate of interventional Shapley values: for each design and random feature
    order, switch features from a background design to the design one at a time and credit
    each feature with the change in f. Batched: one call to f per (design, order)."""
    n, k = X.shape
    phi = np.zeros((n, k))
    for i, x in enumerate(X):
        for _ in range(perms):
            order = rng.permutation(k)
            b = background[rng.integers(len(background))]
            Z = np.repeat(b[None], k + 1, axis=0)
            for j, c in enumerate(order):
                Z[j + 1:, c] = x[c]
            v = f(Z)
            phi[i, order] += np.diff(v)
    return phi / perms


def explain(t: dict, n: int, perms: int) -> dict:
    rng = np.random.default_rng(0)
    ours = (t["source"] == "ours") & ~t["crash"]
    pub = t["source"] == "public"
    qi = CONSTRAINTS.index("log10_qi_le_-3.5")
    have_qi = ~t["crash"] & np.isfinite(t["violations"][:, qi]) & (t["source"] != "public")
    targets = {
        "QI violation": (have_qi, t["violations"][:, qi]),
        "L_gradB": (~t["crash"], t["L"]),
    }
    explained = rng.choice(np.flatnonzero(ours), n, replace=False)     # designs near the front
    background = t["X"][rng.choice(np.flatnonzero(pub), 64, replace=False)]
    names, out = labels(), {}
    for name, (rows, y) in targets.items():
        rows = rows & np.isfinite(y)
        t0 = time.time()
        model = HistGradientBoostingRegressor(**HGB).fit(t["X"][rows], y[rows], sample_weight=t["weight"][rows])
        phi = shapley(model.predict, t["X"][explained], background, perms, rng)
        imp = np.abs(phi).mean(axis=0)
        order = np.argsort(-imp)
        grid = {"R": np.zeros((N_POL, N_TOR)), "Z": np.zeros((N_POL, N_TOR))}
        k = int(_R_FREE.sum())
        grid["R"][_R_FREE], grid["Z"][_Z_FREE] = imp[:k], imp[k:]
        out[name] = {"mean_abs_shap": imp.tolist(), "top": [[names[i], float(imp[i])] for i in order[:15]],
                     "share_top10": float(imp[order[:10]].sum() / imp.sum()),
                     "by_m": {f"m{m}": float(grid["R"][m].sum() + grid["Z"][m].sum()) for m in range(N_POL)},
                     "grid_R": grid["R"].tolist(), "grid_Z": grid["Z"].tolist()}
        print(f"Shapley, {name} ({time.time() - t0:.0f} s): top 5 "
              + ", ".join(f"{a} {b:.3f}" for a, b in out[name]["top"][:5])
              + f"; top 10 carry {out[name]['share_top10']:.0%}", flush=True)
    out["labels"] = names
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip", default="")
    ap.add_argument("--shap-n", type=int, default=200)
    ap.add_argument("--shap-perms", type=int, default=8)
    a = ap.parse_args()

    t = load_table()
    ok = ~t["crash"]
    y_all = t["max_violation"] <= TOL
    ids = t["id"].astype(str)
    gen_rows = ok & (t["source"] == "ours") & np.char.startswith(ids, "gen-")
    w60 = ok & np.isin(ids, list(ids_in("loop_w60_*.jsonl")))
    pilot = ledger_rows(GEN / "pilot_designs.jsonl", GEN / "runs" / "pilot.jsonl")
    repair = ledger_rows(GEN / "repair_designs.jsonl", GEN / "runs" / "repair.jsonl")
    print(f"table: {ok.sum()} non-crashed rows; generated in the loop {gen_rows.sum()}, "
          f"wide blends {w60.sum()}, generator pilot {len(pilot.get('y', []))}, "
          f"repair pilot {len(repair.get('y', []))}", flush=True)

    t0 = time.time()
    dist = Dist()
    fam = np.where(t["source"] == "public", t["family"], -1)
    F_table = dist(t["X"], fam)
    F_pilot = dist(pilot["X"]) if pilot else None
    F_repair = dist(repair["X"]) if repair else None
    print(f"off-surface features computed in {time.time() - t0:.0f} s", flush=True)

    splits = {   # name: (training rows, [(test name, X, features, y, groups)])
        "gen_unseen": (ok & ~gen_rows, [("generator pilot", pilot["X"], F_pilot, pilot["y"], pilot["group"]),
                                        ("generator loop arms", t["X"][gen_rows], F_table[gen_rows],
                                         y_all[gen_rows], None)]),
        "gen_learned": (ok, [("generator pilot", pilot["X"], F_pilot, pilot["y"], pilot["group"])]),
        "blends": (ok & ~w60, [("wide blend arms", t["X"][w60], F_table[w60], y_all[w60], None)]),
    }
    if repair:
        splits["repair"] = (ok, [("repair pilot", repair["X"], F_repair, repair["y"], repair["group"])])

    results = {}
    for name, (make, use_dist) in models(set(filter(None, a.skip.split(",")))).items():
        for split, (rows, tests) in splits.items():
            if split == "repair" and name not in ("hgb", "hgb+dist"):
                continue
            t0 = time.time()
            Xtr = np.c_[t["X"][rows], F_table[rows]] if use_dist else t["X"][rows]
            m = make(Xtr, y_all[rows], t["weight"][rows], t["source"][rows])
            for test, X, F, y, groups in tests:
                p = m.predict_proba(np.c_[X, F] if use_dist else X)[:, 1]
                r = evaluate(p, y)
                if groups is not None:
                    r["by_group"] = {g: evaluate(p[groups == g], y[groups == g]) for g in np.unique(groups)}
                results.setdefault(name, {}).setdefault(split, {})[test] = r
                print(f"{name:9s} {split:12s} {test:20s} AUC {r['auc'] if r['auc'] is None else round(r['auc'], 3)}  "
                      f"predicted {r['predicted']:.0%} vs actual {r['actual']:.0%}  "
                      f"top quarter {r['top_quarter_pass']:.0%}  ({time.time() - t0:.0f} s)", flush=True)

    out = {"models": results, "shapley": explain(t, a.shap_n, a.shap_perms)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
