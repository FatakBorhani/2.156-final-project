"""Screening models: millisecond guesses at what a ~75 s simulation would say.

Four gradient-boosted tree models on the 80 standard-form coefficients
(training_data.py builds the table):

  crash    P(VMEC++ fails on this shape)                dataset + ours
  pass     P(max violation <= 0.01)                     dataset + public + ours
  margin   log10(max violation + 0.001)                 dataset + public + ours
  L_gradB  the buildability objective                   dataset + public + ours

Aspect ratio is not modelled: shapes.aspect_ratio computes it exactly from the shape.
Public designs are weighted 1 / family size, so each family counts equally (one team
supplied 95% of them).

    python scripts/seeded/screen.py validate   # honest accuracy: held-out families, held-out pilot
    python scripts/seeded/screen.py fit        # train on everything -> data/seeded/models/screen.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
DATA = Path(__file__).resolve().parents[2] / "data" / "seeded"
MODEL = DATA / "models" / "screen.joblib"
TOL = 0.01


def load_table() -> dict:
    with np.load(DATA / "train.npz", allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    w = np.ones(len(t["X"]))
    pub = t["source"] == "public"
    sizes = np.bincount(t["family"][pub])
    w[pub] = (pub.sum() / len(sizes)) / sizes[t["family"][pub]]   # equal total per family
    t["weight"] = w
    return t


class Screen:
    """Fit on a row mask of the training table; predict for standard-form vectors."""

    def fit(self, t: dict, rows: np.ndarray | None = None) -> "Screen":
        rows = np.ones(len(t["X"]), bool) if rows is None else rows
        ok = rows & ~t["crash"]
        crash_rows = rows & (t["source"] != "public")
        kw = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, random_state=0)
        self.crash = HistGradientBoostingClassifier(**kw).fit(
            t["X"][crash_rows], t["crash"][crash_rows], sample_weight=t["weight"][crash_rows])
        self.pass_ = HistGradientBoostingClassifier(**kw).fit(
            t["X"][ok], t["max_violation"][ok] <= TOL, sample_weight=t["weight"][ok])
        self.margin = HistGradientBoostingRegressor(**kw).fit(
            t["X"][ok], np.log10(t["max_violation"][ok] + 1e-3), sample_weight=t["weight"][ok])
        self.L = HistGradientBoostingRegressor(**kw).fit(
            t["X"][ok], t["L"][ok], sample_weight=t["weight"][ok])
        return self

    def predict(self, X: np.ndarray) -> dict:
        X = np.atleast_2d(X)
        return {"p_crash": self.crash.predict_proba(X)[:, 1],
                "p_pass": self.pass_.predict_proba(X)[:, 1],
                "max_violation": 10 ** self.margin.predict(X) - 1e-3,
                "L_gradB": self.L.predict(X)}


def validate() -> None:
    t = load_table()
    pub = t["source"] == "public"
    # 1. held-out public families: L_gradB and margin on designs ~novel to the model
    errs, rank = [], []
    for train_pub, test_pub in GroupKFold(5).split(t["X"][pub], groups=t["family"][pub]):
        test = np.flatnonzero(pub)[test_pub]
        rows = ~np.isin(t["id"], t["id"][test])        # also hides our re-scored copies
        p = Screen().fit(t, rows).predict(t["X"][test])
        errs.append(np.abs(p["L_gradB"] - t["L"][test]))
    errs = np.concatenate(errs)
    print(f"L_gradB on held-out public families: median |error| {np.median(errs):.2f}, "
          f"p90 {np.quantile(errs, .9):.2f}  (public L_gradB spans 4.9-19.0)")
    # 2. crashes: random 20% of dataset rows held out
    rng = np.random.default_rng(0)
    ds = np.flatnonzero(t["source"] == "dataset"); test = rng.choice(ds, len(ds) // 5, replace=False)
    rows = np.ones(len(t["X"]), bool); rows[test] = False
    p = Screen().fit(t, rows).predict(t["X"][test])
    print(f"crash on held-out dataset rows: AUC {roc_auc_score(t['crash'][test], p['p_crash']):.3f} "
          f"(base rate {t['crash'][test].mean():.0%})")
    # 3. the real test: our perturbed designs, model trained without any of them
    pilot = [json.loads(line)["id"] for line in (DATA / "pilot.jsonl").open()]
    pert = (t["source"] == "ours") & np.isin(t["id"].astype(str), pilot)   # every proposer uses "pert-" ids
    ok = pert & ~t["crash"]
    if ok.sum() < 30 or len(set(t["max_violation"][ok] <= TOL)) < 2:
        print(f"pilot: only {ok.sum()} scored perturbed designs so far; re-run when the pilot is done")
        return
    p = Screen().fit(t, ~pert).predict(t["X"][pert])
    passed = (t["max_violation"][pert] <= TOL) & ~t["crash"][pert]
    print(f"pilot ({pert.sum()} perturbed designs, {passed.mean():.0%} passed): "
          f"pass AUC {roc_auc_score(passed, p['p_pass']):.3f}, "
          f"margin-rank AUC {roc_auc_score(passed, -p['max_violation']):.3f}")
    top = np.argsort(-p["p_pass"])[: max(1, pert.sum() // 4)]
    print(f"  pass rate if we had simulated only the top quarter by P(pass): {passed[top].mean():.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["validate", "fit"])
    args = ap.parse_args()
    if args.cmd == "validate":
        validate()
    else:
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(Screen().fit(load_table()), MODEL)
        print(f"-> {MODEL}")


if __name__ == "__main__":
    main()
