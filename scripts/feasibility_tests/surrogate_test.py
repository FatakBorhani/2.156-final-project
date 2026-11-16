"""Test 1: can a quick surrogate predict the Benchmark 3 metrics, especially QI?

Trains two off-the-shelf models on the 81 boundary coefficients of 3-field-period designs
(80/20 random split) to predict the 7 metrics, then checks accuracy where it matters:
  - QI for the best-QI designs (the region we need to reach)
  - pass/fail per constraint, overall and near each limit
  - ranking designs by worst violation (can it screen candidates?)
This is a throwaway baseline to test the plan, not the project's surrogate.

    python scripts/feasibility_tests/surrogate_test.py
    -> results/feasibility_tests/surrogate.json, surrogate_qi_parity.png
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CONSTRAINTS, ROOT, TARGETS, load_designs, violations_from_targets  # noqa: E402

SEED = 0
OUT = ROOT / "results" / "feasibility_tests"


def fit_gbt(Xtr, Ytr, Xte):
    pred = {}
    for k in Ytr.columns:
        m = HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=0.05, max_leaf_nodes=63,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
            random_state=SEED,
        )
        m.fit(Xtr, Ytr[k])
        pred[k] = m.predict(Xte)
    return pd.DataFrame(pred, index=Xte.index)


def fit_mlp(Xtr, Ytr, Xte):
    xs, ys = StandardScaler().fit(Xtr), StandardScaler().fit(Ytr)
    m = MLPRegressor(
        hidden_layer_sizes=(256, 256, 256), activation="relu", alpha=1e-5,
        batch_size=256, learning_rate_init=1e-3, max_iter=300,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=SEED,
    )
    m.fit(xs.transform(Xtr), ys.transform(Ytr))
    p = ys.inverse_transform(m.predict(xs.transform(Xte)))
    return pd.DataFrame(p, index=Xte.index, columns=Ytr.columns)


def evaluate(pred: pd.DataFrame, true: pd.DataFrame) -> dict:
    out = {"per_metric": {}}
    for k in TARGETS:
        out["per_metric"][k] = {
            "r2": round(float(r2_score(true[k], pred[k])), 3),
            "mae": round(float(mean_absolute_error(true[k], pred[k])), 4),
            "std_of_true": round(float(true[k].std()), 4),
        }

    q, qp = true["log10_qi"], pred["log10_qi"]
    out["qi_in_the_region_we_need"] = {}
    for thr in (-3.0, -3.25):
        sel = q < thr
        out["qi_in_the_region_we_need"][f"true_log10_qi_below_{thr}"] = {
            "n_test_designs": int(sel.sum()),
            "mae": round(float((qp[sel] - q[sel]).abs().mean()), 3),
            "mean_error (+ = predicts worse QI than true)": round(
                float((qp[sel] - q[sel]).mean()), 3
            ),
        }

    vt, vp = violations_from_targets(true), violations_from_targets(pred)
    out["pass_fail_accuracy"] = {}
    for k in CONSTRAINTS:
        near = vt[k].abs() < 0.2
        out["pass_fail_accuracy"][k] = {
            "all": round(float(((vp[k] <= 0.01) == (vt[k] <= 0.01)).mean()), 3),
            "within_20pct_of_limit": round(
                float(((vp[k] <= 0.01) == (vt[k] <= 0.01))[near].mean()), 3
            ),
            "n_within_20pct": int(near.sum()),
        }

    wt, wp = vt.clip(lower=0).max(axis=1), vp.clip(lower=0).max(axis=1)
    close = wt <= 0.5
    top100_true = set(wt.nsmallest(100).index)
    out["worst_violation"] = {
        "spearman_all": round(float(stats.spearmanr(wp, wt)[0]), 3),
        "mae_for_designs_with_true_worst_le_0.5": round(
            float((wp[close] - wt[close]).abs().mean()), 3
        ),
        "n_designs_with_true_worst_le_0.5": int(close.sum()),
        "recall_of_true_100_closest_in_predicted_top_500": round(
            len(top100_true & set(wp.nsmallest(500).index)) / 100, 2
        ),
        "recall_of_true_100_closest_in_predicted_top_1000": round(
            len(top100_true & set(wp.nsmallest(1000).index)) / 100, 2
        ),
    }
    return out


def main() -> None:
    X, Y, _, _ = load_designs(3)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    print(f"train {len(Xtr)}, test {len(Xte)}, {X.shape[1]} inputs", flush=True)

    results, preds = {"n_train": len(Xtr), "n_test": len(Xte)}, {}
    for name, fit in (("boosted_trees", fit_gbt), ("mlp", fit_mlp)):
        t0 = time.perf_counter()
        preds[name] = fit(Xtr, Ytr, Xte)
        results[name] = {"train_seconds": round(time.perf_counter() - t0)}
        results[name].update(evaluate(preds[name], Yte))
        print(name, json.dumps(results[name], indent=2), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "surrogate.json").write_text(json.dumps(results, indent=2))

    # Parity plot of QI for both models, zoomed on the region that matters
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150, sharey=True)
    q = Yte["log10_qi"]
    for ax, (name, p) in zip(axes, preds.items()):
        ax.scatter(q, p["log10_qi"], s=3, alpha=0.25, color="#2a78d6", linewidths=0)
        lim = (-3.8, -0.5)
        ax.plot(lim, lim, color="#52514e", lw=1)
        ax.axvline(-3.5, color="#0b0b0b", ls="--", lw=1)
        ax.axhline(-3.5, color="#0b0b0b", ls="--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        mae_tail = results[name]["qi_in_the_region_we_need"]["true_log10_qi_below_-3.0"]["mae"]
        ax.set_title(f"{name.replace('_', ' ')}: R² {results[name]['per_metric']['log10_qi']['r2']}, "
                     f"MAE for true < -3: {mae_tail}", fontsize=10, loc="left")
        ax.set_xlabel("true log10(qi)  (physics)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("predicted log10(qi)  (surrogate)")
    fig.suptitle("Test 1: surrogate prediction of QI on held-out designs "
                 "(dashed = Benchmark 3 limit, -3.5)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "surrogate_qi_parity.png")
    print("wrote", OUT / "surrogate_qi_parity.png")


if __name__ == "__main__":
    main()
