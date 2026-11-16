"""Test 2: does a surrogate-guided search improve *real* designs?

1. Train an ensemble of 3 MLPs (Test 1's architecture) on all 68k three-field-period
   designs to predict the 7 metrics from the 81 boundary coefficients.
2. Pick 20 seed designs: the 10 closest to feasible, 5 from the "passes QI" group and
   5 from the "passes the other four" group (see the README figure).
3. For each seed, run CMA-ES on the surrogate to minimize the predicted worst constraint
   violation (with a 0.05 safety margin), staying inside a box of +-r standard deviations
   per coefficient around the seed. Two radii: small (0.25) and large (1.0).
4. Evaluate the 40 resulting designs with real VMEC++ (high fidelity, plus very low
   fidelity to test the filter) and compare with the seeds and with what the surrogate
   predicted.

    python scripts/feasibility_tests/search_test.py
    -> results/feasibility_tests/search.json, search.png, search_ledger.jsonl
"""

import json
import sys
import time
from pathlib import Path

import cma
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import CONSTRAINTS, ROOT, load_designs, violations_from_targets  # noqa: E402
from evaluate import evaluate_designs  # noqa: E402
from fidelity_test import targets_from_metrics  # noqa: E402

OUT = ROOT / "results" / "feasibility_tests"
RADII = {"small": 0.25, "large": 1.0}
MARGIN = 0.05
N_MODELS = 3


class Ensemble:
    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.cols = list(Y.columns)
        self.xs, self.ys = StandardScaler().fit(X), StandardScaler().fit(Y)
        Xs, Ys = self.xs.transform(X), self.ys.transform(Y)
        self.models = []
        for seed in range(N_MODELS):
            m = MLPRegressor(
                hidden_layer_sizes=(256, 256, 256), alpha=1e-5, batch_size=256,
                learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20, random_state=seed,
            )
            t0 = time.perf_counter()
            m.fit(Xs, Ys)
            print(f"  model {seed}: {time.perf_counter() - t0:.0f} s", flush=True)
            self.models.append(m)

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """(n_models, n, n_targets) in physical units."""
        Xs = self.xs.transform(X)
        return np.stack([self.ys.inverse_transform(m.predict(Xs)) for m in self.models])

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(self.predict_all(X).mean(axis=0), columns=self.cols)


def worst(v: pd.DataFrame) -> pd.Series:
    return v.clip(lower=0).max(axis=1)


def pick_seeds(viol: pd.DataFrame) -> pd.DataFrame:
    w = worst(viol)
    fail = viol > 0.01
    qi_group = w[~fail["log10_qi"]].nsmallest(5).index
    other_group = viol.loc[~fail.drop(columns="log10_qi").any(axis=1), "log10_qi"].nsmallest(5).index
    closest = w.drop(qi_group.union(other_group)).nsmallest(10).index
    return pd.DataFrame(
        {"group": ["closest"] * 10 + ["passes_QI"] * 5 + ["passes_other_four"] * 5},
        index=closest.append(qi_group).append(other_group),
    )


def to_boundary_json(x: np.ndarray, cols: list[str], seed_json: str) -> str:
    b = json.loads(seed_json)
    for value, name in zip(x, cols):
        arr, m, n = name.split("_m")[0], *name.split("_m")[1].split("_n")
        b[arr][int(m)][int(n) + 4] = float(value)
    return json.dumps(b)


def get_ensemble(X: pd.DataFrame, Y: pd.DataFrame) -> Ensemble:
    """Train once, then reuse from data/feasibility_tests/ensemble_nfp3.joblib."""
    import joblib

    path = ROOT / "data" / "feasibility_tests" / "ensemble_nfp3.joblib"
    if path.exists():
        return joblib.load(path)
    print("training ensemble on", len(X), "designs", flush=True)
    ens = Ensemble(X, Y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ens, path)
    return ens


def search(ens: Ensemble, x0: np.ndarray, basis: np.ndarray, radius: float) -> np.ndarray:
    """CMA-ES over z in a box [-radius, radius]^d, where the design is x0 + z @ basis.

    basis = diag(per-coefficient std) searches raw coefficients (Test 2);
    basis = top PCA directions scaled by their std searches along real-design
    directions only (Test 2b)."""
    def objective(zs: list[np.ndarray]) -> list[float]:
        X = x0 + np.asarray(zs) @ basis
        v = violations_from_targets(ens.predict(X))
        # soft max over constraints of (violation + margin): pushes past every limit
        a = (v.values + MARGIN) * 20
        return list(np.log(np.exp(a).sum(axis=1)) / 20)

    es = cma.CMAEvolutionStrategy(
        np.zeros(basis.shape[0]), radius / 3,
        {"bounds": [-radius, radius], "popsize": 32, "maxiter": 150, "seed": 1,
         "verbose": -9},
    )
    while not es.stop():
        zs = es.ask()
        es.tell(zs, objective(zs))
    return x0 + es.result.xbest @ basis


def main() -> None:
    X, Y, viol, meta = load_designs(3)
    ens = get_ensemble(X, Y)
    res, candidates = run_search(ens, X, viol, meta, np.diag(X.std().values), RADII)
    finish(res, candidates, RADII, "search",
           "Test 2: surrogate-guided search in raw coefficients, checked with VMEC++")


def run_search(ens, X, viol, meta, basis, radii):
    cols = list(X.columns)
    sigma = X.std().values
    seeds = pick_seeds(viol)
    print(seeds.group.value_counts().to_dict(), flush=True)
    candidates, rows = [], []
    for sid, grp in seeds.group.items():
        x0 = X.loc[sid].values
        for rname, r in radii.items():
            x = search(ens, x0, basis, r)
            cid = f"{sid}__{rname}"
            preds = ens.predict_all(x[None])[:, 0, :]
            pv = violations_from_targets(pd.DataFrame(preds, columns=ens.cols))
            candidates.append({"id": cid, "boundary": to_boundary_json(x, cols, meta.loc[sid, "boundary_json"])})
            rows.append({
                "id": cid, "seed": sid, "group": grp, "radius": rname,
                "seed_true_worst": float(worst(viol.loc[[sid]]).iloc[0]),
                "pred_worst": float(worst(pv.mean().to_frame().T).iloc[0]),
                "pred_worst_ensemble_spread": float(worst(pv).std()),
                **{f"pred_{k}": float(pv[k].mean()) for k in CONSTRAINTS},
                "step_size_std_units": float(np.linalg.norm((x - x0) / sigma)),
            })
        print(f"searched {sid} ({grp})", flush=True)
    return pd.DataFrame(rows).set_index("id"), candidates


def finish(res, candidates, radii, name, title):
    """Verify candidates with VMEC++, write {name}.json / .png / _candidates.csv."""
    RADII = radii  # noqa: N806  (the code below reads RADII)
    ledger = OUT / f"{name}_ledger.jsonl"
    for fid in ("very_low_fidelity", "high_fidelity"):
        evaluated = evaluate_designs(candidates, ledger, fidelity=fid)
        for r in evaluated:
            pre = "hi" if fid == "high_fidelity" else "vlo"
            res.loc[r["id"], f"{pre}_ok"] = r["ok"]
            res.loc[r["id"], f"{pre}_seconds"] = r["seconds"]
            if r["ok"]:
                v = violations_from_targets(pd.DataFrame([targets_from_metrics(r["metrics"])]))
                res.loc[r["id"], f"{pre}_worst"] = float(worst(v).iloc[0])
                res.loc[r["id"], f"{pre}_feasible"] = bool(r["feasible"])
                for k in CONSTRAINTS:
                    res.loc[r["id"], f"{pre}_{k}"] = float(v[k].iloc[0])
    for pre in ("vlo", "hi"):  # keep every column even if no candidate solved
        for col in ("ok", "worst", "feasible", *CONSTRAINTS):
            if f"{pre}_{col}" not in res:
                res[f"{pre}_{col}"] = np.nan
    res.to_csv(OUT / f"{name}_candidates.csv")

    # --- summary ---
    out = {"n_candidates": len(res), "margin": MARGIN, "radii_std_units": RADII}
    for rname in RADII:
        d = res[res.radius == rname]
        ok = d[d.hi_ok == True]  # noqa: E712
        improved = ok.hi_worst < ok.seed_true_worst
        err = ok.hi_worst - ok.pred_worst
        out[rname] = {
            "vmec_failed": int((d.hi_ok != True).sum()),  # noqa: E712
            "vmec_failed_very_low": int((d.vlo_ok != True).sum()),  # noqa: E712
            "feasible_high_fidelity": int(ok.hi_feasible.fillna(False).sum()),
            "improved_vs_seed": f"{int(improved.sum())}/{len(ok)}",
            "median_seed_worst": round(float(ok.seed_true_worst.median()), 3),
            "median_candidate_worst": round(float(ok.hi_worst.median()), 3),
            "best_candidate_worst": round(float(ok.hi_worst.min()), 3),
            "median_predicted_worst": round(float(ok.pred_worst.median()), 3),
            "surrogate_error_worst (actual - predicted): median": round(float(err.median()), 3),
            "surrogate_error_worst: mean abs": round(float(err.abs().mean()), 3),
            "per_constraint_mean_abs_error": {
                k: round(float((ok[f"hi_{k}"] - ok[f"pred_{k}"]).abs().mean()), 3)
                for k in CONSTRAINTS
            },
            "ensemble_spread_vs_abs_error_spearman": round(
                float(stats.spearmanr(ok.pred_worst_ensemble_spread, err.abs())[0]), 2
            ) if len(ok) > 3 else None,
            "very_low_vs_high_worst_spearman": round(
                float(stats.spearmanr(ok.vlo_worst, ok.hi_worst, nan_policy="omit")[0]), 2
            ) if len(ok) > 3 else None,
            "median_step_size_std_units": round(float(d.step_size_std_units.median()), 2),
        }
        by_group = {}
        for g, dg in ok.groupby("group"):
            by_group[g] = {
                "improved": f"{int((dg.hi_worst < dg.seed_true_worst).sum())}/{len(dg)}",
                "median_change_in_worst": round(float((dg.hi_worst - dg.seed_true_worst).median()), 3),
            }
        out[rname]["by_seed_group"] = by_group
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # --- figure: seed -> candidate (real), and predicted vs. real ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    colors = {"small": "#2a78d6", "large": "#eb6834"}
    ax = axes[0]
    for rname, c in colors.items():
        d = res[(res.radius == rname) & (res.hi_ok == True)]  # noqa: E712
        ax.scatter(d.seed_true_worst, d.hi_worst, s=28, color=c, edgecolor="white",
                   linewidth=0.6, label=f"{rname} step (±{RADII[rname]}σ)", zorder=3)
    lim = (0, max(1.0, float(res[["seed_true_worst", "hi_worst"]].max().max()) * 1.05))
    ax.plot(lim, lim, color="#52514e", lw=1)
    ax.fill_between(lim, 0, lim, color="#1baf7a", alpha=0.07)
    ax.axhline(0.01, color="#0b0b0b", ls="--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("seed design: real worst violation")
    ax.set_ylabel("searched design: real worst violation (VMEC++)")
    ax.set_title("Below the diagonal = search made the real design better", loc="left", fontsize=10)
    ax.text(lim[1] * 0.02, 0.025, "feasible below dashed line", fontsize=8, color="#52514e")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axes[1]
    for rname, c in colors.items():
        d = res[(res.radius == rname) & (res.hi_ok == True)]  # noqa: E712
        ax.scatter(d.pred_worst, d.hi_worst, s=28, color=c, edgecolor="white", linewidth=0.6,
                   label=rname, zorder=3)
    ax.plot(lim, lim, color="#52514e", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("surrogate's predicted worst violation")
    ax.set_ylabel("real worst violation (VMEC++)")
    ax.set_title("On the diagonal = surrogate was right about new shapes", loc="left", fontsize=10)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
