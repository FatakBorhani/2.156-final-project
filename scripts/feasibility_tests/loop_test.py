"""Test 5: a small physics-in-the-loop search. Does it improve real designs, and does
the surrogate help compared with plain trial and error?

Same 20 seed designs as Tests 2/2b. Three arms, each with the same VMEC++ budget
(very low fidelity: 8 designs per seed per round, ROUNDS rounds):

  control  no surrogate: 8 random small steps along the top 16 PCA directions,
           keep the best if it beats the current design
  pca      surrogate loop: sample N_POOL small PCA steps, score them with the surrogate,
           send the 8 most promising (and spread out) to VMEC++, keep the best,
           fine-tune the surrogate on what VMEC++ said
  vae      same, but the steps are taken in a 16-dim VAE latent space. Only the *change*
           is applied to the current design, x_new = x + dec(z + dz) - dec(z), so the
           VAE's blurring of fine QI detail (Test 3) doesn't touch the design itself

Both surrogate arms predict relative to the current design: predicted metrics are shifted
so that, at the current design, they equal what VMEC++ measured there. This removes the
surrogate's off-data offset that broke Tests 2/2b. Each arm has a trust region: the step
size grows 1.5x after an improving round and halves after a round with no improvement.

At the end, every arm's final design per seed is re-checked at high fidelity.

    python scripts/feasibility_tests/loop_test.py          # full run (~45 min)
    python scripts/feasibility_tests/loop_test.py --dry    # 1 round, 3 seeds, separate ledger
    -> results/feasibility_tests/loop.json, loop.png, loop_history.csv, loop_final_*.csv,
       loop_final_designs.jsonl (each arm's final design per seed, evaluator input format)
       VMEC++ log: data/feasibility_tests/loop_ledger.jsonl (gitignored, ~7 MB)
"""

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import CONSTRAINTS, ROOT, load_designs, violations_from_targets  # noqa: E402
from compression_test import VAE, GlobalScaler  # noqa: E402
from evaluate import evaluate_designs  # noqa: E402
from fidelity_test import targets_from_metrics  # noqa: E402
from search_test import get_ensemble, pick_seeds, to_boundary_json  # noqa: E402

warnings.filterwarnings("ignore")

ARMS = ["control", "pca", "vae"]
ROUNDS = 10
PER_ROUND = 8          # VMEC++ evaluations per seed per round per arm
N_POOL = 4000          # surrogate-scored proposals per seed per round
K = 16                 # PCA directions / VAE latent size
R0, R_MIN, R_MAX, GROW, SHRINK = 0.03, 0.003, 0.3, 1.5, 0.5  # trust region, per-direction std units
T_SOFT, FLOOR = 50.0, -0.05  # objective: soft max of violations, floored at a 0.05 margin
FEASIBLE = 0.01
REPLAY, REPEAT_NEW, FT_EPOCHS = 5000, 5, 3
OUT = ROOT / "results" / "feasibility_tests"


def J(v: np.ndarray) -> np.ndarray:
    """Soft worst violation: ~max over constraints, but rewards progress on all of them.
    Constraints already 0.05 past their limit stop counting."""
    a = T_SOFT * np.maximum(v, FLOOR)
    m = a.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(a - m).sum(axis=1))) / T_SOFT


def viol_array(targets: np.ndarray, cols: list[str]) -> np.ndarray:
    return violations_from_targets(pd.DataFrame(targets, columns=cols))[CONSTRAINTS].values


def worst_of(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 0, None).max(axis=1)


def diverse_top(score: np.ndarray, Z: np.ndarray, n: int, min_dist: float) -> list[int]:
    picked = []
    for i in np.argsort(score):
        if all(np.linalg.norm(Z[i] - Z[j]) >= min_dist for j in picked):
            picked.append(i)
        if len(picked) == n:
            break
    return picked


def finetune(ens, Xnew, Ynew, X, Y, rng):
    """A few epochs on the loop's own VMEC++ results (repeated) plus a replay sample of
    the dataset, so the surrogate learns the new region without forgetting the rest."""
    idx = rng.choice(len(X), REPLAY, replace=False)
    Xf = np.vstack([np.repeat(Xnew, REPEAT_NEW, axis=0), X[idx]])
    Yf = np.vstack([np.repeat(Ynew, REPEAT_NEW, axis=0), Y[idx]])
    Xs, Ys = ens.xs.transform(Xf), ens.ys.transform(Yf)
    for m in ens.models:
        m.early_stopping = False  # trained with it; partial_fit doesn't allow it
        if getattr(m, "best_loss_", None) is None:
            m.best_loss_ = np.inf
        for _ in range(FT_EPOCHS):
            p = rng.permutation(len(Xs))
            m.partial_fit(Xs[p], Ys[p])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    rounds, n_seeds = (1, 3) if args.dry else (ROUNDS, 20)
    ledger = (ROOT / "data" / "feasibility_tests" / "loop_dry_ledger.jsonl") if args.dry \
        else ROOT / "data" / "feasibility_tests" / "loop_ledger.jsonl"
    t_start = time.perf_counter()

    X, Y, viol, meta = load_designs(3)
    cols, sigma = list(X.columns), X.std().values
    base = get_ensemble(X, Y)
    ycols = base.cols
    Xv, Yv = X.values, Y[ycols].values

    pca = PCA(n_components=K, random_state=0).fit(Xv - Xv.mean(axis=0))
    basis = pca.components_ * np.sqrt(pca.explained_variance_)[:, None]   # (K, 81)
    scaler = GlobalScaler().fit(X)
    t0 = time.perf_counter()
    vae = VAE(K, Xv.shape[1])
    vae.fit(scaler.transform(Xv))
    print(f"VAE-{K} trained in {time.perf_counter() - t0:.0f} s", flush=True)

    seeds = pick_seeds(viol).iloc[:n_seeds]
    sids = list(seeds.index)
    ens = {"pca": copy.deepcopy(base), "vae": copy.deepcopy(base)}
    new_data = {"pca": ([], []), "vae": ([], [])}
    ft_rng = np.random.default_rng(0)

    def to_design(did, x, sid):
        return {"id": did, "boundary": to_boundary_json(x, cols, meta.loc[sid, "boundary_json"])}

    def parse(r):
        if not r["ok"]:
            return None
        t = targets_from_metrics(r["metrics"])
        return np.array([t[c] for c in ycols])

    # --- round 0: the seeds themselves, at very low fidelity (fair starting point) ---
    rows = evaluate_designs([to_design(f"seed_{s}", X.loc[s].values, s) for s in sids],
                            ledger, fidelity="very_low_fidelity")
    seed_t = {r["id"][5:]: parse(r) for r in rows}
    state, history = {}, []
    for s in sids:
        t = seed_t[s]
        v = viol_array(t[None], ycols)
        for arm in ARMS:
            state[(arm, s)] = {"x": X.loc[s].values.copy(), "t": t, "J": float(J(v)[0]),
                               "worst": float(worst_of(v)[0]), "best_worst": float(worst_of(v)[0]),
                               "r": R0, "id": f"seed_{s}", "feasible": False}
        history.append({"arm": "seed", "round": 0, "seed": s, "group": seeds.group[s], "ok": True,
                        "worst": float(worst_of(v)[0]), "true_worst_dataset_hi": float(
                            viol.loc[s].clip(lower=0).max())})
    print("seed worst (very low fid):", {s[:6]: round(state[("control", s)]["worst"], 3) for s in sids},
          flush=True)

    # --- rounds ----------------------------------------------------------------------
    for k in range(1, rounds + 1):
        designs, info = [], {}
        for ai, arm in enumerate(ARMS):
            for si, s in enumerate(sids):
                st = state[(arm, s)]
                if st["feasible"]:
                    continue
                rng = np.random.default_rng([k, ai, si])
                r, x_c = st["r"], st["x"]
                if arm == "control":
                    Z = rng.standard_normal((PER_ROUND, K)) * r
                    Xc = x_c + Z @ basis
                    pred = np.full(PER_ROUND, np.nan)
                else:
                    Z = rng.standard_normal((N_POOL, K)) * r
                    if arm == "pca":
                        Xp = x_c + Z @ basis
                    else:
                        z_c = vae.encode(scaler.transform(x_c[None]))
                        d0 = vae.decode(z_c)
                        Xp = x_c + (vae.decode(z_c + Z) - d0) * scaler.scale_
                    e = ens[arm]
                    P = e.predict_all(np.vstack([x_c[None], Xp])).mean(axis=0)
                    P = P[1:] - P[0] + st["t"]          # anchor to VMEC++ at the current design
                    vp = viol_array(P, ycols)
                    pick = diverse_top(J(vp), Z, PER_ROUND, min_dist=1.5 * r)
                    Xc, pred = Xp[pick], worst_of(vp[pick])
                for j, x in enumerate(Xc):
                    did = f"{arm}_r{k}_{s}_{j}"
                    designs.append(to_design(did, x, s))
                    info[did] = {"arm": arm, "round": k, "seed": s, "group": seeds.group[s],
                                 "x": x, "pred_worst": float(pred[j]), "radius": r,
                                 "step_std_units": float(np.linalg.norm((x - x_c) / sigma))}

        t0 = time.perf_counter()
        rows = evaluate_designs(designs, ledger, fidelity="very_low_fidelity")
        sec = time.perf_counter() - t0

        best = {}
        for r_ in rows:
            inf = info[r_["id"]]
            t = parse(r_)
            rec = {kk: vv for kk, vv in inf.items() if kk != "x"} | {"id": r_["id"], "ok": r_["ok"]}
            if t is not None:
                v = viol_array(t[None], ycols)
                rec |= {"worst": float(worst_of(v)[0]), "J": float(J(v)[0]),
                        **{f"v_{c}": float(v[0, i]) for i, c in enumerate(CONSTRAINTS)}}
                if inf["arm"] in ens:
                    new_data[inf["arm"]][0].append(inf["x"])
                    new_data[inf["arm"]][1].append(t)
                key = (inf["arm"], inf["seed"])
                if key not in best or rec["J"] < best[key][0]:
                    best[key] = (rec["J"], rec["worst"], inf["x"], t, r_["id"])
            history.append(rec)

        for key, st in state.items():
            if st["feasible"]:
                continue
            b = best.get(key)
            if b is not None and b[0] < st["J"]:
                st.update(J=b[0], worst=b[1], x=b[2], t=b[3], id=b[4], r=min(st["r"] * GROW, R_MAX))
                st["best_worst"] = min(st["best_worst"], b[1])
                st["feasible"] = b[1] <= FEASIBLE
            else:
                st["r"] = max(st["r"] * SHRINK, R_MIN)

        for arm in ens:
            if new_data[arm][0]:
                finetune(ens[arm], np.array(new_data[arm][0]), np.array(new_data[arm][1]), Xv, Yv, ft_rng)

        h = pd.DataFrame([x for x in history if x.get("round") == k])
        line = []
        for arm in ARMS:
            ha = h[h.arm == arm]
            bw = [state[(arm, s)]["best_worst"] for s in sids]
            line.append(f"{arm}: median best {np.median(bw):.3f}, failed {int((ha.ok != True).sum())}/{len(ha)}, "
                        f"feasible {sum(state[(arm, s)]['feasible'] for s in sids)}")
        print(f"ROUND {k} ({sec:.0f} s) | " + " | ".join(line), flush=True)

    hist = pd.DataFrame(history)
    if args.dry:
        print(hist.groupby("arm")[["ok", "worst", "pred_worst", "step_std_units"]].agg(
            ["mean", "median"]).round(3).to_string())
        return
    hist.to_csv(OUT / "loop_history.csv", index=False)
    with open(OUT / "loop_final_designs.jsonl", "w") as f:
        for (arm, s), st in state.items():
            if not st["id"].startswith("seed_"):
                f.write(json.dumps({**to_design(st["id"], st["x"], s), "arm": arm, "seed": s,
                                    "worst_very_low_fidelity": st["worst"]}) + "\n")

    # --- high-fidelity check of each arm's final design per seed ------------------------
    hdesigns = [to_design(st["id"], st["x"], s) for (arm, s), st in state.items()
                if not st["id"].startswith("seed_")]
    hi = {r["id"]: r for r in evaluate_designs(hdesigns, ledger, fidelity="high_fidelity")}

    out = {"rounds": rounds, "per_round_per_seed": PER_ROUND, "n_seeds": len(sids),
           "trust_region": {"start": R0, "min": R_MIN, "max": R_MAX, "grow": GROW, "shrink": SHRINK},
           "wall_minutes": round((time.perf_counter() - t_start) / 60, 1)}
    for arm in ARMS:
        ha = hist[hist.arm == arm]
        per_seed = []
        for s in sids:
            st = state[(arm, s)]
            seed_hi = float(viol.loc[s].clip(lower=0).max())
            if st["id"].startswith("seed_"):
                final_hi, final_hi_feasible = seed_hi, False
            else:
                r_ = hi[st["id"]]
                t = parse(r_)
                final_hi = float(worst_of(viol_array(t[None], ycols))[0]) if t is not None else None
                final_hi_feasible = bool(r_.get("feasible")) if t is not None else False
            per_seed.append({"seed": s, "group": seeds.group[s], "seed_worst_hi": seed_hi,
                             "seed_worst_vlo": float(worst_of(viol_array(seed_t[s][None], ycols))[0]),
                             "final_worst_vlo": st["worst"], "final_worst_hi": final_hi,
                             "final_feasible_hi": final_hi_feasible, "final_radius": st["r"]})
        ps = pd.DataFrame(per_seed)
        okh = ps.dropna(subset=["final_worst_hi"])
        curve = []
        for k in range(rounds + 1):
            sub = hist[(hist.arm.isin([arm, "seed"])) & (hist["round"] <= k) & (hist.ok == True)]  # noqa: E712
            curve.append(round(float(sub.groupby("seed").worst.min().median()), 4))
        res = {
            "vmec_evaluations": int(len(ha)),
            "vmec_failed": int((ha.ok != True).sum()),  # noqa: E712
            "median_best_worst_by_round_vlo": curve,
            "seeds_improved_vlo": int((ps.final_worst_vlo < ps.seed_worst_vlo - 1e-9).sum()),
            "median_change_vlo": round(float((ps.final_worst_vlo - ps.seed_worst_vlo).median()), 3),
            "feasible_vlo": int((ps.final_worst_vlo <= FEASIBLE).sum()),
            "seeds_improved_hi": int((okh.final_worst_hi < okh.seed_worst_hi - 1e-9).sum()),
            "median_change_hi": round(float((okh.final_worst_hi - okh.seed_worst_hi).median()), 3),
            "best_final_worst_hi": round(float(okh.final_worst_hi.min()), 3),
            "feasible_hi": int(ps.final_feasible_hi.sum()),
            "hi_check_failed": int(ps.final_worst_hi.isna().sum()),
            "by_group_median_change_hi": okh.assign(d=okh.final_worst_hi - okh.seed_worst_hi)
            .groupby("group").d.median().round(3).to_dict(),
            "median_step_std_units": round(float(ha.step_std_units.median()), 3),
        }
        if arm in ens:
            okr = ha[ha.ok == True]  # noqa: E712
            res["surrogate_error_worst_by_round"] = [
                round(float((g.worst - g.pred_worst).abs().median()), 3) for _, g in okr.groupby("round")]
            res["surrogate_bias_worst_by_round (actual - predicted)"] = [
                round(float((g.worst - g.pred_worst).median()), 3) for _, g in okr.groupby("round")]
        out[arm] = res
        ps.to_csv(OUT / f"loop_final_{arm}.csv", index=False)
    (OUT / "loop.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # --- figure ------------------------------------------------------------------------
    colors = {"control": "#52514e", "pca": "#2a78d6", "vae": "#eb6834"}
    labels = {"control": "no surrogate (random steps)", "pca": "surrogate, PCA steps",
              "vae": "surrogate, VAE steps"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    evals = np.arange(rounds + 1) * PER_ROUND * len(sids)
    for arm in ARMS:
        axes[0].plot(evals, out[arm]["median_best_worst_by_round_vlo"], marker="o", ms=3,
                     color=colors[arm], label=labels[arm])
        ha = hist[hist.arm == arm]
        fr = ha.groupby("round").ok.apply(lambda s: 1 - (s == True).mean())  # noqa: E712
        axes[2].plot(fr.index, fr.values, marker="o", ms=3, color=colors[arm], label=labels[arm])
        if arm in ens:
            axes[1].plot(range(1, rounds + 1), out[arm]["surrogate_error_worst_by_round"],
                         marker="o", ms=3, color=colors[arm], label=labels[arm])
    axes[0].set_xlabel("VMEC++ evaluations per arm (very low fidelity)")
    axes[0].set_ylabel("median over seeds of best worst violation so far")
    axes[0].set_title("Lower = closer to feasible", loc="left", fontsize=10)
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("median |actual - predicted| worst violation")
    axes[1].set_title("Does the surrogate learn the new region?", loc="left", fontsize=10)
    axes[2].set_xlabel("round")
    axes[2].set_ylabel("fraction VMEC++ couldn't solve")
    axes[2].set_title("Shape validity", loc="left", fontsize=10)
    for a in axes:
        a.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Test 5: physics-in-the-loop search, 20 seed designs, same VMEC++ budget per arm",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "loop.png")
    print("wrote", OUT / "loop.png")


if __name__ == "__main__":
    main()
