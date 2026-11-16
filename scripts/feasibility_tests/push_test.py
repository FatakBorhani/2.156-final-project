"""Test 6: a focused push for a feasible design.

Starts from Test 5's 5 best designs at high fidelity (one per seed), and runs a longer
physics-in-the-loop search with two changes from Test 5:

1. Per-requirement penalties (the idea behind the augmented Lagrangian method the paper's
   baseline uses). The loop minimizes  sum_i w_i * max(v_i - TARGET, 0)  over the five
   normalized violations v_i, aiming slightly past each limit (TARGET = -0.02) so that
   very-low-fidelity error doesn't flip a pass into a fail. Each round, the weight of every
   requirement the current design still fails is multiplied by W_GROW, so the loop stops
   trading one stubborn requirement against the others.
2. High-fidelity checks along the way: every CHECK_EVERY rounds, the 3 best not-yet-checked
   candidates per design (by very-low-fidelity worst violation) are re-run at high
   fidelity. Only those can confirm a pass (the cheap setting's error near the limit is
   bigger than the 0.01 tolerance).

Two arms, same VMEC++ budget: "vae" (surrogate-scored VAE steps, Test 5's best) and
"control" (random PCA steps, no surrogate), as in Test 5.

    python scripts/feasibility_tests/push_test.py          # full run (~1 hour)
    python scripts/feasibility_tests/push_test.py --dry    # 1 round, 2 designs, 1 check
    -> results/feasibility_tests/push.json, push.png, push_history.csv,
       push_checked_designs.jsonl (every high-fidelity-checked design, evaluator format)
       VMEC++ log: data/feasibility_tests/push_ledger.jsonl (gitignored)
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
from common import CONSTRAINTS, ROOT, load_designs  # noqa: E402
from compression_test import VAE, GlobalScaler  # noqa: E402
from evaluate import evaluate_designs  # noqa: E402
from fidelity_test import targets_from_metrics  # noqa: E402
from loop_test import K, N_POOL, diverse_top, finetune, viol_array, worst_of  # noqa: E402
from search_test import get_ensemble, to_boundary_json  # noqa: E402

warnings.filterwarnings("ignore")

ARMS = ["control", "vae"]
N_START = 5
ROUNDS = 30
PER_ROUND = 16
CHECK_EVERY, CHECK_TOP = 10, 3
R0, R_MIN, R_MAX, GROW, SHRINK = 0.03, 0.002, 0.3, 1.5, 0.5
TARGET, W_GROW, W_MAX = -0.02, 1.5, 100.0
FEASIBLE = 0.01
OUT = ROOT / "results" / "feasibility_tests"
LOOP_LEDGER = ROOT / "data" / "feasibility_tests" / "loop_ledger.jsonl"


def penalty(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted sum of how far each requirement is from TARGET (0 once all are met)."""
    return (np.maximum(v - TARGET, 0) * w).sum(axis=1)


def x_from_boundary(bjson: str, cols: list[str]) -> np.ndarray:
    b = json.loads(bjson)
    out = []
    for name in cols:
        arr, m, n = name.split("_m")[0], *name.split("_m")[1].split("_n")
        out.append(b[arr][int(m)][int(n) + 4])
    return np.array(out, dtype=float)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)] if path.exists() else []


def pick_starts(ycols: list[str], n: int) -> list[dict]:
    """Test 5's best high-fidelity designs, one per seed."""
    finals = {d["id"]: d for d in read_jsonl(OUT / "loop_final_designs.jsonl")}
    rows = read_jsonl(LOOP_LEDGER)
    vlo = {r["id"]: r for r in rows if r["fidelity"] == "very_low_fidelity" and r["ok"]}
    cands = []
    for r in rows:
        if r["fidelity"] == "high_fidelity" and r["ok"] and r["id"] in finals:
            t = targets_from_metrics(r["metrics"])
            w = float(worst_of(viol_array(np.array([[t[c] for c in ycols]]), ycols))[0])
            cands.append((w, r["id"]))
    starts, used = [], set()
    for w, i in sorted(cands):
        d = finals[i]
        if d["seed"] in used:
            continue
        used.add(d["seed"])
        tv = targets_from_metrics(vlo[i]["metrics"])
        starts.append({"id": i, "seed": d["seed"], "boundary": d["boundary"], "worst_hi": w,
                       "t_vlo": np.array([tv[c] for c in ycols])})
        if len(starts) == n:
            break
    return starts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    rounds, n_start, check_every = (1, 2, 1) if args.dry else (ROUNDS, N_START, CHECK_EVERY)
    ledger = ROOT / "data" / "feasibility_tests" / ("push_dry_ledger.jsonl" if args.dry else "push_ledger.jsonl")
    t_start = time.perf_counter()

    X, Y, _, _ = load_designs(3)
    cols, sigma = list(X.columns), X.std().values
    base = get_ensemble(X, Y)
    ycols = base.cols
    Xv, Yv = X.values, Y[ycols].values
    pca = PCA(n_components=K, random_state=0).fit(Xv - Xv.mean(axis=0))
    basis = pca.components_ * np.sqrt(pca.explained_variance_)[:, None]
    scaler = GlobalScaler().fit(X)
    vae = VAE(K, Xv.shape[1])
    vae.fit(scaler.transform(Xv))
    ens = copy.deepcopy(base)
    new_x, new_t = [], []
    ft_rng = np.random.default_rng(0)

    starts = pick_starts(ycols, n_start)
    print("starting designs (high-fidelity worst):",
          [(s["seed"][:6], round(s["worst_hi"], 3)) for s in starts], flush=True)

    def to_design(did, x, start):
        return {"id": did, "boundary": to_boundary_json(x, cols, start["boundary"])}

    def parse(r):
        if not r["ok"]:
            return None
        t = targets_from_metrics(r["metrics"])
        return np.array([t[c] for c in ycols])

    state, history, xs = {}, [], {}
    for si, s in enumerate(starts):
        x0 = x_from_boundary(s["boundary"], cols)
        v0 = viol_array(s["t_vlo"][None], ycols)[0]
        for arm in ARMS:
            state[(arm, si)] = {"x": x0.copy(), "t": s["t_vlo"], "v": v0, "w": np.ones(5), "r": R0,
                                "best_worst": float(worst_of(v0[None])[0]), "done": False,
                                "best_hi": s["worst_hi"], "best_hi_id": s["id"]}
    checked, hi_rows = set(), []

    for k in range(1, rounds + 1):
        designs, info = [], {}
        for ai, arm in enumerate(ARMS):
            for si, s in enumerate(starts):
                st = state[(arm, si)]
                if st["done"]:
                    continue
                rng = np.random.default_rng([k, ai, si, 6])
                r, x_c = st["r"], st["x"]
                if arm == "control":
                    Z = rng.standard_normal((PER_ROUND, K)) * r
                    Xc, pred = x_c + Z @ basis, np.full(PER_ROUND, np.nan)
                else:
                    Z = rng.standard_normal((N_POOL, K)) * r
                    z_c = vae.encode(scaler.transform(x_c[None]))
                    Xp = x_c + (vae.decode(z_c + Z) - vae.decode(z_c)) * scaler.scale_
                    P = ens.predict_all(np.vstack([x_c[None], Xp])).mean(axis=0)
                    vp = viol_array(P[1:] - P[0] + st["t"], ycols)
                    pick = diverse_top(penalty(vp, st["w"]), Z, PER_ROUND, min_dist=1.5 * r)
                    Xc, pred = Xp[pick], worst_of(vp[pick])
                for j, x in enumerate(Xc):
                    did = f"push_{arm}_r{k}_{s['seed']}_{j}"
                    designs.append(to_design(did, x, s))
                    xs[did] = (x, si)
                    info[did] = {"arm": arm, "round": k, "start": si, "seed": s["seed"],
                                 "pred_worst": float(pred[j]), "radius": r,
                                 "step_std_units": float(np.linalg.norm((x - x_c) / sigma))}

        t0 = time.perf_counter()
        rows = evaluate_designs(designs, ledger, fidelity="very_low_fidelity")
        sec = time.perf_counter() - t0

        best = {}
        for r_ in rows:
            inf = info[r_["id"]]
            rec = inf | {"id": r_["id"], "ok": r_["ok"]}
            t = parse(r_)
            if t is not None:
                v = viol_array(t[None], ycols)[0]
                rec |= {"worst": float(worst_of(v[None])[0]), **{f"v_{c}": float(v[i]) for i, c in enumerate(CONSTRAINTS)}}
                if inf["arm"] == "vae":
                    new_x.append(xs[r_["id"]][0])
                    new_t.append(t)
                key = (inf["arm"], inf["start"])
                p = float(penalty(v[None], state[key]["w"])[0])
                if key not in best or p < best[key][0]:
                    best[key] = (p, v, t, r_["id"])
            history.append(rec)

        for key, st in state.items():
            if st["done"]:
                continue
            p_c = float(penalty(st["v"][None], st["w"])[0])
            b = best.get(key)
            if b is not None and b[0] < p_c:
                st.update(x=xs[b[3]][0], t=b[2], v=b[1], r=min(st["r"] * GROW, R_MAX))
                st["best_worst"] = min(st["best_worst"], float(worst_of(b[1][None])[0]))
            else:
                st["r"] = max(st["r"] * SHRINK, R_MIN)
            failing = st["v"] > TARGET
            st["w"] = np.where(failing, np.minimum(st["w"] * W_GROW, W_MAX), st["w"])

        if new_x:
            finetune(ens, np.array(new_x), np.array(new_t), Xv, Yv, ft_rng)

        # --- high-fidelity checks ------------------------------------------------------
        if k % check_every == 0 or k == rounds:
            h = pd.DataFrame(history)
            h = h[(h.ok == True) & ~h.id.isin(checked)]  # noqa: E712
            todo = []
            for (arm, si), g in h.groupby(["arm", "start"]):
                if state[(arm, si)]["done"]:
                    continue
                for did in g.nsmallest(CHECK_TOP, "worst").id:
                    todo.append(to_design(did, xs[did][0], starts[si]))
                    checked.add(did)
            for r_ in evaluate_designs(todo, ledger, fidelity="high_fidelity"):
                inf = next(x for x in history if x["id"] == r_["id"])
                t = parse(r_)
                rec = {"id": r_["id"], "arm": inf["arm"], "start": inf["start"], "seed": inf["seed"],
                       "round_checked": k, "ok": r_["ok"], "worst_vlo": inf.get("worst"),
                       "boundary": next(d["boundary"] for d in todo if d["id"] == r_["id"])}
                if t is not None:
                    v = viol_array(t[None], ycols)[0]
                    rec |= {"worst_hi": float(worst_of(v[None])[0]), "feasible_hi": bool(r_["feasible"]),
                            "L_gradB": float(t[ycols.index("L_gradB")]),
                            "aspect_ratio": float(t[ycols.index("aspect_ratio")]),
                            **{f"v_{c}": float(v[i]) for i, c in enumerate(CONSTRAINTS)}}
                    st = state[(inf["arm"], inf["start"])]
                    if rec["worst_hi"] < st["best_hi"]:
                        st["best_hi"], st["best_hi_id"] = rec["worst_hi"], r_["id"]
                    if rec["feasible_hi"]:
                        st["done"] = True
                hi_rows.append(rec)
            msg = " | ".join(f"{arm}: best hi {[round(state[(arm, si)]['best_hi'], 3) for si in range(len(starts))]}"
                             for arm in ARMS)
            print(f"  HIGH-FIDELITY CHECK after round {k}: {msg}", flush=True)

        h = pd.DataFrame([x for x in history if x["round"] == k])
        msg = " | ".join(
            f"{arm}: best vlo {[round(state[(arm, si)]['best_worst'], 3) for si in range(len(starts))]}, "
            f"failed {int((h[h.arm == arm].ok != True).sum())}/{int((h.arm == arm).sum())}"  # noqa: E712
            for arm in ARMS)
        print(f"ROUND {k} ({sec:.0f} s) | {msg}", flush=True)

    hist, hi = pd.DataFrame(history), pd.DataFrame(hi_rows)
    if args.dry:
        print(hist.groupby("arm")[["ok", "worst", "pred_worst", "step_std_units"]].mean().round(3))
        print(hi.drop(columns="boundary").round(3).to_string())
        return
    hist.to_csv(OUT / "push_history.csv", index=False)
    with open(OUT / "push_checked_designs.jsonl", "w") as f:
        for rec in hi_rows:
            f.write(json.dumps(rec) + "\n")

    # --- summary ---------------------------------------------------------------------
    out = {"rounds": rounds, "per_round_per_design": PER_ROUND, "n_designs": len(starts),
           "target_margin": TARGET, "weight_growth": W_GROW,
           "starts": [{"seed": s["seed"], "from_test5": s["id"], "worst_hi": round(s["worst_hi"], 4)}
                      for s in starts],
           "wall_minutes": round((time.perf_counter() - t_start) / 60, 1)}
    for arm in ARMS:
        ha, hh = hist[hist.arm == arm], hi[(hi.arm == arm) & (hi.ok == True)]  # noqa: E712
        per = []
        for si, s in enumerate(starts):
            g = hh[hh.start == si]
            best_row = g.loc[g.worst_hi.idxmin()] if len(g) else None
            per.append({
                "seed": s["seed"], "start_worst_hi": round(s["worst_hi"], 4),
                "best_worst_vlo": round(state[(arm, si)]["best_worst"], 4),
                "best_worst_hi": round(float(best_row.worst_hi), 4) if best_row is not None else None,
                "feasible_hi": bool(g.feasible_hi.any()) if len(g) else False,
                "best_hi_violations": {c: round(float(best_row[f"v_{c}"]), 4) for c in CONSTRAINTS}
                if best_row is not None else None,
                "final_weights": dict(zip(CONSTRAINTS, np.round(state[(arm, si)]["w"], 2).tolist())),
            })
        out[arm] = {
            "vmec_evaluations_vlo": int(len(ha)),
            "vmec_failed_vlo": int((ha.ok != True).sum()),  # noqa: E712
            "high_fidelity_checks": int((hi.arm == arm).sum()),
            "high_fidelity_check_failed": int(((hi.arm == arm) & (hi.ok != True)).sum()),  # noqa: E712
            "feasible_designs_hi": int(hh.feasible_hi.sum()) if len(hh) else 0,
            "per_design": per,
        }
        if arm == "vae":
            okr = ha[ha.ok == True]  # noqa: E712
            out[arm]["surrogate_error_worst_by_round"] = [
                round(float((g.worst - g.pred_worst).abs().median()), 3) for _, g in okr.groupby("round")]
    (OUT / "push.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # --- figure ----------------------------------------------------------------------
    colors = {"control": "#52514e", "vae": "#eb6834"}
    labels = {"control": "random steps", "vae": "surrogate, VAE steps"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    for arm in ARMS:
        ha = hist[(hist.arm == arm) & (hist.ok == True)]  # noqa: E712
        for si in range(len(starts)):
            g = ha[ha.start == si].groupby("round").worst.min().cummin()
            g = pd.concat([pd.Series([float(worst_of(viol_array(starts[si]["t_vlo"][None], ycols))[0])], index=[0]), g])
            axes[0].plot(g.index, g.values, color=colors[arm], lw=1.2, alpha=0.8,
                         label=labels[arm] if si == 0 else None)
        hh = hi[(hi.arm == arm) & (hi.ok == True)]  # noqa: E712
        axes[1].scatter(hh.round_checked + (0.4 if arm == "vae" else -0.4), hh.worst_hi, s=22,
                        color=colors[arm], edgecolor="white", linewidth=0.5, label=labels[arm], zorder=3)
    for a in axes:
        a.axhline(FEASIBLE, color="#0b0b0b", ls="--", lw=1)
        a.text(0.02, FEASIBLE * 1.15, "feasible below this line", fontsize=8, color="#52514e",
               transform=a.get_yaxis_transform())
        a.set_yscale("log")
        a.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("best worst violation so far (very low fidelity)")
    axes[0].set_title("One line per starting design", loc="left", fontsize=10)
    axes[1].set_xlabel("round of the high-fidelity check")
    axes[1].set_ylabel("worst violation, high fidelity")
    axes[1].set_title("Only these can confirm a pass", loc="left", fontsize=10)
    fig.suptitle("Test 6: focused push from Test 5's 5 best designs, per-requirement penalties",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "push.png")
    print("wrote", OUT / "push.png")


if __name__ == "__main__":
    main()
