"""Test 12: a conditional VAE as a seed generator (the ML arm against Test 11's chain).

Test 10: the finisher turns a nearly feasible design into feasible ones, but one design
gives no trade-off. Test 11 (the no-ML control) walks the aspect-ratio limit down in
small steps. Here a conditional VAE proposes starting points at each limit directly:

1. Train: x = the 80 boundary coefficients, condition c = the 5 normalized violations
   (clipped), aspect ratio and L_gradB. Data: the 68k dataset designs plus our ~34k
   near-feasible simulations from Tests 7-10 that kept their boundary (very low fidelity
   for most, high fidelity for the 531 checks). Nothing from Test 11, so the arms stay
   independent.
2. Generate, for each aspect-ratio limit of Test 11: "edit" samples (encode our feasible
   designs, decode with the aspect ratio set to the limit and every violation set to just
   passing) and "prior" samples (random latent, same condition).
3. Screen with very-low-fidelity VMEC++ (3 s each), re-check the best 4 per limit at high
   fidelity, then run the same finisher as Test 11 from the best one (MAXIT iterations,
   stop at the first design feasible at that limit).

    python scripts/feasibility_tests/gen_test.py --train   # train + generate only (~10 min)
    python scripts/feasibility_tests/gen_test.py           # screen + finish (~5-7 h)
    python scripts/feasibility_tests/gen_test.py --dry     # tiny version of everything
    python scripts/feasibility_tests/gen_test.py --report  # re-plot only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import constellaration.optimization.augmented_lagrangian_runner  # noqa: F401, E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import multifid_test as mf  # noqa: E402
from chain_test import AR_TOL, MAXIT, STEPS, W_OBJ, belongs, feasible_at, gap_at, step_name  # noqa: E402
from common import CONSTRAINTS, ROOT, load_designs  # noqa: E402
from front_test import pareto, read  # noqa: E402

K, HIDDEN, EPOCHS, BATCH, LR = 16, 256, 60, 512, 1e-3
BETAS = [0.1, 0.5]  # pick the one whose decoder follows the aspect-ratio condition best
V_CLIP = (-0.3, 1.0)
V_TARGET = -0.03  # "just passing" for every requirement
N_EDIT, N_PRIOR, EDIT_NOISE = 160, 160, 0.3
AR_WINDOW = 0.5  # keep samples whose geometric aspect ratio is within this of the limit
TOP_HI = 4
SEED = 0
OUT = ROOT / "results" / "feasibility_tests"
DATA = ROOT / "data" / "feasibility_tests"
# ledgers whose rows kept their boundary; violations in package order (iota, qi, mirror, flux, well)
OUR_SOURCES = ["polish_ledger.jsonl", "alm_ledger.jsonl", "multifid_ledger.jsonl", "front_ledger.jsonl"]
PKG_TO_DATASET = [1, 0, 2, 3, 4]  # -> CONSTRAINTS order (log10_qi, abs_iota, mirror, flux, well)


# ------------------------------------------------------------------------------ data
def boundary_vec(bjson: str, cols: list[str]) -> np.ndarray:
    b = json.loads(bjson)
    out = []
    for c in cols:
        arr, m, n = c.split("_m")[0], *c.split("_m")[1].split("_n")
        a = np.asarray(b[arr])
        nt = (a.shape[1] - 1) // 2
        m, n = int(m), int(n)
        out.append(float(a[m, n + nt]) if m < a.shape[0] and abs(n) <= nt else 0.0)
    return np.array(out)


def cond(v: np.ndarray, ar: np.ndarray, lg: np.ndarray) -> np.ndarray:
    return np.column_stack([np.clip(v, *V_CLIP), (ar - 10.0) / 3.0, (np.clip(lg, 0, 20) - 6.0) / 4.0])


def training_data():
    X, T, V, meta = load_designs(3)
    cols = list(X.columns)
    xs, cs, src = [X.values], [cond(V[CONSTRAINTS].values, T["aspect_ratio"].values, T["L_gradB"].values)], \
        [np.zeros(len(X), int)]
    seen, ours_x, ours_c = set(), [], []
    for s in OUR_SOURCES:
        for r in read(DATA / s):
            if not (r.get("boundary") and r.get("ok") and r.get("violations")):
                continue
            key = r["boundary"][:4000]
            if key in seen:
                continue
            seen.add(key)
            ours_x.append(boundary_vec(r["boundary"], cols))
            ours_c.append(cond(np.array(r["violations"])[PKG_TO_DATASET][None], np.array([r["aspect_ratio"]]),
                               np.array([r["L_gradB"]]))[0])
    xs.append(np.array(ours_x))
    cs.append(np.array(ours_c))
    src.append(np.ones(len(ours_x), int))
    print(f"training data: {len(X):,} dataset + {len(ours_x):,} of ours", flush=True)
    return np.vstack(xs), np.vstack(cs), np.concatenate(src), cols, meta


# ------------------------------------------------------------------------------ model
class CVAE:
    def __init__(self, d: int, dc: int, beta: float):
        import jax

        from compression_test import _init_mlp
        ke, kd = jax.random.split(jax.random.PRNGKey(SEED))
        self.beta = beta
        self.params = {"enc": _init_mlp(ke, [d + dc, HIDDEN, HIDDEN, 2 * K]),
                       "dec": _init_mlp(kd, [K + dc, HIDDEN, HIDDEN, d])}

    def fit(self, S: np.ndarray, C: np.ndarray, epochs: int) -> dict:
        import jax
        import jax.numpy as jnp
        import optax

        from compression_test import _mlp
        beta = self.beta

        def loss(params, x, c, key):
            h = _mlp(params["enc"], jnp.concatenate([x, c], 1))
            mu, logvar = h[:, :K], jnp.clip(h[:, K:], -12, 6)
            z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
            rec = jnp.sum((x - _mlp(params["dec"], jnp.concatenate([z, c], 1))) ** 2, axis=1).mean()
            kl = 0.5 * jnp.sum(jnp.exp(logvar) + mu**2 - 1.0 - logvar, axis=1).mean()
            return rec + beta * kl, (rec, kl)

        opt = optax.adam(LR)
        state = opt.init(self.params)

        @jax.jit
        def step(params, state, x, c, key):
            (_, aux), g = jax.value_and_grad(loss, has_aux=True)(params, x, c, key)
            upd, state = opt.update(g, state)
            return optax.apply_updates(params, upd), state, aux

        rng, key = np.random.default_rng(SEED), jax.random.PRNGKey(SEED)
        Sj, Cj = jnp.asarray(S, jnp.float32), jnp.asarray(C, jnp.float32)
        for _ in range(epochs):
            perm = rng.permutation(len(S))
            for i in range(0, len(S) - BATCH + 1, BATCH):
                key, sub = jax.random.split(key)
                idx = perm[i:i + BATCH]
                self.params, state, (rec, kl) = step(self.params, state, Sj[idx], Cj[idx], sub)
        return {"recon_sse": float(rec), "kl": float(kl)}

    def encode(self, S, C):
        import jax.numpy as jnp

        from compression_test import _mlp
        return np.asarray(_mlp(self.params["enc"], jnp.asarray(np.hstack([S, C]), jnp.float32))[:, :K])

    def decode(self, Z, C):
        import jax.numpy as jnp

        from compression_test import _mlp
        return np.asarray(_mlp(self.params["dec"], jnp.asarray(np.hstack([Z, C]), jnp.float32)))


def geo_aspect_ratio(bjsons: list[str]) -> np.ndarray:
    """Major radius / minor radius from the shape alone (no VMEC++), for screening."""
    from constellaration.geometry import surface_rz_fourier as rz
    out = []
    for bj in bjsons:
        try:
            s = rz.SurfaceRZFourier.model_validate_json(bj)
            r = np.asarray(s.r_cos)
            out.append(float(r[0, (r.shape[1] - 1) // 2] / rz.evaluate_minor_radius(s)))
        except Exception:  # noqa: BLE001  (degenerate shapes)
            out.append(np.nan)
    return np.array(out)


def train_and_generate(dry: bool) -> None:
    from compression_test import GlobalScaler
    from search_test import to_boundary_json
    import pandas as pd

    t0 = time.time()
    Xall, C, src, cols, meta = training_data()
    scaler = GlobalScaler().fit(pd.DataFrame(Xall, columns=cols))
    S = scaler.transform(Xall)
    rng = np.random.default_rng(SEED)
    test = rng.choice(len(S), 1000, replace=False)
    train = np.setdiff1d(np.arange(len(S)), test)
    feas = read(OUT / "front_feasible_designs.jsonl")
    template = feas[0]["boundary"]
    epochs = 2 if dry else EPOCHS

    # calibrate geometric vs. VMEC++ aspect ratio on held-out designs
    probe = test[:200]
    ar_true = C[probe, 5] * 3 + 10
    ar_geo = geo_aspect_ratio([to_boundary_json(Xall[i], cols, template) for i in probe])
    ok = np.isfinite(ar_geo)
    fit = np.polyfit(ar_geo[ok], ar_true[ok], 1)
    calib = {"slope": fit[0], "intercept": fit[1],
             "rmse": float(np.sqrt(np.mean((np.polyval(fit, ar_geo[ok]) - ar_true[ok]) ** 2)))}
    print("geometric aspect ratio calibration:", calib, flush=True)

    def ar_of(bjs):
        return np.polyval(fit, geo_aspect_ratio(bjs))

    models, checks = {}, {}
    for beta in BETAS:
        m = CVAE(S.shape[1], C.shape[1], beta)
        info = m.fit(S[train], C[train], epochs)
        # condition-following: re-decode held-out designs with aspect ratio lowered by 1
        idx = test[:150]
        Z = m.encode(S[idx], C[idx])
        C2 = C[idx].copy()
        C2[:, 5] -= 1 / 3
        rec = scaler.inverse_transform(m.decode(Z, C[idx]))
        moved = scaler.inverse_transform(m.decode(Z, C2))
        a0 = ar_of([to_boundary_json(x, cols, template) for x in rec])
        a1 = ar_of([to_boundary_json(x, cols, template) for x in moved])
        shift = a1 - a0
        checks[beta] = {**info, "recon_rmse_scaled": float(np.sqrt(np.mean((m.decode(Z, C[idx]) - S[idx]) ** 2))),
                        "ar_shift_requested": -1.0, "ar_shift_median": float(np.nanmedian(shift)),
                        "ar_shift_iqr": [float(np.nanpercentile(shift, 25)), float(np.nanpercentile(shift, 75))]}
        models[beta] = m
        print(f"beta {beta}: {checks[beta]} ({time.time() - t0:.0f} s)", flush=True)
    beta = min(BETAS, key=lambda b: abs(checks[b]["ar_shift_median"] + 1.0))
    m = models[beta]
    print(f"using beta {beta}", flush=True)

    # generate per aspect-ratio limit
    fx = np.array([boundary_vec(f["boundary"], cols) for f in feas])
    fc = cond(np.array([f["violations"] for f in feas])[:, PKG_TO_DATASET],
              np.array([f["aspect_ratio"] for f in feas]), np.array([f["L_gradB"] for f in feas]))
    fz = m.encode(scaler.transform(fx), fc)
    n_edit, n_prior = (8, 8) if dry else (N_EDIT, N_PRIOR)
    cands, gen_stats = [], {}
    for ar in (STEPS[:1] if dry else STEPS):
        rows = []
        for mode, n in [("edit", n_edit), ("prior", n_prior)]:
            if mode == "edit":
                z = fz[rng.integers(len(fz), size=n)] + EDIT_NOISE * rng.standard_normal((n, K))
                lg = np.full(n, 8.5)
            else:
                z = rng.standard_normal((n, K))
                lg = rng.uniform(6, 10, n)
            c = cond(np.full((n, 5), V_TARGET), np.full(n, ar), lg)
            xs = scaler.inverse_transform(m.decode(z, c))
            bjs = [to_boundary_json(x, cols, template) for x in xs]
            for i, (bj, a) in enumerate(zip(bjs, ar_of(bjs))):
                rows.append({"id": f"gen_{step_name(ar)}_{mode}_{i}", "ar_limit": ar, "mode": mode,
                             "geo_ar": float(a), "boundary": bj})
        keep = [r for r in rows if np.isfinite(r["geo_ar"]) and abs(r["geo_ar"] - ar) <= AR_WINDOW]
        gen_stats[step_name(ar)] = {"generated": len(rows), "kept": len(keep),
                                    "geo_ar_median": float(np.nanmedian([r["geo_ar"] for r in rows]))}
        print(f"[{step_name(ar)}] generated {len(rows)}, within ±{AR_WINDOW} of {ar}: {len(keep)} "
              f"(median geometric AR {gen_stats[step_name(ar)]['geo_ar_median']:.2f})", flush=True)
        cands += keep
    tag = "_dry" if dry else ""
    with open(DATA / f"gen_candidates{tag}.jsonl", "w") as f:
        for r in cands:
            f.write(json.dumps(r) + "\n")
    (OUT / f"gen_model{tag}.json").write_text(json.dumps(
        {"beta_checks": {str(b): v for b, v in checks.items()}, "beta": beta, "ar_calibration": calib,
         "n_train": int(len(train)), "n_ours": int(src.sum()), "generation": gen_stats,
         "minutes": round((time.time() - t0) / 60, 1)}, indent=2))


# ------------------------------------------------------------------------------ screen + finish
def screen_and_finish(dry: bool) -> None:
    from evaluate import evaluate_designs

    tag = "_dry" if dry else ""
    cands = read(DATA / f"gen_candidates{tag}.jsonl")
    evals = DATA / f"gen_eval_ledger{tag}.jsonl"
    ledger = DATA / f"gen_ledger{tag}.jsonl"
    by_id = {c["id"]: c for c in cands}

    vlo = {r["id"]: r for r in evaluate_designs(cands, evals, fidelity="very_low_fidelity")}
    picks = []
    for ar in (STEPS[:1] if dry else STEPS):
        ok = [(gap_at({"worst": r["max_violation"], "aspect_ratio": r["aspect_ratio"]}, ar), i)
              for i, r in vlo.items() if r["ok"] and by_id[i]["ar_limit"] == ar]
        picks += [by_id[i] for _, i in sorted(ok)[:(2 if dry else TOP_HI)]]
    hi = {r["id"]: r for r in evaluate_designs(picks, evals, fidelity="high_fidelity")}

    mf.W_OBJ, mf.STOP_ON_FEASIBLE, mf.AR_FEAS_TOL = W_OBJ, True, AR_TOL
    done = {r["start"] for r in read(ledger)}
    for ar in (STEPS[:1] if dry else STEPS):
        name = "gen_" + step_name(ar)
        if any(belongs(s, name) for s in done):
            print(f"[{name}] already in ledger; skipping", flush=True)
            continue
        ok = [(gap_at({"worst": r["max_violation"], "aspect_ratio": r["aspect_ratio"]}, ar), i)
              for i, r in hi.items() if r["ok"] and by_id[i]["ar_limit"] == ar]
        if not ok:
            print(f"[{name}] no generated design solved at high fidelity; skipping", flush=True)
            continue
        g, i = min(ok)
        mf.AR_BOUND, mf.MAXIT = ar, (2 if dry else MAXIT)
        print(f"[{name}] AR <= {ar}: start {g:.4f} ({i}, high-fidelity AR {hi[i]['aspect_ratio']:.2f})",
              flush=True)
        mf.run_start(name, by_id[i]["boundary"], dry, ledger)


def report(dry: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from constellaration import problems

    tag = "_dry" if dry else ""
    cands = {c["id"]: c for c in read(DATA / f"gen_candidates{tag}.jsonl")}
    evals = read(DATA / f"gen_eval_ledger{tag}.jsonl")
    rows = read(DATA / f"gen_ledger{tag}.jsonl")
    chain = read(DATA / f"chain_ledger{tag}.jsonl")
    summary = {"model": json.loads((OUT / f"gen_model{tag}.json").read_text()), "limits": {}}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ar in STEPS:
        sn = step_name(ar)
        ev = [r for r in evals if r["id"] in cands and cands[r["id"]]["ar_limit"] == ar]
        if not ev:
            continue
        vl = [r for r in ev if r["fidelity"] == "very_low_fidelity"]
        hs = [r for r in ev if r["fidelity"] == "high_fidelity" and r["ok"]]
        gaps = [gap_at({"worst": r["max_violation"], "aspect_ratio": r["aspect_ratio"]}, ar) for r in vl if r["ok"]]
        mine = [r for r in rows if belongs(r["start"], "gen_" + sn)]
        his = [r for r in mine if r["kind"] == "hi"]
        hit = [j for j, r in enumerate(his) if r["ok"] and feasible_at(r, ar)]
        ch = [r for r in chain if belongs(r["start"], sn)]
        ch_his = [r for r in ch if r["kind"] == "hi"]
        ch_hit = [j for j, r in enumerate(ch_his) if r["ok"] and feasible_at(r, ar)]
        summary["limits"][sn] = {
            "very_low_screened": len(vl), "very_low_solved": sum(r["ok"] for r in vl),
            "very_low_gap_median": float(np.median(gaps)) if gaps else None,
            "very_low_gap_best": float(min(gaps)) if gaps else None,
            "edit_vs_prior_best": {md: min((gap_at({"worst": r["max_violation"], "aspect_ratio": r["aspect_ratio"]}, ar)
                                            for r in vl if r["ok"] and cands[r["id"]]["mode"] == md), default=None)
                                   for md in ("edit", "prior")},
            "high_checked": len(hs),
            "start_gap_high": min((gap_at({"worst": r["max_violation"], "aspect_ratio": r["aspect_ratio"]}, ar)
                                   for r in hs), default=None),
            "chain_start_gap_high": gap_at(ch_his[0], ar) if ch_his and ch_his[0]["ok"] else None,
            "finisher_iterations": sum(r["kind"] == "iteration" for r in mine),
            "reached": bool(hit), "high_runs_to_feasible": (hit[0] + 1 + len(hs)) if hit else None,
            "very_low_runs_to_feasible": (len(vl) + sum(r["kind"] == "oracle" for r in mine[:mine.index(his[hit[0]])])
                                          if hit else None),
            "chain_reached": bool(ch_hit), "chain_high_runs_to_feasible": ch_hit[0] + 1 if ch_hit else None,
        }
        axs[0].scatter([ar] * len(gaps), np.clip(gaps, 1e-3, 10), s=5, alpha=0.3, color="#1f6fd1")
    axs[0].set_yscale("log")
    axs[0].axhline(0.01, color="k", ls=":", lw=1)
    axs[0].set_xlabel("aspect-ratio limit")
    axs[0].set_ylabel("gap to feasible at the limit (very low fidelity)")
    axs[0].set_title("generated designs, screened")
    # combined front
    feas, seen = [], set()
    for src, rs in [("before", read(DATA / "multifid_ledger.jsonl") + read(DATA / "front_ledger.jsonl")),
                    ("chain", chain), ("generator", rows)]:
        for r in rs:
            if r["kind"] == "hi" and r.get("feasible"):
                key = (round(r["aspect_ratio"], 6), round(r["L_gradB"], 6))
                if key not in seen:
                    seen.add(key)
                    feas.append({**r, "src": src})
    for src, color in [("before", "#9a9a9a"), ("chain", "#d9622b"), ("generator", "#1f6fd1")]:
        ps = [p for p in feas if p["src"] == src]
        if ps:
            axs[1].scatter([p["aspect_ratio"] for p in ps], [p["L_gradB"] for p in ps], s=14, color=color,
                           label=f"{src} ({len(ps)})")
    front = pareto(feas)
    hv = (float(problems._hypervolume(np.array([[-p["L_gradB"], p["aspect_ratio"]] for p in front]),
                                      np.array([1.0, 20.0]))) if front else 0.0)
    summary["hypervolume_all"] = hv
    summary["n_feasible_generator"] = sum(p["src"] == "generator" for p in feas)
    axs[1].set_xlabel("aspect ratio")
    axs[1].set_ylabel("L∇B")
    axs[1].set_title(f"feasible designs, front hypervolume {hv:.1f}")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"gen{tag}.png", dpi=150)
    (OUT / f"gen{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["limits"], indent=2), "\nhypervolume", hv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    tag = "_dry" if a.dry else ""
    if not a.report:
        if a.train or not (DATA / f"gen_candidates{tag}.jsonl").exists():
            train_and_generate(a.dry)
        if not a.train:
            screen_and_finish(a.dry)
    if not a.train:
        report(a.dry)


if __name__ == "__main__":
    main()
