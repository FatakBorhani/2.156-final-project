"""Test 15: learn from our own simulations: conditional diffusion + surrogate refinement.

Tests 7-14 were local optimization from one family of shapes; all 262 feasible designs we
have descend from one dataset design and sit at AR 11.26-17. Test 12's conditional VAE
proposed starts at lower aspect ratios, but they began 0.06-0.32 from feasible (worse than
the chain's), because the VAE blurs the fine shape detail QI depends on. Here:

1. Data: Proxima's 68k NFP-3 designs plus every simulation of ours that kept its shape
   (~36k: Tests 7-14, including 2.3k high-fidelity checks at the pass/fail line, and
   Test 12's screened samples). Ours are repeated REP times so they carry about half the
   weight.
2. Surrogate: 3 MLPs (JAX, so differentiable) predict the 5 violations, aspect ratio and
   L_gradB from the shape, with a 'very low fidelity' input flag (predict with it off).
3. Generators, conditioned on (5 violations, aspect ratio, L_gradB):
   - a diffusion model (DDPM, 80 coefficients, classifier-free guidance), sampled from
     noise ('diff_prior') and as edits of our feasible designs ('diff_edit': noise the
     closest feasible design part way, denoise toward the new target);
   - Test 12's cVAE retrained on the same data ('cvae'), the comparison;
   - no generator: our feasible designs and the closest dataset designs, refined by the
     surrogate alone ('surr_only'), the control.
4. Refine every candidate: gradient steps on the surrogate (pessimistic: mean + 1 std of
   the 3 models) toward 'all 5 pass, AR <= target, L_gradB >= target', with a penalty for
   moving far from the generated shape. 'diff_raw' screens some diffusion samples
   unrefined, to measure what refinement adds.
5. Screen diverse picks with very-low-fidelity VMEC++ (per target and arm), re-check the
   best at high fidelity, then run Test 13's finisher (L_gradB floor as a 7th constraint)
   from the best start per target, most hypervolume first, while it is close enough.

Targets: just beyond our current front, compact (AR 9-11, Test 14's line) and simpler
(AR 12-16, L_gradB about 1.25 above our front).

    python scripts/feasibility_tests/diffusion_test.py --train   # train + generate (~15-25 min)
    python scripts/feasibility_tests/diffusion_test.py           # + screen + finish (~8-12 h)
    python scripts/feasibility_tests/diffusion_test.py --dry     # tiny version of everything
    python scripts/feasibility_tests/diffusion_test.py --report  # re-plot only
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import constellaration.optimization.augmented_lagrangian_runner  # noqa: F401, E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import multifid_test as mf  # noqa: E402
from common import CONSTRAINTS, ROOT, load_designs  # noqa: E402
from front_test import pareto, read  # noqa: E402
from gen_test import CVAE, PKG_TO_DATASET, boundary_vec, cond  # noqa: E402

SEED = 0
OUT = ROOT / "results" / "feasibility_tests"
DATA = ROOT / "data" / "feasibility_tests"
TAG = "diffusion"
# our ledgers whose rows (some or all) kept their shape; violations in package order
OUR_LEDGERS = ["polish_ledger.jsonl", "alm_ledger.jsonl", "multifid_ledger.jsonl", "front_ledger.jsonl",
               "chain_ledger.jsonl", "chain_ledger_partial_2026-11-17.jsonl", "gen_ledger.jsonl",
               "ladder_ledger.jsonl", "compact_ledger.jsonl"]
FEAS_LEDGERS = ["multifid_ledger.jsonl", "front_ledger.jsonl", "chain_ledger.jsonl", "gen_ledger.jsonl",
                "ladder_ledger.jsonl", "compact_ledger.jsonl"]
# screened generator samples: shapes in a candidates file, metrics in its eval ledger
EVAL_SETS = [("gen_candidates.jsonl", "gen_eval_ledger.jsonl")]
REP = 2  # our rows repeated this many times in training

# surrogate
S_HIDDEN, S_MODELS, S_EPOCHS, S_BATCH, S_LR = 512, 3, 80, 512, 1e-3
V_LO, V_HI = -0.5, 3.0  # violations clipped to this range for training
# diffusion
D_HIDDEN, D_T, D_STEPS, D_BATCH, D_LR = 512, 500, 60000, 512, 1e-3
P_UNCOND, GUIDE, SAMPLE_STEPS, ETA = 0.15, 1.5, 100, 1.0
EDIT_STRENGTHS = [0.3, 0.5]  # fraction of the noise schedule an edit starts from
# refinement
KAPPA, MARGIN, SMOOTH, TRUST, R_STEPS, R_LR = 1.0, 0.01, 30.0, 0.1, 200, 0.02
R_MAX = 1.0  # max refinement move (standardized units; our feasible family spans ~5.6 from AR 11.3 to 17)
# per target and arm
N_GEN, N_REFINE, N_POOL, N_SCREEN, N_RAW = 2000, 300, 100, 24, 12
TOP_HI_PER_ARM = 2
# finisher
MAXIT, W_OBJ, TOL, FINISH_MAX_GAP = 12, 0.05, 0.01, 0.12
V_TARGET = -0.03  # 'just passing' for every requirement (the generators' condition)
COMPACT = [(9.0, 5.2), (9.5, 6.0), (10.0, 6.8), (10.5, 7.6), (11.0, 8.4)]  # Test 14's line
UPPER_AR, UPPER_LIFT = [12.0, 13.0, 14.0, 15.0, 16.0], 1.25
ARMS = ["diff_prior", "diff_edit", "cvae", "surr_only"]


def tname(ar: float, lgb: float) -> str:
    return f"t15_ar{ar:g}_lgb{lgb:g}".replace(".", "p")


def gap_at(worst: float, ar_val: float, lgb_val: float, ar: float, lgb: float) -> float:
    return max(worst, (ar_val - ar) / ar, (lgb - lgb_val) / lgb)


# ------------------------------------------------------------------------------ data
def feasible_rows() -> list[dict]:
    rows, seen = [], set()
    for s in FEAS_LEDGERS:
        for r in read(DATA / s):
            if r["kind"] == "hi" and r.get("feasible") and r.get("boundary"):
                key = (round(r["aspect_ratio"], 6), round(r["L_gradB"], 6))
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    return rows


def targets(feas: list[dict]) -> list[tuple[float, float]]:
    def front_at(a):
        return max((r["L_gradB"] for r in feas if r["aspect_ratio"] <= a), default=0.0)
    upper = [(a, round((front_at(a) + UPPER_LIFT) * 4) / 4) for a in UPPER_AR]
    return COMPACT + upper


def training_data(cols: list[str]):
    """Rows: (x80, violations in CONSTRAINTS order, AR, L_gradB, is_very_low, is_ours)."""
    X, T, V, _ = load_designs(3)
    xs, vs, ars, lgs = [X[cols].values], [V[CONSTRAINTS].values], [T["aspect_ratio"].values], \
        [T["L_gradB"].values]
    vlo, ours = [np.zeros(len(X))], [np.zeros(len(X))]
    seen = set()

    def add(bjson, viol_pkg, ar, lg, is_vlo):
        key = bjson[:4000]
        if key in seen:
            return
        seen.add(key)
        xs.append(boundary_vec(bjson, cols)[None])
        vs.append(np.asarray(viol_pkg, float)[PKG_TO_DATASET][None])
        ars.append([ar])
        lgs.append([lg])
        vlo.append([float(is_vlo)])
        ours.append([1.0])

    for s in OUR_LEDGERS:
        for r in read(DATA / s):
            if r.get("ok") and r.get("boundary") and r.get("violations") and r.get("kind") in ("hi", "oracle"):
                add(r["boundary"], r["violations"], r["aspect_ratio"], r["L_gradB"], r["kind"] != "hi")
    # screened generator samples (Test 12's, later Test 15's): shapes in the candidates file
    for cf, ef in EVAL_SETS:
        cands = {c["id"]: c["boundary"] for c in read(DATA / cf)}
        for r in read(DATA / ef):
            if r.get("ok") and r["id"] in cands:
                add(cands[r["id"]], list(r["violations"].values()), r["aspect_ratio"], r["L_gradB"],
                    r["fidelity"] != "high_fidelity")
    Xa, Va = np.vstack(xs), np.vstack(vs)
    A, L = np.concatenate([np.ravel(a) for a in ars]), np.concatenate([np.ravel(a) for a in lgs])
    F, O = np.concatenate([np.ravel(a) for a in vlo]), np.concatenate([np.ravel(a) for a in ours])
    ok = np.isfinite(Xa).all(1) & np.isfinite(Va).all(1) & np.isfinite(A) & np.isfinite(L)
    print(f"training data: {int((O[ok] == 0).sum()):,} dataset + {int((O[ok] == 1).sum()):,} of ours "
          f"({int(((O == 1) & (F == 0) & ok).sum()):,} high fidelity)", flush=True)
    return Xa[ok], Va[ok], A[ok], L[ok], F[ok], O[ok]


class Std:
    def fit(self, X):
        self.mean_, self.scale_ = X.mean(0), X.std(0) + 1e-12
        return self

    def transform(self, X):
        return (np.asarray(X) - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        return np.asarray(Z) * self.scale_ + self.mean_


# ------------------------------------------------------------------------------ surrogate
class Surrogate:
    """S_MODELS MLPs on standardized coefficients (+ very-low-fidelity flag) -> 7 outputs:
    5 violations (clipped), aspect ratio, L_gradB."""

    def fit(self, Z, Y, flag, w, epochs, val=None):
        import jax
        import jax.numpy as jnp
        import optax

        from compression_test import _init_mlp, _mlp
        self.ys = Std().fit(Y)
        Ys = self.ys.transform(Y)
        Xin = np.hstack([Z, flag[:, None]]).astype(np.float32)
        steps = epochs * (len(Z) // S_BATCH)
        self.params, info = [], {}
        for m in range(S_MODELS):
            params = _init_mlp(jax.random.PRNGKey(100 + m), [Xin.shape[1], S_HIDDEN, S_HIDDEN, S_HIDDEN, 7])
            opt = optax.adamw(optax.cosine_decay_schedule(S_LR, steps), weight_decay=1e-5)
            state = opt.init(params)

            @jax.jit
            def step(params, state, x, y, ww):
                def loss(p):
                    return jnp.mean(ww[:, None] * (_mlp(p, x) - y) ** 2)
                lv, g = jax.value_and_grad(loss)(params)
                upd, state = opt.update(g, state, params)
                return optax.apply_updates(params, upd), state, lv

            rng = np.random.default_rng(SEED + m)
            Xj, Yj, Wj = jnp.asarray(Xin), jnp.asarray(Ys, jnp.float32), jnp.asarray(w, jnp.float32)
            for _ in range(epochs):
                perm = rng.permutation(len(Z))
                for i in range(0, len(Z) - S_BATCH + 1, S_BATCH):
                    idx = perm[i:i + S_BATCH]
                    params, state, lv = step(params, state, Xj[idx], Yj[idx], Wj[idx])
            self.params.append(params)
            info[f"model{m}_final_loss"] = float(lv)
        return info

    def predict_all(self, Z):
        """(models, n, 7) in physical units, with the high-fidelity flag."""
        import jax.numpy as jnp
        return np.asarray(self.jax_predict_all(jnp.asarray(Z, jnp.float32)))

    def jax_predict_all(self, Zj):
        import jax.numpy as jnp

        from compression_test import _mlp
        x = jnp.concatenate([Zj, jnp.zeros((Zj.shape[0], 1), Zj.dtype)], 1)
        out = jnp.stack([_mlp(p, x) for p in self.params])
        return out * jnp.asarray(self.ys.scale_, jnp.float32) + jnp.asarray(self.ys.mean_, jnp.float32)


def pessimistic_terms(out, ar_t, lg_t):
    """out: (models, n, 7). Terms (n, 7) that must all be <= 0 (violation + margin, AR, L_gradB)."""
    import jax.numpy as jnp
    mu, sd = out.mean(0), out.std(0)
    v = mu[:, :5] + KAPPA * sd[:, :5] + MARGIN
    ar = mu[:, 5] + KAPPA * sd[:, 5]
    lg = mu[:, 6] - KAPPA * sd[:, 6]
    return jnp.concatenate([v, ((ar - ar_t) / ar_t)[:, None], ((lg_t - lg) / lg_t)[:, None]], 1)


def predicted_gap(sur: Surrogate, Z, ar_t, lg_t) -> np.ndarray:
    import jax.numpy as jnp
    t = pessimistic_terms(sur.jax_predict_all(jnp.asarray(Z, jnp.float32)), ar_t, lg_t)
    return np.asarray(t.max(1))


def refine(sur: Surrogate, Z0: np.ndarray, ar_t: float, lg_t: float, steps: int):
    """Gradient steps on the pessimistic smooth-max gap plus a pull back to the start shape."""
    import jax
    import jax.numpy as jnp
    import optax

    Z0j = jnp.asarray(Z0, jnp.float32)

    def loss(Z):
        t = pessimistic_terms(sur.jax_predict_all(Z), ar_t, lg_t)
        g = jax.nn.logsumexp(SMOOTH * t, axis=1) / SMOOTH
        return jnp.sum(g + TRUST * jnp.sum((Z - Z0j) ** 2, axis=1))

    opt = optax.adam(R_LR)
    Z, state = Z0j, opt.init(Z0j)
    grad = jax.jit(jax.grad(loss))
    for _ in range(steps):
        upd, state = opt.update(grad(Z), state)
        Z = optax.apply_updates(Z, upd)
        dz = Z - Z0j  # project back inside the trust radius
        Z = Z0j + dz * jnp.minimum(1.0, R_MAX / (jnp.linalg.norm(dz, axis=1, keepdims=True) + 1e-9))
    Z = np.asarray(Z)
    return Z, float(np.sqrt(np.mean(np.sum((Z - Z0) ** 2, axis=1))))


# ------------------------------------------------------------------------------ diffusion
def _alpha_bar(T: int) -> np.ndarray:
    s = np.arange(T + 1) / T
    ab = np.cos((s + 0.008) / 1.008 * np.pi / 2) ** 2
    return np.clip(ab / ab[0], 1e-5, 1.0)  # ab[0] = 1 (clean), ab[T] ~ 0 (noise)


class Diffusion:
    """DDPM on standardized coefficients with classifier-free guidance on the condition."""

    def __init__(self, d: int, dc: int):
        import jax

        from compression_test import _init_mlp
        self.d, self.dc = d, dc
        self.ab = _alpha_bar(D_T)
        self.params = _init_mlp(jax.random.PRNGKey(SEED + 7), [d + 64 + dc + 1] + [D_HIDDEN] * 4 + [d])

    @staticmethod
    def _temb(t):
        import jax.numpy as jnp
        freqs = jnp.exp(jnp.linspace(0.0, jnp.log(1000.0), 32))
        a = t[:, None] * freqs[None]
        return jnp.concatenate([jnp.sin(a), jnp.cos(a)], 1)

    def _eps(self, params, x, t, c, m):
        import jax.numpy as jnp

        from compression_test import _mlp
        return _mlp(params, jnp.concatenate([x, self._temb(t / D_T), c * m[:, None], m[:, None]], 1))

    def fit(self, Z, C, steps):
        import jax
        import jax.numpy as jnp
        import optax

        ab = jnp.asarray(self.ab, jnp.float32)
        opt = optax.adamw(optax.cosine_decay_schedule(D_LR, steps), weight_decay=1e-5)
        state = opt.init(self.params)

        @jax.jit
        def step(params, state, x0, c, key):
            k1, k2, k3 = jax.random.split(key, 3)
            t = jax.random.randint(k1, (x0.shape[0],), 1, D_T + 1)
            eps = jax.random.normal(k2, x0.shape)
            m = (jax.random.uniform(k3, (x0.shape[0],)) > P_UNCOND).astype(jnp.float32)
            a = ab[t][:, None]
            xt = jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps

            def loss(p):
                return jnp.mean((self._eps(p, xt, t.astype(jnp.float32), c, m) - eps) ** 2)
            lv, g = jax.value_and_grad(loss)(params)
            upd, state = opt.update(g, state, params)
            return optax.apply_updates(params, upd), state, lv

        rng, key = np.random.default_rng(SEED), jax.random.PRNGKey(SEED)
        Zj, Cj = jnp.asarray(Z, jnp.float32), jnp.asarray(C, jnp.float32)
        losses = []
        for i in range(steps):
            key, sub = jax.random.split(key)
            idx = rng.integers(0, len(Z), D_BATCH)
            self.params, state, lv = step(self.params, state, Zj[idx], Cj[idx], sub)
            if i % 1000 == 0:
                losses.append(float(lv))
        return {"loss_first": losses[0], "loss_last_1k_mean": float(np.mean(losses[-5:]))}

    def sample(self, c: np.ndarray, rng: np.random.Generator, x_start=None, strength: float = 1.0):
        """DDIM (eta=ETA) with guidance. x_start + strength < 1: edit (SDEdit) that design."""
        import jax
        import jax.numpy as jnp

        n = len(c)
        t0 = int(round(strength * D_T))
        ts = np.unique(np.linspace(1, t0, max(2, int(SAMPLE_STEPS * strength))).round().astype(int))[::-1]
        ab = self.ab
        if x_start is None:
            x = rng.standard_normal((n, self.d))
        else:
            x = np.sqrt(ab[t0]) * x_start + np.sqrt(1 - ab[t0]) * rng.standard_normal((n, self.d))
        cj = jnp.asarray(c, jnp.float32)
        ones, zeros = jnp.ones(n), jnp.zeros(n)

        @jax.jit
        def eps_g(params, x, t):
            tt = jnp.full((n,), t, jnp.float32)
            e_c = self._eps(params, x, tt, cj, ones)
            e_u = self._eps(params, x, tt, cj, zeros)
            return e_u + (1 + GUIDE) * (e_c - e_u)

        for i, t in enumerate(ts):
            tp = ts[i + 1] if i + 1 < len(ts) else 0
            a, ap = ab[t], ab[tp]
            e = np.asarray(eps_g(self.params, jnp.asarray(x, jnp.float32), float(t)))
            x0 = np.clip((x - np.sqrt(1 - a) * e) / np.sqrt(a), -8, 8)
            sig = ETA * np.sqrt((1 - ap) / (1 - a)) * np.sqrt(max(0.0, 1 - a / ap))
            x = np.sqrt(ap) * x0 + np.sqrt(max(0.0, 1 - ap - sig**2)) * e + sig * rng.standard_normal(x.shape)
        return x


# ------------------------------------------------------------------------------ train + generate
def diverse(Z: np.ndarray, order: np.ndarray, k: int) -> list[int]:
    """Greedy farthest-point picks among the given (best-first) indices, starting with the best."""
    pool = list(order)
    picks = [pool.pop(0)]
    while pool and len(picks) < k:
        d = np.min(np.linalg.norm(Z[pool][:, None] - Z[picks][None], axis=2), axis=1)
        picks.append(pool.pop(int(np.argmax(d))))
    return picks


def train_and_generate(dry: bool) -> None:
    import pandas as pd

    from compression_test import GlobalScaler
    from search_test import to_boundary_json

    t0 = time.time()
    tag = "_dry" if dry else ""
    rng = np.random.default_rng(SEED)
    feas = feasible_rows()
    template = feas[0]["boundary"]
    X, _, _, _ = load_designs(3)
    cols = list(X.columns)
    Xa, Va, A, L, F, O = training_data(cols)
    if dry:
        keep = np.r_[rng.choice(np.where(O == 0)[0], 4000, replace=False), rng.choice(np.where(O == 1)[0], 3000, replace=False)]
        Xa, Va, A, L, F, O = Xa[keep], Va[keep], A[keep], L[keep], F[keep], O[keep]
    zs = Std().fit(Xa)
    Z = zs.transform(Xa)
    Y = np.column_stack([np.clip(Va, V_LO, V_HI), np.clip(A, 3, 25), np.clip(L, 0, 25)])
    C = cond(Va, A, L)

    # --- surrogate, validated on held-out high-fidelity rows of ours
    hi_ours = np.where((O == 1) & (F == 0))[0]
    val = rng.choice(hi_ours, max(1, len(hi_ours) // 10), replace=False)
    tr = np.setdiff1d(np.arange(len(Z)), val)
    tr = np.r_[tr, np.repeat(tr[O[tr] == 1], REP - 1)]
    worst = np.clip(Va, 0, None).max(1)
    w = 1.0 + 4.0 * (worst < 0.5) + 10.0 * ((O == 1) & (F == 0))
    sur = Surrogate()
    s_info = sur.fit(Z[tr], Y[tr], F[tr], w[tr], 2 if dry else S_EPOCHS)
    P = sur.predict_all(Z[val]).mean(0)
    true_w, pred_w = worst[val], np.clip(P[:, :5], 0, None).max(1)
    from scipy.stats import spearmanr
    s_info.update({
        "val_high_fidelity_rows": int(len(val)),
        "mae_violations": np.abs(P[:, :5] - np.clip(Va[val], V_LO, V_HI)).mean(0).round(4).tolist(),
        "mae_aspect_ratio": float(np.abs(P[:, 5] - A[val]).mean()),
        "mae_L_gradB": float(np.abs(P[:, 6] - L[val]).mean()),
        "mae_worst": float(np.abs(pred_w - true_w).mean()),
        "spearman_worst": float(spearmanr(pred_w, true_w)[0]),
        "true_worst_median": float(np.median(true_w)),
    })
    print(f"surrogate: {s_info} ({time.time() - t0:.0f} s)", flush=True)

    # --- generators (training rows: everything, ours repeated)
    gtr = np.r_[np.arange(len(Z)), np.repeat(np.where(O == 1)[0], REP - 1)]
    dif = Diffusion(Z.shape[1], C.shape[1])
    d_info = dif.fit(Z[gtr], C[gtr], 300 if dry else D_STEPS)
    print(f"diffusion: {d_info} ({time.time() - t0:.0f} s)", flush=True)
    gs = GlobalScaler().fit(pd.DataFrame(Xa, columns=cols))
    cv = CVAE(Xa.shape[1], C.shape[1], 0.5)
    c_info = cv.fit(gs.transform(Xa[gtr]), C[gtr], 2 if dry else 60)
    print(f"cvae: {c_info} ({time.time() - t0:.0f} s)", flush=True)
    with open(DATA / f"{TAG}_models{tag}.pkl", "wb") as f:
        pickle.dump({"zs": zs, "sur": sur, "dif_params": dif.params, "cvae": cv.params, "gs": gs,
                     "cols": cols}, f)

    # --- per target: generate, refine, pick
    fz = zs.transform(np.array([boundary_vec(r["boundary"], cols) for r in feas]))
    f_ar = np.array([r["aspect_ratio"] for r in feas])
    n_gen, n_ref, n_pool, n_scr, n_raw = (64, 32, 16, 3, 2) if dry else (N_GEN, N_REFINE, N_POOL, N_SCREEN, N_RAW)
    tgts = targets(feas)[:2] if dry else targets(feas)
    cands, stats = [], {}
    for ar_t, lg_t in tgts:
        name = tname(ar_t, lg_t)
        c = cond(np.full((n_gen, 5), V_TARGET), np.full(n_gen, ar_t), np.full(n_gen, lg_t))
        raw = {"diff_prior": dif.sample(c, rng)}
        near = np.argsort(np.abs(f_ar - ar_t))[:20]  # edits start from the closest feasible designs
        starts = fz[rng.choice(near, n_gen)]
        half = n_gen // 2
        raw["diff_edit"] = np.vstack([dif.sample(c[:half], rng, starts[:half], EDIT_STRENGTHS[0]),
                                      dif.sample(c[half:], rng, starts[half:], EDIT_STRENGTHS[1])])
        raw["cvae"] = zs.transform(gs.inverse_transform(cv.decode(rng.standard_normal((n_gen, 16)), c)))
        # control: closest real shapes by aspect ratio (dataset + ours), no generator
        near_all = np.argsort(np.abs(A - ar_t) / ar_t + np.clip(Va, 0, None).max(1))[:n_gen]
        raw["surr_only"] = Z[near_all]
        st = {}
        for arm in ARMS:
            Zr = raw[arm]
            g0 = predicted_gap(sur, Zr, ar_t, lg_t)
            top = np.argsort(g0)[:n_ref]
            Zf, moved = refine(sur, Zr[top], ar_t, lg_t, 20 if dry else R_STEPS)
            g1 = predicted_gap(sur, Zf, ar_t, lg_t)
            order = np.argsort(g1)[:n_pool]
            picks = diverse(Zf, order, n_scr)
            st[arm] = {"pred_gap_raw_median": float(np.median(g0)), "pred_gap_raw_best": float(g0.min()),
                       "pred_gap_refined_best": float(g1.min()),
                       "pred_gap_picked": [float(g1[i]) for i in picks], "refine_rms_move": moved}
            for j, i in enumerate(picks):
                x = zs.inverse_transform(Zf[i])
                cands.append({"id": f"{name}_{arm}_{j}", "target": name, "ar_t": ar_t, "lg_t": lg_t, "arm": arm,
                              "pred_gap": float(g1[i]), "boundary": to_boundary_json(x, cols, template)})
            if arm == "diff_prior":  # unrefined, to measure what refinement adds
                for j, i in enumerate(diverse(Zr, np.argsort(g0)[:n_pool], n_raw)):
                    x = zs.inverse_transform(Zr[i])
                    cands.append({"id": f"{name}_diff_raw_{j}", "target": name, "ar_t": ar_t, "lg_t": lg_t,
                                  "arm": "diff_raw", "pred_gap": float(g0[i]),
                                  "boundary": to_boundary_json(x, cols, template)})
        stats[name] = st
        print(f"[{name}] " + ", ".join(f"{a}: raw {s['pred_gap_raw_best']:.3f} -> {s['pred_gap_refined_best']:.3f}"
                                       for a, s in st.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    with open(DATA / f"{TAG}_candidates{tag}.jsonl", "w") as f:
        for r in cands:
            f.write(json.dumps(r) + "\n")
    (OUT / f"{TAG}_model{tag}.json").write_text(json.dumps(
        {"n_train": int(len(Z)), "n_ours": int(O.sum()), "n_ours_high": int(((O == 1) & (F == 0)).sum()),
         "surrogate": s_info, "diffusion": d_info, "cvae": c_info, "targets": tgts, "generation": stats,
         "minutes": round((time.time() - t0) / 60, 1)}, indent=2))


# ------------------------------------------------------------------------------ screen + finish
def screen_and_finish(dry: bool) -> None:
    from constellaration import problems

    from evaluate import evaluate_designs

    tag = "_dry" if dry else ""
    cands = read(DATA / f"{TAG}_candidates{tag}.jsonl")
    by_id = {c["id"]: c for c in cands}
    evals = DATA / f"{TAG}_eval_ledger{tag}.jsonl"
    ledger = DATA / f"{TAG}_ledger{tag}.jsonl"

    def gap(r):
        c = by_id[r["id"]]
        return gap_at(r["max_violation"], r["aspect_ratio"], r["L_gradB"], c["ar_t"], c["lg_t"])

    vlo = [r for r in evaluate_designs(cands, evals, fidelity="very_low_fidelity") if r["ok"]]
    picks = []
    for t in dict.fromkeys(c["target"] for c in cands):
        for arm in ARMS + ["diff_raw"]:
            rs = sorted((r for r in vlo if by_id[r["id"]]["target"] == t and by_id[r["id"]]["arm"] == arm), key=gap)
            picks += [by_id[r["id"]] for r in rs[:(1 if dry else TOP_HI_PER_ARM)] if arm != "diff_raw"]
    hi = [r for r in evaluate_designs(picks, evals, fidelity="high_fidelity") if r["ok"]]

    # finisher: per target, best high-fidelity start; most hypervolume gained first
    feas = feasible_rows()
    pts = np.array([[-r["L_gradB"], r["aspect_ratio"]] for r in feas])
    hv0 = float(problems._hypervolume(pts, np.array([1.0, 20.0])))
    plan = []
    for t in dict.fromkeys(c["target"] for c in cands):
        rs = [r for r in hi if by_id[r["id"]]["target"] == t]
        if not rs:
            continue
        best = min(rs, key=gap)
        c = by_id[best["id"]]
        gain = float(problems._hypervolume(np.vstack([pts, [[-c["lg_t"], c["ar_t"]]]]), np.array([1.0, 20.0]))) - hv0
        plan.append((gain, gap(best), t, best, c))
    plan.sort(key=lambda p: -p[0])
    print("finisher plan:", [(t, round(g, 1), round(gp, 4)) for g, gp, t, _, _ in plan], flush=True)
    done = {r["start"] for r in read(ledger)}
    mf.W_OBJ, mf.STOP_ON_FEASIBLE, mf.LOG_ORACLE_BOUNDARY = W_OBJ, True, True
    for gain, gp, t, best, c in (plan[:1] if dry else plan):
        if t in done:
            print(f"[{t}] already in ledger; skipping", flush=True)
            continue
        if gp > FINISH_MAX_GAP:
            print(f"[{t}] best start {gp:.4f} > {FINISH_MAX_GAP}; skipping", flush=True)
            continue
        mf.AR_BOUND, mf.AR_FEAS_TOL = c["ar_t"], TOL * c["ar_t"]
        mf.LGB_MIN, mf.LGB_FEAS_TOL = c["lg_t"], TOL * c["lg_t"]
        mf.MAXIT = 2 if dry else MAXIT
        print(f"[{t}] AR <= {c['ar_t']}, L_gradB >= {c['lg_t']}: start gap {gp:.4f} ({best['id']}: worst "
              f"{best['max_violation']:.4f}, AR {best['aspect_ratio']:.2f}, L_gradB {best['L_gradB']:.2f}), "
              f"hypervolume if reached +{gain:.1f}", flush=True)
        mf.run_start(t, c["boundary"], dry, ledger)


# ------------------------------------------------------------------------------ report
def report(dry: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scipy.stats import spearmanr

    from constellaration import problems

    tag = "_dry" if dry else ""
    cands = {c["id"]: c for c in read(DATA / f"{TAG}_candidates{tag}.jsonl")}
    evals = [r for r in read(DATA / f"{TAG}_eval_ledger{tag}.jsonl") if r["id"] in cands]
    rows = read(DATA / f"{TAG}_ledger{tag}.jsonl")
    model = json.loads((OUT / f"{TAG}_model{tag}.json").read_text())
    arms = ["diff_raw"] + ARMS
    tlist = list(dict.fromkeys(c["target"] for c in cands.values()))

    def gap(r):
        c = cands[r["id"]]
        return gap_at(r["max_violation"], r["aspect_ratio"], r["L_gradB"], c["ar_t"], c["lg_t"])

    summary = {"model": model, "targets": {}, "arms": {}}
    for arm in arms:
        vl = [r for r in evals if cands[r["id"]]["arm"] == arm and r["fidelity"] == "very_low_fidelity"]
        ok = [r for r in vl if r["ok"]]
        gs = [gap(r) for r in ok]
        summary["arms"][arm] = {
            "screened": len(vl), "solved": len(ok),
            "gap_median": float(np.median(gs)) if gs else None,
            "worst_median": float(np.median([r["max_violation"] for r in ok])) if ok else None,
            "n_gap_below_0p05": int(sum(g < 0.05 for g in gs)), "n_gap_below_0p1": int(sum(g < 0.1 for g in gs)),
            "pred_vs_true_gap_spearman": (float(spearmanr([cands[r["id"]]["pred_gap"] for r in ok], gs)[0])
                                          if len(ok) > 2 else None),
            "best_per_target": {t: min((gap(r) for r in ok if cands[r["id"]]["target"] == t), default=None)
                                for t in tlist},
        }
    for t in tlist:
        c0 = next(c for c in cands.values() if c["target"] == t)
        his = [r for r in evals if cands[r["id"]]["target"] == t and r["fidelity"] == "high_fidelity" and r["ok"]]
        mine = [r for r in rows if r["start"] == t]
        hh = [r for r in mine if r["kind"] == "hi" and r["ok"]]
        hit = [r for r in hh if r.get("feasible") and r["aspect_ratio"] <= c0["ar_t"] * (1 + TOL)
               and r["L_gradB"] >= c0["lg_t"] * (1 - TOL)]
        summary["targets"][t] = {
            "ar_t": c0["ar_t"], "lg_t": c0["lg_t"],
            "best_high_start": min(({"gap": gap(r), "arm": cands[r["id"]]["arm"], "worst": r["max_violation"],
                                     "aspect_ratio": r["aspect_ratio"], "L_gradB": r["L_gradB"]} for r in his),
                                   key=lambda d: d["gap"], default=None),
            "finisher_iterations": sum(r["kind"] == "iteration" for r in mine),
            "feasible_designs": sum(1 for r in hh if r.get("feasible")),
            "reached": bool(hit),
            "best_finisher_gap": min((gap_at(r["worst"], r["aspect_ratio"], r["L_gradB"], c0["ar_t"], c0["lg_t"])
                                      for r in hh), default=None),
        }

    # front: everything before + this test (generated designs feasible at high fidelity count too)
    feas, seen = [], set()
    srcs = [(s.split("_ledger")[0], read(DATA / s)) for s in FEAS_LEDGERS] + [(TAG, rows)]
    for src, rs in srcs:
        for r in rs:
            if r["kind"] == "hi" and r.get("feasible"):
                key = (round(r["aspect_ratio"], 6), round(r["L_gradB"], 6))
                if key not in seen:
                    seen.add(key)
                    feas.append({**r, "src": src})
    for r in evals:
        if r["fidelity"] == "high_fidelity" and r.get("feasible"):
            feas.append({"L_gradB": r["L_gradB"], "aspect_ratio": r["aspect_ratio"], "src": TAG,
                         "worst": r["max_violation"], "start": r["id"]})

    def hv(ps):
        f = pareto(ps)
        return float(problems._hypervolume(np.array([[-p["L_gradB"], p["aspect_ratio"]] for p in f]),
                                           np.array([1.0, 20.0]))) if f else 0.0, f
    hv_before, _ = hv([p for p in feas if p["src"] != TAG])
    hv_after, front = hv(feas)
    summary.update(hypervolume_before=hv_before, hypervolume_after=hv_after,
                   n_feasible_new=sum(p["src"] == TAG for p in feas),
                   front=[{k: p.get(k) for k in ("src", "start", "aspect_ratio", "L_gradB", "worst")} for p in front])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    colors = {"diff_raw": "#c4b3e8", "diff_prior": "#7b4fc9", "diff_edit": "#1f6fd1", "cvae": "#d9622b",
              "surr_only": "#9a9a9a"}
    for k, arm in enumerate(arms):
        for i, t in enumerate(tlist):
            gs = [gap(r) for r in evals if r["ok"] and r["fidelity"] == "very_low_fidelity"
                  and cands[r["id"]]["arm"] == arm and cands[r["id"]]["target"] == t]
            axs[0].scatter(np.full(len(gs), i + (k - 2) * 0.15), np.clip(gs, 1e-3, 3), s=6, alpha=0.5,
                           color=colors[arm], label=arm if i == 0 else None)
    axs[0].axhline(0.01, color="k", ls=":", lw=1)
    axs[0].set_yscale("log")
    axs[0].set_xticks(range(len(tlist)))
    tv = {c["target"]: (c["ar_t"], c["lg_t"]) for c in cands.values()}
    axs[0].set_xticklabels([f"{tv[t][0]:g}/{tv[t][1]:g}" for t in tlist], rotation=45, fontsize=7)
    axs[0].set_xlabel("target (AR / L∇B)")
    axs[0].set_ylabel("gap to target (very low fidelity)")
    axs[0].set_title("generated + refined designs, screened", fontsize=9)
    axs[0].legend(fontsize=7, markerscale=2)
    for src, color in [("multifid", "#9a9a9a"), ("front", "#9a9a9a"), ("chain", "#7fb3e6"), ("gen", "#7b4fc9"),
                       ("ladder", "#2a9d6f"), ("compact", "#d9622b"), ("diffusion", "#8c564b"),
                       ("descent", "#e0a030"), (TAG, "#d62728")]:
        ps = [p for p in feas if p["src"] == src]
        if ps:
            axs[1].scatter([p["aspect_ratio"] for p in ps], [p["L_gradB"] for p in ps], s=10, color=color,
                           label=f"{src} ({len(ps)})")
    xs = [p["aspect_ratio"] for p in front] + [20.0]
    ys = [p["L_gradB"] for p in front]
    axs[1].step(xs, ys + [ys[-1]], where="post", color="k", lw=1)
    axs[1].set_xlabel("aspect ratio")
    axs[1].set_ylabel("L∇B")
    axs[1].set_title(f"feasible designs; hypervolume {hv_before:.1f} → {hv_after:.1f}", fontsize=9)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"{TAG}{tag}.png", dpi=150)
    (OUT / f"{TAG}{tag}.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / f"{TAG}_feasible_designs{tag}.jsonl", "w") as f:
        for r in rows:
            if r["kind"] == "hi" and r.get("feasible"):
                f.write(json.dumps({k: r.get(k) for k in ("start", "k", "worst", "L_gradB", "aspect_ratio",
                                                         "violations", "boundary")}) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("front", "model")}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    tag = "_dry" if a.dry else ""
    DATA.mkdir(parents=True, exist_ok=True)
    if not a.report:
        if a.train or not (DATA / f"{TAG}_candidates{tag}.jsonl").exists():
            train_and_generate(a.dry)
        if not a.train:
            screen_and_finish(a.dry)
    if not a.train:
        report(a.dry)


if __name__ == "__main__":
    main()
