"""Test 3: can shapes be compressed (PCA or a VAE) without changing the physics?

Inputs are the 81 boundary coefficients of 3-field-period designs (standardized), split
80/20 like Test 1. For latent sizes k = 4, 8, 16, 32 we fit PCA and train a small VAE
(JAX; untuned throwaway version, beta = BETA), then check four things:

  A. Reconstruction: rebuild 100 held-out designs (50 closest to feasible + 50 random)
     through each model, run VMEC++, and measure how much the physics changed.
  B. Sampling: draw 50 new designs per model from its latent distribution. What fraction
     can VMEC++ solve, and how QI-like are they? (The VAE's main selling point.)
  C. Interpolation: walk from a good-QI design to a good-everything-else design (the two
     groups in the README figure) in raw coefficients, PCA-16 and VAE-16 space. Do the
     in-between shapes stay valid, and do any beat both ends?
  D. Spot check: high-fidelity runs for 10 of the closest designs rebuilt by PCA-16 and
     VAE-16, to confirm the very-low-fidelity conclusions.

A-C use very-low-fidelity VMEC++ (~3 s). Comparing a design with its own reconstruction
at the same fidelity cancels most of that fidelity's error (Test 4).

    python scripts/feasibility_tests/compression_test.py
    -> results/feasibility_tests/compression.json, compression.png, compression_designs.csv
"""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import CONSTRAINTS, ROOT, load_designs, violations_from_targets  # noqa: E402
from evaluate import evaluate_designs  # noqa: E402
from fidelity_test import targets_from_metrics  # noqa: E402
from search_test import to_boundary_json  # noqa: E402

OUT = ROOT / "results" / "feasibility_tests"
LEDGER = ROOT / "data" / "feasibility_tests" / "compression_ledger.jsonl"
SEED = 0
KS = [4, 8, 16, 32]
BETA = 0.05
EPOCHS, BATCH, LR = 80, 512, 1e-3
N_SAMPLES = 50
SPOT_K = 16
METRICS = ["log10_qi", "abs_iota", "mirror", "flux_compression", "vacuum_well",
           "L_gradB", "aspect_ratio"]


# ----------------------------------------------------------------------------- VAE
def _init_mlp(key, sizes):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        params.append((jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / a), jnp.zeros(b)))
    return params


def _mlp(params, x):
    for W, b in params[:-1]:
        x = jax.nn.gelu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


class VAE:
    def __init__(self, k: int, d: int, hidden: int = 256):
        ke, kd = jax.random.split(jax.random.PRNGKey(SEED + k))
        self.k = k
        self.params = {"enc": _init_mlp(ke, [d, hidden, hidden, 2 * k]),
                       "dec": _init_mlp(kd, [k, hidden, hidden, d])}

    @staticmethod
    def _loss(params, x, key, k):
        h = _mlp(params["enc"], x)
        mu, logvar = h[:, :k], jnp.clip(h[:, k:], -12, 6)
        z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
        rec = jnp.sum((x - _mlp(params["dec"], z)) ** 2, axis=1).mean()
        kl = 0.5 * jnp.sum(jnp.exp(logvar) + mu**2 - 1.0 - logvar, axis=1).mean()
        return rec + BETA * kl, (rec, kl)

    def fit(self, X: np.ndarray) -> dict:
        opt = optax.adam(LR)
        state = opt.init(self.params)
        k = self.k

        @jax.jit
        def step(params, state, x, key):
            (loss, aux), g = jax.value_and_grad(self._loss, has_aux=True)(params, x, key, k)
            upd, state = opt.update(g, state)
            return optax.apply_updates(params, upd), state, aux

        rng = np.random.default_rng(SEED)
        key = jax.random.PRNGKey(SEED)
        Xj = jnp.asarray(X, dtype=jnp.float32)
        for _ in range(EPOCHS):
            perm = rng.permutation(len(X))
            for i in range(0, len(X) - BATCH + 1, BATCH):
                key, sub = jax.random.split(key)
                self.params, state, (rec, kl) = step(self.params, state, Xj[perm[i:i + BATCH]], sub)
        return {"final_train_recon_sse": float(rec), "final_train_kl": float(kl)}

    def encode(self, X):  # posterior mean
        return np.asarray(_mlp(self.params["enc"], jnp.asarray(X, jnp.float32))[:, : self.k])

    def decode(self, Z):
        return np.asarray(_mlp(self.params["dec"], jnp.asarray(Z, jnp.float32)))

    def sample(self, n, rng):
        return self.decode(rng.standard_normal((n, self.k)))


class GlobalScaler:
    """Center each coefficient, divide all by ONE number. Keeps the raw relative sizes:
    low-order Fourier modes are up to ~400x larger than high-order ones, and
    per-coefficient standardization would blow the tiny ones up into noise
    (PCA-16 explains 99.2% of raw variance but only 54% of standardized variance)."""

    def fit(self, X):
        self.mean_ = X.values.mean(axis=0)
        self.scale_ = float((X.values - self.mean_).std())
        return self

    def transform(self, X):
        return (np.asarray(X) - self.mean_) / self.scale_

    def inverse_transform(self, S):
        return np.asarray(S) * self.scale_ + self.mean_


class StdPCAModel:
    """Comparison only: PCA after per-coefficient standardization."""

    def __init__(self, k):
        self.k, self.pca = k, PCA(n_components=k, random_state=SEED)

    def fit(self, S):
        self.sd = S.std(axis=0)
        self.pca.fit(S / self.sd)
        return {"explained_variance_of_standardized": round(float(self.pca.explained_variance_ratio_.sum()), 5)}

    def encode(self, S):
        return self.pca.transform(S / self.sd)

    def decode(self, Z):
        return self.pca.inverse_transform(Z) * self.sd

    def sample(self, n, rng):
        return self.decode(rng.standard_normal((n, self.k)) * np.sqrt(self.pca.explained_variance_))


class PCAModel:
    def __init__(self, k):
        self.k, self.pca = k, PCA(n_components=k, random_state=SEED)

    def fit(self, X):
        self.pca.fit(X)
        return {"explained_variance": round(float(self.pca.explained_variance_ratio_.sum()), 5)}

    def encode(self, X):
        return self.pca.transform(X)

    def decode(self, Z):
        return self.pca.inverse_transform(Z)

    def sample(self, n, rng):
        return self.decode(rng.standard_normal((n, self.k)) * np.sqrt(self.pca.explained_variance_))


# ----------------------------------------------------------------------- helpers
def worst(v: pd.DataFrame) -> pd.Series:
    return v.clip(lower=0).max(axis=1)


def run(designs: list[dict], fidelity: str) -> pd.DataFrame:
    """Evaluate and return one row per design: ok + 7 targets + 5 violations + worst."""
    rows = []
    for r in evaluate_designs(designs, LEDGER, fidelity=fidelity):
        row = {"id": r["id"], "ok": r["ok"]}
        if r["ok"]:
            t = targets_from_metrics(r["metrics"])
            v = violations_from_targets(pd.DataFrame([t])).iloc[0]
            row.update(t)
            row.update({f"v_{k}": float(v[k]) for k in CONSTRAINTS})
            row["worst"] = float(np.clip(v.values, 0, None).max())
        rows.append(row)
    return pd.DataFrame(rows).set_index("id")


def main() -> None:
    X, Y, viol, meta = load_designs(3)
    cols = list(X.columns)
    scaler = GlobalScaler().fit(X)
    Xtr, Xte = train_test_split(X, test_size=0.2, random_state=SEED)
    Str, Ste = scaler.transform(Xtr), scaler.transform(Xte)
    rng = np.random.default_rng(SEED)
    out = {"beta": BETA, "latent_sizes": KS}

    # --- fit all models ------------------------------------------------------
    models, fit_info = {}, {}
    for k in KS:
        for name, cls in (("pca", PCAModel), ("vae", VAE)):
            m = cls(k) if name == "pca" else cls(k, Str.shape[1])
            t0 = time.perf_counter()
            info = m.fit(Str)
            Zte = m.encode(Ste)
            rel = np.linalg.norm(Ste - m.decode(Zte), axis=1) / np.linalg.norm(Ste, axis=1)
            info.update(seconds=round(time.perf_counter() - t0),
                        heldout_coeff_rel_error_median=round(float(np.median(rel)), 4))
            models[f"{name}{k}"], fit_info[f"{name}{k}"] = m, info
            print(f"{name}{k}: {info}", flush=True)
    m = StdPCAModel(SPOT_K)
    info = m.fit(Str)
    rel = np.linalg.norm(Ste - m.decode(m.encode(Ste)), axis=1) / np.linalg.norm(Ste, axis=1)
    info["heldout_coeff_rel_error_median"] = round(float(np.median(rel)), 4)
    models[f"pca_standardized{SPOT_K}"], fit_info[f"pca_standardized{SPOT_K}"] = m, info
    print(f"pca_standardized{SPOT_K}: {info}", flush=True)
    out["fit"] = fit_info
    out["scaling"] = "centered, one global scale (raw relative coefficient sizes kept)"

    def to_designs(S: np.ndarray, ids: list[str], seed_ids: list[str]) -> list[dict]:
        Xraw = scaler.inverse_transform(S)
        return [{"id": i, "boundary": to_boundary_json(x, cols, meta.loc[s, "boundary_json"])}
                for i, x, s in zip(ids, Xraw, seed_ids)]

    # --- A. reconstruction ---------------------------------------------------
    w_te = worst(viol.loc[Xte.index])
    pick = w_te.nsmallest(50).index
    pick = pick.append(pd.Index(rng.choice(w_te.index.difference(pick), 50, replace=False)))
    S = scaler.transform(Xte.loc[pick])
    designs = to_designs(S, [f"orig__{i}" for i in pick], list(pick))
    for name, m in models.items():
        designs += to_designs(m.decode(m.encode(S)), [f"{name}__{i}" for i in pick], list(pick))
    print(f"A: evaluating {len(designs)} designs", flush=True)
    A = run(designs, "very_low_fidelity")
    orig = A.loc[[f"orig__{i}" for i in pick]]
    orig.index = pick
    recon = {}
    for name in models:
        r = A.loc[[f"{name}__{i}" for i in pick]]
        r.index = pick
        both = r.ok.astype(bool) & orig.ok.astype(bool)
        recon[name] = {
            "vmec_failed": int((~r.ok.astype(bool)).sum()),
            **{f"mae_{t}": round(float((r.loc[both, t] - orig.loc[both, t]).abs().mean()), 4)
               for t in METRICS},
            "mae_worst_violation": round(float((r.loc[both, "worst"] - orig.loc[both, "worst"]).abs().mean()), 4),
            "mae_worst_violation_50_closest": round(float(
                (r.loc[both & pick.isin(pick[:50]), "worst"] - orig.loc[both & pick.isin(pick[:50]), "worst"]).abs().mean()), 4),
        }
    out["A_reconstruction"] = recon
    print(json.dumps(recon, indent=1), flush=True)

    # --- B. sampling ---------------------------------------------------------
    any_seed = pick[0]  # only used for the JSON template (nfp, symmetry)
    designs = []
    for name, m in models.items():
        designs += to_designs(m.sample(N_SAMPLES, rng), [f"sample_{name}__{j}" for j in range(N_SAMPLES)],
                              [any_seed] * N_SAMPLES)
    print(f"B: evaluating {len(designs)} samples", flush=True)
    B = run(designs, "very_low_fidelity")
    samp = {}
    data_q = Y["log10_qi"]
    for name in models:
        d = B[B.index.str.startswith(f"sample_{name}__")]
        ok = d[d.ok.astype(bool)]
        samp[name] = {
            "vmec_solved": f"{len(ok)}/{len(d)}",
            "median_log10_qi": round(float(ok.log10_qi.median()), 2) if len(ok) else None,
            "share_with_log10_qi_below_-2.5": round(float((ok.log10_qi < -2.5).mean()), 2) if len(ok) else None,
            "median_worst_violation": round(float(ok.worst.median()), 2) if len(ok) else None,
        }
    samp["dataset_reference"] = {
        "median_log10_qi": round(float(data_q.median()), 2),
        "share_with_log10_qi_below_-2.5": round(float((data_q < -2.5).mean()), 2),
        "median_worst_violation": round(float(worst(viol).median()), 2),
    }
    out["B_sampling"] = samp
    print(json.dumps(samp, indent=1), flush=True)

    # --- C. interpolation between the two groups -------------------------------
    fail = viol > 0.01
    qi_group = worst(viol)[~fail["log10_qi"]].nsmallest(5).index
    other_group = viol.loc[~fail.drop(columns="log10_qi").any(axis=1), "log10_qi"].nsmallest(5).index
    ts = [0.25, 0.5, 0.75]
    designs = []
    for p, (a, b) in enumerate(zip(qi_group, other_group)):
        Sa, Sb = scaler.transform(X.loc[[a]]), scaler.transform(X.loc[[b]])
        designs += to_designs(np.vstack([Sa, Sb]), [f"interp{p}_end_a", f"interp{p}_end_b"], [a, b])
        for t in ts:
            designs += to_designs((1 - t) * Sa + t * Sb, [f"interp{p}_raw_t{t}"], [a])
            for name in (f"pca{SPOT_K}", f"vae{SPOT_K}"):
                m = models[name]
                z = (1 - t) * m.encode(Sa) + t * m.encode(Sb)
                designs += to_designs(m.decode(z), [f"interp{p}_{name}_t{t}"], [a])
    print(f"C: evaluating {len(designs)} interpolation designs", flush=True)
    C = run(designs, "very_low_fidelity")
    interp = {}
    for method in ("raw", f"pca{SPOT_K}", f"vae{SPOT_K}"):
        solved, beat_both, best_gain = 0, 0, []
        for p in range(len(qi_group)):
            ends = C.loc[[f"interp{p}_end_a", f"interp{p}_end_b"]]
            mids = C.loc[[f"interp{p}_{method}_t{t}" for t in ts]]
            ok = mids[mids.ok.astype(bool)]
            solved += len(ok)
            if len(ok) and ends.ok.astype(bool).all():
                gain = float(ends.worst.min() - ok.worst.min())
                best_gain.append(gain)
                beat_both += int(gain > 0)
        interp[method] = {
            "vmec_solved": f"{solved}/{len(qi_group) * len(ts)}",
            "paths_with_a_midpoint_better_than_both_ends": f"{beat_both}/{len(qi_group)}",
            "median_improvement_over_best_end": round(float(np.median(best_gain)), 3) if best_gain else None,
        }
    out["C_interpolation"] = interp
    print(json.dumps(interp, indent=1), flush=True)

    # --- D. high-fidelity spot check -----------------------------------------
    spot = pick[:10]
    S = scaler.transform(Xte.loc[spot])
    designs = []
    for name in (f"pca{SPOT_K}", f"vae{SPOT_K}"):
        m = models[name]
        designs += to_designs(m.decode(m.encode(S)), [f"hifi_{name}__{i}" for i in spot], list(spot))
    print(f"D: evaluating {len(designs)} designs at high fidelity", flush=True)
    D = run(designs, "high_fidelity")
    spot_out = {}
    for name in (f"pca{SPOT_K}", f"vae{SPOT_K}"):
        r = D.loc[[f"hifi_{name}__{i}" for i in spot]]
        r.index = spot
        ok = r.ok.astype(bool)
        t_true = Y.loc[spot]
        w_true = worst(viol.loc[spot])
        spot_out[name] = {
            "vmec_failed": int((~ok).sum()),
            "mae_log10_qi_vs_original_high_fidelity": round(float((r.loc[ok, "log10_qi"] - t_true.loc[ok, "log10_qi"]).abs().mean()), 4),
            "mae_worst_violation_vs_original_high_fidelity": round(float((r.loc[ok, "worst"] - w_true[ok]).abs().mean()), 4),
            "very_low_fidelity_estimate_of_same": recon[name]["mae_worst_violation_50_closest"],
        }
    out["D_high_fidelity_spot_check"] = spot_out
    print(json.dumps(spot_out, indent=1), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compression.json").write_text(json.dumps(out, indent=2))
    pd.concat([A, B, C, D]).to_csv(OUT / "compression_designs.csv")

    # --- figure ----------------------------------------------------------------
    colors = {"pca": "#2a78d6", "vae": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    panels = [
        ("mae_log10_qi", "QI change after rebuild\n(mean |Δ log10 qi|)"),
        ("mae_worst_violation", "Worst-violation change after rebuild\n(mean |Δ|)"),
    ]
    for ax, (key, title) in zip(axes[:2], panels):
        for name, c in colors.items():
            ax.plot(KS, [recon[f"{name}{k}"][key] for k in KS], "-o", color=c, lw=2, ms=6,
                    label=name.upper())
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_xlabel("latent size")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
    ax = axes[2]
    x = np.arange(len(KS))
    for j, (name, c) in enumerate(colors.items()):
        vals = [int(samp[f"{name}{k}"]["vmec_solved"].split("/")[0]) / N_SAMPLES for k in KS]
        ax.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=c, label=name.upper())
    ax.set_xticks(x, [str(k) for k in KS])
    ax.set_xlabel("latent size")
    ax.set_ylim(0, 1.05)
    ax.set_title("New samples VMEC++ can solve\n(share of 50)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Test 3: compressing shapes with PCA vs. a VAE (very-low-fidelity VMEC++)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "compression.png")
    print("wrote", OUT / "compression.png")


if __name__ == "__main__":
    main()
