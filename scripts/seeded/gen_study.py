"""Generator trade study: score every generative model and setting offline, without simulating.

For each configuration (generators.py): train on the public passing designs, draw `n`
designs, and score them the way the loop would see them:

  scoreable   share with aspect ratio < 19.9 (exact geometry; the score's reference is 20)
  novel       share >= 0.003 from every public design (not a near-copy)
  far         share > 0.03 from every public design (likely off the passing surface)
  valid       mean screening-model P(no crash) * P(pass), over the novel, scoreable ones
  yield       share that is novel, scoreable and predicted to pass: expected novel passes
              per generated design
  supply      novel designs left after 0.003 spacing, from the first 1000 generated
  top60       the loop's pick score (P(no crash) P(pass) (predicted gain + 0.1)) summed
              over the best 60 spaced picks: what a screened round would expect to add
  a_err       conditional models: median |A asked for - A got|
  val_rms     held-out families: reconstruction error, RMS surface distance (VAE: through
              the latent space; GMM: through the PCA projection)

Diffusion models report their denoising loss on held-out and training families instead
(a widening gap means the model is memorising). Their strategies: "prior" (from noise; for a
conditional model with guidance strength `guide`) and "edit" (SDEdit from a front design;
`temp` holds the strength, the share of the noise schedule).

The screening models are trained once on a snapshot of the training table and cached, so
every configuration is scored by the same judge. Surrogate scores can be fooled by designs
unlike anything simulated, so the best few configurations go on to real simulations.
Blending (surface.blend) is scored the same way as the baseline.

Results: data/gen/study.jsonl (one row per configuration and sampling strategy; re-running
skips rows already there).

    python scripts/seeded/gen_study.py                 # everything
    python scripts/seeded/gen_study.py --only vae,cvae --n 3000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generators as G  # noqa: E402
import screen  # noqa: E402
from families import NOVELTY, nearest_public  # noqa: E402
from loop import hv_gain  # noqa: E402
from shapes import aspect_ratio, to_vector  # noqa: E402
from surface import layer_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "gen"
RESULTS = OUT / "study.jsonl"
A_MAX = 19.9


# ----------------------------------------------------------------------------- the judge
def judge() -> dict:
    """Screening models trained on a frozen snapshot of the training table, plus the front."""
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "screen.pkl"
    if cache.exists():
        with cache.open("rb") as f:
            scr = pickle.load(f)
    else:
        snap = OUT / "snapshot"; snap.mkdir(exist_ok=True)
        for _ in range(5):                              # the running loop rewrites train.npz
            shutil.copy(screen.DATA / "train.npz", snap / "train.npz")
            try:
                screen.DATA = snap; t = screen.load_table(); break
            except Exception:
                time.sleep(5)
        t0 = time.time()
        scr = screen.Screen().fit(t)
        print(f"screening models trained on {len(t['X'])} rows in {time.time() - t0:.0f} s", flush=True)
        with cache.open("wb") as f:
            pickle.dump(scr, f)
    d = G.training_set()
    return {"screen": scr, "d": d, "fA": d["A"], "fL": d["L"],
            "parents": np.flatnonzero(layer_mask(d["A"], d["L"], 3))}


front_height = G.front_height


def targets(n: int, rng, J: dict) -> np.ndarray:
    """Points just above the public front (generators.targets)."""
    return G.targets(n, rng, J["fA"], J["fL"])


def spaced(V: np.ndarray, order, limit: int) -> list[int]:
    chosen, W = [], np.empty((0, V.shape[1]))
    for i in order:
        if len(chosen) >= limit:
            break
        if not len(W) or np.min(np.linalg.norm(W - V[i], axis=1)) >= NOVELTY:
            chosen.append(int(i)); W = np.vstack([W, V[i]])
    return chosen


def score(V: np.ndarray, J: dict, asked=None) -> dict:
    V = np.asarray(V, float)
    A, dist = aspect_ratio(V), nearest_public(V)
    ok, nov = A < A_MAX, dist >= NOVELTY
    p = J["screen"].predict(V)
    valid = (1 - p["p_crash"]) * p["p_pass"]
    s = valid * (hv_gain(A, p["L_gradB"], J["fA"], J["fL"]) + 0.1)
    cand = np.flatnonzero(ok & nov)
    top = spaced(V, cand[np.argsort(-s[cand])], 60)
    out = {"n": len(V), "scoreable": float(ok.mean()), "novel": float(nov.mean()),
           "far": float((dist > 0.03).mean()), "dist_median": float(np.median(dist)),
           "valid": float(valid[cand].mean()) if len(cand) else 0.0,
           "yield": float((valid * (ok & nov)).mean()),
           "supply": len(spaced(V, cand[cand < 1000], 10 ** 9)),
           "top60": float(s[top].sum()), "top60_pred_pass": float(valid[top].mean()) if top else 0.0,
           "A_range": [float(np.quantile(A, 0.05)), float(np.quantile(A, 0.95))]}
    if asked is not None:
        out["a_err"] = float(np.median(np.abs(A - asked[:, 0])))
    return out


# ----------------------------------------------------------------------------- the grid
def grid(only: set[str]) -> list:
    g = []
    if "gmm" in only:
        g += [G.GMM(k=k, n=n) for k in (8, 16, 32) for n in (16, 64, 256)]
        g += [G.GMM(k=16, n=64, weighting=w) for w in ("sqrt", "uniform")]
    if "vae" in only:
        g += [G.VAE(latent=z, beta=b) for z in (4, 8, 16, 32) for b in (0.01, 0.1, 1.0)]
        g += [G.VAE(latent=8, beta=0.1, weighting=w) for w in ("sqrt", "uniform")]
    if "cvae" in only:
        g += [G.VAE(latent=z, beta=b, cond=True) for z in (4, 8, 16) for b in (0.01, 0.1, 1.0)]
        g += [G.VAE(latent=8, beta=0.1, cond=True, weighting=w) for w in ("sqrt", "uniform")]
    if "diff" in only:
        g += [G.Diffusion(width=w, steps=s) for w in (256, 512) for s in (3000, 10000, 30000)]
        g += [G.Diffusion(width=256, steps=10000, scaling="global")]
        # second round, around the best of the first (width 256, 3000 steps): train less, go smaller
        g += [G.Diffusion(width=256, steps=s) for s in (1000, 1500)] + [G.Diffusion(width=128, steps=3000)]
    if "cdiff" in only:
        g += [G.Diffusion(width=256, steps=s, cond=True) for s in (3000, 10000)]
        g += [G.Diffusion(width=512, steps=10000, cond=True)]
        g += [G.Diffusion(width=256, steps=1500, cond=True)]             # second round
    return g


def strategies(model) -> list[tuple]:
    """(strategy, temp or edit strength, guidance strength or None)."""
    if model.kind == "gmm":
        return [("prior", 0.7, None), ("prior", 1.0, None)]
    if model.kind == "diff":
        return [("prior", 1.0, None), ("prior", 0.8, None)] + [("edit", s, None) for s in (0.1, 0.25, 0.5)]
    if model.kind == "cdiff":
        return [("prior", 1.0, g) for g in (0.0, 1.0, 3.0)] + [("edit", 0.25, g) for g in (0.0, 1.0)]
    return [("prior", 0.7, None), ("prior", 1.0, None), ("interp", 0.0, None)]


def key(row: dict) -> str:
    return json.dumps({k: row[k] for k in sorted(row) if k in ("config", "strategy", "temp", "guide")}, sort_keys=True)


def run(only: set[str], n: int) -> None:
    done = {key(json.loads(x)) for x in RESULTS.open()} if RESULTS.exists() else set()
    J = judge()
    sets = {}

    def emit(row):
        with RESULTS.open("a") as f:
            f.write(json.dumps(row) + "\n")
        m = row["metrics"]
        print(f"{row['label']:46s} novel {m['novel']:.2f} far {m['far']:.2f} valid {m['valid']:.2f} "
              f"yield {m['yield']:.3f} supply {m['supply']:4d} top60 {m['top60']:6.2f}"
              + (f" a_err {m['a_err']:.2f}" if "a_err" in m else "")
              + (f"  val_rms {row['fit']['val_rms']:.4f}" if "val_rms" in row.get("fit", {}) else "")
              + (f"  loss val/train {row['fit']['val_loss']:.1f}/{row['fit']['train_loss']:.1f}"
                 if "val_loss" in row.get("fit", {}) else ""),
              flush=True)

    if "blend" in only:
        import surface
        for layers, window in ((1, 1.0), (3, 2.0)):
            row = {"config": {"model": "blend", "layers": layers, "window": window},
                   "strategy": "blend", "temp": 0.0}
            if key(row) in done:
                continue
            V = np.array([to_vector(x["boundary"]) for x in surface.blend(n, 7, None, layers=layers, window=window)])
            emit(dict(row, label=f"blend layers {layers} window {window}", fit={}, metrics=score(V, J)))

    for model in grid(only):
        rows = [dict({"config": model.config(), "strategy": s, "temp": t}, **({"guide": g} if g is not None else {}))
                for s, t, g in strategies(model)]
        rows = [r for r in rows if key(r) not in done]
        if not rows:
            continue
        d = sets.setdefault(model.weighting, G.training_set(model.weighting))
        model.fit(d)
        for r in rows:
            rng = np.random.default_rng(int(hashlib.sha1(key(r).encode()).hexdigest()[:8], 16))
            asked = targets(n, rng, J) if model.kind in ("cvae", "cdiff") and r["strategy"] == "prior" else None
            if r["strategy"] == "interp":
                V = model.interpolate(n, rng, d, J["parents"])
            elif r["strategy"] == "edit":
                V, asked = model.edit(n, rng, d, J["parents"], strength=r["temp"], guide=r.get("guide", 0.0))
            else:
                V = model.sample(n, rng, temp=r["temp"], targets=asked, guide=r.get("guide", 0.0))
            label = " ".join(f"{k} {v}" for k, v in r["config"].items()
                             if k not in ("seed", "lr", "hidden", "depth", "T") and v != "family"
                             and not (k == "steps" and r["config"]["model"] in ("vae", "cvae")))
            emit(dict(r, label=f"{label} {r['strategy']} {r['temp'] or ''}".strip()
                      + (f" guide {r['guide']}" if "guide" in r else ""),
                      fit=model.fit_info, metrics=score(V, J, asked)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="blend,gmm,vae,cvae,diff,cdiff")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    run(set(a.only.split(",")), a.n)


if __name__ == "__main__":
    main()
