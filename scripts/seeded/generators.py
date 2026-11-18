"""Generative models of passing designs: the proposers for the generator arm.

Trained on the public passing NFP-3 4x4 designs (standard form, families.py). Whole
families (~15% of designs) are held out to check that a model learned the shapes in
general rather than memorising its training designs. `weighting` sets how much each design
counts: "family" (every family equal, as the screening models do), "sqrt" (1/sqrt(family
size)) or "uniform" (every design equal). The model types in the
trade study (gen_study.py):

  gmm   PCA to k dimensions, then a Gaussian mixture: the classical density-model baseline
  vae   variational autoencoder (beta-VAE). Sample the latent prior, or interpolate in the
        latent space between two front designs: the learned version of blending
  cvae  conditional VAE on (aspect ratio, L_gradB): ask for designs at a target point,
        e.g. just above the current front
  diff  denoising diffusion model (DDPM). Sample from noise, or "edit" a front design:
        noise it part-way and denoise it (SDEdit), a learned perturbation whose strength
        sets how far the result moves from its parent
  cdiff conditional diffusion on (aspect ratio, L_gradB) with classifier-free guidance

Every model returns standard-form vectors (shapes.py), so the rest of the pipeline
(novelty, aspect ratio, screening, simulation) is unchanged. Vectors are centred per
coefficient and divided by one global scale, so squared error stays proportional to RMS
surface distance (the same choice as Leif's compression test). JAX + optax, like
scripts/feasibility_tests/compression_test.py, so the code can move between tracks.

`temp` scales the latent (or mixture) noise at sampling time: below 1 stays closer to the
training designs (more likely to pass, less novel), above 1 explores further.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from families import load as load_public  # noqa: E402

VAL_SHARE = 0.15


def training_set(weighting: str = "family", seed: int = 0) -> dict:
    """Public passing designs with weights and a held-out split by family: families in random
    order, skipping any that would push the held-out share past 20%, until it reaches 15%."""
    d = load_public()
    fam = d["family"]; size = np.bincount(fam)
    w = {"family": 1.0 / size[fam], "sqrt": 1.0 / np.sqrt(size[fam]),
         "uniform": np.ones(len(fam))}[weighting]
    held, n = [], 0
    for f in np.random.default_rng(seed).permutation(np.unique(fam)):
        if n + size[f] <= 0.20 * len(fam):
            held.append(f); n += size[f]
        if n >= VAL_SHARE * len(fam):
            break
    return dict(d, weight=w / w.sum(), val=np.isin(fam, held), weighting=weighting)


class Scaler:
    """Centre each coefficient; divide everything by one number (keeps relative sizes)."""

    def fit(self, X: np.ndarray, w: np.ndarray) -> "Scaler":
        w = w / w.sum()
        self.mu = (w[:, None] * X).sum(0)
        self.s = float(np.sqrt((w[:, None] * (X - self.mu) ** 2).sum() / X.shape[1]))
        return self

    def to(self, X):
        return (X - self.mu) / self.s

    def back(self, S):
        return np.asarray(S) * self.s + self.mu


def _weighted_rows(d: dict, n: int, rng) -> np.ndarray:
    tr = np.flatnonzero(~d["val"])
    p = d["weight"][tr] / d["weight"][tr].sum()
    return rng.choice(tr, size=n, p=p)


# ----------------------------------------------------------------------------- GMM
class GMM:
    kind = "gmm"

    def __init__(self, k: int = 16, n: int = 64, weighting: str = "family", seed: int = 0):
        self.k, self.n, self.weighting, self.seed = k, n, weighting, seed

    def config(self) -> dict:
        return {"model": "gmm", "k": self.k, "n": self.n, "weighting": self.weighting, "seed": self.seed}

    def fit(self, d: dict) -> "GMM":
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        tr = ~d["val"]
        self.sc = Scaler().fit(d["X"][tr], d["weight"][tr])
        S = self.sc.to(d["X"][_weighted_rows(d, 20000, rng)])   # resample: GMM takes no weights
        self.pca = PCA(self.k, random_state=self.seed).fit(S)
        self.gm = GaussianMixture(self.n, covariance_type="full", reg_covar=1e-6,
                                  random_state=self.seed, max_iter=300).fit(self.pca.transform(S))
        self.chol = np.linalg.cholesky(self.gm.covariances_)
        Sv = self.sc.to(d["X"][d["val"]])
        rec = self.pca.inverse_transform(self.pca.transform(Sv))
        self.fit_info = {"val_rms": float(np.sqrt(((rec - Sv) ** 2).sum(1)).mean() * self.sc.s),
                         "val_loglik": float(self.gm.score(self.pca.transform(Sv))),
                         "train_loglik": float(self.gm.score(self.pca.transform(S))),
                         "fit_s": round(time.time() - t0, 1)}
        return self

    def sample(self, n: int, rng, temp: float = 1.0, **_) -> np.ndarray:
        c = rng.choice(self.n, size=n, p=self.gm.weights_)
        eps = rng.standard_normal((n, self.k))
        Z = self.gm.means_[c] + temp * np.einsum("nij,nj->ni", self.chol[c], eps)
        return self.sc.back(self.pca.inverse_transform(Z))


# ----------------------------------------------------------------------------- VAE / CVAE
def _init(key, sizes):
    import jax
    import jax.numpy as jnp
    out = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        out.append((jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / a), jnp.zeros(b)))
    return out


def _mlp(params, x):
    import jax
    for W, b in params[:-1]:
        x = jax.nn.gelu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


class VAE:
    """beta-VAE; cond=True makes it a conditional VAE on (A, L_gradB)."""

    def __init__(self, latent: int = 8, beta: float = 0.1, hidden: int = 256, depth: int = 2,
                 steps: int = 12000, lr: float = 1e-3, batch: int = 256, cond: bool = False,
                 weighting: str = "family", seed: int = 0):
        self.latent, self.beta, self.hidden, self.depth = latent, beta, hidden, depth
        self.steps, self.lr, self.batch, self.cond, self.seed = steps, lr, batch, cond, seed
        self.weighting = weighting
        self.kind = "cvae" if cond else "vae"

    def config(self) -> dict:
        return {"model": self.kind, "latent": self.latent, "beta": self.beta, "hidden": self.hidden,
                "depth": self.depth, "steps": self.steps, "lr": self.lr, "weighting": self.weighting,
                "seed": self.seed}

    def _c(self, C) -> np.ndarray:
        if not self.cond:
            return np.zeros((len(C), 0), np.float32)
        return ((np.asarray(C, float) - self.cmu) / self.csd).astype(np.float32)

    def fit(self, d: dict) -> "VAE":
        import jax
        import jax.numpy as jnp
        import optax
        t0 = time.time()
        tr = ~d["val"]
        self.sc = Scaler().fit(d["X"][tr], d["weight"][tr])
        S = self.sc.to(d["X"]).astype(np.float32)
        C = np.c_[d["A"], d["L"]]
        self.cmu, self.csd = C[tr].mean(0), C[tr].std(0)
        Cs = self._c(C)
        k, nc, dim = self.latent, Cs.shape[1], S.shape[1]
        ke, kd = jax.random.split(jax.random.PRNGKey(self.seed))
        params = {"enc": _init(ke, [dim + nc] + [self.hidden] * self.depth + [2 * k]),
                  "dec": _init(kd, [k + nc] + [self.hidden] * self.depth + [dim])}
        beta = self.beta

        def loss(params, x, c, key):
            h = _mlp(params["enc"], jnp.concatenate([x, c], 1))
            mu, lv = h[:, :k], jnp.clip(h[:, k:], -12, 6)
            z = mu + jnp.exp(0.5 * lv) * jax.random.normal(key, mu.shape)
            xr = _mlp(params["dec"], jnp.concatenate([z, c], 1))
            rec = jnp.sum((x - xr) ** 2, 1).mean()
            kl = 0.5 * jnp.sum(jnp.exp(lv) + mu ** 2 - 1.0 - lv, 1).mean()
            return rec + beta * kl, (rec, kl)

        opt = optax.adam(optax.cosine_decay_schedule(self.lr, self.steps))
        state = opt.init(params)

        @jax.jit
        def step(params, state, x, c, key):
            (_, aux), g = jax.value_and_grad(loss, has_aux=True)(params, x, c, key)
            upd, state = opt.update(g, state, params)
            return optax.apply_updates(params, upd), state, aux

        rng = np.random.default_rng(self.seed)
        rows = _weighted_rows(d, self.steps * self.batch, rng).reshape(self.steps, self.batch)
        key = jax.random.PRNGKey(self.seed + 1)
        for i in range(self.steps):
            key, sub = jax.random.split(key)
            params, state, (rec, kl) = step(params, state, S[rows[i]], Cs[rows[i]], sub)
        self.params = params
        self._enc = jax.jit(lambda p, x, c: _mlp(p["enc"], jnp.concatenate([x, c], 1)))
        self._dec = jax.jit(lambda p, z, c: _mlp(p["dec"], jnp.concatenate([z, c], 1)))

        def held(mask):                                  # posterior-mean reconstruction, KL
            h = np.asarray(self._enc(self.params, S[mask], Cs[mask]))
            mu, lv = h[:, :k], np.clip(h[:, k:], -12, 6)
            xr = np.asarray(self._dec(self.params, mu.astype(np.float32), Cs[mask]))
            rms = float(np.sqrt(((S[mask] - xr) ** 2).sum(1)).mean() * self.sc.s)
            return rms, float((0.5 * (np.exp(lv) + mu ** 2 - 1 - lv).sum(1)).mean())
        v_rms, v_kl = held(d["val"])
        t_rms, _ = held(tr)
        self.fit_info = {"val_rms": v_rms, "train_rms": t_rms, "val_kl": v_kl,
                         "final_rec": float(rec), "final_kl": float(kl), "fit_s": round(time.time() - t0, 1)}
        return self

    def encode(self, X, C=None) -> np.ndarray:
        S = self.sc.to(np.atleast_2d(X)).astype(np.float32)
        h = np.asarray(self._enc(self.params, S, self._c(C if C is not None else np.zeros((len(S), 2)))))
        return h[:, : self.latent]

    def decode(self, Z, C=None) -> np.ndarray:
        Z = np.asarray(Z, np.float32)
        out = self._dec(self.params, Z, self._c(C if C is not None else np.zeros((len(Z), 2))))
        return self.sc.back(out)

    def sample(self, n: int, rng, temp: float = 1.0, targets=None, **_) -> np.ndarray:
        """Prior samples; a conditional model needs `targets`, an (n, 2) array of (A, L_gradB)."""
        if self.cond and targets is None:
            raise ValueError("conditional VAE needs targets (A, L_gradB)")
        return self.decode(temp * rng.standard_normal((n, self.latent)), targets)

    def interpolate(self, n: int, rng, d: dict, parents: np.ndarray, window: float = 2.0) -> np.ndarray:
        """Learned blending: encode two parents from different families with |dA| < window,
        mix their latents (and, for a conditional model, their targets) at t in [0.3, 0.7]."""
        X, A, L, fam = d["X"], d["A"], d["L"], d["family"]
        a = parents[rng.integers(len(parents), size=n)]
        b = np.empty(n, int)
        for i, ai in enumerate(a):
            o = parents[(fam[parents] != fam[ai]) & (np.abs(A[parents] - A[ai]) < window)]
            b[i] = o[rng.integers(len(o))] if len(o) else parents[rng.integers(len(parents))]
        t = rng.uniform(0.3, 0.7, size=(n, 1))
        C = np.c_[A, L]
        za, zb = self.encode(X[a], C[a]), self.encode(X[b], C[b])
        return self.decode((1 - t) * za + t * zb, (1 - t) * C[a] + t * C[b])


# ----------------------------------------------------------------------------- diffusion
class CoefScaler(Scaler):
    """Standardise each coefficient on its own, so a model's errors stay in proportion to each
    coefficient's natural spread (the usual choice for diffusion on tabular data)."""

    def fit(self, X: np.ndarray, w: np.ndarray) -> "CoefScaler":
        w = w / w.sum()
        self.mu = (w[:, None] * X).sum(0)
        sd = np.sqrt((w[:, None] * (X - self.mu) ** 2).sum(0))
        self.s = np.maximum(sd, 1e-3 * sd.max())          # coefficients the standard form fixes
        return self


def _cosine_schedule(T: int):
    """Nichol & Dhariwal's cosine schedule: beta_t and alpha-bar_t for t = 1..T (index t-1)."""
    t = np.arange(T + 1) / T
    f = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
    beta = np.clip(1 - f[1:] / f[:-1], 1e-5, 0.999)
    return beta, np.cumprod(1 - beta)


def _temb(t, T):
    import jax.numpy as jnp
    f = jnp.exp(jnp.linspace(0.0, jnp.log(200.0), 16))
    x = (t.astype(jnp.float32) / T)[:, None] * f[None]
    return jnp.concatenate([jnp.sin(x), jnp.cos(x)], 1)


class Diffusion:
    """DDPM with an MLP denoiser that predicts the added noise. cond=True conditions on
    (A, L_gradB) with classifier-free guidance: the condition is dropped `p_uncond` of the
    time in training, so one network learns both, and `guide` > 0 at sampling time pushes
    harder towards the request. `scaling` is "coef" (CoefScaler) or "global" (Scaler)."""

    def __init__(self, width: int = 512, depth: int = 3, T: int = 200, steps: int = 20000,
                 lr: float = 1e-3, batch: int = 256, cond: bool = False, scaling: str = "coef",
                 p_uncond: float = 0.1, weighting: str = "family", seed: int = 0):
        self.width, self.depth, self.T, self.steps, self.lr = width, depth, T, steps, lr
        self.batch, self.cond, self.scaling, self.p_uncond = batch, cond, scaling, p_uncond
        self.weighting, self.seed = weighting, seed
        self.kind = "cdiff" if cond else "diff"

    def config(self) -> dict:
        return {"model": self.kind, "width": self.width, "depth": self.depth, "T": self.T,
                "steps": self.steps, "lr": self.lr, "scaling": self.scaling,
                "weighting": self.weighting, "seed": self.seed}

    def fit(self, d: dict) -> "Diffusion":
        import jax
        import jax.numpy as jnp
        import optax
        t0 = time.time()
        tr = ~d["val"]
        self.sc = (CoefScaler() if self.scaling == "coef" else Scaler()).fit(d["X"][tr], d["weight"][tr])
        S = self.sc.to(d["X"]).astype(np.float32)
        C = np.c_[d["A"], d["L"]]
        self.cmu, self.csd = C[tr].mean(0), C[tr].std(0)
        Cs = ((C - self.cmu) / self.csd).astype(np.float32)
        T, dim, nc, p_unc = self.T, S.shape[1], 3 if self.cond else 0, self.p_uncond
        self.dim = dim
        beta, ab = _cosine_schedule(T)
        self._beta, self._ab = beta.astype(np.float32), ab.astype(np.float32)
        bj, aj = jnp.asarray(self._beta), jnp.asarray(self._ab)
        params = _init(jax.random.PRNGKey(self.seed), [dim + 32 + nc] + [self.width] * self.depth + [dim])

        def net(params, x, t, c):                       # c: (n, 3) = (A, L, given) or (n, 0)
            return _mlp(params, jnp.concatenate([x, _temb(t, T), c], 1))

        def cvec(c, keep):
            return jnp.concatenate([c * keep, keep], 1) if nc else c[:, :0]

        def loss(params, x0, c, key, drop=True):
            k1, k2, k3 = jax.random.split(key, 3)
            t = jax.random.randint(k1, (x0.shape[0],), 1, T + 1)
            eps = jax.random.normal(k2, x0.shape)
            a = aj[t - 1][:, None]
            keep = (jax.random.uniform(k3, (x0.shape[0], 1)) >= (p_unc if drop else 0.0)).astype(jnp.float32)
            pred = net(params, jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps, t, cvec(c, keep))
            return jnp.sum((eps - pred) ** 2, 1).mean()

        opt = optax.adam(optax.cosine_decay_schedule(self.lr, self.steps))
        state = opt.init(params)

        @jax.jit
        def step(params, state, x, c, key):
            l, g = jax.value_and_grad(loss)(params, x, c, key)
            upd, state = opt.update(g, state, params)
            return optax.apply_updates(params, upd), state, l

        rng = np.random.default_rng(self.seed)
        rows = _weighted_rows(d, self.steps * self.batch, rng).reshape(self.steps, self.batch)
        key = jax.random.PRNGKey(self.seed + 1)
        for i in range(self.steps):
            key, sub = jax.random.split(key)
            params, state, l = step(params, state, S[rows[i]], Cs[rows[i]], sub)
        self.params = params

        lo, hi = S[tr].min(0), S[tr].max(0)                # clean-design estimates stay within the
        lo, hi = jnp.asarray(lo - 0.25 * (hi - lo)), jnp.asarray(hi + 0.25 * (hi - lo))  # data's range

        def run(params, x, k, c, guide, temp, key):
            """Ancestral sampling from step k (per row) down to 1; rows wait until t <= k. Each
            step estimates the clean design, clips it to the data range, and steps towards it
            (the early, high-noise steps otherwise amplify the denoiser's errors ~30x)."""
            n = x.shape[0]
            abp = jnp.concatenate([jnp.ones(1), aj[:-1]])
            sig = jnp.sqrt(bj * (1 - abp) / (1 - aj))
            c0 = jnp.sqrt(abp) * bj / (1 - aj)                # posterior mean weights on x0, x_t
            ct = jnp.sqrt(1 - bj) * (1 - abp) / (1 - aj)

            def body(carry, t):
                x, key = carry
                key, sub = jax.random.split(key)
                tt = jnp.full((n,), t)
                if nc:
                    e_c = net(params, x, tt, cvec(c, jnp.ones((n, 1))))
                    e_u = net(params, x, tt, cvec(c, jnp.zeros((n, 1))))
                    e = (1 + guide) * e_c - guide * e_u
                else:
                    e = net(params, x, tt, c[:, :0])
                i = t - 1
                x0 = jnp.clip((x - jnp.sqrt(1 - aj[i]) * e) / jnp.sqrt(aj[i]), lo, hi)
                mean = c0[i] * x0 + ct[i] * x
                xn = mean + jnp.where(t > 1, temp * sig[i], 0.0) * jax.random.normal(sub, x.shape)
                return (jnp.where((t <= k)[:, None], xn, x), key), None

            (x, _), _ = jax.lax.scan(body, (x, key), jnp.arange(T, 0, -1))
            return x
        self._run = jax.jit(run)

        def held(mask, reps=4):
            ls = [float(loss(params, S[mask], Cs[mask], jax.random.PRNGKey(100 + r), drop=False)) for r in range(reps)]
            return float(np.mean(ls))
        self.fit_info = {"val_loss": held(d["val"]), "train_loss": held(tr), "final_loss": float(l),
                         "fit_s": round(time.time() - t0, 1)}
        return self

    def _sample(self, x, k, targets, guide, temp, rng) -> np.ndarray:
        import jax
        n = len(x)
        c = (((np.asarray(targets, float) - self.cmu) / self.csd).astype(np.float32)
             if self.cond else np.zeros((n, 2), np.float32))
        key = jax.random.PRNGKey(int(rng.integers(2 ** 31)))
        out = self._run(self.params, np.asarray(x, np.float32), np.asarray(k, np.int32), c,
                        np.float32(guide), np.float32(temp), key)
        return self.sc.back(np.asarray(out, float))

    def sample(self, n: int, rng, temp: float = 1.0, targets=None, guide: float = 0.0, **_) -> np.ndarray:
        """From pure noise; `temp` scales all the noise. A conditional model needs `targets`."""
        if self.cond and targets is None:
            raise ValueError("conditional diffusion needs targets (A, L_gradB)")
        x = temp * rng.standard_normal((n, self.dim))
        return self._sample(x, np.full(n, self.T), targets, guide, temp, rng)

    def edit(self, n: int, rng, d: dict, parents: np.ndarray, strength: float = 0.25,
             guide: float = 0.0, lift=(0.0, 1.0)):
        """SDEdit: noise a front design part-way (strength = share of the schedule) and denoise
        it. Low strength stays near the parent; 1.0 is a fresh sample. A conditional model is
        asked for the parent's aspect ratio and an L_gradB `lift` higher (more buildable).
        Returns (designs, the (A, L_gradB) asked for, or None)."""
        a = parents[rng.integers(len(parents), size=n)]
        k = max(1, int(round(strength * self.T)))
        x0 = self.sc.to(d["X"][a])
        x = np.sqrt(self._ab[k - 1]) * x0 + np.sqrt(1 - self._ab[k - 1]) * rng.standard_normal(x0.shape)
        asked = np.c_[d["A"][a], d["L"][a] + rng.uniform(*lift, n)] if self.cond else None
        return self._sample(x, np.full(n, k), asked, guide, 1.0, rng), asked


# ----------------------------------------------------------------------------- finalists, the loop's proposer
# name: (model factory taking a seed, how to sample). Settings picked by the trade study
# (gen_study.py) and tested with real simulations (gen_pilot.py). strategy "prior" samples
# (temp = noise scale, guide = guidance strength), "edit" is SDEdit (temp = strength).
FINALISTS = {
    "vae_z32_b0.1": (lambda seed=0: VAE(latent=32, beta=0.1, seed=seed), {"temp": 0.7}),
    "cvae_z8_b0.1": (lambda seed=0: VAE(latent=8, beta=0.1, cond=True, seed=seed), {"temp": 1.0}),
    "gmm_k16_n16": (lambda seed=0: GMM(k=16, n=16, seed=seed), {"temp": 1.0}),
    "diff_w256_s3k": (lambda seed=0: Diffusion(width=256, steps=3000, seed=seed), {"temp": 1.0}),
    "cdiff_w256_s3k": (lambda seed=0: Diffusion(width=256, steps=3000, cond=True, seed=seed),
                       {"temp": 1.0, "guide": 0.0}),
    # second pilot: the first one failed mostly on QI, which the surrogate can't see, so try
    # settings that stay closer to real designs: less sampling noise, or small edits
    "diff_w256_s3k_t0.5": (lambda seed=0: Diffusion(width=256, steps=3000, seed=seed), {"temp": 0.5}),
    "diff_w256_s3k_edit0.1": (lambda seed=0: Diffusion(width=256, steps=3000, seed=seed),
                              {"strategy": "edit", "temp": 0.1}),
    "vae_z32_b0.1_t0.3": (lambda seed=0: VAE(latent=32, beta=0.1, seed=seed), {"temp": 0.3}),
}


def front_height(a: np.ndarray, fA: np.ndarray, fL: np.ndarray) -> np.ndarray:
    """L_gradB of the front (fA, fL) at aspect ratio a: the best L among designs with A <= a."""
    o = np.argsort(fA); xa, xl = fA[o], np.maximum.accumulate(fL[o])
    k = np.searchsorted(xa, a, side="right") - 1
    return np.where(k >= 0, xl[np.clip(k, 0, None)], -1.0)


def targets(n: int, rng, fA: np.ndarray, fL: np.ndarray) -> np.ndarray:
    """Points just above the front: A uniform on [7.2, 19.8], L_gradB 0-1 above it."""
    a = rng.uniform(7.2, 19.8, n)
    return np.c_[a, front_height(a, fA, fL) + rng.uniform(0, 1, n)]


def generate(model, spec: dict, n: int, rng, front: dict):
    """n designs from a fitted model, aimed at `front` (arrays X, A, L): conditional models
    ask for points just above it, edits start from its first 3 non-dominated layers.
    Returns (vectors, the (A, L_gradB) asked for or None)."""
    from surface import layer_mask
    strategy, temp, guide = spec.get("strategy", "prior"), spec["temp"], spec.get("guide", 0.0)
    if strategy == "edit":
        parents = np.flatnonzero(layer_mask(front["A"], front["L"], 3))
        return model.edit(n, rng, front, parents, strength=temp, guide=guide)
    asked = targets(n, rng, front["A"], front["L"]) if model.kind in ("cvae", "cdiff") else None
    return model.sample(n, rng, temp=temp, targets=asked, guide=guide), asked


_FITTED: dict = {}


def propose(name: str, n: int, seed: int, ours: dict | None = None, model_seed: int = 0) -> list[dict]:
    """The loop's generator proposer (loop.py --proposer gen:<name>). The model is trained
    once per process on the public passing designs, exactly as in the pilot, then aimed at the
    arm's current front: the public designs plus `ours` (its novel passing designs: X, A, L)."""
    import hashlib
    import json
    from shapes import from_vector
    if (name, model_seed) not in _FITTED:
        make, _ = FINALISTS[name]
        d = training_set()
        _FITTED[(name, model_seed)] = (make(model_seed).fit(d), d)
    model, d = _FITTED[(name, model_seed)]
    front = {k: d[k] for k in ("X", "A", "L")}
    if ours is not None and len(ours["X"]):
        front = {k: np.r_[front[k], ours[k]] for k in front}
    V, asked = generate(model, FINALISTS[name][1], n, np.random.default_rng(seed), front)
    out = []
    for i, v in enumerate(V):
        b = json.dumps(from_vector(v))
        rec = {"id": "gen-" + hashlib.sha1(b.encode()).hexdigest()[:12], "boundary": b, "method": name}
        if asked is not None:
            rec["target_A"], rec["target_L"] = float(asked[i, 0]), float(asked[i, 1])
        out.append(rec)
    return out
