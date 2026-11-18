"""Full-resolution pilot: do blends work at the leaderboard's real front?

Everything else in scripts/seeded/ works in the dataset's design space: NFP 3, 4x4 Fourier
modes (80 coefficients). That front (HV 205.5) is not where the competition is: every design
on the full public front (HV 285.3) is higher resolution (NFP 3 and 4; 8x8 and 9x9 modes).
This pilot blends pairs of full-front designs at their own resolution and simulates a
random sample, plus a few unmodified front designs as a scorer check at that resolution
(the original scorer check used 4x4 designs only).

Designs of any resolution are zero-padded to the largest public one (m = 0..10,
n = -10..10), where distance is still the RMS distance between surfaces. Blends mix two
designs with the same number of field periods. Public designs use two sign conventions
for the same surface (4x4: z_sin[1, 0] < 0; every higher-resolution design: > 0), so
everything is put in the higher-resolution one before blending or measuring distance;
the parents themselves are never changed.

Files go in data/fullres/, so the 4x4 pipeline (training_data, arm_view) never reads them.

    python scripts/seeded/fullres.py propose --n 60 --controls 8 --seed 40
    python scripts/seeded/fullres.py simulate [--only 2]
    python scripts/seeded/fullres.py report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from families import NOVELTY, STRICT  # noqa: E402
from loop import hv_gain, hypervolume  # noqa: E402
from perturb import front_mask  # noqa: E402
from surface import layer_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "seeded" / "public_p3.jsonl"
OUT = ROOT / "data" / "fullres"
DESIGNS, LEDGER = OUT / "pilot_designs.jsonl", OUT / "runs" / "pilot.jsonl"
P, T = 11, 21                      # padded shape: m = 0..10, n = -10..10
MID = (T - 1) // 2
_FREE = np.ones((P, T), bool); _FREE[0, : MID + 1] = False


def standard(boundary) -> tuple[np.ndarray, np.ndarray, int, tuple[int, int]]:
    """(r_cos, z_sin) zero-padded to (P, T), scaled to major radius 1, in the
    higher-resolution sign convention; plus nfp and the original array shape."""
    b = json.loads(boundary) if isinstance(boundary, str) else boundary
    rc, zs = np.array(b["r_cos"], float), np.array(b["z_sin"], float)
    p, t = rc.shape; h = (t - 1) // 2
    R, Z = np.zeros((P, T)), np.zeros((P, T))
    R[:p, MID - h:MID + h + 1], Z[:p, MID - h:MID + h + 1] = rc, zs
    s = R[0, MID]; R, Z = R / s, Z / s
    m = np.arange(P)[:, None]; n = np.arange(T)[None, :] - MID
    if R[1, MID] < 0:                                  # theta -> theta + pi
        R, Z = R * (-1.0) ** m, Z * (-1.0) ** m
    if Z[1, MID] < 0:                                  # theta -> -theta
        R[1:], Z[1:] = R[1:, ::-1].copy(), -Z[1:, ::-1].copy()
    if R[0, MID + 1] < 0:                              # half a field period later
        R, Z = R * (-1.0) ** n, Z * (-1.0) ** n
    return R, Z, int(b["n_field_periods"]), (p, t)


def vec(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Free coefficients; Euclidean distance = RMS surface distance (Parseval)."""
    return np.sqrt(0.5) * np.concatenate([R[..., _FREE], Z[..., _FREE]], axis=-1)


def boundary(R: np.ndarray, Z: np.ndarray, nfp: int, shape: tuple[int, int]) -> dict:
    p, t = shape; h = (t - 1) // 2
    sl = (slice(0, p), slice(MID - h, MID + h + 1))
    return {"r_cos": R[sl].tolist(), "z_sin": Z[sl].tolist(), "r_sin": None, "z_cos": None,
            "n_field_periods": nfp, "is_stellarator_symmetric": True}


def aspect_ratio(R: np.ndarray, Z: np.ndarray, nfp: int, n_phi: int = 48, n_theta: int = 96) -> np.ndarray:
    """VMEC's definition from geometry alone (as shapes.aspect_ratio, any resolution)."""
    R, Z = np.atleast_3d(R) if R.ndim == 3 else R[None], np.atleast_3d(Z) if Z.ndim == 3 else Z[None]
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / nfp, n_phi, endpoint=False)
    m = np.arange(P)[:, None]; n = np.arange(T)[None, :] - MID
    ang = m[None, None] * th[None, :, None, None] - nfp * n[None, None] * ph[:, None, None, None]
    cos, sin = np.cos(ang), np.sin(ang)
    out = []
    for s in range(0, len(R), 500):
        r = np.einsum("imn,ptmn->ipt", R[s:s + 500], cos); z = np.einsum("imn,ptmn->ipt", Z[s:s + 500], sin)
        r2, z2 = np.roll(r, -1, axis=2), np.roll(z, -1, axis=2)
        area = np.abs(0.5 * np.sum(r * z2 - r2 * z, axis=2).mean(axis=1))
        vol = np.abs(np.sum((r**2 + r * r2 + r2**2) / 6 * (z2 - z), axis=2).mean(axis=1)) * 2 * np.pi
        out.append(vol / (2 * np.pi * area) / np.sqrt(area / np.pi))
    return np.concatenate(out)


def load_public() -> dict:
    rows = [json.loads(x) for x in PUBLIC.open()]
    rows = [r for r in rows if r["feasible"]]
    std = [standard(r["boundary"]) for r in rows]
    d = {"R": np.array([s[0] for s in std]), "Z": np.array([s[1] for s in std]),
         "nfp": np.array([s[2] for s in std]), "shape": [s[3] for s in std],
         "A": np.array([r["aspect_ratio"] for r in rows]), "L": np.array([r["L_gradB"] for r in rows]),
         "violation": np.array([r["max_violation"] for r in rows]),
         "id": np.array([r["id"] for r in rows]), "user": np.array([r["sources"][0]["user"] for r in rows]),
         "boundary": [r["boundary"] for r in rows]}
    d["X"] = vec(d["R"], d["Z"])
    d["family"] = np.zeros(len(rows), int); off = 0
    for nfp in np.unique(d["nfp"]):                    # single linkage at NOVELTY, per NFP
        i = np.flatnonzero(d["nfp"] == nfp)
        g = NearestNeighbors().fit(d["X"][i]).radius_neighbors_graph(d["X"][i], radius=NOVELTY)
        k, lab = connected_components(g, directed=False)
        d["family"][i] = lab + off; off += k
    return d


def propose(n: int, controls: int, seed: int, layers: int, window: float, pool: int,
            append: bool = False) -> None:
    """append: add blends to the existing file (no new controls), spaced from its blends."""
    rng = np.random.default_rng(seed)
    pub = load_public()
    X, A, L, nfp, fam = pub["X"], pub["A"], pub["L"], pub["nfp"], pub["family"]
    front = np.flatnonzero(layer_mask(A, L, layers))
    print(f"{len(A)} public passing designs; {len(front)} parents (first {layers} layer(s)): "
          + ", ".join(f"NFP {k} {tuple(s)}: {sum(1 for i in front if nfp[i] == k and pub['shape'][i] == s)}"
                      for k, s in sorted({(int(nfp[i]), pub['shape'][i]) for i in front})))
    nn = {k: NearestNeighbors(n_neighbors=2).fit(X[nfp == k]) for k in np.unique(nfp)}
    for k in nn:                                       # spacing of the public set, per NFP
        dd = nn[k].kneighbors(X[nfp == k])[0][:, 1]
        print(f"  NFP {k}: nearest-neighbour distance median {np.median(dd):.4f}, 99th pct {np.quantile(dd, 0.99):.4f}")
    Ag = np.r_[[aspect_ratio(pub["R"][i], pub["Z"][i], int(nfp[i]))[0] for i in front]]
    print(f"  geometric A vs leaderboard on parents: max |dA| {np.max(np.abs(Ag - A[front])):.4f}")

    cands = {}
    for _ in range(pool):
        a = front[rng.integers(len(front))]
        others = front[(nfp[front] == nfp[a]) & (fam[front] != fam[a]) & (np.abs(A[front] - A[a]) < window)]
        if len(others) == 0:
            continue
        b = others[rng.integers(len(others))]
        t = rng.uniform(0.3, 0.7)
        R = (1 - t) * pub["R"][a] + t * pub["R"][b]; Z = (1 - t) * pub["Z"][a] + t * pub["Z"][b]
        shape = tuple(max(x, y) for x, y in zip(pub["shape"][a], pub["shape"][b]))
        bj = json.dumps(boundary(R, Z, int(nfp[a]), shape))
        did = "full-" + hashlib.sha1(bj.encode()).hexdigest()[:12]
        cands[did] = dict(id=did, boundary=bj, method="blend", nfp=int(nfp[a]), shape=list(shape),
                          parent=str(pub["id"][a]), parent_b=str(pub["id"][b]),
                          parent_user=str(pub["user"][a]), parent_b_user=str(pub["user"][b]),
                          step=round(float(t), 3), seed=seed, layers=layers, window=window,
                          _R=R, _Z=Z)
    cl = list(cands.values())
    V = vec(np.array([c["_R"] for c in cl]), np.array([c["_Z"] for c in cl]))
    k_nfp = np.array([c["nfp"] for c in cl])
    nov = np.empty(len(cl)); Ac = np.empty(len(cl))
    for k in np.unique(k_nfp):
        i = np.flatnonzero(k_nfp == k)
        nov[i] = nn[k].kneighbors(V[i], n_neighbors=1)[0][:, 0]
        Ac[i] = aspect_ratio(np.array([cl[j]["_R"] for j in i]), np.array([cl[j]["_Z"] for j in i]), int(k))
    keep = np.flatnonzero((nov >= NOVELTY) & (Ac < 19.9))
    print(f"{len(cl)} distinct blends, {len(keep)} novel with A < 19.9 "
          f"(novelty median {np.median(nov):.4f})")
    old = [json.loads(x) for x in DESIGNS.open()] if append and DESIGNS.exists() else []
    W = [vec(*standard(d["boundary"])[:2]) for d in old if d["method"] == "blend"]
    chosen = []
    for i in rng.permutation(keep):
        if len(chosen) == n:
            break
        if (cl[i]["id"] not in {d["id"] for d in old}
                and all(np.linalg.norm(V[i] - V[j]) >= NOVELTY for j in chosen)
                and all(np.linalg.norm(V[i] - w) >= NOVELTY for w in W)):
            chosen.append(i)
    out = []
    for i in chosen:
        c = {k: v for k, v in cl[i].items() if not k.startswith("_")}
        out.append(dict(c, novelty=float(nov[i]), A_geom=float(Ac[i])))

    # controls: unmodified public front designs, spread over A, both NFPs
    f1 = np.flatnonzero(front_mask(A, L)) if not append else np.array([], int)
    for k in np.unique(nfp[f1]):
        i = f1[nfp[f1] == k]; i = i[np.argsort(A[i])]
        for j in i[np.linspace(0, len(i) - 1, controls // len(np.unique(nfp[f1]))).round().astype(int)]:
            out.append(dict(id=str(pub["id"][j]), boundary=pub["boundary"][j], method="control",
                            nfp=int(k), shape=list(pub["shape"][j]), user=str(pub["user"][j]),
                            lb_A=float(A[j]), lb_L=float(L[j]), lb_violation=float(pub["violation"][j])))
    OUT.mkdir(parents=True, exist_ok=True)
    with DESIGNS.open("a" if append else "w") as fh:
        for d in out:
            fh.write(json.dumps(d) + "\n")
    print(f"{'appended' if append else 'wrote'} {len(out)} designs ({len(chosen)} blends + {len(out) - len(chosen)} controls) -> {DESIGNS}")


def simulate(only: int | None, workers: int) -> None:
    import run_sims
    designs = [json.loads(x) for x in DESIGNS.open()]
    if only:                                           # timing check: one control + one blend first
        designs = [next(d for d in designs if d["method"] == "control"),
                   next(d for d in designs if d["method"] == "blend")][:only]
    run_sims.run(designs, LEDGER, workers=workers)


def report() -> None:
    designs = {d["id"]: d for d in map(json.loads, DESIGNS.open())}
    rows = {r["id"]: r for r in map(json.loads, LEDGER.open()) if r["id"] in designs}
    pub = load_public(); A, L = pub["A"], pub["L"]
    sec = [r["seconds"] for r in rows.values()]
    print(f"{len(rows)} simulated; seconds median {np.median(sec):.0f}, max {np.max(sec):.0f}; "
          f"timeouts {sum(bool(r.get('timeout')) for r in rows.values())}")
    print("\ncontrols (our scorer vs the leaderboard, at full resolution):")
    for i, r in rows.items():
        d = designs[i]
        if d["method"] != "control":
            continue
        if not r["ok"]:
            print(f"  {i} NFP {d['nfp']} {tuple(d['shape'])}: FAILED {r.get('error')}"); continue
        print(f"  NFP {d['nfp']} {tuple(d['shape'])} A {d['lb_A']:6.2f}: dA {r['aspect_ratio'] - d['lb_A']:+.1e} "
              f"dL {r['L_gradB'] - d['lb_L']:+.1e} violation {d['lb_violation']:.4f} -> {r['max_violation']:.4f} "
              f"pass {d['lb_violation'] <= 0.01} -> {r['feasible']}")
    bl = [(designs[i], r) for i, r in rows.items() if designs[i]["method"] == "blend"]
    ok = [(d, r) for d, r in bl if r["ok"]]
    ps = [(d, r) for d, r in ok if r["feasible"]]
    print(f"\nblends: {len(bl)} simulated, {len(bl) - len(ok)} crashed, {len(ps)} passed "
          f"({len(ps) / max(len(bl), 1):.0%}); strict-novel (>= {STRICT}) passes: "
          f"{sum(d['novelty'] >= STRICT for d, _ in ps)}")
    for key, name in (("nfp", "NFP"), ("layers", "parent layers")):
        for k in sorted({d[key] for d, _ in bl}):
            s = [(d, r) for d, r in bl if d[key] == k]
            print(f"  {name} {k}: {sum(r['feasible'] for _, r in s)}/{len(s)} passed")
    if ps:
        oA = np.array([r["aspect_ratio"] for _, r in ps]); oL = np.array([r["L_gradB"] for _, r in ps])
        base = hypervolume(A, L); both = hypervolume(np.r_[A, oA], np.r_[L, oL])
        g = hv_gain(oA, oL, A, L)
        print(f"\nfull public front HV {base:.3f} -> with our passes {both:.3f} (gain {both - base:+.4f}); "
              f"{int((g > 1e-9).sum())} designs add to it")
        f = front_mask(A, L); o = np.argsort(A[f]); fa, fl = A[f][o], np.maximum.accumulate(L[f][o])
        for (d, r), gi in sorted(zip(ps, g), key=lambda x: -x[1])[:8]:
            h = fl[np.searchsorted(fa, r["aspect_ratio"], side="right") - 1] if r["aspect_ratio"] >= fa[0] else -1
            print(f"  A {r['aspect_ratio']:6.2f} L {r['L_gradB']:6.2f} (front there {h:6.2f}) gain {gi:.4f} "
                  f"viol {r['max_violation']:.4f} novelty {d['novelty']:.4f} NFP {d['nfp']} "
                  f"{d['parent_user']} x {d['parent_b_user']} t={d['step']} layers {d['layers']}")
    fails = {}
    for d, r in ok:
        if not r["feasible"]:
            k = max(r["violations"], key=r["violations"].get); fails[k] = fails.get(k, 0) + 1
    print(f"\nfailing blends, worst constraint: {fails}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--n", type=int, default=60); p.add_argument("--controls", type=int, default=8)
    p.add_argument("--seed", type=int, default=40); p.add_argument("--layers", type=int, default=1)
    p.add_argument("--window", type=float, default=1.0); p.add_argument("--pool", type=int, default=6000)
    p.add_argument("--append", action="store_true")
    s = sub.add_parser("simulate")
    s.add_argument("--only", type=int, default=None); s.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "propose":
        propose(a.n, a.controls, a.seed, a.layers, a.window, a.pool, a.append)
    elif a.cmd == "simulate":
        simulate(a.only, a.workers)
    else:
        report()


if __name__ == "__main__":
    main()
