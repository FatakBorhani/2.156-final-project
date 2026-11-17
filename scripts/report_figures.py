"""Figures for the mid-term report: dataset distributions and sample designs.

No simulations: reads the cached nfp-3 dataset and our saved designs.

    python scripts/report_figures.py   -> results/report/{outputs,inputs,samples}.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "feasibility_tests"), str(HERE)]
from common import ROOT, load_designs  # noqa: E402

OUT = ROOT / "results" / "report"
FEAS = ROOT / "results" / "feasibility_tests" / "first_feasible_design.jsonl"
GREY, BLUE, ORANGE = "#9a9a9a", "#1f6fd1", "#d9622b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

PANELS = [  # (target column, label, limit, passes if value <= limit?)
    ("log10_qi", "QI residual, log10(qi)", -3.5, True),
    ("abs_iota", "edge twist per field period", 0.25, False),
    ("mirror", "edge mirror ratio", 0.25, True),
    ("flux_compression", "flux compression, bad curvature", 0.9, True),
    ("vacuum_well", "vacuum well", 0.0, False),
]


def boundary_arrays(bjson: str) -> tuple[np.ndarray, np.ndarray, int]:
    b = json.loads(bjson)
    return np.array(b["r_cos"]), np.array(b["z_sin"]), b["n_field_periods"]


def cross_section(r, z, nfp, phi, n_theta=200):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    m = np.arange(r.shape[0])[:, None, None]
    n = (np.arange(r.shape[1]) - (r.shape[1] - 1) // 2)[None, :, None]
    ang = m * theta[None, None, :] - n * nfp * phi
    return (r[:, :, None] * np.cos(ang)).sum((0, 1)), (z[:, :, None] * np.sin(ang)).sum((0, 1))


def ours():
    from constellaration import forward_model  # noqa: F401  (only to fail early if env is wrong)
    d = json.loads(FEAS.read_text().splitlines()[0])
    check = ROOT / "data" / "feasibility_tests" / "first_feasible_check.jsonl"
    m = json.loads(check.read_text().splitlines()[0])["metrics"]
    t = {"log10_qi": np.log10(m["qi"]), "abs_iota": abs(m["edge_rotational_transform_over_n_field_periods"]),
         "mirror": m["edge_magnetic_mirror_ratio"],
         "flux_compression": m["flux_compression_in_regions_of_bad_curvature"],
         "vacuum_well": m["vacuum_well"], "L_gradB": m["minimum_normalized_magnetic_gradient_scale_length"],
         "aspect_ratio": m["aspect_ratio"]}
    return d["boundary"], t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, T, V, meta = load_designs(3)
    worst = V.clip(lower=0).max(axis=1)
    best = worst.idxmin()
    our_b, our_t = ours()
    n = len(T)

    # 1. Outputs: the five requirements + the objective space
    fig, axs = plt.subplots(2, 3, figsize=(7.5, 4.6))
    for ax, (col, label, lim, upper) in zip(axs.flat, PANELS):
        v = T[col].clip(*np.nanpercentile(T[col], [0.5, 99.5]))
        ax.hist(v, bins=60, color=GREY)
        frac = (V[col] <= 0.01).mean()
        ax.axvline(lim, color="k", ls="--", lw=1)
        ax.axvline(our_t[col], color=BLUE, lw=1.5)
        ax.set_title(f"{label}\n{'≤' if upper else '≥'} {lim:g}: {frac:.0%} pass", fontsize=8.5)
        ax.set_yticks([])
    ax = axs.flat[5]
    ax.hexbin(T["aspect_ratio"], T["L_gradB"].clip(upper=20), gridsize=40, bins="log", cmap="Greys", mincnt=1)
    ax.plot(T.loc[best, "aspect_ratio"], T.loc[best, "L_gradB"], "o", color=ORANGE, ms=6, label="dataset closest")
    ax.plot(our_t["aspect_ratio"], our_t["L_gradB"], "*", color=BLUE, ms=11, label="our feasible design")
    ax.set_xlabel("aspect ratio")
    ax.set_ylabel("L∇B")
    ax.set_title("objectives (all designs)", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.suptitle(f"Outputs: {n:,} dataset designs with 3 field periods, none passing all five "
                 f"(dashed: limit, blue: our feasible design)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "outputs.png", dpi=200)

    # 2. Inputs: coefficient spread by mode + PCA map colored by distance to feasible
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})
    cols = list(X.columns)
    grid = np.full((2 * 5, 9), np.nan)
    for c in cols:
        kind, m, nn = c.split("_m")[0], int(c.split("_m")[1].split("_n")[0]), int(c.split("_n")[1])
        grid[(0 if kind == "r_cos" else 5) + m, nn + 4] = np.log10(X[c].abs().quantile(0.95) + 1e-6)
    im = axs[0].imshow(grid, cmap="Greys", aspect="auto")
    axs[0].set_xticks(range(9), [str(k) for k in range(-4, 5)])
    axs[0].set_yticks(range(10), [f"R m={m}" for m in range(5)] + [f"Z m={m}" for m in range(5)], fontsize=7)
    axs[0].set_xlabel("toroidal mode n")
    axs[0].set_title("81 inputs: 95th-percentile |coefficient| (log10)", fontsize=8.5)
    fig.colorbar(im, ax=axs[0], fraction=0.05)
    pca = PCA(2).fit(X.values)
    P = pca.transform(X.values)
    order = np.argsort(-worst.values)  # closest to feasible drawn on top
    sc = axs[1].scatter(P[order, 0], P[order, 1], c=np.log10(worst.values[order] + 1e-3), s=1.5,
                        cmap="viridis_r", rasterized=True)
    ours_x = X.iloc[[0]].copy()
    r, z, _ = boundary_arrays(our_b)
    for c in cols:
        kind, m, nn = c.split("_m")[0], int(c.split("_m")[1].split("_n")[0]), int(c.split("_n")[1])
        ours_x[c] = (r if kind == "r_cos" else z)[m, nn + 4]
    po = pca.transform(ours_x.values)[0]
    pb = P[X.index.get_loc(best)]
    axs[1].plot(*pb, "o", mfc="none", mec=ORANGE, ms=9, mew=1.5, label="dataset closest (0.203)")
    axs[1].plot(*po, "*", color=BLUE, ms=12, label="our feasible design (0.0087)")
    axs[1].set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    axs[1].set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.0%})")
    axs[1].set_title("inputs projected to 2 PCA directions", fontsize=8.5)
    axs[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.colorbar(sc, ax=axs[1], fraction=0.05, label="log10 worst violation")
    fig.tight_layout()
    fig.savefig(OUT / "inputs.png", dpi=200)

    # 3. Sample designs: cross-sections over half a field period
    rng = np.random.default_rng(0)
    rand = X.index[rng.integers(len(X))]
    samples = [(f"random dataset design ({worst[rand]:.2f})", meta.loc[rand, "boundary_json"]),
               (f"dataset closest ({worst[best]:.3f})", meta.loc[best, "boundary_json"]),
               ("our feasible design (0.0087)", our_b)]
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.9), sharey=True)
    for ax, (title, bj) in zip(axs, samples):
        r, z, nfp = boundary_arrays(bj)
        for k, shade in zip(range(3), [1.0, 0.6, 0.3]):
            phi = k * np.pi / nfp / 2  # 0, 1/4 and 1/2 of a field period
            R, Z = cross_section(r, z, nfp, phi)
            ax.plot(R, Z, color=BLUE if "our" in title else "k", alpha=shade, lw=1.2,
                    label=f"{k / 4:g} of a period")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("R")
    axs[0].set_ylabel("Z")
    axs[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Sample designs: plasma cross-sections at three angles", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "samples.png", dpi=200)
    print("pass rates:", {c: round(float((V[c] <= 0.01).mean()), 3) for c, *_ in PANELS},
          "n", n, "best", best, round(float(worst[best]), 3))


if __name__ == "__main__":
    main()
