"""Why does no dataset design pass Benchmark 3? Breakdown + figure.

Splits the five Benchmark 3 constraints into QI vs. "the other four" and shows that the
designs that pass QI and the designs that pass the other four are different designs.

    python scripts/why_infeasible.py     # writes results/why_infeasible.json
                                         # and Docs/figures/why_no_feasible.png

"Violation" below is the benchmark's normalized constraint violation: how far a metric
is past its limit, as a fraction of the limit (0.11 = 11% past). <= 0.01 counts as a pass.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = "proxima-fusion/constellaration"
TOL = 0.01
M = "metrics."
COLS = [
    M + "edge_rotational_transform_over_n_field_periods",
    M + "qi",
    M + "edge_magnetic_mirror_ratio",
    M + "flux_compression_in_regions_of_bad_curvature",
    M + "vacuum_well",
]

files = [
    f
    for f in HfApi().list_repo_files(REPO, repo_type="dataset")
    if f.endswith(".parquet") and f.startswith("data/")
]
df = pd.concat(
    [
        pq.read_table(hf_hub_download(REPO, f, repo_type="dataset"), columns=COLS)
        .to_pandas()
        for f in files
    ],
    ignore_index=True,
).dropna()
df = df[df[M + "qi"] > 0]

# Same formulas as problems.MHDStableQIStellarator._normalized_constraint_violations
v = pd.DataFrame(
    {
        "QI: log10(qi) <= -3.5": (np.log10(df[COLS[1]]) + 3.5) / 3.5,
        "twist: |edge iota| >= 0.25": (0.25 - df[COLS[0]].abs()) / 0.25,
        "mirror ratio <= 0.25": (df[COLS[2]] - 0.25) / 0.25,
        "flux compression <= 0.9": (df[COLS[3]] - 0.9) / 0.9,
        "vacuum well >= 0": (0.0 - df[COLS[4]]) / 0.1,
    }
)
qi = v.columns[0]
others = list(v.columns[1:])
fail = v > TOL
pass_qi = ~fail[qi]
pass_others = ~fail[others].any(axis=1)

out = {
    "valid_designs": len(v),
    "feasible (pass all 5)": int((pass_qi & pass_others).sum()),
    "number_of_constraints_failed": {
        str(k): int(n) for k, n in fail.sum(axis=1).value_counts().sort_index().items()
    },
    "pass_QI": int(pass_qi.sum()),
    "pass_QI__pct_failing_each_other_constraint": {
        k: round(100 * fail.loc[pass_qi, k].mean(), 1) for k in others
    },
    "pass_QI__median_violation_of_failed_other_constraints": {
        k: round(float(v.loc[pass_qi & fail[k], k].median()), 2) for k in others
    },
    "pass_other_four": int(pass_others.sum()),
    "pass_other_four__median_log10_qi": round(
        float(np.log10(df.loc[pass_others, COLS[1]]).median()), 2
    ),
    "pass_other_four__median_QI_violation": round(
        float(v.loc[pass_others, qi].median()), 2
    ),
}
print(json.dumps(out, indent=2))
(ROOT / "results" / "why_infeasible.json").write_text(json.dumps(out, indent=2))

# --- Figure ---------------------------------------------------------------------
Y_MAX = 150  # % ; worse values are drawn at the top edge
x = 100 * v[qi].clip(lower=0)
y = (100 * v[others].max(axis=1).clip(lower=0)).clip(upper=Y_MAX)
INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (reference palette)

fig, ax = plt.subplots(figsize=(9, 6.2), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
hb = ax.hexbin(x, y, gridsize=70, bins="log", cmap="Greys", mincnt=1,
               extent=(0, 105, 0, Y_MAX), linewidths=0, alpha=0.55)
ax.scatter(x[pass_qi], y[pass_qi], s=16, color=BLUE, edgecolor="#fcfcfb",
           linewidth=0.6, zorder=3,
           label=f"Pass QI, fail others ({pass_qi.sum()})")
ax.scatter(x[pass_others], y[pass_others], s=16, color=ORANGE, edgecolor="#fcfcfb",
           linewidth=0.6, zorder=3,
           label=f"Pass the other four, fail QI ({pass_others.sum()})")

# The target: both violations ~0
ax.add_patch(plt.Rectangle((-1.5, -2.5), 6, 9, fill=False, ec=INK, lw=1.5, ls="--",
                           zorder=4))
ax.annotate("Feasible = this corner\n(passes all 5)\n0 designs",
            xy=(4.5, 6.5), xytext=(14, 30), fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=1))

ax.set_xlim(-2, 105)
ax.set_ylim(-3, Y_MAX + 3)
ax.set_xlabel("QI constraint: % past the limit  (0 = passes)", color=INK)
ax.set_ylabel("Worst of the other four constraints:\n% past the limit  (0 = passes)",
              color=INK)
ax.set_title("No dataset design passes all five Benchmark 3 constraints",
             loc="left", fontsize=13, color=INK, pad=22)
ax.text(0, 1.015,
        f"{len(v):,} valid designs (gray = density). Designs that pass QI and designs "
        f"that pass everything else are different designs.",
        transform=ax.transAxes, fontsize=9, color=MUTED)
ax.text(1, -0.11, f"y values above {Y_MAX}% drawn at the top edge.",
        transform=ax.transAxes, fontsize=8, color=MUTED, ha="right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#b5b4af")
ax.tick_params(colors=MUTED)
ax.grid(color="#e8e7e3", lw=0.6, zorder=0)
ax.set_axisbelow(True)
leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.035)
cb.set_label("designs per cell (log)", color=MUTED, fontsize=8)
cb.ax.tick_params(colors=MUTED, labelsize=7)
cb.outline.set_visible(False)

out_png = ROOT / "Docs" / "figures" / "why_no_feasible.png"
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out_png, facecolor=fig.get_facecolor())
print("wrote", out_png)
