"""Track A results figure for the mid-term report: progress to feasibility, then the front.

Left: worst violation of the best design after each Track A stage (and the paper's
baseline at a matched budget). Right: Track A's feasible designs (Tests 9-10) in the
benchmark's objective space, with their hypervolume next to the paper's baseline
submission and the public feasible designs in the same design space (3 field periods,
4x4 modes). No simulations: reads our results and the cached leaderboard dataset.

    python scripts/report_front_figure.py   -> results/report/track_a.png
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from constellaration import problems  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "feasibility_tests"), str(HERE)]
from common import ROOT  # noqa: E402

OUT = ROOT / "results" / "report"
TESTS = ROOT / "results" / "feasibility_tests"
REF = np.array([1.0, 20.0])  # benchmark reference point: L_gradB -1, aspect ratio 20
BASELINE_FILE = "2025-07-02T07-38-21.817193_mhd_stable_results.json"  # paper's ALM-NGOpt, 129.8
GREY, BLUE, ORANGE, INK = "#9a9a9a", "#1f6fd1", "#d9622b", "#333333"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

STAGES = [  # (label, worst violation, ours?)  values from results/feasibility_tests/SUMMARY.md
    ("Dataset best (182k designs)", 0.203, True),
    ("Test 5: physics-in-the-loop", 0.096, True),
    ("Test 6: longer local push", 0.078, True),
    ("Test 8: retuned ALM polish", 0.033, True),
    ("Test 9: multi-fidelity finish", 0.0087, True),
    ("Paper baseline, usual start", 0.72, False),
    ("Paper baseline, dataset best", 0.126, False),
]


def _loads(x):
    return json.loads(x) if isinstance(x, str) else x


def leaderboard() -> tuple[list[dict], list[dict]]:
    """(public feasible designs, 3 field periods, 4x4 modes; paper baseline feasible designs)."""
    root = Path(snapshot_download("proxima-fusion/constellaration-bench-results", repo_type="dataset",
                                  allow_patterns=["mhd_stable/*.json"]))
    public, baseline, seen = [], [], set()
    for f in sorted((root / "mhd_stable").glob("*.json")):
        sub = json.loads(f.read_text())
        if not sub.get("evaluated") or sub.get("objectives") is None:
            continue
        for b, obj, viol in zip(_loads(sub["boundary_json"]), _loads(sub["objectives"]),
                                _loads(sub["feasibilities"])):
            b = b if isinstance(b, str) else json.dumps(b)
            if float(viol) > 0.01:
                continue
            (lgb, _), (ar, _) = obj
            p = {"L_gradB": float(lgb), "aspect_ratio": float(ar)}
            if f.name == BASELINE_FILE:
                baseline.append(p)
            d = json.loads(b)
            if (d["n_field_periods"] == 3 and np.shape(d["r_cos"]) == (5, 9) and b not in seen):
                seen.add(b)
                public.append(p)
    return public, baseline


def front(points: list[dict]) -> tuple[float, np.ndarray]:
    """Hypervolume and the non-dominated points (sorted by aspect ratio)."""
    x = np.array([[-p["L_gradB"], p["aspect_ratio"]] for p in points])
    hv = float(problems._hypervolume(x, REF))
    nd = [i for i in range(len(x)) if not any((x[j] <= x[i]).all() and (x[j] < x[i]).any()
                                              for j in range(len(x)))]
    f = x[nd][np.argsort(x[nd][:, 1])]
    return hv, np.column_stack([f[:, 1], -f[:, 0]])  # (aspect ratio, L_gradB)


def draw_front(ax, f: np.ndarray, color: str, label: str, fill: float, ls: str = "-",
               lw: float = 1.2) -> None:
    """Step line of the front and the region it dominates up to the reference point."""
    xs, ys = [f[0, 0]], [-1.0]
    for i, (ar, lgb) in enumerate(f):
        ys.append(lgb)
        xs.append(ar)
        nxt = f[i + 1, 0] if i + 1 < len(f) else REF[1]
        xs.append(nxt)
        ys.append(lgb)
    xs, ys = np.array(xs[1:]), np.array(ys[1:])
    ax.fill_between(np.r_[f[0, 0], xs], np.r_[-1, ys], -1, step=None, color=color, alpha=fill, lw=0)
    ax.plot(np.r_[f[0, 0], xs], np.r_[-1, ys], color=color, lw=lw, ls=ls, label=label)


def main() -> None:
    public, baseline = leaderboard()
    ours = [json.loads(line) for line in open(TESTS / "front_feasible_designs.jsonl")]
    hv_pub, f_pub = front(public)
    hv_base, f_base = front(baseline)
    hv_ours, f_ours = front(ours)
    print(f"public {len(public)} feasible, HV {hv_pub:.1f}; baseline {len(baseline)}, HV {hv_base:.1f}; "
          f"ours {len(ours)}, HV {hv_ours:.1f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.15]})

    # left: progress to feasibility
    ys = np.arange(len(STAGES))[::-1].astype(float)
    ys[-2:] -= 0.5  # gap before the baseline rows
    for (label, v, mine), y in zip(STAGES, ys):
        c = BLUE if label.startswith("Test 9") else (GREY if mine else ORANGE)
        a1.plot([0.005, v], [y, y], color="#e6e6e6", lw=1, zorder=0)
        a1.scatter(v, y, s=26, color=c, zorder=2)
        a1.annotate(f"{v:g}", (v, y), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK)
    a1.axvline(0.01, color=INK, ls="--", lw=0.8)
    a1.text(0.0104, ys[0] + 0.55, "feasible at 0.01", fontsize=7, color=INK)
    a1.set_xscale("log")
    a1.set_xlim(0.005, 1.6)
    a1.set_ylim(ys[-1] - 0.6, ys[0] + 0.9)
    a1.set_xticks([0.01, 0.03, 0.1, 0.3, 1])
    a1.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
    a1.set_yticks(ys)
    a1.set_yticklabels([s[0] for s in STAGES], fontsize=7)
    for t, (_, _, mine) in zip(a1.get_yticklabels(), STAGES):
        t.set_color(INK if mine else ORANGE)
    a1.spines["left"].set_visible(False)
    a1.tick_params(axis="y", length=0)
    a1.set_xlabel("worst normalized violation (log, lower is better)")
    a1.set_title("(a) Getting to feasible", fontsize=8.5, loc="left")

    # right: objective space and hypervolume
    a2.scatter([p["aspect_ratio"] for p in public], [p["L_gradB"] for p in public], s=2, color=GREY,
               alpha=0.25, lw=0, zorder=1)
    draw_front(a2, f_pub, GREY, "", 0.12)
    draw_front(a2, f_base, ORANGE, "", 0.15)
    draw_front(a2, f_ours, BLUE, "", 0.30)
    a2.scatter([p["aspect_ratio"] for p in ours], [p["L_gradB"] for p in ours], s=8, color=BLUE, zorder=3)
    a2.scatter([REF[1]], [-1], marker="x", color=INK, s=18, zorder=3)
    a2.set_xlim(0, 20.5)
    a2.set_ylim(-1.6, None)
    a2.set_xlabel("aspect ratio (lower = more compact)")
    a2.set_ylabel("L∇B (higher = simpler coils)")
    for y, color, text in [(16.4, "#555555", f"public: HV {hv_pub:.1f}"),
                           (9.9, ORANGE, f"paper baseline\nHV {hv_base:.1f}"),
                           (3.6, BLUE, f"Track A ({len(ours)} feasible)\nHV {hv_ours:.1f}")]:
        a2.text(19.7, y, text, ha="right", va="center", fontsize=7, color=color, fontweight="bold",
                linespacing=1.1)
    a2.set_title("(b) Feasible designs and hypervolume (HV)", fontsize=8.5, loc="left")

    fig.tight_layout(w_pad=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "track_a.png", dpi=200)
    print("wrote", OUT / "track_a.png")


if __name__ == "__main__":
    main()
