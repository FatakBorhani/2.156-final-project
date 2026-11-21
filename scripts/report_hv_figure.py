"""Hypervolume figure for the report: leaderboard yardsticks, Track A and Track B together.

(a) Objective space: the public leaderboard's feasible designs (3 field periods, 4x4 modes),
    Proxima's ALM-NGOpt leaderboard entry, Track A's feasible designs, and Track B's.
(b) Hypervolume after each stage of each track, against the two leaderboard yardsticks.

Track B's designs come from Joel's results/seeded/track_b_feasible_designs.csv (written by
scripts/seeded/report_numbers.py: every novel feasible design, columns aspect_ratio,
L_gradB, nearest_public, improves_public_front). Without that file, Track B is drawn as a
placeholder. --track-b points at a copy elsewhere (e.g. before his branch is merged).

No simulations: reads our results and the cached leaderboard dataset.

    python scripts/report_hv_figure.py [--track-b CSV]   -> results/report/hypervolume.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from report_front_figure import (BLUE, GREY, INK, ORANGE, OUT, REF, ROOT, TESTS,  # noqa: E402
                                 draw_front, front, leaderboard)

DATA = ROOT / "data" / "feasibility_tests"
TRACK_B_FILE = ROOT / "results" / "seeded" / "track_b_feasible_designs.csv"
PURPLE, LILAC = "#7b4fc9", "#c4b3e8"  # Track B: designs that improve the public front, the rest


def ledger_feasible(name: str) -> list[dict]:
    rows = [json.loads(line) for line in open(DATA / name)]
    return [r for r in rows if r["kind"] == "hi" and r.get("feasible")]


def track_b_designs(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return [{"aspect_ratio": float(r["aspect_ratio"]), "L_gradB": float(r["L_gradB"]),
                 "improves": r.get("improves_public_front") == "1"} for r in csv.DictReader(f)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track-b", type=Path, default=TRACK_B_FILE)
    a = p.parse_args()
    public, baseline = leaderboard()
    t9 = ledger_feasible("multifid_ledger.jsonl")  # Test 9: first feasible design(s)
    t10 = [json.loads(line) for line in open(TESTS / "front_feasible_designs.jsonl")]
    track_b = track_b_designs(a.track_b)

    hv_pub, f_pub = front(public)
    hv_base, f_base = front(baseline)
    hv_t9, _ = front(t9)
    hv_t10, f_t10 = front(t9 + t10)
    hv_b, f_b = front(public + track_b) if track_b else (None, None)
    hv_b_alone = front(track_b)[0] if track_b else None
    print(f"public {hv_pub:.1f}, Proxima entry {hv_base:.1f}, Test 9 {hv_t9:.1f} ({len(t9)}), "
          f"Test 10 {hv_t10:.1f} ({len(t10)}), Track B "
          + (f"{hv_b:.1f} ({len(track_b)})" if track_b else "placeholder"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) objective space
    a1.scatter([p["aspect_ratio"] for p in public], [p["L_gradB"] for p in public], s=2, color=GREY,
               alpha=0.25, lw=0, zorder=1)
    draw_front(a1, f_pub, GREY, "", 0.12)
    draw_front(a1, f_base, ORANGE, "", 0.15)
    draw_front(a1, f_t10, BLUE, "", 0.30)
    a1.scatter([p["aspect_ratio"] for p in t10], [p["L_gradB"] for p in t10], s=7, color=BLUE, zorder=3)
    if track_b:
        rest = [p for p in track_b if not p["improves"]]
        a1.scatter([p["aspect_ratio"] for p in rest], [p["L_gradB"] for p in rest], s=2,
                   color=LILAC, alpha=0.5, lw=0, zorder=2)
        top = [p for p in track_b if p["improves"]]
        a1.scatter([p["aspect_ratio"] for p in top], [p["L_gradB"] for p in top], s=5,
                   color=PURPLE, lw=0, zorder=3)
        draw_front(a1, f_b, PURPLE, "", 0.0)
        draw_front(a1, f_pub, "#555555", "", 0.0, ls="--", lw=0.8)  # public front on top: the gap is Track B's gain
    else:
        a1.text(1.0, 16.0, "Track B designs:\nplaceholder", fontsize=7, color=PURPLE, style="italic",
                va="center", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PURPLE, ls="--", lw=0.8))
    a1.scatter([REF[1]], [-1], marker="x", color=INK, s=18, zorder=3)
    labels = [(19.7, 16.6, "right", "#555555", "public leaderboard"),
              (19.7, 9.9, "right", ORANGE, "Proxima's entry"), (19.7, 3.6, "right", BLUE, "Track A")]
    if track_b:  # the purple front runs over the right-hand labels: name both fronts top-left
        labels[0] = (0.6, 17.0, "left", "#555555", "public front (dashed)")
        labels.insert(0, (0.6, 18.9, "left", PURPLE, "Track B (public + new)"))
    for x, y, ha, color, text in labels:
        a1.text(x, y, text, ha=ha, va="center", fontsize=7, color=color, fontweight="bold")
    a1.set_xlim(0, 20.5)
    a1.set_ylim(-1.6, 20.5)
    a1.set_xlabel("aspect ratio (lower = more compact)")
    a1.set_ylabel("L∇B (higher = simpler coils)")
    a1.set_title("(a) Feasible designs (3 field periods, 4×4 modes)", fontsize=8.5, loc="left")

    # (b) hypervolume by stage
    rows = [  # (label, value or None for placeholder, color, group)
        ("Proxima's ALM-NGOpt entry", hv_base, ORANGE, "Leaderboard"),
        ("All public designs, pooled", hv_pub, GREY, "Leaderboard"),
        ("Dataset (no feasible designs)", 0.0, BLUE, "Track A: from the dataset"),
        (f"Test 9: first feasible", hv_t9, BLUE, "Track A: from the dataset"),
        (f"Test 10: {len(t10)} feasible", hv_t10, BLUE, "Track A: from the dataset"),
        (f"Track B's {len(track_b):,} new designs alone" if track_b else "Track B designs alone",
         hv_b_alone, LILAC, "Track B: from public designs"),
        ("Public + Track B designs", hv_b, PURPLE, "Track B: from public designs"),
    ]
    y, ys, heads = 0.0, [], []
    for i, (_, _, _, g) in enumerate(rows):
        if i == 0 or g != rows[i - 1][3]:
            y -= 0.9 if i else 0.0
            heads.append((y, g))
            y -= 0.85
        ys.append(y)
        y -= 1.0
    for (label, v, color, _), yy in zip(rows, ys):
        if v is None:
            a2.barh(yy, hv_pub, height=0.62, color="white", ec=PURPLE, ls="--", lw=0.8, hatch="///",
                    alpha=0.6)
            a2.text(4, yy, "placeholder: Track B to add", va="center", fontsize=6.5, color=PURPLE,
                    style="italic", bbox=dict(fc="white", ec="none", pad=0.6))
            continue
        a2.barh(yy, v, height=0.62, color=color, alpha=0.6 if color == GREY else 0.85)
        a2.text(v + 3, yy, "0" if v == 0 else f"{v:.1f}", va="center", fontsize=7, color=INK)
    for yy, g in heads:
        a2.text(-2, yy, g, fontsize=7.2, fontweight="bold", color=INK, ha="left", va="center",
                transform=a2.get_yaxis_transform() if False else a2.transData)
    for v, color in [(hv_base, ORANGE), (hv_pub, GREY)]:
        a2.axvline(v, color=color, ls=":", lw=0.9, zorder=0)
    a2.set_yticks(ys)
    a2.set_yticklabels([r[0] for r in rows], fontsize=7)
    a2.tick_params(axis="y", length=0)
    a2.spines["left"].set_visible(False)
    a2.set_xlim(0, 240)
    a2.set_ylim(min(ys) - 0.7, 0.5)
    a2.set_xlabel("hypervolume (higher is better)")
    a2.set_title("(b) Hypervolume by stage", fontsize=8.5, loc="left")

    fig.tight_layout(w_pad=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "hypervolume.png", dpi=200)
    print("wrote", OUT / "hypervolume.png")


if __name__ == "__main__":
    main()
