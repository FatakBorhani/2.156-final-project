"""Track A progress figure for the update after the mid-term report (Tests 10-22).

Left: Track A's hypervolume after each test. Right: all Track A feasible designs and their
front, next to the three feasible designs of Proxima's ALM-NGOpt baseline entry (values
typed in from the leaderboard; nothing is read from the leaderboard cache).
No simulations: reads our ledgers in data/feasibility_tests.

    python scripts/report_progress_figure.py   -> results/report/track_a_progress.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from report_front_figure import BLUE, GREY, INK, ORANGE, OUT, REF, draw_front, front  # noqa: E402

DATA = HERE.parent / "data" / "feasibility_tests"
LEDGERS = ["multifid", "front", "chain", "gen", "ladder", "compact", "diffusion", "descent",
           "diffusion2", "descent2", "modes5", "finemodes", "raise", "stretch", "fatten", "stretch2", "fatten2"]
BASELINE = [(7.93, 5.60), (9.98, 8.45), (11.9, 11.1)]  # (aspect ratio, L_gradB), Proxima's entry, rounded
HV_BASE = 129.8  # official score of that entry (the rounded points give 129.7)
STEPS = [  # (label, hypervolume after it), from results/feasibility_tests/<test>.json
    ("Test 10: spread from first feasible", 76.5),
    ("Test 11: compactness chain", 81.4),
    ("Test 12: conditional VAE seeds", 87.6),
    ("Test 13: simplicity ladder", 125.3),
    ("Test 14: compactness ladder", 128.2),
    ("Test 15: diffusion seeds", 136.0),
    ("Test 16: small-step descent", 138.6),
    ("Test 17: diffusion seeds, round 2", 138.6),
    ("Test 18: 5×5 modes", 144.4),
    ("Test 19: 4 field periods", 144.4),
    ("Test 20: fine modes, then raise", 147.2),
    ("Test 21: stretch and fatten", 157.9),
    ("Test 22: second pass", None),  # filled from the ledgers
]


def feasible() -> list[dict]:
    out = {}
    for name in LEDGERS:
        p = DATA / f"{name}_ledger.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("kind") == "hi" and r.get("feasible"):
                out[(round(r["aspect_ratio"], 6), round(r["L_gradB"], 6))] = r
    return list(out.values())


def main() -> None:
    ours = feasible()
    base = [{"aspect_ratio": a, "L_gradB": lg} for a, lg in BASELINE]
    hv_ours, f_ours = front(ours)
    _, f_base = front(base)
    hv_base = HV_BASE
    steps = [(lab, hv if hv is not None else hv_ours) for lab, hv in STEPS]
    print(f"ours {len(ours)} feasible, HV {hv_ours:.2f}; baseline HV {hv_base:.2f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1, 1.1]})

    y = np.arange(len(steps))[::-1]
    a1.barh(y, [h for _, h in steps], color=BLUE, height=0.62)
    for yi, (_, h) in zip(y, steps):
        a1.text(h + 2, yi, f"{h:.1f}", va="center", fontsize=7, color=INK)
    a1.axvline(hv_base, color=ORANGE, ls=":", lw=1)
    a1.text(hv_base, y[0] + 0.9, "Proxima's entry", color=ORANGE, fontsize=7, ha="center")
    a1.set_yticks(y)
    a1.set_yticklabels([s for s, _ in steps], fontsize=7)
    a1.set_xlim(0, 175)
    a1.set_ylim(-0.7, y[0] + 1.4)
    a1.spines["left"].set_visible(False)
    a1.tick_params(axis="y", length=0)
    a1.set_xlabel("Track A hypervolume (higher is better)")
    a1.set_title("(a) Hypervolume after each test", fontsize=8.5, loc="left")

    draw_front(a2, f_base, ORANGE, "", 0.15)
    draw_front(a2, f_ours, BLUE, "", 0.30)
    a2.scatter([p["aspect_ratio"] for p in ours], [p["L_gradB"] for p in ours], s=3, color=BLUE,
               alpha=0.5, lw=0, zorder=3)
    a2.scatter(*zip(*BASELINE), s=16, color=ORANGE, zorder=4)
    a2.scatter([REF[1]], [-1], marker="x", color=INK, s=18, zorder=3)
    a2.set_xlim(7, 20.5)
    a2.set_ylim(-1.6, 21)
    a2.set_xlabel("aspect ratio (lower = more compact)")
    a2.set_ylabel("L∇B (higher = simpler coils)")
    a2.text(19.7, 2.0, f"Track A ({len(ours)} feasible)\nHV {hv_ours:.1f}", ha="right", fontsize=7,
            color=BLUE, fontweight="bold", linespacing=1.1)
    a2.text(8.2, 12.2, f"Proxima's entry\nHV {hv_base:.1f}", fontsize=7, color=ORANGE,
            fontweight="bold", linespacing=1.1)
    a2.set_title("(b) Track A's feasible designs now", fontsize=8.5, loc="left")

    fig.tight_layout(w_pad=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "track_a_progress.png", dpi=200)
    print("wrote", OUT / "track_a_progress.png")


if __name__ == "__main__":
    main()
