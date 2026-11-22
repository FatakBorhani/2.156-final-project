import sys, json
from pathlib import Path
sys.path[:0] = ["scripts", "scripts/feasibility_tests"]
from report_front_figure import leaderboard, front, draw_front, BLUE, GREY, ORANGE, INK
import matplotlib.pyplot as plt
from front_test import read
D = Path("data/feasibility_tests")
srcs = [("Tests 9–11 (first feasible, chain)", ["multifid", "front", "chain"], "#9a9a9a"),
        ("Test 12 (VAE seeds)", ["gen"], "#7b4fc9"),
        ("Test 13 (simplicity ladder)", ["ladder"], "#2a9d6f")]
public, baseline = leaderboard()
hv_pub, f_pub = front(public); hv_base, f_base = front(baseline)
ours, seen = [], set()
fig, ax = plt.subplots(figsize=(7.2, 4.4))
draw_front(ax, f_pub, GREY, f"public leaderboard front (NFP-3 4×4), HV {hv_pub:.1f}", 0.0, ls=":", lw=1.0)
draw_front(ax, f_base, ORANGE, f"Proxima's baseline entry, HV {hv_base:.1f}", 0.0, ls="--", lw=1.0)
pts_by = []
for label, names, c in srcs:
    ps = []
    for n in names:
        for r in read(D / f"{n}_ledger.jsonl"):
            if r["kind"] == "hi" and r.get("feasible"):
                k = (round(r["aspect_ratio"], 6), round(r["L_gradB"], 6))
                if k not in seen:
                    seen.add(k); ps.append(r)
    ours += ps; pts_by.append((label, ps, c))
hv_ours, f_ours = front(ours)
draw_front(ax, f_ours, BLUE, f"Track A front now, HV {hv_ours:.1f}", 0.15, lw=1.6)
for label, ps, c in pts_by:
    ax.scatter([p["aspect_ratio"] for p in ps], [p["L_gradB"] for p in ps], s=10, color=c, alpha=0.8,
               label=f"{label}: {len(ps)}", zorder=3)
ax.set_xlim(5, 20.5); ax.set_ylim(-1, 22)
ax.set_xlabel("aspect ratio  (lower = more compact)"); ax.set_ylabel("L∇B  (higher = simpler coils)")
ax.set_title(f"Track A feasible designs and hypervolume, Nov 18 10:40 pm  ({len(ours)} distinct feasible)", fontsize=9, loc="left")
ax.legend(fontsize=7, loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=170)
print(hv_ours, hv_base, hv_pub, len(ours))
