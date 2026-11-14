# Plan feasibility tests: results so far

*Run 2026-11-14 on Leif's M3 Max. These are quick, throwaway checks of the plan's
riskiest assumptions, not the project's real surrogates or search. All use the 68,191
valid designs with 3 field periods.*

| Test | Question | Status | Verdict |
|---|---|---|---|
| 1 | Can a surrogate predict the metrics, especially QI? | ✅ done | **Go.** Use a neural net, not boosted trees, for QI |
| 2 | Does surrogate-guided search improve *real* designs? | not run | (the most important one; needs Test 1) |
| 3 | Can shapes be compressed (PCA) without changing the physics? | not run | |
| 4 | Can low-fidelity VMEC++ stand in for high fidelity? | ✅ done | **Yes, as a filter.** Not for the final score |

## Test 1: surrogate accuracy

`scripts/feasibility_tests/surrogate_test.py` → `surrogate.json`, `surrogate_qi_parity.png`

Two off-the-shelf models (scikit-learn boosted trees and a 3×256 neural net, MLP),
trained in ~2–3 min on 54.5k designs, tested on 13.6k held-out designs. Inputs are the
81 raw shape coefficients.

**Overall, both are accurate.** R² is 0.92–0.996 on all 7 metrics.

| Metric | R² (trees / MLP) | Typical error (MLP) |
|---|---|---|
| log10(qi) | 0.948 / 0.957 | 0.046 |
| Edge twist | 0.994 / 0.993 | 0.006 |
| Mirror ratio | 0.971 / 0.974 | 0.012 |
| Flux compression | 0.945 / 0.954 | 0.030 |
| Vacuum well | 0.986 / 0.988 | 0.008 |
| L∇B (goal) | 0.924 / 0.916 | 0.30 |
| Aspect ratio (goal) | 0.996 / 0.989 | 0.09 |

**Where it matters (best-QI designs), the MLP is much better.**

| Held-out designs with true log10(qi) < -3 (131 designs) | Trees | MLP |
|---|---|---|
| Average QI error | 0.21 | **0.12** |
| Bias (+ = predicts worse QI than true) | +0.20 | **+0.05** |

Trees flatten out at the good end: they can't predict QI better than what they've seen
often, which is fatal when the target is *beyond* the data. The MLP tracks the diagonal
(see the figure). For scale: the closest design needs log10(qi) to improve by 0.71 to
pass, so the MLP's error is about 1/6 of the gap.

![QI parity](surrogate_qi_parity.png)

**It can screen candidates.** Ranking held-out designs by predicted worst violation, the
MLP's top 500 (3.7% of the test set) contains **95%** of the truly 100 closest designs.

**Weak spots:** pass/fail right at the limit for **vacuum well (~70% correct within 20%
of the limit)** and **mirror ratio (~85%)**. Vacuum well's limit is normalized by 0.1, so
small absolute errors become large violations. These are two of the constraints the
good-QI designs fail, so they need attention.

**Caveat:** this is a random split, so test designs look like training designs. The real
risk is accuracy on *new* shapes the search proposes, outside the data. That's what Test 2
checks.

## Test 4: low vs. high fidelity

`scripts/feasibility_tests/fidelity_test.py` → `fidelity.json`

50 designs (the 25 closest to feasible + 25 random), evaluated at very low and low
fidelity and compared with high fidelity (the dataset's stored values, which match a fresh
high-fidelity run).

| | Very low | Low | High |
|---|---|---|---|
| Time per design | **3.2 s** | 8.1 s | ~130 s |
| Speedup vs. high | **40×** | 16× | 1× |
| Worst-violation ranking agreement, all 50 (Spearman) | 0.97 | 0.98 | |
| Same, 25 closest only | 0.80 | 0.84 | |
| Worst-violation error, 25 closest (mean / max) | 0.020 / 0.057 | 0.018 / 0.054 | |
| Pass/fail agreement per constraint | 92–100% | 92–100% | |

- **Low fidelity isn't worth it.** It's barely more accurate than very low and 2.5× slower.
- **Very low fidelity is a good filter:** 40× cheaper, and it ranks designs well.
- **But it can't give the final verdict.** Near the limit its error (up to ~0.06) is far
  bigger than the 0.01 pass tolerance, so a design counts as feasible only after a
  high-fidelity check. The **L∇B goal** is its least reliable metric (one design off by
  5.4), so scoring must be high fidelity too.
- Matters for the baseline comparison: ALM-NGOpt optimizes at very low fidelity, so its
  designs must be re-scored at high fidelity.

## What this means for the plan

1. **Surrogate:** a neural net is viable for QI. Boosted trees are a poor fit for the
   region we need to reach.
2. **Physics loop:** use the surrogate (milliseconds) to propose, very low fidelity
   (3 s) to filter, and high fidelity (~2 min) to confirm. That's a three-tier funnel.
3. **Give vacuum well and mirror ratio extra attention** in the surrogate, e.g. a margin
   or uncertainty estimate.
4. **Next, run Test 2:** does a search guided by this surrogate actually move real designs
   closer to feasible?
