# What separates compact, simple (high L∇B) QI designs: an empirical look

Benchmark 3 data analysis, 2026-11-21. The sources are the ConStellaration training dataset (182,222 rows; 158,685 have all Benchmark 3 metrics and qi > 0) and our own high-fidelity evaluations (`kind == "hi"`, `ok`) from `data/feasibility_tests/*_ledger.jsonl`. Our runs give 609 feasible nfp-3 designs (223 of them nondominated) and 284 nfp-4 evaluations. No leaderboard data and no new simulations were used. The scripts are in the session scratchpad (`common.py`, `load_ds.py`, `add_tgt.py`, `q1.py`, `q1b.py`, `q2.py`, `q2b.py`, `q3.py`, `q4.py`, `q5.py`, `figs.py`). The scratchpad is temporary, so copy the scripts into the repo if they need to be kept.

Conventions:
- Violations are normalized exactly as in the brief. "Pass" means violation ≤ 0.01, and "worst" is the largest positive violation.
- Hypervolume (HV) uses points (−L∇B, AR) with reference (1, 20). It reproduces our 144.40.
- "HV gain" of a dataset point is the exclusive HV it would add to our current feasible front if it were feasible.

## Headline findings

1. **No dataset design is feasible, and almost none is close.** Of 158,685 designs, 0 are feasible, 35 have worst ≤ 0.3, 856 have worst ≤ 0.4 and 4,490 have worst ≤ 0.5. The QI constraint passes in 0.11% of designs. The median log10(qi) is −1.78 against a required −3.5, so the generator never targeted this QI level. The dataset is therefore a map of geometry and of the trade-offs between the other constraints, not a source of near-feasible designs.
2. **Nearly every point in the dataset that is "far above our front" has zero rotational transform.**
   - All 1,228 designs with L∇B ≥ 12 at AR ≤ 10 violate something by at least 0.53. Their median |ι|/nfp is 0.035, against a limit of 0.25.
   - All 48 designs with L∇B ≥ 12 at AR ≤ 9 have |ι|/nfp < 0.03 and elongation of about 1.4. These are nearly axisymmetric tori, not stellarators.
   - In the AR 8–10 slice, nfp 3–4 designs with |ι|/nfp < 0.05 have median L∇B of about 10.7. Designs with |ι|/nfp of 0.1–0.3 have medians of 4–7 (fig3).
   - Low rotational transform and high L∇B are the same physical knob: less helical shaping gives longer gradient scale lengths.
3. **Once iota is required, nfp 4–5 have the best L∇B envelope at AR 9–10.5, and nfp 3 has the best one at AR 6–9.**
   - Among designs passing both iota and mirror, nfp 5 reaches L∇B 13.2 at AR 9–10 and 13.7 at AR 10–11. nfp 4 reaches 12.6 and 13.3.
   - nfp 3 tops out at 9.5 and 8.8 in those bins, but it is the only nfp with iota+mirror passes below AR 7, and it reaches 9.9 at AR 7–8 (fig2).
   - The nfp 4/5 designs at AR ≤ 10.5 with L∇B ≥ 11 (n = 150, all from the DESC initial-guess generator) fail on flux compression (median value 1.29 against ≤ 0.9) and on vacuum well (median −0.083 against ≥ 0). They sit right at |ι|/nfp ≈ 0.26.
4. **After QI, vacuum well is the constraint that blocks high-L∇B and low-AR designs; mirror ratio blocks low AR.**
   - Among dataset designs whose HV gain would be ≥ 20, the well is the largest violation for 81%, mirror for 15% and iota for 3%. Ignoring QI gives the same ranking.
   - The well pass rate falls to 0.1–2% for nfp ≥ 3. Among iota-passing designs, the median well goes from −0.02 (nfp 1) to −1.6 (nfp 5, AR < 6).
   - Mirror ≤ 0.25 essentially never passes below AR 8: 0–3% for every nfp. The median edge mirror among iota-passing designs rises from about 0.36 at AR 9–10 to 0.54–0.69 at AR < 6.
5. **"Simple" boundaries have higher L∇B at fixed AR.** In every model, the strongest predictor of L∇B beyond what AR explains is high-order spectral content: energy in |n| ≥ 3 modes and the RMS of the (m ≥ 2 or |n| ≥ 3) coefficients. More of it means lower L∇B.
   - For nfp-3 iota-passing designs, a gradient-boosted model on boundary features plus generation targets explains 81% (cross-validated R²) of the residual L∇B after an AR fit. The |n| ≥ 3 energy fraction alone carries most of that, with a permutation importance of 0.82.
   - The sign is the same within each generator. For rms_hi/a, Spearman ρ is −0.30 (DESC initial guess), −0.77 (NAE) and −0.58 (other).
6. **On our front, the compact end is held by QI, mirror and vacuum well, all three active, with iota and flux compression slack.**
   - At AR 9.5–11, 100% of front points have QI, mirror and well within 0.03 of the limit. Iota is at −0.04 to −0.065 (|ι|/nfp ≈ 0.26) and flux compression is at about −0.09.
   - From AR 13 upward, the well becomes slack (about −0.05) and flux compression becomes active, while QI and mirror stay active.
   - The compact end also has the highest m = 1 helical share and the largest axis excursion relative to minor radius, which is expected because a ∝ 1/AR.
7. **The 5×5 run did not use its extra modes.**
   - In every modes5 evaluation, the m = 5 and |n| = 5 coefficients stay below 1.3e-5 in absolute value (R00 = 1). The m = 4 coefficients reach 1.8e-3. The share of energy in m = 5/n = 5 is about 1e-10.
   - The cause is the finisher's spectrum scaling, exp(−1.5·max(|m|,|n|)). It makes steps in mode-5 coefficients 0.0025× the size of mode-1 steps and 0.22× the size of mode-4 steps.
   - The Test 18 gain (AR 10.18 → 9.53) therefore came from the smaller, more patient step schedule, not from added shape freedom.
   - To test the 5×5 hypothesis properly, lower alpha or give the new modes their own scale.
8. **Our nfp-4 designs sit at higher L∇B than our nfp-3 designs at the same AR, with a smaller axis excursion.**
   - At AR 10.2–11.2, nfp-4 designs with worst ≤ 0.1 (n = 179) have median L∇B 9.26 and maximum 9.41. Feasible nfp-3 designs have median 7.83 and maximum 8.53. That is about 1.4 units (≈ 18%) higher.
   - The nfp-4 axis excursion is 0.114 R00 against 0.262, and m1/m0 is 0.86 against 0.35. Elongation is the same (proxy about 5.4).
   - The nfp-4 residual violations are spread out: the largest is mirror in 67 cases, QI in 62, flux compression in 37 and iota in 13. No single constraint is the wall.

## Q1. L∇B envelope per nfp and the "as-if-feasible" HV

### Dataset counts and HV of each nfp's subset, treated as if feasible (reference (1, 20))

| subset | HV nfp1 | nfp2 | nfp3 | nfp4 | nfp5 |
|---|---|---|---|---|---|
| all designs (no constraints) | 121.9 | 218.2 | 266.3 | 252.7 | **330.7** |
| worst ≤ 0.5 (n = 2323 / 637 / 1088 / 340 / 102) | 93.3 | 133.4 | **177.4** | 141.6 | 121.4 |
| worst ≤ 0.4 (n = 482 / 115 / 200 / 53 / 6) | 83.3 | 102.9 | **157.6** | 119.7 | 102.3 |
| worst ≤ 0.3 (n = 13 / 2 / 18 / 2 / 0) | 56.0 | 80.9 | **105.7** | 80.2 | 0 |
| iota passes | 109.2 | 148.0 | 174.5 | 195.4 | **200.1** |
| iota + mirror pass | 100.4 | 131.2 | 155.3 | **174.5** | 166.9 |
| iota + flux pass | 105.4 | 118.4 | **165.5** | 116.1 | 144.7 |
| iota + well pass | 94.1 | **104.3** | 104.0 | 90.2 | 70.7 |
| iota, mirror, flux, well all pass (all but QI; n = 148 / 12 / 6 / 0 / 0) | 73.1 | 83.2 | **93.2** | 0 | 0 |
| all but QI ≤ 0.2 (n = 893 / 57 / 177 / 5 / 0) | 94.5 | 106.5 | **133.3** | 127.5 | 0 |

For comparison, our feasible nfp-3 set has HV 144.4. Merging dataset subsets into our front, as if those designs were feasible, gives:

| subset merged into our front | HV |
|---|---|
| iota+mirror, nfp 3 | 189.7 |
| iota+mirror, nfp 4 | 187.0 |
| iota+mirror, nfp 5 | 177.8 |
| iota+mirror, all nfp | 205.4 |
| iota+mirror+flux, all nfp | 171.5 |
| all-but-QI ≤ 0.2 | 175.4 |

None of these reaches the roughly 285 that other groups report. Even an optimistic reading of the dataset tops out around 200 unless iota is dropped. Reaching 285 needs designs that the dataset generator did not sample.

Interpretation: without constraints, nfp 5 has the highest envelope, but that comes from zero-iota designs. With any realistic subset of constraints, nfp 3 has the best near-feasible envelope. nfp 4–5 win only on the iota(+mirror) envelope at AR 9–11. The well constraint removes nfp 4–5 almost entirely: 14 and 4 designs pass both iota and well.

### Upper envelope of L∇B by AR bin

The first column of each pair is the p99 over all designs (bins with ≥ 100 designs). The second is the maximum among iota-passing designs.

| AR bin | nfp1 p99 / iota max | nfp2 | nfp3 | nfp4 | nfp5 |
|---|---|---|---|---|---|
| 4–5 | 3.2 / 2.7 (3–6) | 4.7 / 5.2 | 4.7 / 6.6 | 5.4 / 6.6 | 5.5 / 6.1 |
| 5–6 | 3.9 | 5.5 | 6.4 | 7.6 | 6.0 |
| 6–7 | 4.6 / 3.7 | 6.2 / 7.1 | 7.7 / 7.7 | 6.9 / 8.2 | 7.1 / 8.6 |
| 7–8 | 5.3 / 4.3 | 6.8 / 7.5 | 9.2 / 9.9 | 8.2 / 8.2 | 8.2 / 8.3 |
| 8–9 | 6.0 / 5.1 | 9.6 / 7.5 | 11.0 / 10.0 | 10.8 / 11.0 | 9.8 / 10.9 |
| 9–10 | 6.0 / 6.4 | 7.7 / 8.4 | 12.6 / 9.9 | 12.3 / 12.6 | 12.9 / 13.3 |
| 10–11 | 6.4 / 6.5 | 7.7 / 9.4 | 11.9 / 10.2 | 12.4 / 13.3 | 13.9 / 13.7 |
| 11–12 | 7.1 / 7.1 | 8.7 / 8.8 | 9.6 / 11.3 | 11.2 / 9.6 | 11.6 / 9.9 |

Maximum L∇B among designs with worst ≤ 0.4:

| AR bin | nfp1 | nfp2 | nfp3 | nfp4 | nfp5 |
|---|---|---|---|---|---|
| 7–8 | 4.2 | — | 5.3 | — | — |
| 8–9 | 4.4 | 6.9 | 7.5 | 7.1 | 6.8 |
| 9–10 | 5.9 | 7.7 | 9.4 | 9.5 | 8.3 |
| 10–11 | — | 0.0 | 9.0 | — | — |
| 11–12 | 6.2 | 8.0 | 13.0 | — | — |

The near-feasible fronts (worst ≤ 0.4) stop at AR ≥ 7.2 for every nfp. The only point above our front is nfp 3 at AR 11.29, L∇B 13.02 (worst 0.36). Full tables are in `q1_envelope.csv` in the scratchpad. See fig1 and fig2.

## Q2. Which constraints are hard at low AR and at high L∇B

Pass rate (%) per constraint, all nfp:

| AR bin | iota | QI | mirror | flux | well | n |
|---|---|---|---|---|---|---|
| ≤ 6 | 60.7 | 0.0 | 0.8 | 33.5 | 6.4 | 14,920 |
| 6–7 | 45.3 | 0.0 | 1.4 | 35.4 | 4.5 | 14,769 |
| 7–8 | 37.6 | 0.2 | 1.9 | 32.7 | 4.6 | 19,523 |
| 8–9 | 34.0 | 0.4 | 4.0 | 25.9 | 5.9 | 24,545 |
| 9–10 | 28.2 | 0.0 | 14.8 | 22.3 | 5.9 | 55,910 |
| 10–11 | 28.7 | 0.0 | 50.0 | 12.9 | 0.8 | 23,941 |

| L∇B bin | iota | QI | mirror | flux | well | n |
|---|---|---|---|---|---|---|
| ≤ 4 | 40.1 | 0.1 | 6.6 | 27.3 | 7.9 | 55,480 |
| 4–6 | 38.4 | 0.2 | 13.3 | 22.4 | 5.0 | 47,919 |
| 6–8 | 38.7 | 0.0 | 21.9 | 21.2 | 1.6 | 29,830 |
| 8–10 | 30.2 | 0.0 | 22.3 | 21.5 | 1.0 | 16,363 |
| 10–12 | 7.4 | 0.0 | 22.1 | 39.7 | 2.3 | 6,947 |
| 12–14 | 9.1 | 0.0 | 33.3 | 42.1 | 2.9 | 2,063 |
| ≥ 14 | 0.0 | 0.0 | 62.7 | 38.6 | 7.2 | 83 |

Per-nfp highlights (full per-nfp tables are in the `q2.py` output):

- **Mirror at low AR:** the pass rate is ≤ 3% below AR 8 for every nfp. It rises to 25–78% only at AR 10–11.
- **Iota at high L∇B:** nfp 3 passes iota 51–56% for L∇B < 10 but **0.2% at L∇B 10–12 and 0% above 12**. nfp 4/5 still pass 3.5–15% at L∇B 10–14. This is why the iota-passing envelope (Q1) favours nfp 4–5 at AR about 10.
- **Flux compression** gets easier at high L∇B (nfp 3: about 99% pass above L∇B 10). It is hard for nfp 4–5 at L∇B 8–12 (2–24% pass).
- **Vacuum well:** the pass rate is 0.5% (nfp 5), 2.1% (nfp 4) and 2.8% (nfp 3) overall. It is worst at AR 10–11 (0–2%) and at low AR for high nfp. For nfp 3, the median well violation goes from 0.87 at AR 11–13 to 4.3 at AR < 6.

Blockers among designs that would add HV to our front:

| HV-gain threshold | n | largest violation: well / mirror / flux / iota / QI | fail %: iota / QI / mirror / flux / well | min worst |
|---|---|---|---|---|
| ≥ 1 | 99,308 | 62.5 / 23.6 / 7.0 / 4.9 / 2.0 % | 63 / 99.8 / 94 / 72 / 95 | 0.234 |
| ≥ 5 | 69,647 | 60.2 / 23.9 / 8.5 / 5.7 / 1.7 % | 59 / 99.8 / 96 / 66 / 95 | 0.234 |
| ≥ 20 | 5,250 | 81.1 / 14.7 / 0.2 / 3.2 / 0.8 % | 86 / 100 / 97 / 25 / 94 | 0.529 |
| ≥ 50 | 27 (nfp 3/4/5 = 1/10/16) | iota 96%, well 4% | 100 / 100 / 93 / 0 / 85 | 0.951 |

QI fails in essentially all of them, but it is rarely the largest violation (median violation about 0.5). It is the only blocker for 70 designs. Excluding QI, the blocker ranking is well > mirror > flux > iota for moderate gains. For the largest gains it is iota (designs with ι ≈ 0).

## Q3. What correlates with high L∇B at fixed AR

Method: compute the residual L∇B after a per-nfp quadratic fit in AR. Then fit a histogram gradient-boosted regressor (5-fold CV R²) and compute permutation importances (drop in R²) and Spearman ρ with the residual. Note that the generation "settings" columns (torsion, initial-guess mirror and elongation, `qp.*`, `nae.*`, `ig.*`) are **constant** across the dataset: torsion 0, mirror 0.2, elongation 3, AR 10, iota 0.5. They carry no information. The varying targets are `omnigenous_field_and_targets.{aspect_ratio, max_elongation, rotational_transform}`, the target-field mirror ratio (computed from the |B| spline knots, 0.10–0.40), the x_lmn norm, and the generator type (qp, nae, desc_ig, other).

| set | n | CV R² (boundary + targets) | CV R² (+ physics metrics) | top importances (boundary + targets) |
|---|---|---|---|---|
| A: worst ≤ 0.5, all nfp | 4,490 | 0.30 | 0.34 | Efrac_m3 0.35, rms_hi/a 0.18, b/a (m=1) 0.15, Efrac_m4 0.11, target AR 0.11 |
| B: iota and mirror pass, all nfp | 4,974 | 0.56 | 0.58 | Efrac_m3 0.25, AR 0.23, rms_hi/a 0.22, Efrac_n≥2 0.18, axis R excursion 0.17, target iota 0.15 |
| C: nfp 3, iota passes | 36,136 | **0.81** | 0.82 | **Efrac_n≥3 0.82**, rms_hi/a 0.20, Efrac_n≥2 0.16 |

Spearman ρ with the residual L∇B in set C:

| feature | ρ |
|---|---|
| Efrac_n≥3 | −0.66 |
| rms_hi/a | −0.63 |
| \|ι_axis\| | +0.56 |
| elongation | −0.56 |
| generator = qp | +0.54 |
| axis excursion | −0.53 |
| edge mirror | −0.49 |
| Efrac_m2 | −0.48 |
| m1/m0 | +0.35 |
| well | +0.32 |

Linear model, all nfp, iota-passing (n = 57,824, R² = 0.40), standardized coefficients: mirror −0.46, rms_hi/a −0.44, axis excursion −0.31, Efrac_n≥3 −0.20, target max-elongation −0.20, realized elongation +0.22, target mirror +0.34.

Binned effects (set C, mean residual L∇B by quintile):

| feature | quintile ranges → mean residual L∇B |
|---|---|
| elongation | 2.3–3.4: +0.67; 3.4–3.7: +0.56; 3.7–4.4: +0.25; > 4.4: −0.74 |
| edge mirror | < 0.48: +0.3 to +0.4; > 0.575: −0.95 |
| axis excursion | < 0.027: +0.60; > 0.027: −0.4 to −1.0 |
| m1/m0 | < 0.72: −1.04; 1.8–4.5: +0.5 to +0.75 |

Interpretation:
- **Spectral simplicity is the main controllable lever.** At fixed AR, boundaries with less |n| ≥ 3 and m ≥ 2 content have longer gradient scale lengths.
- A **small axis excursion relative to helical (m = 1) shaping**, meaning a large m1/m0, also helps.
- **Lower elongation** helps. Our front sits at elongation (proxy) 5.3–6.0, which is in the low-L∇B part of the dataset distribution.
- **Higher target mirror** helps a little in sets A and B, and in the linear model. Realized mirror is negative in the pooled linear fit, and its sign is mixed across sets.
- The effect of **nfp** is mostly absorbed by the per-nfp AR fit. Raw envelopes show nfp 4–5 ahead at AR about 10 (Q1).
- Caveat: generator type is confounded with spectral content. qp designs have no |n| ≥ 3 content and the highest mean residual (+0.61 for nfp 3). Within-generator correlations keep the same sign for rms_hi/a and Efrac_n≥3, but are weaker for the DESC initial-guess generator.

## Q4. Our front

Constraint activity along the 223 nondominated feasible points. "% active" means violation ≥ −0.02; the median violation follows in parentheses.

| AR bin | n | iota | QI | mirror | flux | well |
|---|---|---|---|---|---|---|
| 9.5–10 | 61 | 0% (−0.065) | 100% (0.008) | 100% (0.004) | 0% (−0.089) | 100% (0.001) |
| 10–10.5 | 45 | 0% (−0.039) | 100% | 100% | 0% (−0.099) | 100% |
| 10.5–11 | 7 | 0% | 100% | 100% | 0% | 100% |
| 11–12 | 13 | 0% (−0.10) | 100% | 100% | 0% | 38% (−0.027) |
| 12–13 | 12 | 0% (−0.17) | 100% | 100% | 0% | 0% (−0.039) |
| 13–14 | 13 | 0% (−0.23) | 100% | 38% | 62% (−0.016) | 0% (−0.050) |
| 14–15 | 25 | 0% (−0.24) | 88% | 72% | 100% (−0.002) | 0% |
| 15–16 | 18 | 0% (−0.24) | 100% | 89% | 78% | 0% |
| 16–17.5 | 29 | 0% (−0.19) | 100% | 17% | 0% (−0.087) | 0% (−0.074) |

Geometry along all 609 feasible designs:
- Medians from compact to large AR: elongation (calibrated proxy) 5.3 → 5.9 (AR 12–13) → 5.2 (AR 13–14) → 5.8. The two branches come from different run families.
- The m1/m0 ratio falls monotonically with AR (Spearman ρ = −0.99: 0.37 at AR 9.5–10, 0.24 at AR 16–17). Most of that is a = R/AR.
- rms_hi/a rises with AR (ρ = +0.99), because the absolute high-order shaping stays nearly fixed while a shrinks.
- The |n| ≥ 3 energy fraction is about 7e-4 at AR < 13 and 3e-4 above. It is tiny everywhere, so our designs are already spectrally simple compared with the dataset. For nfp-3 iota-passing dataset designs, Efrac_n≥3 has median 0 (all qp designs), 75th percentile 3.9e-3 and 95th percentile 3.7e-2.
- Among our feasible designs, the residual L∇B after a linear AR fit (L ≈ 1.215·AR − 4.79, residual std 0.50) correlates with elongation at −0.25, Efrac_n≥2 at −0.45, rms_hi/a at −0.33 and Efrac_m2 at +0.82. The last one mostly separates run families, so treat it with caution.

modes5 (Test 18) compared with the 4×4 descents at the compact end:

| run | n feasible at AR < 10.5 | AR range | L∇B range | rms_hi/a | elongation proxy |
|---|---|---|---|---|---|
| modes5 | 202 | 9.53–10.20 | 7.13–7.72 | 0.214 | 5.32 |
| descent | 78 | 10.22–10.41 | 7.70–7.93 | 0.220 | 5.45 |
| descent2 | 17 | 10.18–10.22 | 7.68–7.71 | 0.218 | 5.42 |

As AR decreases within modes5, the m = 4 energy decreases (ρ(AR, Efrac_m4+) = +0.97). The m = 5/n = 5 values rise slightly but stay about 1e-10 of the energy (finding 7). The compact end is reached with slightly *lower* high-order content and lower elongation. That is consistent with the Q3 trend and suggests the wall is not a lack of high modes.

nfp 4 compared with nfp 3 at AR 10.2–11.2 (medians):

| | nfp 3 feasible (n = 91) | nfp 4, worst ≤ 0.1 (n = 179) |
|---|---|---|
| L∇B | 7.83 (max 8.53) | **9.26 (max 9.41)** |
| AR | 10.31 | 10.80 |
| axis excursion / R00 | 0.262 | **0.114** |
| axis R / Z excursion | 0.063 / 0.254 | 0.016 / 0.114 |
| m1/m0 | 0.35 | 0.86 |
| Efrac_m2 | 0.0048 | 0.0175 |
| elongation proxy | 5.45 | 5.40 |
| violations iota / QI / mirror / flux / well | −0.04 / 0.006 / 0.002 / −0.10 / −0.002 | 0.031 / 0.037 / 0.033 / 0.036 / 0.014 |

The nfp-4 designs buy about 1.4 units of L∇B at fixed AR. They do it through a much smaller axis excursion and more m = 2 shaping, but every constraint is slightly violated (worst 0.027 at best). If all of them were feasible, they would add only HV 0.9 to our front (best single design: 0.66). Our nfp-3 front already reaches L∇B 8.1–8.5 at AR 10.6–11.0, and the step up to L∇B 9.3 at AR 11.6 is close by. The nfp-4 gain matters only if it holds at lower AR, below about 9.5.

## Q5. Do designs far above our front exist in the dataset?

| region | n | nfp | generator | min worst | why they fail |
|---|---|---|---|---|---|
| L∇B ≥ 12, AR ≤ 10 | 1,228 | 3: 590, 4: 163, 5: 474, 2: 1 | NAE 736, desc_ig 490 | 0.53 | median \|ι\|/nfp 0.035 (iota passes in 9%); mirror median 0.38; well |
| L∇B ≥ 12, AR ≤ 9 | 48 | 3: 16, 4: 14, 5: 18 | NAE 47 | 0.84 | iota ≈ 0 for all (max \|ι\|/nfp 0.03), elongation about 1.4 |
| L∇B ≥ 15, AR ≤ 11 | 25 | 3: 6, 4: 5, 5: 14 | NAE 22 | 0.74 | iota ≈ 0 |
| L∇B ≥ 10, AR ≤ 8 | 47 | 3: 17, 4: 15, 5: 14, 2: 1 | NAE 47 | 0.84 | iota ≈ 0, elongation about 1.4 |
| L∇B ≥ 8, AR ≤ 7 | 53 | mostly nfp 3–4 | NAE 50 | 0.82 | iota ≈ 0 |
| L∇B ≥ 20, AR ≤ 8 | 0 | | | | |

- The best "far above" designs that do have iota are nfp 4, DESC initial guess, AR 9.4–9.8, L∇B 12.0–14.2. Their worst is 0.53–0.60, with iota borderline, QI about 0.53, flux 0.2–0.5 and well 0.53–0.6.
- Their generation targets are AR 9.0–9.5, iota 0.76–1.09 (total; |ι|/nfp about 0.24–0.27), max elongation 4.3–4.9 and target mirror 0.13–0.19. Their realized elongation is 4.5–5.0.
- The largest unconstrained L∇B values per nfp, all at worst about 0.95–1.0 with ι ≈ 0:

| nfp | max L∇B | AR bin |
|---|---|---|
| 5 | 23.1 | 9–10 |
| 5 | 20.1 | 8–9 |
| 3 | 17.8 | 8–10 |
| 4 | 15.9 | 8–9 |

- With iota passing, the maximum is 13.7 (nfp 5, AR 10–11) and 9.9 (nfp 3, AR 7–8).

So the dataset contains **no evidence** that designs with L∇B ≥ 12 at AR ≤ 10 can satisfy iota together with the well. Its high-L∇B compact designs are low-shear, nearly tokamak-like shapes. The physically plausible route the dataset points to is nfp 4–5 at AR about 9.5–10.5 with L∇B 12–13.7, where |ι|/nfp sits right at 0.25. Those designs would need flux compression cut from about 1.3 to 0.9 and the well moved from about −0.08 to ≥ 0. The dataset never achieves either at nfp ≥ 4 (iota+well+flux passes at nfp 4: 0 designs).

## Caveats

- **Fidelity:** dataset metrics come from Proxima's forward model at its own settings, while ours are `high_fidelity` evaluations. They are not guaranteed to match exactly. Our earlier fidelity tests showed differences of a few percent in L∇B and QI.
- **Sampling bias:** the dataset was generated to target QI fields with randomized AR (4–12), max elongation (4–7), total iota (0.1–1.5) and target mirror (0.1–0.4). It was not optimized for vacuum well, flux compression, or log10(qi) ≤ −3.5. Pass rates therefore reflect the generator, not physical limits. Our own optimizer reaches well ≥ 0 and log10(qi) ≤ −3.5 routinely at nfp 3. All boundaries are at most 4×4 modes, and AR is capped at about 12.
- **The iota target is total iota, while the constraint is per field period.** Higher nfp needs proportionally more total iota (nfp 4 needs ι ≥ 1.0, nfp 5 needs ι ≥ 1.25). Most targets fall below that, which explains the low iota pass rate for nfp 4–5 (15–16%).
- **Elongation for our designs** is a boundary-only proxy: second moments of the cylindrical-φ cross-sections, scaled by 1.15. It is not the axis-perpendicular ellipse fit in `constellaration`. On 400 dataset designs it has Spearman ρ = 0.85 and Pearson r = 0.80 against `metrics.max_elongation`. Use it for trends only.
- **Spectral features** are normalized by the total non-R00 energy, so m = 1 fractions scale roughly as 1/AR. rms_hi/a and m1/m0 are the less AR-confounded versions.
- **Small samples:** "Near-feasible" counts are small (35 designs at worst ≤ 0.3). The per-bin maxima for those sets are anecdotal.

## Figures

- `fig1_dataset_L_vs_AR_by_nfp.png`: L∇B against AR per nfp (all, iota-passing, worst ≤ 0.5/0.4), with our front and our nfp-4 points.
- `fig2_envelopes_iota_mirror_by_nfp.png`: maximum L∇B per 0.5-AR bin per nfp for iota-passing and iota+mirror-passing designs, against our front.
- `fig3_iota_vs_L_AR8to10.png`: L∇B against |ι|/nfp at AR 8–10 for nfp 3–5. High L∇B lives at ι ≈ 0.
- `fig4_spectral_vs_residual_L.png`: residual L∇B against high-order spectral content (nfp 3, iota passing), coloured by generator.
- `fig5_our_front_active_constraints.png`: violations along our nondominated front, and the elongation proxy for nfp-3 feasible and nfp-4 designs.
