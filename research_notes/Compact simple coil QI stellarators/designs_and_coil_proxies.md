# Compact QI stellarator designs and coil-simplicity proxies

Scope: published quasi-isodynamic (QI) and QI-like designs, with their parameters, and the coil-simplicity proxies used to judge them (L∇B and alternatives). These notes support ConStellaration Benchmark 3: maximize normalized L∇B, minimize aspect ratio, subject to QI and MHD constraints. Physics scaling theory and ML methods are out of scope. No ConStellaration leaderboard submissions are used.

Notation: nfp is the number of field periods, A the aspect ratio, Δ the mirror ratio, ι the rotational transform, ε the maximum elongation, and a the minor radius.

---

## Q1. Which compact QI designs exist, what are their parameters, and which field-period counts do the most compact ones use?

### Takeaway
Reactor-grade QI designs with full coil sets (W7-X, Stellaris, Infinity Two, CIEMAT-QI4/4X) all sit at nfp 4–5 and A ≈ 10. They reach about 1 minor radius of coil-plasma distance only by adding more coils (48–50 nonplanar coils), and none goes below A ≈ 9.8. Compactness has been shown mainly at low nfp: A = 8 for nfp 1 (Jorge 2022) and nfp 3 (Goodman 2023), and A = 6.7 for nfp 2 and 3 (Goodman 2024 compact SQuIDs). Stellaris Appendix C also mentions unpublished-in-detail SQuIDs at A ≈ 8–9. The largest coil-plasma distances in the Kappel database all come from nfp 1–2 QI configurations. Thea Energy's Eos/Helios (A = 6 / 4.5) are quasi-axisymmetric (QA), not QI.

### Cited Findings

**Design table** (numbers as published; "n/r" means not reported in the sources I could access)

| Design (source, year) | Type | nfp | A | Δ (mirror) | ι | Max elongation | Coil numbers / coil-simplicity data |
|---|---|---|---|---|---|---|---|
| W7-X (design, 1990s–2010) | QI-like (HELIAS, optimized, not precisely QI) | 5 | ≈10 (R = 5.5 m, a ≈ 0.53–0.55 m) | n/r in sources found | 5/6 – 5/4 range; ι = 1 at edge in standard config | n/r | 50 nonplanar + 20 planar superconducting coils |
| Jorge et al., "A single-field-period QI stellarator" (JPP 2022) | QI, near-axis direct construction | 1 | 8 | n/r in abstract | n/r | n/r | Coil set found; "significantly fewer coils" than W7-X. In the Kappel database, scaled to a = 1.704 m: L_REGCOIL = 4.713 m, L*∇B = 7.536 m, among the largest in the database |
| Goodman et al., "Constructing precisely QI magnetic fields" (JPP 2023), nfp = 1 | Precise QI | 1 | ≈10 | just below 0.19 | flat, just above 0.601 | just above 5.5 | Kappel DB: L_REGCOIL = 4.156 m, L*∇B = 6.699 m |
| Goodman 2023, nfp = 2 | Precise QI | 2 | 10 | 0.22 | [0.61, 0.62] after island avoidance (initially [0.45, 0.55]) | 6 | Kappel DB: L_REGCOIL = 3.005 m, L*∇B = 5.192 m |
| Goodman 2023, nfp = 3 | Precise QI | 3 | 8 | 0.2 | 0.75–0.8 | 6.2 | Kappel DB: L_REGCOIL = 1.371 m, L*∇B = 2.563 m. The Kappel table lists this entry as NFP = 4, which conflicts with its description |
| Goodman et al. 2024 SQuID (main), "QI stellarators with low turbulence" (arXiv 2024 / PRX Energy) | QI, max-J, ITG-optimized | 4 | ≈10 | ≈0.25 | ≈[0.80, 0.95] (variant [0.86, 0.95]) | "less elongated" than the elongated-QI comparison; no number | Mercier limit ⟨β⟩ ≈ 2.5%. No coil data |
| Goodman 2024 compact SQuIDs | QI, max-J | 2 and 3 | 6.7 | n/r | Avoids low-order rationals inside; approaches 1 at edge (island divertor) | n/r | The nfp = 2 design was additionally optimized with the Kappel L∇B figure of merit. Coils "still underway" |
| Stellaris (Lion et al., Proxima, Fusion Eng. Des. 214, 114868, 2025) | QI (SQuID family), HTS | 4 | 9.8 (R = 12.7 m, a = 1.3 m) | 26.66% (edge) | axis 0.86, edge 0.98 | n/r | 48 nonplanar coils (6 unique per half period). Center filament: min coil-plasma 1.37 m (≈1.05a), min coil-coil 0.67 m, min curvature radius 0.64 m. Winding-pack casing: casing-to-LCFS 1.04–1.11 m, max curvature 1.42–2.12 m⁻¹, max torsion 3.76–5.27 rad/m. Peak conductor field 24.9 T at B_axis = 9 T |
| Stellaris Appendix C variants (2025) | QI (SQuID) | n/r | ≈9 and ≈8 | n/r | n/r | n/r | Described as ideal-MHD stable, "allow for sufficiently distant coils", and "verified for engineering feasibility, though not to the same depth" |
| CIEMAT-QI4 (Sánchez et al., Nucl. Fusion 63, 066037, 2023) | QI | 4 | n/r in sources fetched; QI4X "closely resembles" it | n/r | lowest rational 8/9 near s ≈ 0.7 | n/r | Several filamentary coil sets. Min coil-plasma ≈ 1·a. Kappel DB (CIEMAT QI, nfp 4): L_REGCOIL = 1.413 m, L*∇B = 3.263 m |
| CIEMAT-QI4X (Sánchez et al., arXiv 2512.08825 / Nucl. Fusion 2026) | QI, island divertor | 4 | 10.4 (R = 18.5 m, a = 1.77 m) | n/r | core lowest rational 8/9; edge 4/4 island | κ = 6.9 at φ = 0 | 48 coils. Min coil-plasma ≈ 0.84a (≈1.48 m). Mean coil length 30.7 m. Curvature max 1.85 m⁻¹, mean 0.38 m⁻¹. Mean torsion 0.20 m⁻¹. Min coil-coil 0.84 m (text and table are ambiguous) |
| Type One Energy Infinity Two (JPP 2025 special issue) | QI, max-J | 4 | 10 | n/r | n/r | n/r | 9 T. "Consistent with a set of buildable coils"; no coil numbers found |
| Thea Energy Eos (APS DPP 2024 poster; Nucl. Fusion 2025) | **QA, not QI** | 2 | 6 | n/a | n/r | n/r | Planar coils only: encircling coils give ~97% of the field, plus planar shaping coils. R ≈ 3.24 m, ~5 T |
| Thea Energy Helios (arXiv 2512.08027) | **QA, not QI** | 2 | 4.5 | n/a | n/r | n/r | 12 encircling + 324 planar shaping coils |
| ConStellaration Benchmark 3 baselines (Cadena et al., NeurIPS 2025) | QI-like | 3 | 6.02 / 7.93 / 9.98 / 11.9 | constrained | constrained | n/a | L̃∇B = 2.98 / 5.60 / 8.45 / 11.1. Normalized constraint violation 0.104 / 0.0013 / 0.0 / 0.0021, so only A ≈ 10 is fully feasible |

Sources for the table:
- W7-X: 50 nonplanar + 20 planar coils; ι range 5/6–5/4; ι = 1 at edge in the standard configuration — [IAEA FT/3-5](https://www-pub.iaea.org/MTCD/Meetings/PDFplus/fusion-20-preprints/FT_3-5.pdf); [JSPF PFR 2010](https://www.jspf.or.jp/PFR/PDF/pfr2010_05-S1011.pdf). A ≈ 10, R = 5.5 m, a = 0.55 m — [EPFL W7-X report](https://www.epfl.ch/labs/bpe/wp-content/uploads/2018/10/w7X-report.pdf). Note: A ≈ 10.5–11 is often quoted; the source found gives ≈10, so treat it as approximate.
- Jorge 2022: nfp = 1, A = 8, fewer coils — [arXiv 2205.05797](https://arxiv.org/abs/2205.05797v2)
- Goodman 2023 configuration values — [arXiv 2211.09829 (PDF)](https://arxiv.org/pdf/2211.09829), Sec. IV A–C
- Kappel-database values for Goodman, Jorge, and CIEMAT entries — [Kappel, Landreman, Malhotra, arXiv 2309.11342 appendix table](https://ar5iv.labs.arxiv.org/html/2309.11342). All configurations are scaled to a = 1.704 m and ⟨B⟩ = 5.865 T.
- Goodman 2024 SQuIDs — [arXiv 2405.19860](https://arxiv.org/html/2405.19860v1)
- Stellaris Tables 2, 3, 8 and Appendix C — [Lion et al., FED 2025 (KIT repository PDF)](https://publikationen.bibliothek.kit.edu/1000179851/172386752). nfp 4, A 9.8, and 6 coils per half period are also confirmed in [arXiv 2507.12681](https://arxiv.org/pdf/2507.12681).
- CIEMAT-QI4X — [arXiv 2512.08825](https://arxiv.org/html/2512.08825v1). CIEMAT-QI4 is nfp 4 with several filamentary coil sets — [EFTC 2023 poster](https://indico.global/event/7904/contributions/69676/attachments/33894/63688/EFTC_poster.pdf)
- Infinity Two — [JPP 2025, alpha confinement paper](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/539CC34E6A16076ADE0AF11F474D7E6D/S0022377825000352a.pdf/alpha-particle-confinement-in-infinity-two-fusion-pilot-plant-baseline-plasma-design.pdf). A 2026 Sherwood abstract calls Infinity Two nfp 3, which conflicts with the JPP papers — [Sherwood 2026](https://www.sherwoodtheory.org/sw2026/abstract.php?view=328).
- Thea Eos — [Thea APS DPP 2024 physics poster](https://thea.energy/wp-content/uploads/2024/10/Physics-Design-of-the-Eos-Neutron-Source-Stellarator_APS-DPP-24.pdf); [Nucl. Fusion 2025 (DOAJ)](https://doaj.org/article/33703fca2b2b43aca88411b15b8a65ea). Helios — [arXiv 2512.08027](https://arxiv.org/abs/2512.08027v3)
- ConStellaration baselines — [arXiv 2506.19583v2](https://arxiv.org/html/2506.19583v2), Table 5

**Other findings**
- Goodman 2023: "When optimizing configurations with nfp = 1, we had very good results with every initial condition we tried, but the success of optimizations with nfp > 1 was quite sensitive to the chosen initial conditions." Near-axis constructions also "seem to degrade with increased field-periods." — [arXiv 2211.09829](https://arxiv.org/pdf/2211.09829)
- Goodman 2023 penalizes mirror ratio, elongation, and aspect ratio with "cut-in" thresholds (Δ*, ε*, A*). The final configurations sit right at those thresholds, so the reported A, Δ, ε are design choices, not intrinsic limits. — [arXiv 2211.09829](https://arxiv.org/pdf/2211.09829)
- Goodman 2023 gives the reason elongation is penalized: earlier QI searches "resulted in very elongated flux surfaces." — [arXiv 2211.09829](https://arxiv.org/pdf/2211.09829)
- Kappel 2023: "the largest coil-plasma distances are possible only with a small number of field periods". The two largest separations are single-period configurations, and the nine largest all have nfp 1 or 2. — [arXiv 2309.11342](https://ar5iv.labs.arxiv.org/html/2309.11342)
- Stellaris is a SQuID "adapted to meet the requirements of a power plant by dedicated optimization for larger coil-plasma distances". The paper cites Kappel's L∇B as the cost function that addresses insufficient plasma-coil distance. — [Lion et al. 2025](https://publikationen.bibliothek.kit.edu/1000179851/172386752)
- Stellaris Appendix C: an A = 6 configuration at the same minor radius would give R0 ≈ 7.8 m but only about 3/5 of the fusion power. Lower-A SQuIDs (A ≈ 9, 8) show "slightly higher" neoclassical transport. — [Lion et al. 2025](https://publikationen.bibliothek.kit.edu/1000179851/172386752)
- The ConStellaration dataset samples nfp 1–5 (successful configurations: 15k, 20k, 68k, 27k, 28k). The baselines use nfp 3. — [arXiv 2506.19583v2](https://arxiv.org/html/2506.19583v2)
- Near-axis QI database: more than 800,000 stable, approximately QI vacuum configurations "span a range of field period numbers". The abstract gives no A or coil metrics. — [Rodriguez & Plunk, arXiv 2601.08400](https://arxiv.org/abs/2601.08400)
- Camacho Mata, Plunk & Jorge (JPP 2022) extend first-order near-axis direct construction to stellarator-symmetric QI fields and stress control of axis curvature. This method underlies the Jorge nfp = 1 design. — [arXiv 2206.08417](https://ar5iv.labs.arxiv.org/html/2206.08417)

### Inferences
- **What made the compact ones compact:**
  1. Low nfp (1–3). Each period has more toroidal length to fit the QI "well", so less shaping per unit length is needed.
  2. Precise QI targets that do not chase extreme elongation. Goodman caps ε at about 5.5–6.2.
  3. Accepting moderate mirror ratios (0.19–0.25).
  4. Accepting lower ι for nfp 1–2 (≈0.6 vs ≈0.9 at nfp 4).

  Reactor designs (nfp 4–5) instead buy coil distance with more coils, about 6 unique coils per half period, and stay at A ≈ 10.
- The most compact QI designs with *demonstrated* coil sets in the literature found are still A ≈ 10 (Stellaris, CIEMAT-QI4X). Compact A = 6.7–8 QI designs exist mostly as stage-1 boundaries. Coils for the Goodman A = 6.7 nfp 2 design were "underway" in 2024; I found no published coil set. This is the open niche a student team could contribute to.
- For coil-plasma distance, the evidence favors nfp 1–2, but those designs carry lower ι and reported sensitivity to initial conditions (for nfp > 1). nfp 2–3 looks like the practical compromise; the Goodman A = 6.7 designs use nfp 2 and 3.

### Gaps
- Could not access W7-X mirror ratio and maximum elongation from a primary source. Commonly quoted values, roughly 10% mirror in the standard configuration, are unverified here.
- Stellaris maximum elongation, vacuum well depth, and L∇B value were not given in tabulated form in the text extracted. L∇B appears only normalized in a spider plot (Fig. 4).
- No published L∇B or coil-plasma distance for Infinity Two or for the Goodman 2024 compact SQuIDs.
- Mirror ratio, ι, and coil data for the Jorge 2022 nfp = 1 design could not be pulled from the abstract; the full text is needed.
- Precise CIEMAT-QI4 (2023) aspect ratio was not found; QI4X is A = 10.4.
- Did not find a published "compact nfp = 1 or 2 QI with full coils" from Plunk et al. beyond Jorge 2022 and the Goodman entries.

---

## Q2. How is L∇B defined, what evidence links it to coil-plasma distance and coil complexity, what other proxies exist, and where do they disagree?

### Takeaway
L∇B = √2·B/‖∇B‖_F is strongly predictive of the minimum coil-plasma distance for current-potential (REGCOIL) coils across diverse configurations (R² = 0.941). The fit is also strong for filamentary coils in a single-stage dataset (R² = 0.944), but only when enough coil length is allowed. L∇B is essentially uncorrelated with coil non-planarity in QI coil sets built from ConStellaration boundaries (Spearman ≈ −0.10). Geodesic torsion of field lines predicts non-planarity much better (≈ +0.73). The proxies therefore measure different aspects of "simple coils": distance versus shape.

### Cited Findings

**Definition (Kappel, Landreman, Malhotra, PPCF 66, 025018, 2024; arXiv 2309.11342)**
- L∇B = √2·B / ‖∇B‖_F, where ‖·‖_F is the Frobenius norm of the 3×3 tensor ∇B. The √2 makes L∇B equal the distance R for an infinite straight wire. The proxy is the minimum of L∇B on the last closed flux surface (L*∇B). The authors chose the Frobenius form over B/|∇B|, the curvature radius 1/|b·∇b|, and B/max σᵢ because it uses all components and is rotationally invariant. — [ar5iv 2309.11342](https://ar5iv.labs.arxiv.org/html/2309.11342)
- How "coil distance" was defined: REGCOIL places a uniform-offset winding surface. The regularization is set so B_RMS = 0.01 T, and the offset is increased until max current density ‖K‖∞ = 17.16 MA/m, derived from the ARIES-CS minimum coil spacing of 77.32 cm. That offset is L_REGCOIL. All configurations are scaled to a = 1.704 m and ⟨B⟩ = 5.865 T. — [ar5iv 2309.11342](https://ar5iv.labs.arxiv.org/html/2309.11342)
- Result: L*∇B vs L_REGCOIL gives R² = 0.941 over 45 configurations (QA, QH, QI, tokamak, others). L_REGCOIL spans 0.406–4.713 m. Changing the B_RMS and ‖K‖∞ targets mainly shifts slope and intercept, not correlation strength. Alternative scale lengths correlate more weakly. — [ar5iv 2309.11342](https://ar5iv.labs.arxiv.org/html/2309.11342)
- Outliers and caveats:
  - The ITER-like tokamak falls below the fit.
  - HSX with coil ripple falls above the fit; truncating n > 4 modes moved it closer. L∇B captures large-scale gradients and may miss small-scale features, so ∇∇B was suggested.
  - The CTH high-ι configuration is attributed to VMEC errors.
  - β ≤ 5% only, so high-β behavior is untested.
  - Stage-II solutions are non-unique, so the correlation depends on fixing field-accuracy and current-density targets.
  - The minimum L∇B typically sits on the inboard side of the bean cross-section.

  — [ar5iv 2309.11342](https://ar5iv.labs.arxiv.org/html/2309.11342)

**ConStellaration's normalized version (as implemented)**
- The installed `constellaration` package computes L∇B = B·√(2/‖∇B‖²_F), divides by `Aminor_p`, then takes the minimum over the boundary and multiplies by n_field_periods. The code comment reads: "We normalize by the number of field periods to make the metric independent of the configuration number of field periods." So L̃∇B = Nfp · min(L∇B)/a. — local package source `constellaration/forward_model.py` lines 172–181 and `constellaration/mhd/magnetics_utils.py` lines 58–181 (version installed in the team's `stellarator` conda env). The paper's description matches: "normalized by a/N_fp" — [arXiv 2506.19583v2](https://arxiv.org/html/2506.19583v2)
- The ConStellaration paper footnote derives L∇B ∝ R0/Nfp under a quasi-poloidal single-well assumption. It also states that larger A reduces compactness while helping L̃∇B, which is the core Benchmark 3 trade-off. — [arXiv 2506.19583v2](https://arxiv.org/html/2506.19583v2)

**Filamentary-coil evidence (Kappel, Landreman, Jurašić, Henneberg, arXiv 2602.18974, 2026)**
- QUASR subset: 3,027 single-stage vacuum equilibria with 12 coils each. min(L∇B)/a vs min coil-surface distance/a gives R² = 0.944. Normalizing by R0 instead gives R² = 0.644. The second-gradient scale L∇∇B correlates more strongly, and its minimum is closer to the location of minimum coil distance (mean offset 0.14 vs 0.18 for L∇B). — [arXiv 2602.18974](https://arxiv.org/html/2602.18974v1)
- QH nfp-4 family at A = 8: 42 equilibria, 6 coils per half period, 20 length budgets. At fixed min coil-coil distance, larger min L∇B gives larger min coil-surface distance and lower normal-field error. The trend is monotonic. — [arXiv 2602.18974](https://arxiv.org/html/2602.18974v1)
- Random-boundary set: nfp 2, A = 6, 98 cases, β up to 3.46%. Correlation is only "moderate" with larger scatter. — [arXiv 2602.18974](https://arxiv.org/html/2602.18974v1)
- Breakdown: for short coils (the "high ripple regime"), min coil-surface distance is set by coil length relative to cross-section perimeter, and min L∇B is not correlated with it. Larger L∇B configurations need more total coil length. The authors conclude min(L∇B) "works better as an optimization objective than as a metric for comparing widely different configurations". — [arXiv 2602.18974](https://arxiv.org/html/2602.18974v1)
- Alpha-particle "sweet spot" near min(L∇B)/a ≈ 2.04 for QH. Too low gives coil ripple losses; too high degrades stage-1 quasisymmetry. — [arXiv 2602.18974](https://arxiv.org/html/2602.18974v1)
- APS DPP 2024/2025: targeting L∇B in stage I allows larger coil-plasma distance in stage II, and generally improves both min plasma-coil and min coil-coil distance "at sufficient total coil length". — [APS DPP 2024](https://meetings-archive.aps.org/dpp/2024/jp12/60/); [APS DPP 2025](https://archive.aps.org/dpp/2025/uo05/5)

**Non-planarity evidence where L∇B fails (Pavone, Kwak, Warmer, arXiv 2604.26763, 2026)**
- Dataset: 3,836 QI-like ConStellaration boundaries (vacuum, positive well, nfp 2–5). The analysis uses 3,110 "optimised family" boundaries. Filamentary coils were optimized in SIMSOPT with an augmented Lagrangian, using 7 constraint sets and 54 campaigns: min coil-surface distance, coil-coil distance, max curvature, length target, and B̃_n < 5×10⁻³. The non-planarity metric is σ₃/(σ₁+σ₂+σ₃) of each coil's point cloud. — [arXiv 2604.26763](https://arxiv.org/html/2604.26763)
- Spearman correlation with mean coil non-planarity (nfp-demeaned medians):

  | Predictor | Spearman |
  |---|---|
  | Geodesic torsion | +0.729 (pooled 0.723 over 23,849 coil sets) |
  | Normal curvature κ_n·a | +0.650 |
  | Gaussian curvature \|K\|·a² | +0.464 |
  | Edge mirror ratio | +0.333 |
  | QI residual | +0.229 |
  | Vacuum well | +0.177 |
  | **L∇B** | **−0.099** (IQR −0.147 to −0.003; sign flips with nfp: −0.247 at nfp 2, −0.104 at nfp 3, +0.120 at nfp 4) |

  — [arXiv 2604.26763](https://arxiv.org/html/2604.26763)
- Geodesic torsion also reproduces the non-planarity ranking of precise QA, precise QI at nfp 2 and 3, and W7-X. — [arXiv 2604.26763](https://arxiv.org/abs/2604.26763)

**Other proxies**
- REGCOIL current-potential complexity: winding-surface current-potential methods (Landreman 2017 REGCOIL) are the basis for L_REGCOIL. Carlton-Jones, Paul & Dorland computed shape gradients of REGCOIL complexity metrics with respect to the plasma boundary on a uniformly offset winding surface. They caution that "the complexity of the current potential solution does not necessarily correlate with the complexity of filamentary coils." — [arXiv 2011.03702](https://ar5iv.labs.arxiv.org/html/2011.03702)
- QUADCOIL (Fu et al., Nucl. Fusion 2025) notes that few coil-complexity proxies exist at the equilibrium stage and uses winding-surface methods as the natural proxy (from search summary; not read in full). — [search hit via arXiv 2602.18974 references](https://arxiv.org/pdf/2602.18974)
- Hudson et al. (2018): simultaneous optimization of plasma boundary and filamentary coils, minimizing coil complexity at fixed confinement (cited in Carlton-Jones et al.). — [arXiv 2011.03702](https://ar5iv.labs.arxiv.org/html/2011.03702)
- Proxy coils from equilibria: Rodriguez & Sengupta use contours of constant Boozer toroidal angle as "proxy coils" and find non-planarity "strongly governed by local field properties". No validation numbers are in the abstract. — [arXiv 2604.12339](https://arxiv.org/abs/2604.12339)
- Single-stage coil optimization (Jorge et al. 2023, combining coils with fixed-boundary equilibria) optimizes coils and boundary together. This avoids proxies but costs more compute. — [arXiv 2302.10622](https://arxiv.org/pdf/2302.10622)
- Planar-coil route: Thea Energy's planar coil stellarator (QA) replaces nonplanar modular coils with encircling planar coils plus planar shaping coils, sidestepping non-planarity. — [Nucl. Fusion 2025](https://doaj.org/article/33703fca2b2b43aca88411b15b8a65ea)
- Permanent-magnet approaches: no sources found in this search pass (see Gaps).

### Inferences
- **The proxies measure different things.**
  - L∇B predicts how far coils can sit from the plasma at a fixed field-accuracy and complexity budget.
  - Geodesic torsion and normal curvature predict how non-planar the coils will be.
  - Both matter for "simple coils", but they are nearly independent in QI datasets. A Benchmark-3 design that maximizes L̃∇B is not guaranteed simpler-shaped coils.
- **Interaction of the Nfp factor with the physical meaning.** Kappel's physical evidence relates min(L∇B)/a, without an Nfp factor, to coil distance/a (R² = 0.944 when normalized by a, versus 0.941 in absolute units for the scaled database). Benchmark 3 multiplies by Nfp. At fixed A, a higher-nfp design can score the same L̃∇B with a physically smaller coil distance/a. Kappel's nfp trend points the same way: the largest distances are at nfp 1–2. Teams should report raw min(L∇B)/a alongside L̃∇B to avoid over-crediting high-nfp designs. This is my reasoning from the code and the cited scalings, not a published result.
- In the ConStellaration baselines L̃∇B rises almost linearly with A (2.98 at A = 6 to 11.1 at A = 11.9), consistent with L∇B ∝ R0/Nfp. A contribution would therefore be designs that sit *above* this line, with more L̃∇B per unit A.
- L∇∇B and geodesic torsion are cheap additional stage-1 diagnostics a team could compute and report alongside L̃∇B.

### Gaps
- Did not find a quantitative paper on Paul/Landreman coil-sensitivity (shape-gradient) metrics vs. filamentary coil outcomes beyond the Carlton-Jones caveat.
- Did not retrieve Landreman 2017 REGCOIL complexity metrics (e.g., max K, ∫K²) directly. They are cited indirectly.
- No sources retrieved in this pass on permanent-magnet stellarator coil-simplicity proxies (e.g., Zhu/Hammond/Landreman permanent magnet work, MUSE). Flag as unresearched.
- The QUADCOIL details were not read in full.

---

## Q3. Has anyone designed actual coils for ConStellaration-style designs and compared them to L∇B?

### Takeaway
Yes, as of 2026. Pavone et al. (2026) optimized tens of thousands of SIMSOPT filamentary coil sets for about 3,100 ConStellaration QI boundaries and found L∇B nearly uncorrelated with coil non-planarity. Proxima's CoilStellaration (Nov 2026) released 177k filamentary coil sets for ConStellaration equilibria. However, its abstract reports no L∇B correlation. The original ConStellaration paper did no quantitative coil design.

### Cited Findings
- The ConStellaration paper (NeurIPS 2025) covers stage 1 only. Fig. 9 shows "a representative coilset" for the simple-to-build problem without coil metrics. It states that L̃∇B "has proven effective" for guiding toward simpler coils, citing Kappel. — [arXiv 2506.19583v2](https://arxiv.org/html/2506.19583v2)
- Pavone, Kwak, Warmer (arXiv 2604.26763, v1 Apr 2026, v6 Oct 2026): SIMSOPT augmented-Lagrangian coils for ConStellaration QI boundaries (nfp 2–5). L∇B Spearman with non-planarity ≈ −0.10, with sign changing across nfp. Geodesic torsion is ≈ +0.73. — [arXiv 2604.26763](https://arxiv.org/html/2604.26763)
- CoilStellaration (Ewers, Cadena, Foppiani, Merlo; arXiv 2611.12689, 9 Nov 2026; NeurIPS 2026 E&D track): 177,000 optimized filamentary coil sets targeting ConStellaration equilibria across multiple coil counts and nfp. The benchmark conditions on coil-coil spacing, coil-plasma spacing, curvature, and field error. "Magnetic-field accuracy is the main challenge"; learned models rarely meet all requirements, and using learned coils to initialize conventional optimization speeds early progress. The abstract reports no L∇B or A correlation. — [arXiv 2611.12689](https://arxiv.org/abs/2611.12689)
- Augmented-Lagrangian coil optimization (arXiv 2507.12681) re-optimized Stellaris coils as a benchmark. One 6-coil-per-half-period alternative roughly halved field error versus Proxima's coils and reduced linear forces on HTS turns by 29%; a 5-coil alternative was also found. The Stellaris min casing-to-casing distance is 11.4 mm. — [arXiv 2507.12681](https://arxiv.org/pdf/2507.12681) (from search summary)

### Inferences
- The CoilStellaration dataset, if it releases per-equilibrium coil metrics, is the natural place for the team to test L̃∇B against actual achievable coil-plasma distance on Benchmark-3-like boundaries. That analysis does not appear to be published yet.
- The two 2026 studies are complementary. Kappel 2026 shows L∇B predicts distance given enough coil length; Pavone 2026 shows L∇B does not predict non-planarity in QI. Neither alone validates L̃∇B as a full "coil simplicity" score for Benchmark 3.

### Gaps
- Full CoilStellaration paper body not read. It may contain L∇B correlations not in the abstract; check the PDF at [arXiv 2611.12689](https://arxiv.org/pdf/2611.12689).
- No published REGCOIL or FOCUS study specifically on ConStellaration boundaries was found.

---

## Q4. What trade-offs do the Stellaris and Goodman papers report between compactness, coil distance, MHD stability (vacuum well), and QI quality?

### Takeaway
- Improving stability or turbulence costs QI precision: the SQuID QI penalty is about 1.6× that of the elongated QI comparison.
- Max-J and well-based stability comes with a lower Mercier β limit for the low-flux-compression SQuID (≈2.5% vs ≈8.5%).
- Coil distance is bought with design effort (dedicated L∇B optimization in Stellaris) and coil count.
- Lower A costs fusion power at fixed a and slightly raises neoclassical transport.
- CIEMAT notes that closer coils are simpler but raise ripple and fast-ion losses.

### Cited Findings
- **Goodman 2024 SQuID vs elongated-QI configuration**
  - SQuID has lower ITG growth rates and heat fluxes ("the clear winner overall").
  - Its QI penalty is about 1.6× higher, "the cost of reduced flux compression".
  - It needs a_min ≳ 0.6 m for ≤1% fast-ion loss (vs ≳ 0.5 m).
  - Its Mercier-stable limit is ⟨β⟩ ≈ 2.5% vs ≈ 8.5%.
  - Max a_min|∇s| in bad-curvature regions is about 2.3 vs 6.5.

  — [arXiv 2405.19860](https://arxiv.org/html/2405.19860v1)
- **Goodman 2024 compact A = 6.7 nfp 2 and 3 SQuIDs** "have excellent QI quality and negligible fast-ion losses at reactor scales". Compactness is called "an attractive feature in an economical fusion reactor". The nfp-2 design was co-optimized with the Kappel L∇B figure of merit. — [arXiv 2405.19860](https://arxiv.org/html/2405.19860v1)
- **Goodman 2023**
  - Penalizes mirror ratio and elongation because "excessive values of these quantities are undesirable for fusion reactors", with high mirror ratio implying inefficient use of magnetic energy.
  - States that "it is not clear what exactly the highest allowable value of these parameters should be."
  - The nfp 2 and 3 configurations become maximum-J when enough pressure is added.
  - Fast-particle losses rise near β ≃ 1% before max-J sets in.

  — [arXiv 2211.09829](https://arxiv.org/pdf/2211.09829)
- **Stellaris**
  - W7-X cannot simply be scaled up because of fast-particle losses, "insufficient plasma-coil distance to fit a neutron shield and blanket at moderate machine sizes", and ITG transport.
  - Stellaris addresses coil distance via L∇B-based optimization, giving a min coil-plasma distance of 1.37 m ≈ 1.05a at A = 9.8.
  - Plasma-coil distance has "been identified as the primary constraint on overall machine size".
  - Blanket and shield thickness vary with the local coil distance; the minimum inboard neutron shield thickness is 100 mm.
  - Listed future work includes using "a lower aspect ratio stellarator configuration to optimize economic viability" and "optimization of a Stellaris device to maximize coil tolerances, assessing trade-offs with plasma performance".

  — [Lion et al. 2025](https://publikationen.bibliothek.kit.edu/1000179851/172386752)
- **Stellaris peak field**: peak coil field "can vary significantly from one coil to the next, especially in QI stellarators with a large magnetic mirror ratio". Peak is 24.6 T on coil 0 vs 19.5 T on coil 5. This is a mirror-ratio vs coil-engineering trade-off. — [Lion et al. 2025](https://publikationen.bibliothek.kit.edu/1000179851/172386752)
- **Stellaris Appendix C**: A ≈ 8–9 SQuIDs keep minimal Shafranov shift, fast-particle confinement, low bootstrap current, max-J, ideal-MHD stability, and "sufficiently distant coils". Neoclassical transport is slightly higher. — [Lion et al. 2025](https://publikationen.bibliothek.kit.edu/1000179851/172386752)
- **CIEMAT-QI4X**
  - Closer coils are simpler (lower curvature and torsion, better coil-coil spacing) but raise coil ripple and fast-ion losses; 0.84a was a compromise.
  - A magnetic well alone was judged insufficient for stability, so the Mercier criterion was added.
  - Tighter ι control conflicts with omnigenity and required iterative re-optimization.

  — [arXiv 2512.08825](https://arxiv.org/html/2512.08825v1)
- Landreman (APS DPP 2023) gives about 1.5 m between the LCFS and the nearest coil as a rough reactor requirement for blanket and shield (search summary). — [Landreman 2023 APS DPP slides](https://terpconnect.umd.edu/~mattland/assets/presentations/Landreman_2023_APS-DPP_Theory_supporting_FPP_v04.pdf)

### Inferences
- In Benchmark 3 terms:
  - The vacuum-well and flux-compression constraints (the SQuID-style turbulence proxy) push toward designs with a larger QI residual and lower Mercier margin.
  - Compactness pushes L̃∇B down roughly in proportion to A.
  - The published reactor designs resolve this at A ≈ 10 with nfp 4. Goodman's A = 6.7 nfp 2–3 SQuIDs are the best published evidence that QI + well + low flux compression is achievable at A < 7. Their coils are not yet published.
- In physical units, Stellaris (≈1.05a) and CIEMAT-QI4X (≈0.84a) provide calibration points for coil-plasma distance/a in successful QI reactor designs. A coil-plasma distance near 1·a is a sensible "simple enough" target when comparing compact designs.

### Gaps
- The PRX Energy published version of Goodman 2024 may contain updated numbers for the compact SQuIDs (mirror ratio, well depth, L∇B). Only the arXiv v1 was read.
- Stellaris vacuum well depth and elongation were not found in the extracted text.
- Quantitative Stellaris Appendix C numbers (L∇B, coil distance for the A ≈ 8–9 variants) are shown only in a normalized spider plot, and no values were extracted.
