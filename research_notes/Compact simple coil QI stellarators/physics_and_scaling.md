# Physics and Scaling of Compact, Large-L∇B Quasi-Isodynamic Stellarators (ConStellaration Benchmark 3)

Scope: physics trade-offs and scaling laws that set how compact (low aspect ratio A) a QI stellarator can be and how large its normalized magnetic gradient scale length L̃∇B can be, under the Benchmark 3 constraints. ML methods and company designs are covered only as data points.

Benchmark 3 definitions, checked against the installed `constellaration` 0.2.6 source (forward_model.py, problems.py, mhd/magnetics_utils.py, mhd/turbulent_transport.py; public repo [proximafusion/constellaration](https://github.com/proximafusion/constellaration)):

| Quantity | Exact definition in code | Constraint |
|---|---|---|
| Aspect ratio A | VMEC `aspect` = R0/a | objective (minimize) |
| L̃∇B | `N_fp × min over LCFS of [L∇B / a]`, with L∇B = √2·B/‖∇B‖_F and a = VMEC `Aminor_p` | objective (maximize) |
| Edge ι per field period | `|iota(ρ=1)| / N_fp`, evaluated from VMEC `iotaf` | ≥ 0.25 |
| QI residual | sum of squared residuals from `qi.quasi_isodynamicity_residual` on a Boozer transform (default Boozer surface s = 1.0) | log10 ≤ −3.5 |
| Edge mirror ratio Δ | (B_max − B_min)/(B_max + B_min) on the ρ = 1 surface | ≤ 0.25 |
| Flux compression in bad curvature | ⟨Θ(B×κ·∇α)·|∇r|²⟩ (Jacobian-weighted surface average), with |∇r|² = a²|∇s|²/(4s), evaluated at s = 0.49 (ρ = 0.7) | ≤ 0.9 |
| Vacuum well W | [V′(0) − V′(1)]/V′(0), where V′ = dV/ds (same as simsopt) | ≥ 0 |

Default thresholds in code: ι/N_fp ≥ 0.25, log10 QI ≤ −3.5, Δ ≤ 0.25, flux compression ≤ 0.9, W ≥ 0. These match the user's statement.

---

## Q1. How do the minimum achievable aspect ratio and L∇B of QI configurations scale with nfp, elongation, mirror ratio, rotational transform, and axis shape? What does near-axis QI theory say about the limits?

### Takeaway
Near-axis theory gives one robust scaling: the error between an ideal first-order QI field and the real finite-A equilibrium grows like δB ≈ C_g (N·a/R)² = C_g (N/A)². So at a fixed level of QI error, A_min ∝ N·√C_g, which makes compact designs easier at low N. The geometric constant C_g is set by axis shape (torsion, curvature, inclination) and elongation. QI also requires the axis curvature to vanish at every |B| extremum. Torsion grows with N, and QI quality improves with more elongation and more mirror ratio, but these are exactly the levers that the Benchmark 3 constraints cap (Δ ≤ 0.25) or penalize (flux compression, L∇B).

### Cited Findings
**Axis-shape requirements (zeroth/first order)**
- For poloidal closure of |B| contours, a first-order QI axis must have flattening points (zero curvature) at the field-strength extrema. In the "(i,j)" classification, i is the zero order at maxima and j at minima. The standard construction is (2,3), which gives a "half-helix" axis (per-period helicity m = 1/2; half-integer helicity needs i + j odd). Signed curvature is odd about field minima (an inflection) and torsion is even. — [Plunk & Rodríguez 2025, arXiv:2508.12820](https://arxiv.org/html/2508.12820)
- Example (2,3) axis family for N ≥ 2: κ = κ₁ cos²(Nℓ/2) sin(Nℓ/2) sin(Nℓ) and τ = τ₀ + τ₁ cos(Nℓ). Closure fixes two of the three constants, leaving τ₁ free. On-axis field: B₀ = 1 + Δ cos(Nℓ) + (Δ/4) cos(2Nℓ), with Δ = 0.25 in their survey. — [Plunk & Rodríguez 2025](https://arxiv.org/html/2508.12820)
- Curvature zeros must sit at the extrema of B₀ and must be of odd order, matching the zeros of the function d. In cylindrical Fourier form, a zero-curvature point imposes linear constraints such as R₂ = R₀ (order 0), z₃ = 2z₁ (order 1), and R₄ = 5R₀ (order 2). Quote: "control of the curvature and torsion is crucial to obtain omnigenous configurations." — [Camacho Mata, Plunk & Jorge 2022, JPP, arXiv:2206.08417](https://ar5iv.labs.arxiv.org/html/2206.08417)
- In a constant-torsion axis family with max curvature fixed (κ = 3), "maximum torsion grows linearly with field period number." Torsion growth with N is identified as "an underlying reason for the deterioration of the approximation at finite aspect ratio." Also: "Finding closed curves with torsion and curvature remaining under a certain value gets more difficult when increasing the number of field periods." — [Camacho Mata et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.08417)
- Approximate N-scaling symmetry: with φ̂ = Nφ, τ̂ = τ/N and ῑ₀ = ι₀/N, the σ-equation is invariant, so σ (and hence elongation) is roughly N-independent if κ and τ/N are held fixed per period. This holds only for weakly non-planar axes. Torsion is observed to increase with N, and τ₁ is normalized by N. — [Plunk & Rodríguez 2025](https://arxiv.org/html/2508.12820)

**Compactness scaling (the key quantitative law)**
- Field truncation error: δB ≈ C_g (aN/R)². C_g is a geometric constant that depends on shaping (ρ₀, τ₁, …) and is "largely independent of R/a and N for N ≥ 2". The paper concludes "low field period configurations can be designed more compactly." N = 1 (Möbius) scales worse than N ≥ 2, with absolute δB between the N = 2 and N = 3 values. The authors state the most compact designs will likely not be N = 1. — [Plunk & Rodríguez 2025](https://arxiv.org/html/2508.12820)
- Independent confirmation: across 1680 half-helicity QI near-axis fields run in VMEC at r/R = 0.07 (A ≈ 14), δB/N² collapses the data from different N onto one curve (empirical quadratic N scaling), and the near-axis estimate correlates with VMEC at > 0.99. — [Rodríguez & Plunk 2025, "Near-axis measures of QI configurations", arXiv:2505.02465](https://arxiv.org/html/2505.02465v1)
- The RMS difference between near-axis and VMEC fields scales as 1/A² and grows with N at fixed A. At fixed A, ε_eff was "optimal, below 8×10⁻³, for two field periods" and "significantly worse" for N = 4. Quote: "Configurations with more field periods have proved challenging to find, even when using the optimiser." — [Camacho Mata et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.08417)
- Sharp toroidal derivatives d/dφ ∼ 1/ε invalidate the expansion, "especially felt at high field period number". "Smoothness is especially important for obtaining solutions that remain accurate at lower aspect ratio." — [Camacho Mata et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.08417)
- The singularity radius (minimum aspect ratio at which near-axis surfaces stop being nested) at first order is r̂_c(φ) = 1/(κ√(X₁s² + X₁c²)). Large axis curvature κ or large first-order shaping X₁ therefore lowers the reachable minor radius, i.e. raises the minimum A. — [Landreman 2021, "Figures of merit for stellarators near the magnetic axis", JPP, arXiv:2012.00865](https://ar5iv.labs.arxiv.org/html/2012.00865)
- Second-order omnigenizing shaping (needed for low ε_eff) lowers the shaping-limited aspect ratio A_c. A_c ranged 1.59–5.84, and A_w (the largest A at which secondary ripple wells first appear) ranged 12.9–37.0 across their benchmark set. — [Rodríguez & Plunk 2025](https://arxiv.org/html/2505.02465v1)

**Elongation**
- First-order elongation: E = ½(ρ + √(ρ² − 4)), with ρ = ē + (1 + σ²)/ē, equivalently ρ = 2/sin 2e with E = tan e. Existence requires ρ² ≥ 4(1 + σ²), which fails for large σ (large axis torsion or large ē). — [Plunk & Rodríguez 2025](https://arxiv.org/html/2508.12820)
- With ē = B₀d̄²/2, Eq. 50 gives e = ē/2 + (1 + σ²)/(2ē) + √(ē²/4 + (σ² − 1)/2 + (1 + σ²)/(4ē²)). "d̄ cannot be made too small without causing large elongation." Examples: N = 2, A = 10 → e_max = 4.4; N = 3, A = 20 → e_max = 3.2. Quote: "high elongation is not a necessary feature of QI stellarators." — [Camacho Mata et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.08417)
- In global optimization, increasing elongation reduces the first-order correction B₁ and improves QI quality, but adds shaping that complicates coils. Exact QI near the axis is approached only with infinitely elongated surfaces. — [Goodman et al. 2023, "Constructing precisely QI magnetic fields", JPP, arXiv:2211.09829](https://ar5iv.labs.arxiv.org/html/2211.09829)
- "QI field is easier to achieve at large aspect ratio, highly elongated surfaces, and large mirror ratios." — [Cadena et al. 2025, ConStellaration, arXiv:2506.19583](https://arxiv.org/html/2506.19583v2)

**Rotational transform and axis helicity**
- In a constant-torsion family with κ₀ = 1.6, the per-period transform ι/N = 0.4565 (N = 2), 0.4674 (N = 3), 0.4707 (N = 4). Omnigenity of the buffer regions requires ι to be "nearly rational", via |Nm/ι − 1| ≪ 1 (Eq. 46). Quoted examples: N = 2, A = 10 has ι = 0.107; N = 3, A = 20 has ι = 0.375 (0.125 per period). — [Camacho Mata et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.08417)
- The buffer-region (zeroth-order) contribution to ε_eff vanishes when ι₀ = M (no buffers needed). It scales roughly as δ^(u/2 + v + 1) with buffer width δ. — [Rodríguez & Plunk 2025](https://arxiv.org/html/2505.02465v1)
- Optimized QI configurations tend to converge onto low-order rational ι profiles. The N = 1 case had almost zero shear. — [Goodman et al. 2023](https://ar5iv.labs.arxiv.org/html/2211.09829)

**Data points (A, N, Δ, ι, elongation)**

| Source | N | A | Δ | ι (ι/N) | Max elongation | Notes |
|---|---|---|---|---|---|---|
| [Goodman et al. 2023](https://ar5iv.labs.arxiv.org/html/2211.09829) | 1 | ≈10 | ≈0.19 | ≈0.60 (0.60) | ≈5.5 | — |
| [Goodman et al. 2023](https://ar5iv.labs.arxiv.org/html/2211.09829) | 2 | 10 | 0.22 | 0.61–0.62 (≈0.31) | 6 | — |
| [Goodman et al. 2023](https://ar5iv.labs.arxiv.org/html/2211.09829) | 3 | 8 | 0.20 | 0.75–0.80 (0.25–0.27) | 6.2 | — |
| SQuID, [Goodman et al. 2024 PRX Energy, arXiv:2405.19860](https://arxiv.org/html/2405.19860v1) | 4 | ≈10 | ≈0.25 | 0.80–0.95 (0.20–0.24) | "less elongated" | — |
| SQuID compact variants, [same source](https://arxiv.org/html/2405.19860v1) | 2, 3 | 6.7 | — | — | — | — |
| Sánchez et al. 2023, NF ([arXiv:2212.01143](https://arxiv.org/pdf/2212.01143); figures as summarized in search results) | 4 | 9.94 | — | — | — | "significant magnetic well" |
| ConStellaration baselines, ALM-NGOpt ([Cadena et al. 2025](https://arxiv.org/html/2506.19583v2)) | — | 6.02 / 7.93 / 9.98 / 11.9 | — | — | — | L̃∇B = 2.98 / 5.60 / 8.45 / 11.1; normalized constraint violation 0.104 / 0.0013 / 0.0 / 0.0021 |

The 0.104 violation at A ≈ 6 means the baseline was infeasible there. All baselines start from a rotating ellipse with up to 4 poloidal and toroidal Fourier modes (80 DOF) ([Cadena et al. 2025](https://arxiv.org/html/2506.19583v2)).

### Inferences
- Take δB ∝ (N/A)² and treat the QI residual as roughly ∝ δB² (the residual is a squared field mismatch). A fixed target of log10 QI ≤ −3.5 then gives A_min ∝ N·C_g^{1/2}. Naively, going from N = 3 to N = 4 raises the minimum A by a factor of about 4/3, e.g. from A ≈ 9.5 to about 12.7 if C_g stayed the same. Moving to N = 4 can only beat this if the extra freedom lowers C_g (gentler axis, smaller torsion per period, better elongation profile), or if the objective rewards N enough to compensate (see Q2). This is my extrapolation; neither paper evaluates the ConStellaration QI residual.
- Benchmark 3 caps Δ ≤ 0.25 and penalizes elongation through flux compression and L∇B. That removes the two easiest ways to get good QI at low A (large mirror ratio, large elongation). The binding limit on compactness should therefore be QI quality, set by axis shape (curvature and torsion per period) and second-order shaping.
- The ConStellaration baselines show L̃∇B/A rising from 0.50 (A = 6) through 0.71, 0.85 to 0.93 (A = 12). The team's N = 3 designs (A 9.5 → L̃ 7.1, ratio 0.75; A 17 → 16.4, ratio 0.96) sit near the same curve. This suggests the team is on a front similar to the baseline's, and that compactness costs L̃∇B faster than linearly below A ≈ 10.

### Gaps
- I did not read Plunk, Landreman & Helander 2019 ("Direct construction III") or the Camacho Mata & Plunk 2023 axis-helicity paper in full. The curvature-zero requirement above is cited from the later papers that restate it.
- No paper found gives a closed-form A_min(N, Δ, E) for QI at a fixed residual. The constant C_g is only characterized numerically.
- Jorge et al. 2022 ("Single-field-period QI stellarator") was not fetched, so its A and elongation numbers are not included.
- I found no published scan of the ConStellaration QI residual against N at fixed A.

---

## Q2. What is L∇B exactly, how is it normalized, what shape features control it, and how does it scale with A and nfp?

### Takeaway
L∇B = √2·B/‖∇B‖_F (Frobenius norm of the 3×3 gradient tensor). It equals the distance to a straight wire and correlates with REGCOIL plasma-coil distance at R² ≈ 0.94. Benchmark 3 uses L̃∇B = N_fp·min_LCFS(L∇B)/a, and the factor N_fp is there because Proxima expects L∇B ∝ aA/N_fp in QI/QP configurations. The minimum usually sits on the inboard, bean-shaped side, where boundary curvature is highest. L∇B rises with A and falls with strong shaping, and raising it costs symmetry quality.

### Cited Findings
**Definition and normalization**
- L∇B = √2 B/‖∇B‖_F (Eq. 12). In vacuum ∇B is symmetric, and the definition is equivalent to √2 B/(Σσᵢ²)^{1/2} over the singular values. The √2 makes L∇B equal the distance R to an infinite straight wire. Simpler candidates (B/|∇B|, field-line radius of curvature) correlate less well. — [Kappel, Landreman & Malhotra 2024, PPCF 66, 025018, arXiv:2309.11342](https://ar5iv.labs.arxiv.org/html/2309.11342)
- The near-axis ∇B tensor contains the term s_G B₀ κ (tn + nt) (axis curvature) and s_G B₀′/ℓ′ tt (the on-axis field variation, i.e. the mirror term), plus nn, bn, nb and bb terms built from X₁, Y₁, X₁′, Y₁′, τ and ι. Normalized L∇B = 1 for a purely toroidal field. "Strongly shaped configurations give smaller normalized scale lengths." The second-derivative scale length is L∇∇B = √(4B/‖∇∇B‖). — [Landreman 2021](https://ar5iv.labs.arxiv.org/html/2012.00865)
- Benchmark 3 code: `minimum_normalized_magnetic_gradient_scale_length = min(L∇B/Aminor_p) × n_field_periods`, evaluated on a (θ, φ) grid over one half-period of the LCFS. Code comment: "In a QI/QP configuration, the magnetic gradient scale length should scale proportionally to (a A)/n_field_periods … We normalize by the number of field periods to make the metric independent of the configuration number of field periods." — [constellaration source, forward_model.py](https://github.com/proximafusion/constellaration); restated as "L∇B ∝ R0/N_fp = aA/N_fp" in [Cadena et al. 2025](https://arxiv.org/html/2506.19583v2)

**Correlation with coil distance**
- Over more than 40 configurations (QA, QH, QI, tokamak, others) scaled to a = 1.704 m and ⟨B⟩ = 5.865 T, min L∇B correlates with REGCOIL plasma-coil separation at R² = 0.941. L_REGCOIL varies by about 10× across the set. — [Kappel et al. 2024](https://ar5iv.labs.arxiv.org/html/2309.11342)
- Across 3027 QUASR vacuum QA/QH equilibria with filamentary coils, min L∇B correlates with coil-surface distance at R² = 0.944 when normalized by a, but only R² = 0.644 when normalized by R0. L∇∇B correlates even better. The location of the L∇B minimum predicts where coils come closest: 44.5% of cases within Δ < 0.1, against 3.05% for random points. — [Kappel, Landreman, Jurašić & Henneberg 2026, arXiv:2602.18974](https://arxiv.org/html/2602.18974v1)

**Shape features that set L∇B**
- The minimum of L∇B (and of the alternative scale lengths) sits on the inboard side of the bean-shaped cross-section, where coil engineering is hardest. This coincides with the smallest radius of curvature. Configurations with 1–2 field periods have the largest plasma-coil distances: the top 9 by separation over minor radius all have N = 1 or 2. — [Kappel et al. 2024](https://ar5iv.labs.arxiv.org/html/2309.11342)
- Analytic limit, purely toroidal axisymmetric field: min(L∇B)/a = R0/a − 1 = A − 1. "min(L∇B) scales with aspect ratio for near-axisymmetric equilibria." — [Kappel et al. 2026](https://arxiv.org/html/2602.18974v1)
- In a QH N = 4 DESC scan at A = 8, raising min L∇B by penalizing max(a‖∇B‖_F/B − C, 0)² worsened quasisymmetry error (a Pareto trade-off). It also increased the curvature of the bean tips, so "surface curvature does not track the B scale length" one-to-one. Random N = 2, A = 6 boundaries gave min L∇B/a of 0.56–2.95. — [Kappel et al. 2026](https://arxiv.org/html/2602.18974v1)
- Named configurations from Appendix D (L* = min L∇B on the LCFS, all scaled to a = 1.704 m): W7-X without ripple, N = 5, L* = 2.19 m; W7-X standard with ripple 3.47 m; HSX, N = 4, 4.91 m; NCSX LI383, N = 3, 1.93 m; ARIES-CS, N = 3, 1.84 m; Landreman–Paul precise QH, N = 4, 3.54 m; QH with well 3.19 m; precise QA, N = 2, 5.30 m; QA with well 5.55 m. — [Kappel et al. 2024](https://ar5iv.labs.arxiv.org/html/2309.11342)

### Inferences
- Converting Kappel's table into the Benchmark 3 metric (L̃ = N·L*/a, with a = 1.704 m) gives: W7-X (no ripple) ≈ 6.4; HSX ≈ 11.5; NCSX ≈ 3.4; ARIES-CS ≈ 3.2; LP-QH ≈ 8.3 (with well ≈ 7.5); LP-QA ≈ 6.2 (with well ≈ 6.5). These are my calculations. They are only indicative, because Kappel evaluated L∇B on VMEC equilibria and the grid may differ.
- Adding a magnetic well slightly lowered L∇B for the QH case (3.54 → 3.19 m) and slightly raised it for the QA case (5.30 → 5.55 m). So a vacuum well is not intrinsically fatal to L∇B, but it can cost about 10%.
- An axisymmetric upper bound is L̃ = N(A − 1), i.e. 25.5 for N = 3 at A = 9.5. The team's 7.1 is about 3.5× lower, so 3D shaping, not toroidicity, sets L∇B in these designs.
- Why N = 4 can help on L̃∇B: the metric multiplies by N. If the binding scale is set by toroidal variation (period length ≈ 2πR/N, mirror term B₀′ ∼ ΔBN/R, torsion growing ∝ N), then L ∝ R/N and L̃ ∝ A, roughly independent of N, which is the benchmark's design intent. If the binding scale is instead set by cross-sectional shaping (the inboard bean curvature, ∼ a), then L/a is O(1) and independent of N, so L̃ ∝ N. In that case N = 4 earns a 4/3 bonus over N = 3 at the same cross-section shape. Compact designs (low A) are more likely in the cross-section-limited regime, which would explain why N = 4 looks promising at the compact end despite the harder QI scaling in Q1.
- Mirror-term bound: from the tt component, ‖∇B‖ ≳ |dB/dℓ| ∼ ΔB·N/R. Alone this gives L ≲ √2 R/(NΔ), i.e. L̃ ≲ √2 A/Δ ≈ 5.7A at Δ = 0.25. That is far above observed values, so the mirror ratio is not what limits L̃∇B directly. Curvature and shaping of the boundary are.
- Practical levers that raise min L∇B: reduce inboard bean indentation and concavity; avoid sharp boundary curvature at the toroidal location of the minimum (often the bean-shaped section at φ = 0, the B_max straight section in QI); keep axis curvature low (∇B ⊇ B₀κ term, so L ≲ 1/κ on axis); avoid high-order Fourier content.

### Gaps
- I found no QI-specific scan of min L∇B against N or A in the literature. Kappel 2024/2026 datasets are dominated by QA/QH.
- Landreman 2021 gives no closed-form L∇B(κ, E, N). The bound L ≲ 1/κ from the always-present κ term is my inference.
- Kappel et al. 2026 has an internal inconsistency in coil lengths (14.96 vs 19.36 m per half period for the same case), which does not affect the conclusions here.

---

## Q3. Why do vacuum magnetic well and flux compression in bad curvature compete with compactness? What shaping produces a vacuum well in QI, and at what cost?

### Takeaway
First-order near-axis QI fields are generically magnetic-hill (W < 0). A vacuum well must come from second-order shaping (triangularity and elongation phase, B₂₀). That shaping lowers the shaping-limited aspect ratio and trades against ripple and QI quality. Flux compression ⟨Θ_bad|∇r|²⟩ grows with elongation, while QI quality improves with elongation, so the turbulence constraint (≤ 0.9) and the QI constraint pull in opposite directions. Low-turbulence QI designs (SQuIDs) pay with roughly 1.6× worse QI residual and a lower Mercier β limit.

### Cited Findings
**Vacuum magnetic well**
- Benchmark 3 uses W = [V′(s=0) − V′(s=1)]/V′(s=0), with V′ = dV/ds (s = normalized toroidal flux), extrapolated linearly to axis and LCFS from VMEC `gmnc`. Positive W is favorable to interchange stability, because d²V/ds² < 0 is a well. — [constellaration source, magnetics_utils.vacuum_well](https://github.com/proximafusion/constellaration)
- Near axis, the vacuum well V″ is controlled by B₂₀ (the poloidally averaged radial gradient of |B|). The field is unstable when V″ < 0, and the normalized well is W = B̄ r_ref²/(4πG₀)·[∫dφ/B₀²]⁻¹·V″. All benchmark near-axis QI configurations were MHD unstable (W magnitude ≈ 4.1–9.1%; B₂₀,min from −5.3% to +0.8%). Stabilized versions needed shaping T̂^mhd ≈ 0.13–0.22 and had shaping-limited aspect ratio A_c^mhd ≈ 4.7–6.4, compared with A_c 1.59–5.84 for ripple-optimized shaping. Sensitivity |ΔW/ΔA_c| was about 1.2–8.5%. — [Rodríguez & Plunk 2025](https://arxiv.org/html/2505.02465v1)
- Tokamak-limit formula: 𝒢₂c^W = (3r_ref²/πR)·(d̄⁴ − 1)/(d̄⁴ − 3). Circular sections (d̄ = 1) are insensitive to triangularity. Vertical elongation (d̄ < 1) combined with increasing X₂c (triangular shaping) favors the well. The authors expect flatter B₀ minima (good for MHD and maximum-J) to be more prone to secondary ripple wells. — [Rodríguez & Plunk 2025](https://arxiv.org/html/2505.02465v1)
- A 4-period QI configuration at A = 9.94 has a "significant magnetic well ensuring ideal MHD stability" and good fast-ion confinement at low β. — [Sánchez et al. 2023, NF (arXiv:2212.01143)](https://arxiv.org/pdf/2212.01143) (figures as summarized in search results)
- SQuIDs tie MHD stability to the maximum-J property (∂J/∂s < 0, target ⟨∂_s J̃⟩ ≤ −0.06). Higher β makes them more max-J. — [Goodman et al. 2024](https://arxiv.org/html/2405.19860v1)

**Flux compression in bad curvature**
- Benchmark 3 code: Θ(B×κ·∇α · sign ψ_edge) × |∇r|², with |∇r|² = a²|∇s|²/(4s), averaged over a field-line grid weighted by |√g| at s = 0.49 (ρ = 0.7). The paper writes this as χ∇r = Θ(B×κ·∇α)‖∇r‖². — [constellaration source, turbulent_transport.py](https://github.com/proximafusion/constellaration); [Cadena et al. 2025](https://arxiv.org/html/2506.19583v2)
- In a dataset of more than 200,000 nonlinear gyrokinetic ITG simulations, the top geometric predictor of heat flux is flux-surface compression in bad curvature. Spearman correlations: variance(Θ|∇x|²/B) = 0.775 and mean(Θ|∇x|⁴/B) = 0.772. Second is the magnitude of geodesic curvature. At fixed gradients, heat flux varies by orders of magnitude between geometries. — [Landreman et al. 2025, JPP 91 E120, arXiv:2502.11657](https://arxiv.org/html/2502.11657v2)
- SQuID (N = 4, A ≈ 10, Δ ≈ 0.25, ι 0.80–0.95) against an "extremely elongated" QI reference: max a|∇s| in bad curvature ≈ 2.3 versus ≈ 6.5; f_QI about 1.6× higher (worse) for the SQuID; Mercier β limit ≈ 2.5% versus ≈ 8.5% (W7-X ≈ 5%). The turbulence-target weight was set "to the largest value that kept other properties acceptable." — [Goodman et al. 2024](https://arxiv.org/html/2405.19860v1)

### Inferences
- Why elongation raises flux compression (geometric argument, my derivation): for an ellipse of elongation E, with r normalized to √(area/π), |∇r|² ≈ E on the flat broad sides and 1/E at the tips. A circular section gives ⟨Θ|∇r|²⟩ ≈ (bad-curvature fraction) ≈ 0.5. The 0.9 cap therefore allows only modest compression in the bad-curvature (mostly outboard) region, roughly E ≲ 2 there if bad curvature covered half the surface. QI theory and optimization want E ≈ 4–6 (Goodman 2023, Camacho Mata 2022). The way through is to put the elongated, flux-compressed sections where curvature is good, or to rotate elongation so the compressed sides avoid bad curvature, which is what SQuIDs do.
- Why the well competes with compactness: the well needs second-order shaping (triangularity, vertically elongated phase, B₂₀ > 0). Second-order shaping amplitude limits the radius at which surfaces stay nested and the ripple stays controlled (A_c, A_w). Pushing A down shrinks the room for that shaping, and the shaping itself sharpens boundary curvature, which lowers min L∇B (Kappel's minima sit at the bean inboard indentation). The bean indentation that helps the well in W7-X-like sections is the same feature that sets min L∇B.
- Because W in Benchmark 3 is a vacuum (β = 0) quantity, the finite-β max-J stabilization exploited by SQuIDs does not help directly. A positive vacuum well must be built from geometry alone.

### Gaps
- I did not fetch Landreman & Jorge 2020 ("Magnetic well and Mercier stability of stellarators near the magnetic axis"), which gives the general near-axis V″ formula. Only the QI-specific version from Rodríguez & Plunk 2025 is cited.
- I found no quantitative relation between vacuum W and min L∇B for QI configurations. The small LP-QA/QH with-well effects in Kappel's table are the only data.
- The exact sign convention of the W values quoted from Rodríguez & Plunk 2025 Table 3 was not visible in the summary. They are reported as unstable, so presumably negative.

---

## Q4. Role of the edge mirror ratio and iota per field period in QI; why a Δ ≤ 0.25 cap constrains compact designs

### Takeaway
QI needs a toroidal mirror (poloidally closed B_max contours), and larger Δ improves QI quality at finite A. Δ ≈ 0.25 is a common design value (SQuIDs, the Plunk & Rodríguez survey). The cap therefore removes a lever that would otherwise buy compactness. The per-period iota requirement ι/N ≥ 0.25 means edge ι ≥ 1.0 at N = 4, against ≥ 0.75 at N = 3. Several published QI designs sit at or below this threshold (Goodman N = 3: 0.25–0.27; SQuID N = 4: 0.20–0.24), so it is likely an active constraint, more so at N = 4.

### Cited Findings
- Mirror ratio definition, same in Goodman 2023 and Benchmark 3: Δ = (B_max − B_min)/(B_max + B_min), on the boundary. — [Goodman et al. 2023](https://ar5iv.labs.arxiv.org/html/2211.09829); [constellaration source](https://github.com/proximafusion/constellaration)
- "Larger Δ should help QP-symmetry quality, but if too large, confinement degrades where B is reduced." The optimization penalty includes w_Δ max(0, Δ − Δ*)². — [Goodman et al. 2023](https://ar5iv.labs.arxiv.org/html/2211.09829)
- QI is easier at large A, high elongation and large mirror ratio. — [Cadena et al. 2025](https://arxiv.org/html/2506.19583v2)
- Survey axes used Δ = 0.25 in B₀ = 1 + Δ cos(Nℓ) + (Δ/4) cos(2Nℓ). — [Plunk & Rodríguez 2025](https://arxiv.org/html/2508.12820)
- Near-axis examples used on-axis modulation amplitudes of 0.15 (N = 2) and 0.25 (N = 3) — [Camacho Mata et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.08417). The paper does not quote Δ explicitly. With Benchmark 3's definition, an amplitude ε_B gives Δ = ε_B, so these are Δ = 0.15 and 0.25 (my calculation).
- SQuID: Δ ≃ 0.25 and ι ≃ 0.80–0.95 at N = 4. A variant at 0.86–0.95 was run to avoid the 4/5 rational surface. — [Goodman et al. 2024](https://arxiv.org/html/2405.19860v1)
- Per-period ι in near-axis constant-torsion QI families is ≈ 0.46–0.47 for N = 2–4 — [Camacho Mata et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.08417). ι/N is the natural invariant under the approximate N-scaling symmetry — [Plunk & Rodríguez 2025](https://arxiv.org/html/2508.12820).
- Benchmark 3 uses |ι_edge|/N_fp, so the sign of ι does not matter. — [constellaration source, problems.py](https://github.com/proximafusion/constellaration)

### Inferences
- The mirror term B₀′ enters ∇B directly (tt component), so a larger Δ would slightly reduce L∇B. As estimated in Q2, it is a weak effect. The Δ cap matters for compactness mainly through QI quality: with Δ fixed at the cap, the remaining levers are elongation (penalized by flux compression) and axis/second-order shaping.
- Δ is evaluated at the edge, and at low A the edge |B| variation picks up a large toroidal (1/R) component. Poloidal variation adds to B_max − B_min: inboard/outboard variation ∼ ε = 1/A, so roughly Δ_edge ≳ Δ_axis + O(1/A). At A ≈ 6–8 this could consume a large part of the 0.25 budget, forcing a smaller on-axis mirror and hence worse QI. This is a plausible reason designs stall near A ≈ 9.5; it is my reasoning, not a published result.
- ι/N ≥ 0.25 is well below the near-axis "natural" half-helix value (≈ 0.46), but finite-A optimized QI designs sit lower (0.20–0.31 per period in the table above). Raising ι per period requires more torsion or helical excursion of the axis (or more rotating elongation), which Q1 identifies as raising C_g. So at N = 4 the iota floor and the QI residual are likely to be simultaneously active.

### Gaps
- I found no paper giving Δ_edge as a function of Δ_axis and A for QI. The O(1/A) toroidal correction is a heuristic.
- I did not retrieve the Camacho Mata & Plunk 2023 axis-helicity paper, which discusses how axis helicity sets ι for QI.

---

## Q5. Which constraints are usually active (binding) at the compact end of QI optimization?

### Takeaway
At the compact end the binding constraints in the literature are QI quality, together with mirror ratio and elongation caps (Goodman 2023), and the trade-off among turbulence proxy, QI and MHD stability (Goodman 2024). The ConStellaration baseline could not reach feasibility at A ≈ 6 (normalized violation 0.104). From theory, expect the QI residual and vacuum well to bind first at low A, with flux compression and ι/N binding as shaping is added to fix them. The Δ cap binds whenever an optimizer tries to buy QI with mirror ratio.

### Cited Findings
- The ALM-NGOpt baseline was feasible at A* = 10 (violation 0.0) and nearly feasible at A* = 8 and 12 (0.0013, 0.0021), but clearly infeasible at A* = 6 (0.104). Only ALM-NGOpt found feasible solutions; the SciPy baselines did not. The reported Benchmark 3 score was 130.0. — [Cadena et al. 2025](https://arxiv.org/html/2506.19583v2)
- The Goodman 2023 optimization used weights w_QI ≪ w_Δ ∼ w_ε ∼ w_A. That is, the mirror ratio, elongation and aspect ratio caps were treated as hard, and the converged designs sit at or near their caps (Δ 0.19–0.22, E 5.5–6.2, A 8–10). Optimizations with N > 1 were sensitive to initial conditions. — [Goodman et al. 2023](https://ar5iv.labs.arxiv.org/html/2211.09829)
- SQuID: the turbulence weight was set to the maximum that kept other properties acceptable, meaning the turbulence target was pushed until QI, MHD and other targets became binding. The result was f_QI 1.6× worse and Mercier limit 2.5% against 8.5%. — [Goodman et al. 2024](https://arxiv.org/html/2405.19860v1)
- Near-axis: all unshaped QI fields lack a well, so stabilization is an active requirement requiring dedicated second-order shaping. Lower ε_eff requires more shaping, which reduces A_c. — [Rodríguez & Plunk 2025](https://arxiv.org/html/2505.02465v1)
- Raising min L∇B trades directly against symmetry error (QS case). — [Kappel et al. 2026](https://arxiv.org/html/2602.18974v1)

### Inferences
- Expected binding set for the team's N = 3 front near A ≈ 9.5: (1) log10 QI ≤ −3.5, which tightens as (N/A)⁴ under the δB² heuristic; (2) W ≥ 0, since vacuum well shaping fights compactness; (3) Δ ≤ 0.25, because the optimizer will want more mirror; then (4) flux compression ≤ 0.9 if elongation is used to fix QI.
- At N = 4, add (5) ι/N ≥ 0.25 (edge ι ≥ 1). The QI constraint should bind at a higher A than at N = 3 under the δB ∝ (N/A)² law, while the N-multiplier in L̃∇B raises the objective. At equal L̃, an N = 4 design may therefore sit at a slightly higher A but dominate on L̃. The hypervolume gain could come mainly from the L̃ axis.
- Diagnostic suggestion: at each Pareto point, log which constraints have zero slack. If the QI residual is always at −3.5 while W, Δ and flux compression have slack, the limit is the axis shape (C_g), and the fix is axis-level (lower torsion per period, smoother curvature zeros). If W and flux compression are at their bounds, the limit is cross-section shaping.

### Gaps
- I found no published Lagrange-multiplier or active-set analysis for Benchmark 3 specifically. The binding-set claims for the team's designs are inferences.
- I did not find a QI study that reports active constraints as a function of N at fixed A.
