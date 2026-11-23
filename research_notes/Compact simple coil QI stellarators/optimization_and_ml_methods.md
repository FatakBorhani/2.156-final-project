# Optimization and Machine-Learning Methods for Compact QI Stellarator Boundary Design (ConStellaration Benchmark 3)

Scope note: methods and evidence only. Research date 2026-11-21. "Peer-reviewed" means a journal or main-track conference; "preprint" means arXiv only or workshop. Several sources carry 2026 arXiv IDs (2601-2611); they are real listings fetched this session, but most are preprints.

## 1. Differentiable / gradient-based equilibrium optimization (DESC, SIMSOPT, vmec_jax, single-stage)

### Takeaway
DESC is the most directly useful gradient-based tool for Benchmark 3: it is JAX-autodiff, GPU-capable, ships ready-made objectives for every Benchmark-3 quantity except flux compression (Omnigenity, MirrorRatio, MagneticWell, AspectRatio, Elongation, BScaleLength), and has native augmented-Lagrangian constrained optimizers. Proxima itself used DESC (lsq_auglag) to generate the ConStellaration dataset at about 3 minutes per run. SIMSOPT+VMEC uses finite-difference gradients (MPI-parallel). vmec_jax/vmex is a newer, less-proven differentiable VMEC clone.

### Cited Findings
**DESC: omnigenity and QI**
- Dudt, Goodman, Conlin, Panici, Kolemen, "Magnetic fields with general omnigenity," J. Plasma Phys. 90 (2024), peer-reviewed (arXiv 2305.08026, 2023). It parameterizes general omnigenous fields in DESC and optimizes for them. It produced examples in six classes: OP (poloidally closed contours, i.e. QI), QP, OH, QH, OT, QA. — [arXiv](https://arxiv.org/html/2305.08026v3); [JPP](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/magnetic-fields-with-general-omnigenity/8E3A81AF1CCBBB9CFBC9F3B0EFF6053D)
- Method in that paper: the initial guess comes from a near-axis expansion, the equilibrium is re-solved in DESC, then optimized with a multi-objective cost. Each optimization took **under 30 minutes on one NVIDIA A100 GPU**. — [arXiv 2305.08026](https://arxiv.org/html/2305.08026v3)
- A 2024 APS-DPP poster describes a database of about **50k omnigenous equilibria generated with DESC**. This is a conference abstract, not peer-reviewed. — [APS DPP 2024 JP12.36](https://meetings-archive.aps.org/dpp/2024/jp12/36/)
- Gaur et al., PPCF 67 (2025) 125015, peer-reviewed. They used DESC to optimize omnigenous equilibria with poloidal, toroidal and helical closure for Mercier and ideal-ballooning stability (and, as a result, improved kinetic-ballooning stability). Force balance is enforced at every iteration. — [Gaur 2025 PPCF](https://eprints.whiterose.ac.uk/id/eprint/237029/1/Gaur_2025_Plasma_Phys._Control._Fusion_67_125015.pdf); [Princeton listing](https://collaborate.princeton.edu/en/publications/omnigenous-stellarators-with-improved-ideal-and-kinetic-balloonin/)

**DESC: available objectives and optimizers (current docs)**
- `desc.objectives` includes Omnigenity, Isodynamicity, MirrorRatio, MagneticWell, MercierStability, BallooningStability, AspectRatio, Elongation, BScaleLength (a magnetic-field scale-length target, the closest analogue to L∇B), EffectiveRipple, GammaC, QuasisymmetryTwoTerm/Boozer/TripleProduct, and curvature objectives. It also has `ExternalObjective`, `ObjectiveFromUser`, `GenericObjective`, and `DeflationOperator`. — [DESC objectives API](https://desc-docs.readthedocs.io/en/latest/api_objectives.html)
- Optimizers: `lsqtr` (least-squares trust region), `fmintr`, `fmin_auglag` and `lsq_auglag` (augmented Lagrangian, constrained), and `sgd`. There are also helpers for fixing degrees of freedom (FixBoundaryR/Z, FixMode, FixOmniWell/Map/Bmax) and near-axis constraints (`get_NAE_constraints`). — [DESC objectives API](https://desc-docs.readthedocs.io/en/latest/api_objectives.html)
- Conlin et al., "Stellarator optimization with constraints," J. Plasma Phys. (2024), peer-reviewed. It surveys constraint handling for stellarator optimization and adds a new augmented-Lagrangian algorithm to DESC, motivated by engineering constraints such as aspect ratio, coil-plasma distance, and boundary or coil curvature. — [JPP PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/0D9718262A99FF856EC654FB8AB29124/S0022377824000655a_hi.pdf/stellarator-optimization-with-constraints.pdf); [arXiv 2403.11033](https://arxiv.org/html/2403.11033v1)
- ConStellaration used DESC's omnigenity objective inside `lsq-auglag` to generate dataset boundaries. Each DESC run averaged about **3 minutes**. The same text reports an average VMEC++ run of about **1 hour**; this probably means a VMEC++-in-the-loop optimization, not a single solve, so the comparison is ambiguous. — [ConStellaration arXiv 2506.19583](https://arxiv.org/html/2506.19583v2)
- There is a DESC paper on extending near-axis equilibria (arXiv 2506.05170, 2025). Its details were not fetched. — [arXiv 2506.05170](https://arxiv.org/pdf/2506.05170)

**SIMSOPT / VMEC finite-difference and QI target**
- Goodman et al., "Constructing precisely quasi-isodynamic magnetic fields," J. Plasma Phys. (2023), peer-reviewed. It used SIMSOPT to couple VMEC (zero pressure and current), a new QI target, and SciPy's trust-region-reflective least squares with **finite-difference gradients**. Objective terms: (a) a QI residual, built by squashing, stretching and shuffling the |B| wells into a perfectly QI reference field and penalizing the squared difference; (b) a one-sided mirror-ratio penalty Δ > Δ*; (c) a max-elongation penalty ε > ε*; (d) an aspect-ratio penalty A > A*. w_QI is set much smaller than the other weights so the physical limits are respected. This is essentially the ConStellaration QI residual lineage. — [Goodman 2023 JPP](https://www.cambridge.org/core/product/6601E449C8DD3B3FEB361DA2C5732EFC/core-reader)
- In Goodman 2023, the nfp=2 start came from a near-axis configuration (Camacho Mata et al. 2022) **truncated to modes ≤ 2**. The nfp=3 start reused the optimized nfp=2 boundary with nfp increased and truncated to mpol = ntor = 2. A full-fidelity near-axis boundary needed about 10 poloidal × 90 toroidal modes. — [Goodman 2023 JPP](https://www.cambridge.org/core/product/6601E449C8DD3B3FEB361DA2C5732EFC/core-reader)
- Follow-ups in the same VMEC-in-the-loop SIMSOPT lineage: Rodríguez, Helander & Goodman, "The maximum-J property in QI stellarators," JPP 2024, peer-reviewed, reached a vacuum field in which **99.6% of trapped particles satisfy maximum-J**. Also SQuIDs (Goodman et al., arXiv 2405.19860, 2024). — [arXiv 2311.14439](https://arxiv.org/pdf/2311.14439); [arXiv 2405.19860](https://arxiv.org/html/2405.19860v1)
- SIMSOPT does gradient-based optimization with MPI parallelization of finite differences. Jorge et al.'s single-field-period QI near-axis optimization ran in **30 s on one CPU core** in SIMSOPT (JPP 2022, peer-reviewed). — [Jorge et al. 2022 JPP](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9B2A5FDCCD7774E4F91BE45E75FDC6B0/S0022377822000873a.pdf/singlefieldperiod_quasiisodynamic_stellarator.pdf)
- Landreman & Paul, "Magnetic fields with precise quasisymmetry," PRL 128, 035001 (2022), peer-reviewed. Symmetry-breaking amplitudes were about 50 µT (for 1 T) over an aspect-ratio-6 volume. This shows how much precision VMEC+SIMSOPT least squares can reach. — [arXiv 2108.03711](https://arxiv.org/abs/2108.03711)

**vmec_jax / vmex**
- `vmec_jax` (uwplasma, PyPI 0.0.x) is an end-to-end differentiable JAX reimplementation of VMEC2000 (fixed and free boundary). It uses a discrete adjoint via a checkpointed iteration tape instead of per-coefficient finite differences. Its residuals include aspect ratio, iota bounds, and QA/QH/QI. Differentiable Mercier and Redl bootstrap are listed as future work. The successor repo `vmex` claims iteration-for-iteration agreement with VMEC2000. It is pre-1.0 software and not peer-reviewed. — [PyPI vmec-jax](https://pypi.org/project/vmec-jax/0.0.13/); [GitHub vmex](https://github.com/uwplasma/vmex)

**Single-stage plasma+coil**
- Jorge, Goodman, Landreman, Rodrigues, Wechsung, "Single-stage stellarator optimization: combining coils with fixed boundary equilibria," PPCF 65 (2023) 074003, peer-reviewed. Boundary and coil degrees of freedom are optimized together, with a cost combining stage-1 and stage-2 objectives. It reached smaller plasma-objective values than two-stage optimization for QS and QI targets. — [arXiv 2302.10622](https://ar5iv.labs.arxiv.org/html/2302.10622)
- Jorge, Giuliani, Loizu, "Simplified and flexible coils for stellarators using single-stage optimization" (arXiv 2406.07830, 2024; SIMSOPT). — [arXiv 2406.07830](https://arxiv.org/html/2406.07830v1)

### Inferences
- For a team already on VMEC++, the cheapest gradient route is a **DESC polish stage**: import a feasible or near-feasible VMEC++ boundary, re-solve in DESC, then run `lsq_auglag`. Aspect ratio, MirrorRatio, MagneticWell, Elongation and Omnigenity/Isodynamicity go in as constraints or objectives, and BScaleLength serves as an L∇B proxy. Re-score with the official VMEC++ evaluator afterwards, because DESC's metric definitions (for example Omnigenity vs. the ConStellaration QI residual) differ. This replaces surrogate gradients with true physics gradients, which removes surrogate exploitation entirely in that stage.
- Flux compression in regions of bad curvature (⟨χ∇r⟩) has no off-the-shelf DESC objective. It would need `ObjectiveFromUser`/`GenericObjective` (JAX-traceable) or an `ExternalObjective` (finite differences).
- DESC's 3-minute runs vs. 18-34 h for ALM-NGOpt on 96 vCPUs (Section 3) suggest at least an order-of-magnitude wall-clock advantage for gradient methods near feasibility. No head-to-head on Benchmark 3 was found, so this is an inference, not a measurement.
- From prior knowledge, not re-verified this session: Paul, Antonsen and Landreman developed adjoint methods for VMEC-based shape gradients (JPP 2019-2021). These are largely superseded in practice by DESC autodiff and vmec_jax.

### Gaps
- No published wall-clock or success-rate comparison of DESC vs. VMEC++/NGOpt on the exact Benchmark 3 metrics.
- I found no source on whether DESC's `BScaleLength` matches ConStellaration's normalized minimum L∇B definition.
- The vmec_jax/vmex speed and robustness numbers for QI were not found.

## 2. Near-axis expansion (NAE) initialization for QI across nfp and aspect ratio

### Takeaway
Near-axis codes (pyQIC/`qicna`, and second-order extensions) build approximately QI configurations analytically in seconds. They are the standard seed for global QI optimization, both for DESC (Dudt 2024) and for VMEC/SIMSOPT (Goodman 2023). A new database of more than 800k near-axis QI configurations (Rodríguez & Plunk, 2026) is explicitly meant to supply initial conditions across nfp and geometry.

### Cited Findings
- pyQIC (GitHub `rogeriojorge/pyQIC`; PyPI `qicna`) is pure Python with few dependencies and generates QI configurations by expansion about the magnetic axis. A second-order extension is in the fork `SebereX/pyQIC`. — [GitHub pyQIC](https://github.com/rogeriojorge/pyQIC); [PyPI qicna](https://pypi.org/project/qicna/); [arXiv 2409.20328](https://arxiv.org/pdf/2409.20328)
- Camacho Mata, Plunk, Jorge, "Direct construction of stellarator-symmetric QI magnetic configurations" (JPP 2022; arXiv 2206.08417). It is first order: the inputs are the axis shape and a set of toroidal-angle functions. — [arXiv 2206.08417](https://ar5iv.labs.arxiv.org/html/2206.08417)
- The second-order near-axis description of stellarator-symmetric QI configurations is benchmarked against global equilibria. — [arXiv 2409.20328](https://arxiv.org/pdf/2409.20328)
- Plunk & Rodríguez (arXiv 2508.12820, 2025, preprint) recast the near-axis QI theory in geometric inputs and give a way to construct suitable magnetic axes without extra optimization. — [arXiv 2508.12820](https://arxiv.org/pdf/2508.12820)
- Rodríguez & Plunk, "Near-axis quasi-isodynamic database" (arXiv 2601.08400, v1 Jan 2026, v3 May 2026). The listing has a DOI 10.1017/S00223778261... that suggests JPP acceptance, but this was not confirmed. The database holds **more than 800,000 stable, approximately QI vacuum configurations** across several nfp values, axis shapes and elongations. Each is scored on effective ripple, Shafranov-shift sensitivity, maximum-J fraction, and Rosenbluth-Hinton residual. The authors apply ML and statistics to find key descriptors, and state that the database is meant to supply "baseline configurations and initial conditions for optimization." — [arXiv 2601.08400](https://arxiv.org/abs/2601.08400)
- Rodríguez & Plunk, "Near-axis measures of QI configurations" (arXiv 2505.02465). — [arXiv 2505.02465](https://arxiv.org/pdf/2505.02465)
- Usage pattern in practice:
  - Dudt et al. 2024 seeded DESC omnigenity optimizations from NAE. — [arXiv 2305.08026](https://arxiv.org/html/2305.08026v3)
  - Goodman 2023 took an NAE nfp=2 boundary, truncated it to |m|,|n| ≤ 2, optimized it in VMEC, then changed nfp to 3 and reused it. — [Goodman 2023](https://www.cambridge.org/core/product/6601E449C8DD3B3FEB361DA2C5732EFC/core-reader)
  - Jorge 2022 optimized within near-axis parameter space in 30 s on one CPU before the global solve. — [Jorge 2022](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9B2A5FDCCD7774E4F91BE45E75FDC6B0/S0022377822000873a.pdf/singlefieldperiod_quasiisodynamic_stellarator.pdf)
- Near-axis QI boundaries at full fidelity can need about 10 poloidal × 90 toroidal modes, so truncation to low modes is mandatory before handing them to a 5×5-mode optimizer. Truncation gives "poor QI but a similar boundary shape." — [Goodman 2023](https://www.cambridge.org/core/product/6601E449C8DD3B3FEB361DA2C5732EFC/core-reader)

### Inferences
- NAE is most accurate at large aspect ratio, and Benchmark 3 rewards low A. A practical pattern: (1) generate NAE seeds at the target nfp with moderate A (for example 8-10, where ConStellaration's own baselines are feasible); (2) truncate to 3×3 or 4×4; (3) run aspect-ratio continuation downward in the global optimizer (Section 5).
- The Rodríguez-Plunk database (filtered by nfp, max-J fraction, low ripple) could replace or augment the team's diffusion/VAE seed generator with physics-derived seeds. Whether the database is publicly downloadable is unconfirmed.
- Changing nfp by reusing an optimized boundary (Goodman's nfp 2→3) is a cheap way to bootstrap nfp-4 or nfp-5 seeds from nfp-3 results.

### Gaps
- Database availability, file format, and nfp/A ranges for arXiv 2601.08400 could not be confirmed (PDF text not extractable).
- I found no quantitative study of the success rate when NAE seeds are pushed to A ≈ 6 or below.

## 3. Surrogate-based and ML methods in stellarator design

### Takeaway
Published ML-for-stellarator work is young (2025-2026), mostly preprints or workshop papers. Generative models (diffusion, MDN, conditional generators) reach roughly 5% error on conditioned geometric targets (aspect ratio), are worse on iota, and VMEC fails on many generated samples. ConStellaration's own ML baseline is a PCA → random-forest feasibility classifier → GMM → MCMC sampler. High-dimensional BO (TuRBO and successors) has now been benchmarked on ConStellaration. None of these reports an end-to-end Benchmark-3 hypervolume improvement over ALM-NGOpt.

### Cited Findings
**ConStellaration baselines (NeurIPS 2025 Datasets & Benchmarks, peer-reviewed; arXiv 2506.19583)**
- Benchmark 3 setup:
  - Objectives: minimize −L̃∇B and A.
  - Constraints: ι̃ ≥ ι̃*, QI ≤ QI*, mirror ratio Δ ≤ Δ*, vacuum well W ≥ 0, ⟨χ∇r⟩ ≤ ⟨χ∇r⟩*.
  - Scoring: hypervolume against a fixed reference point.
  - Dataset: about 158k boundaries, modes up to 4. Baselines use nfp = 3 and are seeded from rotating ellipses.
  - [arXiv 2506.19583](https://arxiv.org/html/2506.19583v2)
- Optimizers: only ALM-NGOpt found feasible solutions; scipy trust-constr and COBYQA were infeasible after about 40 h on 4 vCPUs.
- ALM-NGOpt runtime: 18 h (geometric) and 34 h (simple-to-build) on 96 vCPUs, and **15-24 h per multi-objective instance on 96 vCPUs**.
- Benchmark-3 front from ALM-NGOpt (one run per A* ∈ {6, 8, 10, 12}):

  | A | L̃∇B | normalized violation |
  |---|------|----------------------|
  | 6.02 | 2.98 | 0.104 (infeasible) |
  | 7.93 | 5.60 | 0.0013 |
  | 9.98 | 8.45 | 0 |
  | 11.9 | 11.1 | 0.0021 |

  The leaderboard reference score for MHD-stable is 130.0. — [arXiv 2506.19583](https://arxiv.org/html/2506.19583v2)
- Constraint-handling tricks in the baseline: ALM with a non-Euclidean proximal (trust-region-like) term, which the authors say helps evolutionary methods converge; diagonal preconditioning with exponentially decaying scales; and a log10 transform on the QI constraint. Stage-1 VMEC++ runs use lower fidelity (fewer surfaces, looser force tolerance) than scoring. — [arXiv 2506.19583](https://arxiv.org/html/2506.19583v2)
- ML generative baseline: PCA → random-forest feasibility classifiers → GMM fit to feasible points → adaptive MCMC. Only 41 and 52 feasible seeds (relaxed geometric and simple-to-build problems) existed out of about 160k. Samples with classifier confidence ≥ 0.99 were taken as predicted-feasible. The paper does not report an overall oracle feasibility rate. — [arXiv 2506.19583](https://arxiv.org/html/2506.19583v2)
- Surrogate baseline: an ensemble of 10 MLPs (3×256, tanh) on about 23k points. The text claims R² > 0.97 and RMSE < 0.1, but Table 7 has some RMSE above 0.1 (for example 0.581 for aspect_ratio_over_edge_rotational_transform). This is an internal inconsistency in the source. — [arXiv 2506.19583](https://arxiv.org/html/2506.19583v2)

**Diffusion and other generative models**
- Padidar, Huang, Giuliani, Spivak, "Diffusion for Fusion: Designing Stellarators with Generative AI" (arXiv 2511.20445; NeurIPS 2025 AI4Science workshop, non-archival).
  - Model: conditional DDPM trained on **QUASR** (QA/QH, not QI). The denoiser is an MLP with 4×2048 GELU layers; PCA reduces 661 dimensions to 50; 200 diffusion steps.
  - Conditioning: mean iota, A, helicity, nfp.
  - Results: aspect-ratio error is usually under 5%; iota is within 10-12% about 75% of the time; QS deviation is under 2.5% in-sample and under 6-10% out-of-sample (the 1% target is generally not reached).
  - **VMEC failed on 57% of in-sample generated stellarators** (vs. 41% of the training set).
  - Code and data are on Zenodo (10.5281/zenodo.17713019). The authors list guided sampling, physics-informed losses and multi-fidelity models as future work.
  - [arXiv 2511.20445](https://arxiv.org/html/2511.20445)
- Curvo et al. (2025) are credited by Padidar et al. as the first generative-model stellarator design: a mixture density network producing quasisymmetric configurations. Not fetched. — [cited in arXiv 2511.20445](https://arxiv.org/html/2511.20445)
- Han, Chen, Cao, Lu, Chen, Xu, Wan, "Data-Driven Generation of Compact Quasi-Isodynamic Stellarators" (arXiv 2608.16938, Aug 2026, preprint).
  - Approach: conditional boundary generation for **nfp-4 QI** starting from ConStellaration. The model first learns shared geometry from known QI equilibria, then **fine-tunes on a small high-fidelity compact (low-A) dataset**.
  - Results: about 87% reduction in compact-domain test loss; converged "ultra-compact" candidates consistent with the conditions; one candidate is proposed as a seed for further QI optimization.
  - Not reported on the abstract page: architecture, numeric A, feasibility rate, code.
  - This is the closest published analogue to the team's approach.
  - [arXiv 2608.16938](https://arxiv.org/abs/2608.16938)
- Weißberg, Heep, Lähner, Bernard, "Towards Stellarator Geometry Optimisation for Nuclear Fusion" (arXiv 2608.28224; ECCV 2026 workshop extended abstract). A local geometry-refinement method plus a 2D latent representation reached **#1 on the ConStellaration geometric-task leaderboard in May 2026**. No Benchmark-3 claims. — [arXiv 2608.28224](https://arxiv.org/abs/2608.28224)
- CoilStellaration (arXiv 2611.12689; listed as NeurIPS 2026): 177k optimized filamentary coilsets for ConStellaration equilibria. — [arXiv 2611.12689](https://arxiv.org/abs/2611.12689)
- Other surrogates (out of scope but relevant to the pattern):
  - A neural VMEC surrogate for W7-X (arXiv 2105.08467, 2021). — [arXiv 2105.08467](https://arxiv.org/pdf/2105.08467)
  - A GX gyrokinetic neural surrogate, used because it is differentiable and avoids finite-difference GX runs (arXiv 2609.28730). — [arXiv 2609.28730](https://arxiv.org/html/2609.28730)
  - A fast-ion generative and surrogate benchmark (arXiv 2610.37910). — [arXiv 2610.37910](https://arxiv.org/html/2610.37910v1)

**Datasets for training**
- QUASR (Giuliani et al.): about 370k vacuum QA/QH stellarators with coil sets (Zenodo v4, Sept 2024), in SIMSOPT and VMEC formats, browsable at quasr.flatironinstitute.org. It is quasisymmetric only, not QI. About 30% (mostly high-nfp QH) failed to converge in VMEC per a later study. — [Zenodo 13717741](https://zenodo.org/records/13717741); [arXiv 2409.04826](https://arxiv.org/pdf/2409.04826); [arXiv 2512.24544](https://arxiv.org/pdf/2512.24544)
- Rodríguez-Plunk near-axis QI database (>800k) is listed in Section 2. DESC omnigenity database (about 50k) is listed in Section 1.

**Bayesian optimization on ConStellaration**
- RAMBO (Zhang et al., arXiv 2601.20043, Jan 2026, preprint) is regime-adaptive BO using Dirichlet-process mixtures of GPs.
  - Setup: the **81-D** simple-to-build problem (maximize QI quality), 500 iterations, 5 seeds.
  - Baselines: SGP, TuRBO, SAASBO, BAxUS, ALEBO, HEBO, Bounce, SMAC.
  - Results: about 51.6% higher QI objective than the best baseline; 3-5 distinct "regimes".
  - Claims: embedding methods (ALEBO, SAASBO) fail because the physics breaks their assumptions, and **TuRBO plateaus when its trust region straddles a stability (VMEC-convergence) boundary**.
  - Absolute values and constraint handling were not in the fetched text.
  - [arXiv 2601.20043](https://arxiv.org/html/2601.20043v1)

### Inferences
- The Han et al. 2026 result (pretrain on all QI, fine-tune on a small compact high-fidelity set) supports the team's conditional VAE/DDPM direction. It suggests the biggest gain may come from **fine-tuning on the team's own low-A feasible simulations** rather than training only on Proxima's data, which is sparse at low A.
- Diffusion samples should be treated as **seeds, not answers**: Padidar et al. saw more than 50% VMEC failure and about 10% iota error. Expect a VMEC++ convergence filter plus a local optimizer to be needed.
- Discontinuities from VMEC non-convergence are a documented failure mode for trust-region BO (RAMBO). This argues for modeling "VMEC converged" as a separate classifier constraint.

### Gaps
- No published Benchmark-3 (MHD-stable, multi-objective) results from any ML or BO method beyond the ALM-NGOpt baseline were found. Leaderboard standings for Benchmark 3 were not retrieved.
- Flow matching, normalizing flows, RL, and equivariant (field-period-symmetric) networks applied to stellarator boundaries: no sources found in this session.
- The architecture and code for Han et al. 2026 are unknown.

## 4. Preventing surrogate exploitation

### Takeaway
The established remedies are to keep proposals near data or near the incumbent (trust regions, proximal terms), model constraints and feasibility explicitly with uncertainty (separate constraint GPs or classifiers, ensembles), and always verify with the true solver. The stellarator-specific evidence is thin. ConStellaration's own pipeline gates samples by classifier confidence ≥ 0.99, and its ALM uses a proximal trust-region term.

### Cited Findings
- SCBO (Eriksson & Poloczek, AISTATS 2021, peer-reviewed; arXiv 2002.08526):
  - Trust region: a TuRBO-style hyperrectangle centered on the best feasible point, or on the **least-violating point if none is feasible**. It expands on success and shrinks on failure.
  - Candidates come from Thompson sampling inside the region.
  - Constraints: **separate GPs per constraint** within the trust region.
  - Built for high-dimensional expensive objectives with expensive constraints, with large-batch and asynchronous support. It outperformed prior methods on high-dimensional constrained problems.
  - [PMLR](https://proceedings.mlr.press/v130/eriksson21a.html); [arXiv](https://arxiv.org/abs/2002.08526)
- A 2026 follow-up on trust-region constrained BO with penalized constraint handling exists (arXiv 2603.24567; not fetched). There is also latent-space GP BO with large-scale constraints (arXiv 2412.15679). — [arXiv 2603.24567](https://arxiv.org/pdf/2603.24567); [arXiv 2412.15679](https://arxiv.org/pdf/2412.15679)
- TuRBO can stall when the trust region straddles a stability or convergence boundary on ConStellaration; regime-aware (mixture) surrogates helped. — [RAMBO arXiv 2601.20043](https://arxiv.org/html/2601.20043v1)
- ConStellaration ALM-NGOpt adds a non-Euclidean proximal term, i.e. a trust region, to the augmented Lagrangian. Its generative pipeline accepts only MCMC samples with classifier confidence ≥ 0.99. — [arXiv 2506.19583](https://arxiv.org/html/2506.19583v2)
- Padidar et al. list guided sampling (classifier or physics guidance) as future work for stellarator diffusion. No published stellarator result yet. — [arXiv 2511.20445](https://arxiv.org/html/2511.20445)

### Inferences
These are concrete recipes for the team; they are inferred, not tested in stellarator papers.
1. **Trust-region-bounded surrogate steps.** Limit each surrogate-gradient refinement to a ball (in normalized Fourier or PCA space) around the last VMEC++-verified point. Expand or shrink it based on whether VMEC++ confirms improvement, as in TuRBO/SCBO. This directly addresses "leaving the data manifold."
2. **Uncertainty penalty.** Optimize mean − κ·σ, using the MLP-ensemble spread σ that the team already has, and add a learned "VMEC converges" classifier as a hard constraint.
3. **Manifold constraint.** Penalize distance to the data, for example VAE reconstruction error, the DDPM denoising loss at small t, or Mahalanobis distance in the PCA space ConStellaration already uses. Alternatively, refine in the VAE latent rather than in raw coefficients.
4. **Active learning loop.** Add every VMEC++-evaluated point, especially failures and near-misses, back into the surrogate training set each round.
5. **Replace surrogate gradients near feasibility with true gradients** (DESC, or vmec_jax adjoints). The true gradient cannot be exploited.

### Gaps
- No stellarator-specific paper quantifying how much exploitation-prevention techniques improve feasible-hit rates was found.
- Classifier-guided or constraint-guided diffusion for stellarators: none published that I could find.

## 5. Continuation strategies (Fourier resolution, aspect ratio, homotopy)

### Takeaway
Low-mode-first resolution continuation is standard practice in the QI literature. Goodman 2023 truncated near-axis seeds to |m|,|n| ≤ 2 and stepped nfp 2→3 by reusing boundaries. ConStellaration's front was built as separate runs per A* rather than by continuation. Low-to-high fidelity (VMEC++ low resolution in the loop, high resolution for scoring) is used by Proxima.

### Cited Findings
- Goodman 2023:
  - Started from a near-axis seed truncated to modes ≤ 2.
  - The nfp=3 run started from the optimized nfp=2 boundary truncated to mpol = ntor = 2.
  - A Boozer-angle approximation used in early, low-resolution stages worked for nfp 1-2 and failed at nfp 3. This is a caution that cheap approximations used early in continuation can break as the problem changes.
  - [Goodman 2023 JPP](https://www.cambridge.org/core/product/6601E449C8DD3B3FEB361DA2C5732EFC/core-reader)
- ConStellaration:
  - Dataset modes go up to 4.
  - The multi-objective front came from separate ALM-NGOpt runs at A* = 6, 8, 10, 12. **The A* = 6 run ended infeasible (violation 0.104)**, while A ≥ 8 runs were feasible or near-feasible.
  - Lower-fidelity VMEC++ is used in the optimization loop and high fidelity for scoring. Some high-fidelity VMEC++ runs did not converge, so lower-fidelity values were reported.
  - Mode continuation is not mentioned.
  - [arXiv 2506.19583](https://arxiv.org/html/2506.19583v2)
- DESC provides `DeflationOperator` (an objective that repels known solutions, useful for finding multiple distinct optima) and Fix* constraints that support staged freeing of modes. — [DESC objectives API](https://desc-docs.readthedocs.io/en/latest/api_objectives.html)

### Inferences
- Since ALM-NGOpt from scratch failed at A* = 6 but succeeded at A* = 8, a **homotopy in A*** is the natural next step: warm-start the A* = 7 run from the feasible A* = 8 solution, then A* = 6.5, and so on, keeping the ALM multipliers. This tests whether the A = 6 failure was a starting-point problem or a true feasibility limit. It is untested in the literature found.
- **Resolution continuation for a 5×5 parameterization:** optimize at 2×2 or 3×3 (few degrees of freedom, where NGOpt converges much faster), then free 4×4 and 5×5 modes with small initial amplitudes and a tight trust region. Gradient-free cost scales poorly with dimension, so this matters most for NGOpt. Gradient methods like DESC are less sensitive.
- Using nfp-swap reuse (Goodman) plus near-axis seeds gives a cheap path to nfp 4-5 starting points.
- From prior knowledge, not re-verified this session: DESC's own equilibrium solver uses continuation in resolution, pressure and boundary perturbation (Conlin, Dudt, Panici, Kolemen, "The DESC stellarator code suite Part II: perturbation and continuation methods," JPP 2023). SIMSOPT examples typically step `max_mode` 1→N in an outer loop.

### Gaps
- I found no published quantitative comparison of mode-continuation vs. direct full-resolution optimization for QI, and no aspect-ratio homotopy study for QI fronts.
- Multi-fidelity offset schemes (as the team uses) have no published stellarator benchmark beyond ConStellaration's low/high-fidelity split.
