# ConStellaration benchmark (Proxima Fusion, 2025): Benchmark 3 definition, dataset, baselines, and published participant methods

Notes compiled 2026-11-21. Scope: methods and aggregate scores only. No leaderboard boundary shapes or Fourier coefficients were downloaded or recorded. Several numbers below came from automated page summaries, mainly the leaderboard Space config and code files. These are flagged where they were not checked against a second source.

## Q1. Exact Benchmark 3 definition (objectives, constraints, normalizations, HV reference point, fidelity, nfp/mode restrictions); is the leaderboard score the same HV?

### Takeaway
Benchmark 3 ("MHD-stable QI") maximizes L∇B and minimizes aspect ratio A, subject to five constraints:
- |ι_edge|/nfp ≥ 0.25
- log10(QI residual) ≤ −3.5
- edge mirror ratio ≤ 0.25
- flux compression in bad curvature ≤ 0.9
- vacuum well ≥ 0

A design counts as feasible when every normalized violation is ≤ 1e-2. The score is the 2-D hypervolume of the feasible points in (−L∇B, A) space, against reference point (1.0, 20.0). The leaderboard score is this same hypervolume, computed with `constellaration==0.2.6` at high-fidelity VMEC++. The scoring code checks neither nfp nor Fourier resolution.

### Cited Findings
- **Paper.** "ConStellaration: A dataset of QI-like stellarator plasma boundaries and optimization benchmarks," Cadena, Merlo, Laude, Bauer, Agrawal, Pascu, Savtchouk, Guiraud, Bonauer, Hudson, Kaiser (Proxima Fusion). arXiv:2506.19583, v1 24 Jun 2025, v2 26 Nov 2025. NeurIPS 2025 Datasets & Benchmarks track. — [arXiv abs](https://arxiv.org/abs/2506.19583); [NeurIPS poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/121685)
- **Problem statement in the paper.** Minimize (−L̃∇B, A) subject to:
  - ι̃ ≥ ι̃*
  - QI ≤ QI*
  - Δ ≤ Δ* (mirror ratio)
  - W_MHD ≥ 0 (vacuum magnetic well)
  - ⟨χ∇r⟩ ≤ ⟨χ∇r⟩*

  χ∇r, the flux-compression / turbulence proxy, is evaluated on one flux surface at ρ = 0.7. Violations are normalized as c̃ᵢ = (cᵢ − cᵢ*)/cᵢ*. Scoring is the hypervolume (HV) over feasible solutions with "a fixed reference point in objectives space." — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **Exact thresholds in code** (`problems.py`, class `MHDStableQIStellarator`):
  - `edge_rotational_transform_over_n_field_periods` ≥ 0.25, applied to the absolute value
  - log10(qi) ≤ −3.5
  - `edge_magnetic_mirror_ratio` ≤ 0.25
  - `flux_compression_in_regions_of_bad_curvature` ≤ 0.9
  - `vacuum_well` ≥ 0.0. The vacuum-well violation is normalized by max(0.1, bound) = 0.1, since its target is 0.

  Objectives are `minimum_normalized_magnetic_gradient_scale_length` (maximized) and `aspect_ratio` (minimized). Score is the hypervolume of feasible points `(-L∇B, A)` against reference point `[1.0, 20.0]`, or 0 if none are feasible. Evaluation uses `default_high_fidelity()`. — [problems.py (main)](https://raw.githubusercontent.com/proximafusion/constellaration/main/src/constellaration/problems.py)
- **Feasibility tolerance.** `_DEFAULT_RELATIVE_TOLERANCE = 1e-2`. The feasibility value is the largest positive normalized violation, where each violation is divided by |target|. — [problems.py](https://raw.githubusercontent.com/proximafusion/constellaration/main/src/constellaration/problems.py)
- **Other benchmarks, for context.**
  - Benchmark 1 (geometric): minimize max elongation subject to A ≤ 4.0, average triangularity ≤ −0.5 and |ι_edge|/nfp ≥ 0.3. Score = 1 − normalize(ε_max, 1, 10).
  - Benchmark 2 (simple-to-build QI): maximize L∇B subject to A ≤ 10, |ι|/nfp ≥ 0.25, log10 QI ≤ −4.0, mirror ≤ 0.2 and ε_max ≤ 5. Score = L∇B normalized on [0, 20].

  — [problems.py](https://raw.githubusercontent.com/proximafusion/constellaration/main/src/constellaration/problems.py)
- **L∇B normalization.**
  - Code: `minimum_normalized_magnetic_gradient_scale_length` is the minimum over a θ–φ grid of the normalized gradient scale length, multiplied by `n_field_periods`. The code comment says this makes the metric "independent of the configuration number of field periods." — [forward_model.py v0.2.6](https://raw.githubusercontent.com/proximafusion/constellaration/v0.2.6/src/constellaration/forward_model.py)
  - Paper: L̃∇B is described as normalized with a/N_fp. — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **Where each metric is evaluated.**
  - Flux compression: normalized toroidal flux 0.7², 101 field lines, 64 toroidal points.
  - QI residual: sum of squared residuals from a Boozer transform at normalized toroidal flux 1.0 (the edge).
  - Mirror ratio: at the edge.

  — [forward_model.py](https://raw.githubusercontent.com/proximafusion/constellaration/main/src/constellaration/forward_model.py)
- **VMEC++ fidelity presets** (v0.2.6 `vmec_settings.py`):

  | Preset | Multigrid ns | Force tolerance | Max iterations | mpol/ntor |
  |---|---|---|---|---|
  | `high_fidelity` (scoring) | 25 / 51 / 99 | 1e-17 / 1e-17 / 1e-15 | 2000 / 2000 / 40000 | max(10, boundary's largest nonzero mode) |
  | `low_fidelity` (default during optimization) | 25 / 71 | 1e-17 / 1e-13 | 2000 / 20000 | boundary max mode + 2 |

  — [vmec_settings.py v0.2.6](https://raw.githubusercontent.com/proximafusion/constellaration/v0.2.6/src/constellaration/mhd/vmec_settings.py)
- **No nfp or mode checks.** Per a summary of the code, none of the three problem classes validates `n_field_periods` or Fourier mode counts. — [problems.py](https://raw.githubusercontent.com/proximafusion/constellaration/main/src/constellaration/problems.py)
- **Higher resolution is used in practice.** The Weißberg et al. geometric-task entry optimized in a 306-dimensional Fourier space, which exceeds the dataset's 4×4 (D = 80) parameterization. — [arXiv 2608.28224](https://arxiv.org/html/2608.28224)
- **Leaderboard version.** The leaderboard (HF Space `proxima-fusion/constellaration-bench`) uses `constellaration==0.2.6`. NeurIPS 2025 experiments used 0.2.1. Version 0.2.6 applies the iota constraint to |ι|, while 0.2.1 penalized negative iota. — [README](https://raw.githubusercontent.com/proximafusion/constellaration/main/README.md)
- **Why v0.3.0 is not comparable** (released 2 Oct 2026).
  - It includes a flux-compression fix (PR #109, switching the Jacobian from VMEC to PEST coordinates) that "changes feasibility." On re-scoring the first three boundaries of the top two MHD-stable submissions, feasibility moved from 0.009 to 0.177 under v0.3.0.
  - PR #127, merged 8 Oct 2026, pinned the Space back to 0.2.6. The two submissions scored with 0.3.0 between 2 and 8 Oct 2026 were re-scored with 0.2.6.

  — [PR #127](https://github.com/proximafusion/constellaration/pull/127)
- **Other v0.3.0 additions.** "Medium" and "Loose" tightness variants of the MHD-stable problem (#111), plus diversity and performance scoring utilities (#114–#116). — [Release v0.3.0](https://github.com/proximafusion/constellaration/releases/tag/v0.3.0)
- **Runner iota defect.** A community issue reports that the 0.2.6 optimization runners constrain the signed ι while the scorer uses |ι|. — [Issue #139](https://github.com/proximafusion/constellaration/issues/139); [DrWorkhard issue #114](https://github.com/DrWorkhard/nuclear-fusion-at-home/issues/114)
- **Score scale cross-check.** ExLLM reports the official P3 baseline as 129.796 ("best single point: 97"). The paper's Table 3 lists the ALM-NGOpt MHD-stable baseline as 130.0. The leaderboard score is therefore the raw HV, not a normalized [0, 1] score. — [ExLLM arXiv 2502.12845v4](https://arxiv.org/html/2502.12845v4); [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)

### Inferences
- **Hypervolume formula.** With reference point (−L∇B = 1, A = 20), the score is the area dominated by the feasible set:

  HV = ∫ from A_min to 20 of [1 + max{L∇B of feasible points with A' ≤ A}] dA

  A single point contributes (L∇B + 1)(20 − A).
  - **Check against the paper.** The three feasible ALM-NGOpt points in Table 5, (A, L∇B) = (7.93, 5.60), (9.98, 8.45) and (11.9, 11.1), give 6.6·2.05 + 9.45·1.92 + 12.1·8.1 ≈ 129.7. This matches 129.796 / 130.0, which confirms the formula. My own arithmetic.
  - **What 285 implies.** A score near 285 requires feasible points with very large L∇B at low-to-moderate A. For example, a single point needs (L∇B + 1)(20 − A) ≈ 285, such as L∇B ≈ 19 at A ≈ 6, or L∇B ≈ 27 at A ≈ 10. In practice a front sweeps this region.
- **Where the gains are.** Because the HV integrand runs from A_min to 20, the large-A region (A ≈ 12–20) contributes a lot of area. Pushing L∇B high at moderate-to-large A, and extending the front to low A, both pay. A team at 144 is likely missing either high-L∇B points or coverage at low A.
- **Fourier resolution.** The scoring code does not check it, and high-fidelity VMEC sets mpol/ntor ≥ 10. Submitting boundaries with more than 4×4 modes appears allowed. I found no explicit rule text confirming this.

### Gaps
- I could not read the Space's rules text, such as any maximum number of boundaries per MHD-stable submission. The Space is JS-rendered and its README is metadata only.
- I did not confirm whether the paper's v2 Table 2 lists the same thresholds as `problems.py` on `main`. The thresholds were taken from code, which the leaderboard actually uses.
- I did not resolve the exact normalization definition inside `normalized_magnetic_gradient_scale_length`, beyond the ×nfp factor and the paper's "a/N_fp" wording.

## Q2. How the dataset was generated, its coverage, and why so few dataset designs are feasible for Benchmark 3

### Takeaway
The roughly 158k boundaries came from sampling about 100k omnigenous-field target sets and turning them into boundaries by four routes: heuristic QP initialization, near-axis expansion, DESC optimization and VMEC++ optimization. nfp spans 1–5 with heavy oversampling of nfp = 3. Boundaries are at most 4×4 Fourier modes (80 DOF). Almost none of them meet the strict Benchmark 2/3 constraints. Participants found that leaderboard-level results needed their own optimization or generation rather than mining the dataset.

### Cited Findings
- **Omnigenous-field targets.** Following Dudt et al., with stellarator symmetry imposed. Knot positions come from Beta CDFs, and the mean field is fixed at 1 T. Sampling ranges (paper Table 6):

  | Quantity | Range |
  |---|---|
  | N_fp | 1–5 |
  | ι̃ (iota per field period) | 0.1–0.3 |
  | A | 4.0–12.0 |
  | ε_max (max elongation) | 4.0–7.0 |
  | α_Beta, β_Beta | 2.0–6.0 each |
  | Δ_edge (mirror) | 0.1–0.4 |

  — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **Generation routes and counts.**
  - About 100k target sets were sampled.
  - These yielded about 30k heuristic (QP model, Goodman et al.) candidates, about 49k near-axis (pyQSC) candidates, about 88k DESC-optimized boundaries, and a VMEC++-optimized subset of about 15k seeded by rotating ellipses.
  - In total about 182k candidates were produced, and about 158k evaluated without errors.

  — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **By nfp.** 15k (1), 20k (2), 68k (3), 27k (4), 28k (5). Fourier: poloidal and toroidal mode numbers ≤ 4, giving 80 DOF with R₀,₀ = 1. Runtime: DESC about 3 min per run on 32 vCPU; VMEC++ optimization about 1 h on 32 vCPU. — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **Equilibria.** Computed with VMEC++ in vacuum plus volume-averaged β = 1–5% with a linear pressure profile. The HF dataset has configs for vacuum and each β; the finite-β configs range from about 113k rows (1%) to about 27k rows (5%). — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2); [HF dataset card mirror](https://huggingface.co/datasets/SII-LWY/constellaration-data/blob/main/README.md)
- **Feasible dataset points are rare.** Only 41 (geometric) and 52 (simple-to-build) dataset points out of about 160k are feasible (paper Fig. 8 caption, generative-model section). — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **ExLLM sweep.** The ExLLM authors swept "about 180k released designs with nfp=3" and found no nonzero scores under the strict tolerances (P2/P3). — [ExLLM v4](https://arxiv.org/html/2502.12845v4)
- **Mercier stability.** Separately, only 14 filtered and scaled ConStellaration configurations are Mercier-stable for s > 0.2. This is a different stability criterion from Benchmark 3's vacuum well. — [Landreman et al. arXiv 2606.19523](https://arxiv.org/html/2606.19523)

### Inferences
- **Why so few are feasible.** The dataset targets were sampled with mirror ratio up to 0.4 and elongation up to 7, and were optimized toward QI fields. They were not optimized for vacuum well ≥ 0, flux compression ≤ 0.9, or log10 QI ≤ −3.5 at high fidelity. Hitting all five Benchmark 3 constraints simultaneously is therefore rare in the raw data.
- **Implication for the team.** The dataset is better used as a prior or initializer than as a source of feasible points. I did not find an explicit statement of this reasoning in the paper.

### Gaps
- I found no explicit count of Benchmark 3–feasible points in the raw dataset, and no paper statement explaining why.
- Iota-on-axis and other distribution histograms (paper figures) were not extracted numerically.

## Q3. Baselines (ALM-NGOpt and others) on Benchmark 3, with compute

### Takeaway
The only baseline that produced feasible points was the augmented-Lagrangian method with Nevergrad's NGOpt (ALM-NGOpt). It used nfp = 3, a rotating-ellipse start and 4×4 modes, and scored HV ≈ 130. It was run as four separate ε-constraint runs (A* ∈ {6, 8, 10, 12}) at 15–24 h on 96 vCPUs each. SciPy trust-constr and COBYQA failed.

### Cited Findings
- **Leaderboard baseline values (paper Table 3, ALM-NGOpt).** Geometric 0.969; simple-to-build 0.431; MHD-stable 130.0. — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **Paper Table 5 (MHD-stable).** Four ALM-NGOpt runs with aspect-ratio caps A* ∈ {6, 8, 10, 12} gave (A, L̃∇B, violation):

  | A | L̃∇B | Violation | Feasible (tolerance 1e-2) |
  |---|---|---|---|
  | 6.02 | 2.98 | 0.104 | No |
  | 7.93 | 5.60 | 0.0013 | Yes |
  | 9.98 | 8.45 | 0.0 | Yes |
  | 11.9 | 11.1 | 0.0021 | Yes |

  The resulting Pareto front is described as "sparse." — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **Setup and compute.**
  - All baseline experiments use nfp = 3 and start from rotating ellipses, with up to 4 poloidal/toroidal modes (D = 80), and run VMEC++ at low fidelity during optimization.
  - ALM-NGOpt used 96 vCPUs for 18 h (geometric) and 34 h (simple-to-build), and 15–24 h on 96 vCPUs per multi-objective run.
  - The SciPy optimizers ran about 40 h on 4 vCPUs.

  — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **ALM-NGOpt hyperparameters (MHD-stable).**
  - Penalty: initial ρ = 10, ρ_max = 1e8, increase factor σ = 5, feasibility factor τ = 0.8.
  - Trust region: initial δ₀ = 0.33, decay γ = 0.95, minimum δ = 0.05.
  - NGOpt budget per subproblem: min{20,000, 1500 + 300k} forward calls.
  - Penalty update: increase ρ_i unless c̃_i decreased by factor τ. Trust-region update: δ_{k+1} = max(δ_min, γδ_k).

  — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **Single-objective baselines (Table 4).** On simple-to-build, trust-constr and COBYQA ended infeasible; only ALM-NGOpt was feasible (L̃∇B 8.61). — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)
- **Generative-model experiment (not a Benchmark 3 baseline).** PCA, then random-forest feasibility classifiers (soft threshold 0.8), then a GMM prior, then adaptive MCMC. It found regions where samples were feasible under both the classifier and VMEC++ for P1/P2. No HV is reported. — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2)

### Inferences
- The baseline is an ε-constraint scalarization (one run per A cap) at nfp = 3 with 4×4 modes and no ML.
- Leaderboard scores of 280+ are more than 2× this baseline. The gains plausibly come from many more front points, higher-L∇B shapes and possibly other nfp or resolutions (see Q4–Q5).

### Gaps
- I found no published compute cost for any leaderboard-top method.

## Q4. Published participant methods and reported hypervolumes

### Takeaway
As of 21 Nov 2026, published method descriptions for Benchmark 3 are scarce. The top MHD-stable entrants, all around 283–285, have not published methods that I could find. Three groups of documented methods exist:
- an LLM-driven evolutionary optimizer (ExLLM, HV 133.6, nfp = 3, Aug–Oct 2025);
- a conditional flow-matching generator trained on about 48k self-generated feasible boundaries (Starwell, HV about 223 on its "best known front," nfp 3 and 4);
- for the geometric task only, ALM with L-BFGS plus finite differences and MDS latent recombination (Weißberg et al.).

The leading MHD-stable owner of a tracker dataset explicitly declined to share methods.

### Cited Findings
- **Leaderboard snapshot.** Read from the Space's Gradio config on 2026-11-21 by automated extraction; verify by eye before citing. Best MHD-stable score per user:

  | User | Best score | Date of best |
  |---|---|---|
  | SII-LWY | 285.27 | 2026-11-11 |
  | MehmetAkifOltulu | 285.25 | 2026-11-04 |
  | forge-fusion | 284.55 | 2026-11-02 |
  | AnonymousAussie | 283.33 | 2026-10-29 |
  | phanerozoic | 249.67 | 2026-10-26 |
  | Eppy@EP-Hard | 249.14 | 2026-10-23 |
  | Wenyang Li | 248.23 | 2026-10-18 |
  | mmdeuss | 242.94 | 2026-10-11 |
  | mart1401 | 233.74 | 2026-11-10 |
  | canyon-ant | 205.82 | 2026-09-09 |
  | davidkh | 187.91 | 2026-07-10 |
  | scadena-pf (Proxima) | 178.96 | 2026-10-08 |
  | doublemcyber | 138.72 | 2026-10-24 |
  | CreativeEngineer | 135.42 | 2026-01-08 |
  | NianRan1 (ExLLM author) | 133.50 | 2025-08-12 |

  The first score above 200 was canyon-ant on 2026-09-09. The first above 250 was AnonymousAussie at 268.40 on 2026-10-28. — [Leaderboard Space config](https://proxima-fusion-constellaration-bench.hf.space/config); [Space](https://huggingface.co/spaces/proxima-fusion/constellaration-bench)
- **Pace.** The jump from about 250 to about 285 happened within about two weeks (late Oct to early Nov 2026), across four different users. — same source as above
- **ExLLM (Experience-Enhanced LLM Optimization).** Ran, Wang, Zhang, Li, Ran, Li (Zhongguancun Academy) and Allmendinger (Manchester). arXiv:2502.12845, v4 dated 8 Oct 2025.
  - Method: an LLM-as-optimizer evolutionary loop with an "experience" memory. nfp was fixed at 3 "according to official code."
  - Key trick: promoting the feasibility measure to an explicit objective, which sped convergence to nonzero scores and gave "hundreds of nonzero feasible points per run."
  - Results: P2 0.505 vs 0.431 for ALM-NGOpt. P3 133.634 (best single point 103) vs 129.796 (best single point 97), about 3% better.
  - The evaluation budget for the stellarator task is not stated.

  — [ExLLM v4](https://arxiv.org/html/2502.12845v4)
- **Starwell (HF model by `phanerozoic`).**
  - Model: a conditional flow-matching transformer with one token per boundary Fourier mode, conditioned (adaptive layer norm) on nfp, log A and log L∇B. Sampling takes 64 integration steps.
  - Training data: 48,071 boundaries passing all five `mhd_stable` constraints at high fidelity, 33,561 at nfp = 3 and 14,510 at nfp = 4.
  - Auxiliary heads: a metric-regression head; a validity critic (MLP ensemble, corrected using nearby scored boundaries from a bundled library) used to rank samples; and a "chain" head grading extra physics (Mercier, ballooning, ripple, Γ_c, alpha losses). It also has a coil head and an L-BFGS coil polish.
  - Reported HV: the best known front has 1,386 boundaries and HV 223.49; 159 valid standalone samples reach 220.57; an evenly spaced 256-point subset reaches 222.94.
  - Release date and compute are not stated.

  — [Starwell model card](https://huggingface.co/phanerozoic/starwell); HV figures via [search summary of the same card](https://huggingface.co/phanerozoic/starwell)
- **phanerozoic declined to share methods.** The same user (`phanerozoic`) runs a tracker dataset. On 15 May 2026, SII-LWY (now #1) asked whether the user's strong mhd_stable result came from CMA-ES, Bayesian optimization, evolutionary search or manual tuning, and whether it searched raw Fourier coefficients or a latent space. The owner declined to disclose. — [HF discussion](https://huggingface.co/datasets/phanerozoic/constellaration-tracker-results/discussions/1)
- **Weißberg, Heep, Lähner, Bernard (Univ. Bonn / Lamarr), arXiv:2608.28224, 28 Aug 2026.** Geometric task only.
  - Optimizer: ALM in the style of Cadena et al., with NGOpt/CMA-ES replaced by L-BFGS on forward finite-difference gradients through VMEC++, in a 306-dimensional Fourier space. Compute: 96 concurrent VMEC++ instances per gradient, about 1 h per run.
  - Latent space: an MDS 2-D latent of top solutions with mean-value-coordinate decoding allows "recombination of existing solutions."
  - Results: topped the geometric leaderboard in May 2026 (0.9743); a refined blend reached 0.9763.
  - Note: it refines previous top-5 public entries, which would conflict with the team's no-seeding rule.

  — [arXiv 2608.28224](https://arxiv.org/html/2608.28224)
- **Community "nuclear-fusion-at-home" (DrWorkhard), Oct 2026.** Exploratory nfp = 2 work with local Gauss–Newton/trust-region, Broyden, ALM-NGOpt and DESC.
  - Findings: vacuum well can be fixed cheaply, but flux compression floors at about 1.06–1.09 for its nfp = 2 seeds. QI and flux compression trade off through elongation. The best ALM run reached FC 0.838 but log10 QI −2.66.
  - Also: the default 4×4 truncation degraded one seed's log10 QI from −4.62 to −3.45 at high fidelity.
  - No HV reported.

  — [Issue #114](https://github.com/DrWorkhard/nuclear-fusion-at-home/issues/114)
- **Proxima "Bringing Fusion Down to Earth" blog (HF, 2 Jul 2025).** Frames the challenge around surrogate models that replace VMEC++ calls. No method results. — [HF blog](https://huggingface.co/blog/cgeorgiaw/constellaration-fusion-challenge)
- **Related ML work using ConStellaration.** Not Benchmark 3 entries:
  - RAMBO, Bayesian optimization with DP mixtures of GPs, evaluated on the simple-to-build task. — [arXiv 2601.20043](https://arxiv.org/html/2601.20043v1)
  - Landreman et al., data-informed (PCA) parameter spaces for Bayesian optimization of alpha confinement. — [arXiv 2606.19523](https://arxiv.org/html/2606.19523)
  - CoilStellaration, Proxima's coil dataset (Ewers, Cadena, Foppiani, Merlo, 9 Nov 2026), with no leaderboard analysis. — [arXiv 2611.12689](https://arxiv.org/html/2611.12689)

### Inferences
- **Documented methods top out well below the leaders.** Gradient-free ALM at nfp 3 with 4×4 modes reaches about 130, LLM-evolutionary search about 134, and a generative model on about 48k self-generated feasible points about 223. The 283–285 cluster is undocumented.
- **Likely ingredients behind the 280+ scores.** Not confirmed by any source:
  - large-scale constraint-satisfying optimization producing thousands of feasible points (Starwell's front alone had 1,386);
  - nfp 3–4 or higher;
  - more than 4×4 modes;
  - gradient-based or finite-difference refinement in the style of Weißberg;
  - feasibility-first merit functions (as in ExLLM and the DrWorkhard pilot).
- **Tactic that respects the no-seeding rule.** Build a large self-generated feasible library (Starwell-style), train a conditional generator or critic on it, then refine its samples locally at high fidelity.

### Gaps
- I found no paper, README or blog describing the methods of SII-LWY, MehmetAkifOltulu, forge-fusion, AnonymousAussie, Eppy@EP-Hard, Wenyang Li, mmdeuss, mart1401, canyon-ant or davidkh for Benchmark 3. Searches on several of these handles returned nothing relevant.
- I found no NeurIPS 2025 / ML4PS workshop write-up of a Benchmark 3 method beyond ExLLM.
- Starwell's compute, dates and how its 48k training boundaries were generated are not described on the card. Its relationship to phanerozoic's 249.67 leaderboard score is unstated.

## Q5. Which nfp and shapes populate the best part of the Benchmark 3 Pareto front

### Takeaway
No official statement exists. Indirect evidence points to nfp 3 and 4, possibly 5, as the populated region: all leaderboard submissions are reportedly nfp 3–5, Starwell's feasible library is nfp 3 (70%) and nfp 4 (30%), and the baselines and ExLLM used nfp = 3. Because L∇B is multiplied by nfp, the objective does not mechanically favor high nfp.

### Cited Findings
- **Leaderboard nfp (third-party aggregate).**
  - An October 2026 aggregate analysis of `constellaration-bench-results` counted 21,049 submitted boundaries from 7 users. Every submission was nfp 3, 4 or 5, with none at nfp 2.
  - At A 9–11, 96 nfp-3 boundaries were exactly feasible; the best had A 10.74 and L∇B 12.67 (vs the baseline's 9.98 / 8.45).

  — [DrWorkhard issue #114](https://github.com/DrWorkhard/nuclear-fusion-at-home/issues/114)
- **Starwell's feasible set** is nfp 3 (33,561) and nfp 4 (14,510); example usage uses nfp 3 and 4. — [Starwell card](https://huggingface.co/phanerozoic/starwell)
- **Baselines and ExLLM** used nfp = 3. — [arXiv HTML v2](https://arxiv.org/html/2506.19583v2); [ExLLM v4](https://arxiv.org/html/2502.12845v4)
- **nfp normalization.** L∇B is multiplied by nfp so the metric is nfp-independent. — [forward_model.py v0.2.6](https://raw.githubusercontent.com/proximafusion/constellaration/v0.2.6/src/constellaration/forward_model.py)

### Inferences
- **Why nfp 2 is hard.** At nfp = 2, achieving flux compression ≤ 0.9 together with QI ≤ 10^−3.5 appears difficult (DrWorkhard's floor of about 1.06–1.09). This suggests nfp ≥ 3 for feasibility. That is one group's exploratory evidence, not a general result.
- **Compactness pull.** The iota/nfp ≥ 0.25 constraint and the A objective pull toward compact shapes. Very low A with QI at high nfp is geometrically hard. nfp 3–4 is a reasonable starting range for the team.

### Gaps
- I found no official or peer-reviewed breakdown of the Benchmark 3 front by nfp, aspect ratio or elongation.
- The 7-user / 21k-boundary count is from Oct 2026 and predates later entrants. The leaderboard config extraction listed 15 users with MHD-stable scores by 21 Nov 2026, so the user count may reflect a different snapshot or problem filter.
