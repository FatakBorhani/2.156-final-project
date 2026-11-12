# Project plan: learned search for MHD-stable QI stellarators

*Drafted 2026-11-11 by Leif, with Claude. Up for discussion, so edit freely.*

## Objective

**Can learned models replace most of the expensive physics search in stellarator design?**

We measure that on ConStellaration **Benchmark 3 (MHD-stable QI)**: the number of
feasible designs, and their hypervolume, as a function of **VMEC++ evaluations spent**.
We compare against the paper's ALM-NGOpt baseline on the same evaluation budget.

Success means feasible Benchmark 3 designs for a small fraction of the baseline's
compute. Fallback: if nothing gets fully feasible, we report the smallest constraint
violation reached vs. evaluations spent, against the baseline. That's still a
complete result.

## What the benchmark asks (v0.2.6, the leaderboard version)

Two objectives, scored as the hypervolume of the *feasible* designs only
(reference point (1, 20)):

- **Maximize L∇B**, the minimum normalized magnetic gradient scale length. A smoother
  field can be made by coils farther from the plasma, so they're simpler to build.
- **Minimize aspect ratio**, which makes the device more compact.

Five constraints. A design is feasible if every normalized violation is ≤ 0.01:

| Constraint | Physics |
|---|---|
| log10(QI residual) ≤ -3.5 | Close to quasi-isodynamic: trapped particles (e.g. alphas) stay confined |
| \|edge ι\| per field period ≥ 0.25 | Enough field-line twist for confinement |
| Edge mirror ratio ≤ 0.25 | Limits field-strength variation on the boundary |
| Flux compression in bad curvature ≤ 0.9 | Geometric proxy for turbulent transport |
| Vacuum well ≥ 0 | Stability against ideal-MHD interchange modes |

## What we found (2026-11-11)

- **The dataset has no feasible designs.** 0 of 158,685 evaluated designs pass Benchmark 3,
  and none pass Benchmarks 1 or 2 either. We checked with the package's own
  `compute_feasibility` (`results/official_feasibility.json`). The closest design misses
  by 0.20; 10 are within 0.25 and 4,490 within 0.5.
- **QI is the bottleneck.** Only 0.11% of designs pass the QI constraint within the
  benchmark's tolerance (126 meet log10(qi) ≤ -3.5 exactly). The other constraints pass
  5–36% of the time (`results/b3_feasibility.json`).
- **The dataset was physics-generated, but not for Benchmark 3.** Each row samples a
  target QI field and geometry, then builds a boundary toward it. 88k rows come from
  DESC optimization, 49k from near-axis expansion, 30k from QP initial guesses and 16k
  from VMEC optimization. Those optimizations targeted QI quality, aspect ratio,
  elongation and ι, never vacuum well or flux compression. About 23.5k boundaries
  failed the forward model.
- **The evaluator runs locally and matches the dataset exactly.** One high-fidelity
  evaluation takes ~105–160 s on one core and ~3.6 GB of RAM (M3 Max); about 12 can run
  in parallel. Re-scoring dataset designs reproduces the stored metrics to ~1e-7
  (`results/evaluator_check_4designs.jsonl`).
- **The baseline is expensive.** `launch_alm_mhd_stable_stellarator.py` runs 50 outer
  ALM iterations, each with a Nevergrad budget that starts at 1,500 and grows by 300
  per iteration up to 20,000. That's up to ~442k evaluations per aspect-ratio bound, and
  the script runs 4 bounds (up to ~1.8M evaluations), at *very low* fidelity with 48
  workers. Very-low-fidelity evaluations took 1.5–11 s on 6 designs here, so a full
  baseline run is roughly 1,000–3,000 CPU-hours. Not a laptop job.
- **Low fidelity ≠ high fidelity.** On the same 4 designs, very-low-fidelity violations
  differed from high fidelity by up to ±0.025. Baseline outputs must be re-scored at
  high fidelity. The upside: low fidelity could also be a cheap pre-filter for us.

## Method

1. **Surrogates.** Predict the 5 constraint metrics and 2 objectives from the boundary
   Fourier coefficients, especially log10(qi). Add a classifier for VMEC failure,
   trained on the ~23.5k failed rows. Compare boosted trees with an MLP, with and
   without PCA on the coefficients. Split train/test by generation method too, to test
   generalization honestly.
2. **Latent-space search.** Compress shapes (PCA or a VAE trained on QI-like designs),
   then search that space with the surrogates for predicted-feasible designs spread
   along the L∇B vs. aspect-ratio front.
3. **Physics in the loop.** Evaluate the top ~100 candidates with VMEC++
   (`scripts/evaluate.py`), add them to the training set, refit, and repeat a few
   rounds. This matters because the dataset has no feasible points, so the surrogates
   are extrapolating exactly where we care.
4. **Baseline.** ALM-NGOpt from the benchmark repo, given the same budget of
   evaluations, then re-scored at high fidelity. The full budget is out of reach
   locally (see above), so options are a reduced-budget run on one aspect-ratio bound,
   MIT ORCD Engaging, or the paper's reported numbers. **Decision needed.**

Out of scope unless time allows: diffusion models (no good examples to learn from),
the target → boundary inverse problem, the DESC-equilibrium and 1%-β dataset configs,
and chasing the leaderboard.

## Who does what (proposed)

| Area | Owner |
|---|---|
| Surrogates + VMEC-failure classifier | ? |
| Latent space + search | ? |
| Evaluation pipeline, baseline, final comparison | Leif |

## Milestones

| Date | Deliverable |
|---|---|
| Wed Nov 18 | Mid-term report: objective, data findings, evaluator + timing, first surrogate accuracy, plan |
| ~Nov 25 | First round of searched candidates evaluated with VMEC++ |
| Wed Dec 2 | Final physics-in-the-loop rounds and baseline comparison done |
| Fri Dec 4 | Poster due |
| Wed Dec 9 | Poster session (Hart Nautical Gallery) + final report |

## Open questions

- Baseline: reduced local run, Engaging, or the paper's numbers?
- Do we restrict to one number of field periods or keep all? Evaluated designs by nfp:
  1: 14.6k, 2: 19.9k, 3: 68.2k, 4: 27.8k, 5: 28.1k. The baseline starts from nfp = 3.
- Ask the course staff in our project meeting: is "feasible designs per CPU-hour vs. ALM-NGOpt" a convincing enough headline?
- Before the report: check the physics descriptions above against the ConStellaration
  paper (arXiv 2506.19583).
