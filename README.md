# 2.156 — Final Project

Fall 2026. MIT. 2.156 AI & ML for Engineering Design

Fatak Borhani, Joel Pederson, & Leif Akerley

**Project:** Generative design of easier-to-build stellarators. Can learned models
(surrogates, latent-space search) find feasible
[ConStellaration](https://github.com/proximafusion/constellaration) Benchmark 3
(MHD-stable QI) designs for a fraction of the physics evaluations the ALM-NGOpt
baseline needs? Plan, findings so far, and open questions: **[Docs/plan.md](Docs/plan.md)**.

Key dates: mid-term report **Nov 18** · poster **Dec 4** · poster session + final report **Dec 9**.

## Setup

Needs a conda installer ([Miniforge](https://github.com/conda-forge/miniforge) recommended)
and ~3 GB of disk (2.3 GB env + 0.6 GB dataset).

```bash
git clone https://github.com/FatakBorhani/2.156-final-project.git
cd 2.156-final-project
./setup_env.sh                  # creates the "stellarator" env and a Jupyter kernel (several minutes)
conda activate stellarator
```

`setup_env.sh` installs **`constellaration==0.2.6`**, the exact version the leaderboard
scores with. Don't upgrade it: 0.3.0 changed the Benchmark 3 feasibility math. Tested on
macOS (Apple Silicon). On Linux it should work as-is; the upstream package is tested on
Ubuntu 22.04/24.04 with Python 3.10.

Why a script instead of `mamba env create -f environment.yml`: the package pins
`booz-xform==0.0.9`, which has no macOS wheel and won't build with pybind11 3.x. The
script builds it against the env's pybind11 2.x first.

## Scripts

All scripts run from the repo root with the `stellarator` env active. Each one's
docstring has details.

| Script | What it does | Time |
|---|---|---|
| `scripts/evaluate.py` | **The evaluator.** Scores boundaries against Benchmark 3 with VMEC++ in parallel, appending each result to a JSONL ledger (resumable). Every physics evaluation we count should go through this. | ~2 min/design/core |
| `scripts/export_designs.py` | Exports the N most nearly-feasible dataset designs as evaluator input | ~2 min |
| `scripts/count_feasible.py` | Per-constraint pass rates and feasibility counts over the dataset | ~1 min after download |
| `scripts/check_feasibility.py` | Same counts using the package's own `compute_feasibility`, for all 3 benchmarks | ~2 min |
| `scripts/time_eval.py` | Times one official evaluation of saved sample designs | ~2 min/design |

Evaluator input is one design per line, `{"id": ..., "boundary": <SurfaceRZFourier JSON>}`:

```bash
python scripts/export_designs.py data/near_feasible_100.jsonl --n 100
python scripts/evaluate.py data/near_feasible_100.jsonl results/runs/seed100.jsonl --workers 12
```

From a notebook or your own script: `sys.path.append("scripts")`, then
`from evaluate import evaluate_designs, summarize`. The evaluator uses worker processes,
so in a `.py` file call it under `if __name__ == "__main__":`. Piping code into
`python -` won't work on macOS.

The first data script downloads the dataset (~600 MB) into the Hugging Face cache.
`data/` is gitignored; small results we want to keep go in `results/`.

## Layout

```
Docs/          plan.md, presentations
scripts/       evaluator + dataset analysis
results/       small JSON/JSONL outputs worth keeping
data/          local inputs and scratch outputs (gitignored)
```

## AI use

Claude (Claude Code) helped set up the environment, write the scripts in `scripts/`,
analyze the dataset, and draft `Docs/plan.md`. The course requires an AI-use statement
on every submission, so keep this section current.
