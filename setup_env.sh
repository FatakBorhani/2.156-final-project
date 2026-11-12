#!/usr/bin/env bash
# Create the `stellarator` conda env with the leaderboard version of the benchmark.
#
#   ./setup_env.sh                 # env named "stellarator"
#   ./setup_env.sh my-env-name     # or pick a name
#
# Why not just `pip install constellaration`: it pins booz-xform==0.0.9, which has no
# macOS wheel. pip's isolated build pulls pybind11 3.x and fails, so we build it against
# the env's pybind11 2.x with --no-build-isolation first. On Linux this also works.
set -euo pipefail

ENV_NAME="${1:-stellarator}"
CONSTELLARATION_VERSION="0.2.6"   # the version the leaderboard scores with; don't bump

cd "$(dirname "$0")"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
SOLVER=conda
command -v mamba >/dev/null 2>&1 && SOLVER=mamba

$SOLVER env create -y -n "$ENV_NAME" -f environment.yml
set +u  # the compilers package's activation hooks read unset variables
conda activate "$ENV_NAME"
set -u

pip install --no-build-isolation "booz-xform==0.0.9"
pip install "constellaration==${CONSTELLARATION_VERSION}"

python - <<'EOF'
import importlib.metadata as md
import booz_xform, vmecpp  # noqa: F401  (fails here if the native builds are broken)
from constellaration import problems  # noqa: F401
for p in ("constellaration", "vmecpp", "booz-xform", "pybind11", "numpy", "jax"):
    print(f"{p:16s} {md.version(p)}")
EOF

python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
echo "Done. Activate with: conda activate $ENV_NAME"
