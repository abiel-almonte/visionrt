#!/bin/bash
set -euo pipefail

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc

if [ -n "${VIRTUAL_ENV-}" ]; then
  export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH-}"
fi

if [[ "${1-}" == "release" ]]; then
  echo "Building wheel for release..."
  uv pip install build twine
  uv run python -m build --wheel
else
  echo "Dev install (editable)..."
  uv sync
  uv pip install -e . --no-build-isolation
  PYTHONPATH=. uv run pybind11-stubgen visionrt._visionrt -o .
fi