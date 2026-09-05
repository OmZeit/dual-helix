#!/usr/bin/env bash
# Create the Linux CUDA environment used by the controlled-ablation runners.
#
# Run from a cloned repository on an NVIDIA Linux host or in a Colab terminal:
#   bash true_dna/scripts/setup_cuda_env.sh --rtx5080
#   bash true_dna/scripts/setup_cuda_env.sh --a100
#
# This deliberately does not install system packages.  CUDA (including nvcc),
# an NVIDIA driver, a C++ compiler, and Python 3.10--3.12 must already exist.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash true_dna/scripts/setup_cuda_env.sh --rtx5080|--a100 [--python PATH]

Creates or reuses .venv at the repository root, installs the CUDA 12.8
runtime and Mamba kernels, installs this repository, then verifies CUDA and
Mamba imports.  Set TRUE_DNA_MAX_JOBS to limit Mamba build parallelism.
EOF
}

if [[ "${1:-}" == "--rtx5080" ]]; then
  PROFILE="rtx5080_16gb"
  ARCH="12.0"
  shift
elif [[ "${1:-}" == "--a100" ]]; then
  PROFILE="a100_80gb"
  ARCH="8.0"
  shift
else
  usage >&2
  exit 2
fi

PYTHON_BIN="${TRUE_DNA_PYTHON:-python3}"
if [[ "${1:-}" == "--python" ]]; then
  [[ $# -ge 2 ]] || { echo "--python requires a path." >&2; exit 2; }
  PYTHON_BIN="$2"
  shift 2
fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

[[ "$(uname -s)" == "Linux" ]] || {
  echo "True DNA CUDA training requires Linux. Use WSL2, a native Linux host, or Colab." >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not available; install/use an NVIDIA GPU runtime." >&2; exit 1; }
command -v g++ >/dev/null || { echo "g++ is not available; install a C++ compiler before building Mamba kernels." >&2; exit 1; }

# NVIDIA's WSL packages put nvcc in /usr/local/cuda-12.8/bin without adding
# it to a new shell's PATH. Prefer the project's pinned toolkit, then accept
# an explicit CUDA_HOME, the /usr/local/cuda symlink, or an existing PATH.
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  CUDA_NVCC="$CUDA_HOME/bin/nvcc"
elif [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
  CUDA_NVCC=/usr/local/cuda-12.8/bin/nvcc
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  CUDA_NVCC=/usr/local/cuda/bin/nvcc
else
  CUDA_NVCC="$(command -v nvcc || true)"
fi
[[ -n "$CUDA_NVCC" && -x "$CUDA_NVCC" ]] || {
  echo "nvcc is not available; install the CUDA 12.8 toolkit before building Mamba kernels." >&2
  exit 1
}
RESOLVED_NVCC="$(readlink -f "$CUDA_NVCC")"
CUDA_HOME="$(dirname "$(dirname "$RESOLVED_NVCC")")"

"$PYTHON_BIN" - <<'PY'
import sys
import sysconfig
from pathlib import Path

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Python 3.10--3.12 is required; found {sys.version.split()[0]}. "
        "On Ubuntu, install python3.12 and rerun with --python python3.12."
    )

header = Path(sysconfig.get_paths()["include"]) / "Python.h"
if not header.is_file():
    raise SystemExit(
        f"Python development headers are missing ({header}). "
        f"On Ubuntu, run: sudo apt install python{sys.version_info.major}.{sys.version_info.minor}-dev"
    )
PY

if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
  echo "The NVIDIA driver did not expose a usable GPU." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || {
  echo "CUDA_HOME does not contain nvcc: $CUDA_HOME" >&2
  exit 1
}
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS="${TRUE_DNA_MAX_JOBS:-4}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -r "$REPO_ROOT/true_dna/requirements_rtx5080.txt"
"$VENV_PYTHON" -m pip install --no-build-isolation \
  -r "$REPO_ROOT/true_dna/requirements_rtx5080-kernels.txt"
"$VENV_PYTHON" -m pip install --no-deps "$REPO_ROOT"

TRUE_DNA_PROFILE="$PROFILE" "$VENV_PYTHON" - <<'PY'
import os

import torch
from mamba_ssm import Mamba2  # noqa: F401
import causal_conv1d  # noqa: F401

if not torch.cuda.is_available():
    raise SystemExit("PyTorch was installed, but CUDA is unavailable to it.")

index = torch.cuda.current_device()
name = torch.cuda.get_device_name(index)
memory_gib = torch.cuda.get_device_properties(index).total_memory / 1024**3
print(f"Environment ready for {os.environ['TRUE_DNA_PROFILE']}: {name} ({memory_gib:.1f} GiB)")
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; capability {torch.cuda.get_device_capability(index)}")
PY

cat <<EOF

Next, activate the environment and run the no-data smoke test:
  source "$VENV_DIR/bin/activate"
  cd "$REPO_ROOT/true_dna"
  bash scripts/run_controlled_ablation.sh --${PROFILE%%_*} --smoke
EOF
