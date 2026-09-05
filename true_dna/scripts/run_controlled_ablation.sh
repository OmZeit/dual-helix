#!/usr/bin/env bash
set -euo pipefail

# Linux launcher for the two validated experiment profiles. The required first
# flag picks an explicitly conservative memory envelope for the GPU class.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${1:-}" in
  --rtx5080)
    PROFILE="rtx5080_16gb"
    shift
    ;;
  --a100)
    PROFILE="a100_80gb"
    shift
    ;;
  *)
    echo "Usage: $0 --rtx5080|--a100 [run_controlled_ablation.py options]" >&2
    exit 2
    ;;
esac

if [[ -x .venv/bin/python ]]; then
  PYTHON_CMD=.venv/bin/python
else
  PYTHON_CMD=python3
fi

if [[ "${1:-}" == "--smoke" ]]; then
  shift
  exec "$PYTHON_CMD" scripts/smoke_backbones.py --profile "$PROFILE" "$@"
fi

exec "$PYTHON_CMD" run_controlled_ablation.py --profile "$PROFILE" "$@"
