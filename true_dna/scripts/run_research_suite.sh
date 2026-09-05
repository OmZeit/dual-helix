#!/usr/bin/env bash
set -euo pipefail

# Thin native-WSL wrapper around the staged Python research suite.  The Python
# runner defaults to a no-GPU-work plan; add --run only after inspecting it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_research_suite.sh --rtx5080|--a100 [options]

Examples:
  # Create the default three-seed Transformer tokenizer-study plan.
  bash scripts/run_research_suite.sh --rtx5080

  # Execute it after inspecting experiments/research_suite_v1/suite_manifest.json.
  bash scripts/run_research_suite.sh --rtx5080 --run

  # Plan an architecture study; no GPU work until --run is added.
  bash scripts/run_research_suite.sh --rtx5080 --stages architecture

All remaining options are passed to scripts/run_research_suite.py.
EOF
}

PROFILE=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rtx5080)
      PROFILE="rtx5080_16gb"
      shift
      ;;
    --a100)
      PROFILE="a100_80gb"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$PROFILE" ]]; then
  echo "Select one hardware profile: --rtx5080 or --a100" >&2
  exit 2
fi
if [[ "$PROJECT_DIR" == /mnt/* ]]; then
  echo "Run from the native WSL workspace (for example ~/dual-helix-native/true_dna), not /mnt/<drive>." >&2
  exit 2
fi

if [[ -x "$WORKSPACE_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$WORKSPACE_DIR/.venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$PROJECT_DIR/.venv/bin/python"
else
  PYTHON_CMD=python3
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Credentials are loaded only for an actual W&B-backed training run.  Planning
# and explicit local --no-wandb smoke runs remain usable offline.
if [[ " ${ARGS[*]} " == *" --run "* && " ${ARGS[*]} " != *" --no-wandb "* ]]; then
  source "$SCRIPT_DIR/export_wandb_credentials.sh"
fi

exec "$PYTHON_CMD" scripts/run_research_suite.py --profile "$PROFILE" "${ARGS[@]}"
