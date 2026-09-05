#!/usr/bin/env bash
set -euo pipefail

# Continue from an already verified corpus. This is intentionally separate
# from run_fresh_pretrain_and_ablation.sh so a download that completed before
# training can be resumed without deleting or fetching data again.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_validated_corpus_ablation.sh --rtx5080|--a100 [options] [-- TRAINING_ARGS...]

Requires the verified domain FASTAs and dataset manifest already present in
data/domain_fastas/. It builds an assembly-held-out split, then runs the three
single-base architecture baselines plus a matched Transformer BPE control.

Options:
  --no-wandb  Disable Weights & Biases logging (enabled by default)
  -h, --help  Show this help

Everything after -- is passed to run_controlled_ablation.py.
EOF
}

PROFILE=""
WANDB_ARGS=()
TRAINING_ARGS=()

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
    --no-wandb)
      WANDB_ARGS=(--no-wandb)
      shift
      ;;
    --)
      shift
      TRAINING_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PROFILE" ]]; then
  echo "Select one hardware profile: --rtx5080 or --a100" >&2
  exit 2
fi
if [[ "$PROJECT_DIR" == /mnt/* ]]; then
  echo "Run this from the native WSL workspace (for example ~/dual-helix-native/true_dna), not /mnt/<drive>." >&2
  exit 2
fi

if [[ -x "$WORKSPACE_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$WORKSPACE_DIR/.venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$PROJECT_DIR/.venv/bin/python"
else
  PYTHON_CMD=python3
fi

FASTAS=(data/domain_fastas/bacteria.fa data/domain_fastas/archaea.fa data/domain_fastas/eukaryotes.fa)
for fasta in "${FASTAS[@]}" data/domain_fastas/dataset_manifest.json; do
  if [[ ! -s "$fasta" ]]; then
    echo "Missing or empty verified corpus artifact: $fasta" >&2
    exit 2
  fi
done

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Prompt before work that can take time, rather than discovering missing
# experiment credentials after the held-out split has been generated.
if [[ ${#WANDB_ARGS[@]} -eq 0 ]]; then
  source "$SCRIPT_DIR/export_wandb_credentials.sh"
fi

if [[ ! -s dna_model/bpe_vocab/tokenizer.json ]]; then
  echo "[tokenizer] Building the deterministic 4,096-token BPE vocabulary from the verified corpus."
  "$PYTHON_CMD" scripts/train_bpe.py --fasta "${FASTAS[@]}" --vocab_size 4096
fi

"$PYTHON_CMD" scripts/prepare_ablation_data.py \
  --fasta "${FASTAS[@]}" \
  --output data/domain_fastas/split_manifest.jsonl \
  --group-by assembly \
  --seed 43

echo "[train] Starting base-token architecture baselines and the matched Transformer BPE control."
exec "$PYTHON_CMD" run_controlled_ablation.py \
  --profile "$PROFILE" \
  --fasta "${FASTAS[@]}" \
  --split-manifest data/domain_fastas/split_manifest.jsonl \
  --only-baselines \
  --include-tokenizer-ablation \
  --allow-dirty \
  "${WANDB_ARGS[@]}" \
  "${TRAINING_ARGS[@]}"
