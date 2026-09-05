#!/usr/bin/env bash
set -euo pipefail

# Fast end-to-end training check for protocol v2. This uses a four-record
# synthetic FASTA, a tiny Transformer, mask-only corruption, and no W&B.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
FIXTURE="$PROJECT_DIR/tests/fixtures/smoke_protocol_v2.fa"

if [[ "$PROJECT_DIR" == /mnt/* ]]; then
  echo "Run this from the native WSL checkout, not /mnt/<drive>." >&2
  exit 2
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "Missing smoke FASTA fixture: $FIXTURE" >&2
  exit 2
fi

if [[ -x "$WORKSPACE_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$WORKSPACE_DIR/.venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$PROJECT_DIR/.venv/bin/python"
else
  PYTHON_CMD=python3
fi

STEPS="${SMOKE_STEPS:-5}"
if [[ ! "$STEPS" =~ ^[0-9]+$ ]] || (( STEPS < 3 || STEPS > 7 )); then
  echo "SMOKE_STEPS must be an integer from 3 through 7 (default: 5)." >&2
  exit 2
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PARENT="${SMOKE_OUTPUT_ROOT:-$PROJECT_DIR/experiments/protocol_v2_smoke}"
RUN_ROOT="$OUTPUT_PARENT/$RUN_STAMP"
DATA_DIR="$RUN_ROOT/input"
RUN_DIR="$RUN_ROOT/run"
FASTA="$DATA_DIR/smoke.fa"
SPLIT="$DATA_DIR/split_manifest.jsonl"

mkdir -p "$DATA_DIR" "$RUN_DIR"
cp "$FIXTURE" "$FASTA"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ! "$PYTHON_CMD" scripts/train.py --help 2>&1 | grep -q -- "--mask-replacement-strategy"; then
  echo "This native checkout does not contain the protocol-v2 masking changes. Sync it before running the smoke test." >&2
  exit 2
fi

"$PYTHON_CMD" scripts/build_split_manifest.py \
  --fasta "$FASTA" \
  --output "$SPLIT" \
  --group-by assembly \
  --balance-by bases \
  --holdout-fraction 0.5 \
  --seed 43

DEVICE_ARGS=(--amp --tf32 --pin_memory)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found; this training path requires CUDA." >&2
  exit 2
fi

echo "Starting protocol-v2 smoke training: $STEPS optimizer steps"
echo "Output: $RUN_ROOT"

exec "$PYTHON_CMD" scripts/train.py \
  --fasta "$FASTA" \
  --split-manifest "$SPLIT" \
  --save_dir "$RUN_DIR" \
  --tokenizer-type base \
  --backbone transformer \
  --hidden_size 64 \
  --high_level_layers 2 \
  --num_attention_heads 4 \
  --batch_size 2 \
  --epochs 1 \
  --steps_per_epoch "$STEPS" \
  --max_length 128 \
  --stride 64 \
  --max_chunks_per_file 32 \
  --workers 0 \
  --lr 5e-4 \
  --lr_schedule constant \
  --warmup_steps 2 \
  --grad_accum 1 \
  --grad_clip 1.0 \
  --dropout 0.0 \
  --drop_path_rate 0.0 \
  --frameshift_prob 0.0 \
  --use_reverse_prob 0.0 \
  --lambda_rc_consist 0.0 \
  --rc_loss_prob 0.0 \
  --no-kmer-mix \
  --mask-coordinate-system base \
  --mask-replacement-strategy mask \
  --mask_fraction 0.15 \
  --mean_span_length 3.0 \
  --label_smoothing 0.0 \
  --eval_interval "$STEPS" \
  --eval_max_batches 2 \
  --save_every "$STEPS" \
  --log_interval 1 \
  --seed 43 \
  --no_wandb \
  "${DEVICE_ARGS[@]}"
