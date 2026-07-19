#!/usr/bin/env bash
set -euo pipefail

echo "================================================================="
echo " True DNA Fresh RefSeq Full-Pretrain Pipeline"
echo "================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${PYTHON_CMD:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_CMD=".venv/bin/python"
  elif [[ -x "/mnt/g/DNA/true_dna/.venv/bin/python" ]]; then
    PYTHON_CMD="/mnt/g/DNA/true_dna/.venv/bin/python"
  else
    PYTHON_CMD="python3"
  fi
fi

FASTA_GLOB="data/domain_fastas/*.fa"
SAVE_DIR="experiments/full_pretrain_v1"
PREFLIGHT_DIR="experiments/full_pretrain_v1_preflight"
MAX_LENGTH=4096
STRIDE=2048
BATCH_SIZE=4
GRAD_ACCUM=8
HIDDEN_SIZE=768
LAYERS=16
ATTENTION_HEADS=16
STEPS=100000
PREFLIGHT_STEPS=500
EPOCHS=1
LR="0.0003"
WARMUP_STEPS=5000
SAVE_EVERY=10000
EVAL_MAX_BATCHES=1024
EVAL_SCHEDULE="0:1000,10000:5000,50000:10000"
NUM_GPUS=1
TARGET_MB_PER_DOMAIN=4096
PARALLEL_DOWNLOADS=4
MAX_ASSEMBLIES=20000
OVERSHOOT_MB=10
REFRESH_DATA=false
HARD_DELETE_DATA=false
RUN_PREFLIGHT=true
RESUME=false
RUN_BENCHMARKS=true
BENCHMARK_MILESTONES="10000 50000 100000"
ENABLE_WANDB=false
BENCHMARK_EVERY_CHECKPOINT=false
BENCHMARK_MANIFEST=""
BIOLOGY_FASTAS=()
BIOLOGY_LABELS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --refresh_data) REFRESH_DATA=true ;;
    --hard_delete_data) HARD_DELETE_DATA=true ;;
    --skip_preflight) RUN_PREFLIGHT=false ;;
    --resume) RESUME=true ;;
    --wandb) ENABLE_WANDB=true ;;
    --no_benchmarks) RUN_BENCHMARKS=false ;;
    --fa_glob) FASTA_GLOB="$2"; shift ;;
    --save_dir) SAVE_DIR="$2"; shift ;;
    --preflight_dir) PREFLIGHT_DIR="$2"; shift ;;
    --batch_size) BATCH_SIZE="$2"; shift ;;
    --grad_accum) GRAD_ACCUM="$2"; shift ;;
    --max_length) MAX_LENGTH="$2"; shift ;;
    --stride) STRIDE="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    --preflight_steps) PREFLIGHT_STEPS="$2"; shift ;;
    --num_gpus) NUM_GPUS="$2"; shift ;;
    --target_mb_per_domain) TARGET_MB_PER_DOMAIN="$2"; shift ;;
    --parallel_downloads) PARALLEL_DOWNLOADS="$2"; shift ;;
    --max_assemblies) MAX_ASSEMBLIES="$2"; shift ;;
    --overshoot_mb) OVERSHOOT_MB="$2"; shift ;;
    --benchmark_milestones) BENCHMARK_MILESTONES="$2"; shift ;;
    --benchmark_every_checkpoint) BENCHMARK_EVERY_CHECKPOINT=true ;;
    --benchmark_manifest) BENCHMARK_MANIFEST="$2"; shift ;;
    --biology_fasta) BIOLOGY_FASTAS+=("$2"); shift ;;
    --biology_label) BIOLOGY_LABELS+=("$2"); shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

if [[ "$REFRESH_DATA" == true ]]; then
  echo ""
  echo "[1/5] Rebuilding fresh RefSeq corpus"
  DATA_ARGS=(
    "$PYTHON_CMD" scripts/prepare_refseq_pretrain_data.py
    --species-json configs/species.json
    --target-mb-per-domain "$TARGET_MB_PER_DOMAIN"
    --run-download
    --rebuild-tokenizer
    --verify
    --write-manifest
    --refseq-only
    --parallel-downloads "$PARALLEL_DOWNLOADS"
    --max-assemblies "$MAX_ASSEMBLIES"
    --overshoot-mb "$OVERSHOOT_MB"
  )
  if [[ "$HARD_DELETE_DATA" == true ]]; then
    DATA_ARGS+=(--hard-delete-generated-data)
  fi
  "${DATA_ARGS[@]}"
else
  echo ""
  echo "[1/5] Data refresh skipped. Using existing FASTA files: $FASTA_GLOB"
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCHER=(torchrun --nproc_per_node="$NUM_GPUS")
else
  LAUNCHER=("$PYTHON_CMD")
fi

run_train() {
  local out_dir="$1"
  local steps="$2"
  local save_every="$3"
  local eval_interval="$4"
  local eval_max_batches="$5"
  shift 5
  local wandb_args=()
  if [[ "$ENABLE_WANDB" != true ]]; then
    wandb_args=(--no_wandb)
  fi
  mkdir -p "$out_dir"
  "${LAUNCHER[@]}" scripts/train.py \
    --fasta "$FASTA_GLOB" \
    --save_dir "$out_dir" \
    --max_length "$MAX_LENGTH" \
    --stride "$STRIDE" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --hidden_size "$HIDDEN_SIZE" \
    --high_level_layers "$LAYERS" \
    --num_attention_heads "$ATTENTION_HEADS" \
    --backbone dual_helix \
    --k_mer_sizes 3 \
    --steps_per_epoch "$steps" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --warmup_steps "$WARMUP_STEPS" \
    --lr_schedule cosine \
    --save_every "$save_every" \
    --eval_interval "$eval_interval" \
    --eval_max_batches "$eval_max_batches" \
    --gradient_checkpointing \
    --amp \
    --tf32 \
    --pin_memory \
    --use_curriculum \
    --use_moe \
    --moe_num_experts 4 \
    --use_reverse_prob 0.5 \
    --lambda_rc_consist 0.1 \
    --rc_loss_prob 0.3 \
    --frameshift_prob 0.05 \
    --label_smoothing 0.05 \
    "${wandb_args[@]}" \
    "$@" 2>&1 | tee -a "$out_dir/training.log"
}

if [[ "$RUN_PREFLIGHT" == true && "$RESUME" == false ]]; then
  echo ""
  echo "[2/5] Running preflight: $PREFLIGHT_STEPS optimizer steps"
  run_train "$PREFLIGHT_DIR" "$PREFLIGHT_STEPS" "$PREFLIGHT_STEPS" 100 128
  "$PYTHON_CMD" scripts/tools/check_preflight.py \
    --run-dir "$PREFLIGHT_DIR" \
    --expected-steps-per-epoch "$PREFLIGHT_STEPS" \
    --forbid-npy-cache
else
  echo ""
  echo "[2/5] Preflight skipped"
fi

echo ""
echo "[3/5] Starting full pretrain"
RESUME_ARGS=()
if [[ "$RESUME" == true ]]; then
  if [[ -f "$SAVE_DIR/checkpoint_latest.pt" ]]; then
    RESUME_ARGS=(--resume_from "$SAVE_DIR/checkpoint_latest.pt")
  elif [[ -f "$SAVE_DIR/checkpoint_best.pt" ]]; then
    RESUME_ARGS=(--resume_from "$SAVE_DIR/checkpoint_best.pt")
  else
    echo "[ERROR] --resume requested but no checkpoint_latest.pt or checkpoint_best.pt found in $SAVE_DIR"
    exit 1
  fi
fi

run_train "$SAVE_DIR" "$STEPS" "$SAVE_EVERY" 1000 "$EVAL_MAX_BATCHES" \
  --eval_schedule "$EVAL_SCHEDULE" \
  "${RESUME_ARGS[@]}"

echo ""
echo "[4/5] Discovering checkpoint"
CHECKPOINT=""
if [[ -f "$SAVE_DIR/checkpoint_best.pt" ]]; then
  CHECKPOINT="$SAVE_DIR/checkpoint_best.pt"
elif [[ -f "$SAVE_DIR/checkpoint_latest.pt" ]]; then
  CHECKPOINT="$SAVE_DIR/checkpoint_latest.pt"
else
  CHECKPOINT="$(ls -t "$SAVE_DIR"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
fi

if [[ -z "$CHECKPOINT" ]]; then
  echo "[ERROR] No checkpoint found in $SAVE_DIR"
  exit 1
fi
echo "[Info] Selected checkpoint: $CHECKPOINT"

if [[ "$RUN_BENCHMARKS" == true ]]; then
  echo ""
  echo "[5/5] Running checkpoint benchmarks"
  declare -A GENOMIC_RESULT_CACHE=()
  declare -A BIOLOGY_RESULT_CACHE=()
  BENCHMARK_WANDB_ARGS=()
  if [[ "$ENABLE_WANDB" == true ]]; then
    SAFE_WANDB_ID="$(basename "$SAVE_DIR" | tr -c 'A-Za-z0-9_-' '_')_benchmarks"
    BENCHMARK_WANDB_ARGS=(
      --wandb
      --wandb_run_name "$(basename "$SAVE_DIR") benchmarks"
      --wandb_id "$SAFE_WANDB_ID"
    )
  fi

  run_benchmark() {
    local checkpoint_path="$1"
    local label="$2"
    local genomic_out_json="$SAVE_DIR/genomic_benchmark_${label}.json"
    local biology_out_json="$SAVE_DIR/biology_benchmark_${label}.json"
    local checkpoint_key

    if [[ ! -f "$checkpoint_path" ]]; then
      echo "[Benchmark] Missing $label checkpoint: $checkpoint_path"
      return 0
    fi
    checkpoint_key="$(readlink -f "$checkpoint_path" 2>/dev/null || true)"
    if [[ -z "$checkpoint_key" ]]; then
      checkpoint_key="$checkpoint_path"
    fi

    echo "[Benchmark] genomic/$label: $checkpoint_path"
    if [[ -n "${GENOMIC_RESULT_CACHE[$checkpoint_key]:-}" ]]; then
      cp "${GENOMIC_RESULT_CACHE[$checkpoint_key]}" "$genomic_out_json"
      GENOMIC_OK=true
    else
      GENOMIC_ARGS=(
        "$PYTHON_CMD" scripts/benchmark_genomic.py
        --checkpoint "$checkpoint_path"
        --tokenizer_path "dna_model/bpe_vocab/tokenizer.json"
        --max_length 1024
        --batch_size 8
        --pooling mean
        --include_random_model
        --out_json "$genomic_out_json"
      )
      if [[ -n "$BENCHMARK_MANIFEST" ]]; then
        GENOMIC_ARGS+=(--manifest "$BENCHMARK_MANIFEST")
      else
        GENOMIC_ARGS+=(--preset taxonomy_biotype)
      fi

      if "${GENOMIC_ARGS[@]}"; then
        GENOMIC_RESULT_CACHE[$checkpoint_key]="$genomic_out_json"
        GENOMIC_OK=true
      else
        GENOMIC_OK=false
      fi
    fi

    if [[ "$GENOMIC_OK" == true ]]; then
      "$PYTHON_CMD" -m dna_model.benchmark_logging \
        --run_dir "$SAVE_DIR" \
        --checkpoint "$checkpoint_path" \
        --result_json "$genomic_out_json" \
        --label "$label" \
        --benchmark_type genomic \
        "${BENCHMARK_WANDB_ARGS[@]}" || \
        echo "[Warning] Failed to record genomic benchmark metadata for $label."
    else
      echo "[Warning] Genomic benchmark failed for $label; checkpoint is still available."
    fi

    if [[ "${#BIOLOGY_FASTAS[@]}" -gt 0 ]]; then
      echo "[Benchmark] biology/$label: $checkpoint_path"
      if [[ -n "${BIOLOGY_RESULT_CACHE[$checkpoint_key]:-}" ]]; then
        cp "${BIOLOGY_RESULT_CACHE[$checkpoint_key]}" "$biology_out_json"
        BIOLOGY_OK=true
      else
        BIOLOGY_ARGS=(
          "$PYTHON_CMD" scripts/benchmark_biology.py
          --checkpoint "$checkpoint_path"
          --tokenizer_path "dna_model/bpe_vocab/tokenizer.json"
          --max_length 1024
          --batch_size 8
          --out_json "$biology_out_json"
        )
        for fasta in "${BIOLOGY_FASTAS[@]}"; do
          BIOLOGY_ARGS+=(--fasta "$fasta")
        done
        for label_name in "${BIOLOGY_LABELS[@]}"; do
          BIOLOGY_ARGS+=(--label "$label_name")
        done

        if "${BIOLOGY_ARGS[@]}"; then
          BIOLOGY_RESULT_CACHE[$checkpoint_key]="$biology_out_json"
          BIOLOGY_OK=true
        else
          BIOLOGY_OK=false
        fi
      fi

      if [[ "$BIOLOGY_OK" == true ]]; then
        "$PYTHON_CMD" -m dna_model.benchmark_logging \
          --run_dir "$SAVE_DIR" \
          --checkpoint "$checkpoint_path" \
          --result_json "$biology_out_json" \
          --label "$label" \
          --benchmark_type biology \
          "${BENCHMARK_WANDB_ARGS[@]}" || \
          echo "[Warning] Failed to record biology benchmark metadata for $label."
      else
        echo "[Warning] Biology benchmark failed for $label; checkpoint is still available."
      fi
    fi
  }

  if [[ "$BENCHMARK_EVERY_CHECKPOINT" == true ]]; then
    for checkpoint_path in "$SAVE_DIR"/checkpoint_step*.pt; do
      [[ -f "$checkpoint_path" ]] || continue
      checkpoint_name="$(basename "$checkpoint_path")"
      step_label="${checkpoint_name#checkpoint_step}"
      step_label="${step_label%.pt}"
      run_benchmark "$checkpoint_path" "step${step_label}"
    done
  else
    FINAL_MILESTONE_PRESENT=false
    for milestone in $BENCHMARK_MILESTONES; do
      if [[ "$milestone" == "$STEPS" ]]; then
        FINAL_MILESTONE_PRESENT=true
      fi
      run_benchmark "$SAVE_DIR/checkpoint_step${milestone}.pt" "step${milestone}"
    done
    if [[ "$FINAL_MILESTONE_PRESENT" == false ]]; then
      run_benchmark "$SAVE_DIR/checkpoint_step${STEPS}.pt" "step${STEPS}"
    fi
  fi
  if [[ -f "$SAVE_DIR/checkpoint_latest.pt" ]]; then
    run_benchmark "$SAVE_DIR/checkpoint_latest.pt" "latest"
  fi
  if [[ -f "$SAVE_DIR/checkpoint_best.pt" ]]; then
    run_benchmark "$SAVE_DIR/checkpoint_best.pt" "best"
  fi
  run_benchmark "$CHECKPOINT" "selected"
else
  echo ""
  echo "[5/5] Benchmarks skipped"
fi

echo "================================================================="
echo " Pipeline complete. Results saved to $SAVE_DIR"
echo "================================================================="
