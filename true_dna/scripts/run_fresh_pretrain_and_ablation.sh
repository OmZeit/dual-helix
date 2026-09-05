#!/usr/bin/env bash
set -euo pipefail

# Build a new public RefSeq corpus and, only after every validation succeeds,
# start the three primary controlled-ablation backbones.  This launcher is
# deliberately separate from run_controlled_ablation.sh: the latter supports
# small smoke-preparation defaults, whereas this script has an explicit,
# destructive full-corpus contract.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_fresh_pretrain_and_ablation.sh --rtx5080|--a100 [options] [-- TRAINING_ARGS...]

Builds a fresh 5 GB-per-domain RefSeq corpus, validates it, creates an
assembly-held-out split, removes generated intermediates, then runs the three
single-base architecture baselines and a matched Transformer BPE control.

Options:
  --target-gb-per-domain N  Desired final corpus size per domain (default: 5)
  --parallel-downloads N    Bounded NCBI package downloads (default: 4)
  --max-assemblies N        Candidate cap per selected TaxID (default: 1000)
  --scratch-dir PATH        Temporary NCBI package location (default: ~/.cache/true_dna_ncbi)
  --selection-manifest PATH Pinned TaxID selection manifest (default: data/metadata/taxonomic_selection_v1.json)
  --allow-underfilled-domain NAME  Permit and document a nonempty domain below the lower size tolerance (default: archaea)
  --keep-intermediates      Preserve per-species FASTAs and QC copies after validation
  --no-wandb                Disable Weights & Biases logging (enabled by default)
  -h, --help                Show this help

Everything after -- is passed to run_controlled_ablation.py, for example:
  ... --rtx5080 -- --steps 1000

This intentionally removes generated corpus, split, tokenizer, and generated
corpus-metadata artifacts before downloading. It never removes source code,
experiments, or checkpoints.
EOF
}

PROFILE=""
TARGET_GB=5
PARALLEL_DOWNLOADS=4
MAX_ASSEMBLIES=1000
SCRATCH_DIR="${TRUE_DNA_NCBI_SCRATCH:-$HOME/.cache/true_dna_ncbi}"
SELECTION_MANIFEST="data/metadata/taxonomic_selection_v1.json"
SELECTION_MANIFEST_EXPLICIT=false
KEEP_INTERMEDIATES=false
WANDB_ARGS=()
TRAINING_ARGS=()
UNDERFILLED_DOMAIN_ARGS=(--allow-underfilled-domain archaea)

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
    --target-gb-per-domain)
      TARGET_GB="$2"
      shift 2
      ;;
    --parallel-downloads)
      PARALLEL_DOWNLOADS="$2"
      shift 2
      ;;
    --max-assemblies)
      MAX_ASSEMBLIES="$2"
      shift 2
      ;;
    --scratch-dir)
      SCRATCH_DIR="$2"
      shift 2
      ;;
    --selection-manifest)
      SELECTION_MANIFEST="$2"
      SELECTION_MANIFEST_EXPLICIT=true
      shift 2
      ;;
    --allow-underfilled-domain)
      UNDERFILLED_DOMAIN_ARGS+=(--allow-underfilled-domain "$2")
      shift 2
      ;;
    --keep-intermediates)
      KEEP_INTERMEDIATES=true
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
  usage >&2
  exit 2
fi

if [[ "$PROJECT_DIR" == /mnt/* ]]; then
  echo "Run this from the native WSL workspace (for example ~/dual-helix-native/true_dna), not /mnt/<drive>." >&2
  exit 2
fi

SCRATCH_DIR="$(python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$SCRATCH_DIR")"
if [[ "$SCRATCH_DIR" == "/" || "$SCRATCH_DIR" == "$HOME" || ${#SCRATCH_DIR} -lt 12 ]]; then
  echo "Refusing unsafe scratch directory: $SCRATCH_DIR" >&2
  exit 2
fi
SCRATCH_MARKER="$SCRATCH_DIR/.true_dna_scratch"
if [[ -e "$SCRATCH_DIR" && ! -d "$SCRATCH_DIR" ]]; then
  echo "Refusing non-directory scratch path: $SCRATCH_DIR" >&2
  exit 2
fi
if [[ -d "$SCRATCH_DIR" && -n "$(find "$SCRATCH_DIR" -mindepth 1 -maxdepth 1 -print -quit)" && ! -f "$SCRATCH_MARKER" ]]; then
  echo "Refusing nonempty scratch directory not owned by this pipeline: $SCRATCH_DIR" >&2
  exit 2
fi
mkdir -p "$SCRATCH_DIR"
touch "$SCRATCH_MARKER"

if [[ -x "$WORKSPACE_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$WORKSPACE_DIR/.venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$PROJECT_DIR/.venv/bin/python"
else
  PYTHON_CMD=python3
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Authenticate logging first, while the user is still at the terminal. This
# avoids discovering a missing W&B key only after a multi-hour corpus build.
if [[ ${#WANDB_ARGS[@]} -eq 0 ]]; then
  source "$SCRIPT_DIR/export_wandb_credentials.sh"
fi
source "$SCRIPT_DIR/export_research_credentials.sh"

if [[ "$SELECTION_MANIFEST_EXPLICIT" != true ]]; then
  echo "[selection] Revalidating the versioned selection config and rebuilding its pinned manifest."
  "$PYTHON_CMD" scripts/build_taxonomic_manifest.py \
    --config configs/species_taxonomically_diverse_v1.json \
    --output "$SELECTION_MANIFEST"
elif [[ ! -f "$SELECTION_MANIFEST" ]]; then
  echo "Pinned selection manifest not found: $SELECTION_MANIFEST" >&2
  exit 2
fi

echo "[fresh] Rebuilding a ${TARGET_GB} GB-per-domain corpus; training starts only after preparation, verification, and split creation succeed."
"$PYTHON_CMD" scripts/prepare_refseq_pretrain_data.py \
  --hard-delete-generated-data \
  --selection-manifest "$SELECTION_MANIFEST" \
  --run-download \
  --target-gb-per-domain "$TARGET_GB" \
  --parallel-downloads "$PARALLEL_DOWNLOADS" \
  --max-assemblies "$MAX_ASSEMBLIES" \
  --scratch-dir "$SCRATCH_DIR" \
  --exclude-plasmids \
  "${UNDERFILLED_DOMAIN_ARGS[@]}" \
  --rebuild-tokenizer \
  --write-manifest \
  --verify

"$PYTHON_CMD" scripts/prepare_ablation_data.py \
  --fasta data/domain_fastas/*.fa \
  --output data/domain_fastas/split_manifest.jsonl \
  --group-by assembly \
  --seed 43

if [[ "$KEEP_INTERMEDIATES" != true ]]; then
  echo "[cleanup] Preserving compact QC/provenance records and removing generated intermediates."
  mkdir -p data/metadata/qc_reports data/metadata/corpus_provenance
  for domain in bacteria archaea eukaryotes; do
    domain_marker="data/$domain/.true_dna_generated"
    if [[ ! -f "$domain_marker" ]]; then
      echo "Refusing to delete unowned generated-data directory: data/$domain" >&2
      exit 2
    fi
    if [[ -f "data/$domain/qc_results/summary.json" ]]; then
      cp "data/$domain/qc_results/summary.json" "data/metadata/qc_reports/${domain}_summary.json"
    fi
    if [[ -f "data/$domain/qc_results/qc_summary.csv" ]]; then
      cp "data/$domain/qc_results/qc_summary.csv" "data/metadata/qc_reports/${domain}_summary.csv"
    fi
    if [[ -f "data/$domain/domain_download_manifest.json" ]]; then
      cp "data/$domain/domain_download_manifest.json" "data/metadata/corpus_provenance/${domain}_domain_download_manifest.json"
    fi
    rm -rf "data/$domain"
  done
  if [[ ! -f "$SCRATCH_MARKER" ]]; then
    echo "Refusing to delete unowned scratch directory: $SCRATCH_DIR" >&2
    exit 2
  fi
  rm -rf "$SCRATCH_DIR"
fi

echo "[train] Starting base-token architecture baselines and the matched Transformer BPE control."
exec "$PYTHON_CMD" run_controlled_ablation.py \
  --profile "$PROFILE" \
  --fasta data/domain_fastas/*.fa \
  --split-manifest data/domain_fastas/split_manifest.jsonl \
  --only-baselines \
  --include-tokenizer-ablation \
  --allow-dirty \
  "${WANDB_ARGS[@]}" \
  "${TRAINING_ARGS[@]}"
