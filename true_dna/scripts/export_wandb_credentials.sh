#!/usr/bin/env bash

# Source this file before a run to export a W&B API key for the current shell.
# It deliberately never writes the key to disk or shell history.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this helper so its export stays in your current shell:" >&2
  echo "  source scripts/export_wandb_credentials.sh" >&2
  exit 2
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[credentials] WANDB_API_KEY is already set."
else
  _true_dna_wandb_key=""
  read -r -s -p "Weights & Biases API key (leave blank to use an existing W&B login): " _true_dna_wandb_key
  echo
  if [[ -n "$_true_dna_wandb_key" ]]; then
    export WANDB_API_KEY="$_true_dna_wandb_key"
    echo "[credentials] WANDB_API_KEY set for this shell."
  else
    echo "[credentials] WANDB_API_KEY not set; W&B will use any existing local login."
  fi
  unset _true_dna_wandb_key
fi
