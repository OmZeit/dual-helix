#!/usr/bin/env bash

# Source this file to place optional NCBI credentials in the current shell
# without writing them to the repository, shell history, or a dotenv file.
# Use export_wandb_credentials.sh for the W&B key. Existing environment values
# are left unchanged.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this helper so its exports stay in your current shell:" >&2
  echo "  source scripts/export_research_credentials.sh" >&2
  exit 2
fi

_true_dna_prompt_secret() {
  local variable_name="$1"
  local prompt="$2"
  local value=""
  if [[ -n "${!variable_name:-}" ]]; then
    echo "[credentials] ${variable_name} is already set."
    return
  fi
  read -r -s -p "$prompt (leave blank to skip): " value
  echo
  if [[ -n "$value" ]]; then
    export "$variable_name=$value"
    echo "[credentials] ${variable_name} set for this shell."
  else
    echo "[credentials] ${variable_name} not set."
  fi
}

_true_dna_prompt_text() {
  local variable_name="$1"
  local prompt="$2"
  local value=""
  if [[ -n "${!variable_name:-}" ]]; then
    echo "[credentials] ${variable_name} is already set."
    return
  fi
  read -r -p "$prompt (leave blank to skip): " value
  if [[ -n "$value" ]]; then
    export "$variable_name=$value"
    echo "[credentials] ${variable_name} set for this shell."
  else
    echo "[credentials] ${variable_name} not set."
  fi
}

_true_dna_prompt_secret "NCBI_API_KEY" "NCBI API key"
_true_dna_prompt_text "NCBI_TAXONOMY_EMAIL" "NCBI contact email"

unset -f _true_dna_prompt_secret _true_dna_prompt_text
