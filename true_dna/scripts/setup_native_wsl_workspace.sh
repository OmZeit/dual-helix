#!/usr/bin/env bash
# Create a native-WSL working copy for faster large-file corpus builds/training.
# It intentionally refuses a non-empty target and never deletes the Windows
# source workspace.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash true_dna/scripts/setup_native_wsl_workspace.sh [--target PATH]

Copies the repository source into a new, empty directory on the Linux
filesystem. Generated corpus data and the virtual environment are excluded so
the native workspace starts clean. The source workspace under /mnt/<drive> is
never changed or removed.
EOF
}

target="${HOME}/dual-helix-native"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "--target needs a path" >&2; exit 2; }
      target="$2"
      shift 2
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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd -- "${script_dir}/../.." && pwd)"
target="$(python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$target")"

case "$target" in
  /mnt/*)
    echo "Refusing target on a Windows-mounted filesystem: $target" >&2
    echo "Choose a native WSL path such as \$HOME/dual-helix-native." >&2
    exit 2
    ;;
esac

if [[ -e "$target" ]] && [[ -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty target: $target" >&2
  exit 2
fi

command -v rsync >/dev/null || { echo "rsync is required; install it with: sudo apt install rsync" >&2; exit 1; }
mkdir -p "$target"

rsync -a \
  --exclude='/.venv/' \
  --exclude='/build/' \
  --exclude='/true_dna/data/' \
  --exclude='/__pycache__/' \
  --exclude='*.pyc' \
  "$source_root/" "$target/"

cat <<EOF
Native WSL workspace created: $target

Next:
  cd "$target"
  bash true_dna/scripts/setup_cuda_env.sh --rtx5080 --python python3.12
  cd true_dna
  source ../.venv/bin/activate

Then rerun the corpus command from this native workspace. Its default scratch
directory is already native WSL storage: ~/.cache/true_dna_ncbi
EOF
