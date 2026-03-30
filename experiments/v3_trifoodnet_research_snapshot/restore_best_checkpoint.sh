#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./restore_best_checkpoint.sh [run_id]

Restores weights/best_checkpoint.tar into:
  checkpoints/<run_id>/joint/epoch_038
  checkpoints/<run_id>/joint/best

Default run_id: trial-20260321-cleandata1
EOF
  exit 0
fi

RUN_ID="${1:-trial-20260321-cleandata1}"
EPOCH_DIR="$ROOT_DIR/checkpoints/$RUN_ID/joint/epoch_038"
BEST_DIR="$ROOT_DIR/checkpoints/$RUN_ID/joint/best"
ARCHIVE="$ROOT_DIR/weights/best_checkpoint.tar"

mkdir -p "$EPOCH_DIR" "$BEST_DIR"
tar -xf "$ARCHIVE" -C "$EPOCH_DIR" --strip-components=1
tar -xf "$ARCHIVE" -C "$BEST_DIR" --strip-components=1

printf 'Restored best checkpoint to:\n- %s\n- %s\n' "$EPOCH_DIR" "$BEST_DIR"
