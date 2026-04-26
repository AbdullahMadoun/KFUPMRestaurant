#!/usr/bin/env bash
# Push this repository's Stage 1 code to the Vast instance.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[stage1_push] FAIL: SSH_HOST not set. Run stage1_01_launch.sh first." >&2
    exit 1
fi

SSH_BASE=(ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"${SSH_HOST}")
RSYNC_BASE=(rsync -azh --delete --progress -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")

echo "[stage1_push] waiting for remote bootstrap..."
for i in $(seq 1 60); do
    if "${SSH_BASE[@]}" "test -f ${REMOTE_WORK}/.ready" 2>/dev/null; then
        break
    fi
    sleep 5
    echo "[stage1_push] poll ${i}/60"
done

"${SSH_BASE[@]}" "mkdir -p ${REMOTE_WORK}/code"
"${RSYNC_BASE[@]}" \
    --include="stage1_kcfd/" --include="stage1_kcfd/**/*.py" \
    --include="scripts/" --include="scripts/vast/" \
    --include="scripts/vast/00_state.sh" --include="scripts/vast/00b_pull_dataset_from_drive.sh" \
    --include="scripts/vast/06_destroy.sh" --include="scripts/vast/99_attach_tmux.sh" \
    --include="scripts/vast/stage1_*.sh" \
    --include="tests/" --include="tests/test_stage1_kcfd*.py" \
    --include="docs/" --include="docs/STAGE1_KCFD.md" \
    --include="train.py" --include="requirements-stage1.txt" --include="*.py" \
    --include="*.md" --include="*.json" --include="*/" \
    --exclude="outputs/" --exclude="logs/" --exclude="checkpoints/" \
    --exclude="__pycache__/" --exclude=".pytest_cache/" --exclude="*.pyc" \
    --exclude="*" \
    "${REPO_DIR}/" \
    "root@${SSH_HOST}:${REMOTE_WORK}/code/"

"${SSH_BASE[@]}" "cd ${REMOTE_WORK}/code && test -f train.py && test -f requirements-stage1.txt && test -d stage1_kcfd"
echo "[stage1_push] code pushed to ${REMOTE_WORK}/code"
echo "[stage1_push] next: bash scripts/vast/00b_pull_dataset_from_drive.sh"
echo "[stage1_push] then: bash scripts/vast/stage1_03_prepare_preview.sh"
