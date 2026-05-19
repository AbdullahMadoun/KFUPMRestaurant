#!/usr/bin/env bash
# Pull Stage 1 run artifacts from the Vast instance.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[stage1_pull] FAIL: SSH_HOST not set." >&2
    exit 1
fi

export STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${RUN_NAME:-stage1-qwen7b}}"
LOCAL_DIR="${REPO_DIR}/outputs/${STAGE1_RUN_NAME}"
mkdir -p "${LOCAL_DIR}"

RSYNC_BASE=(rsync -azh --progress -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")

"${RSYNC_BASE[@]}" "root@${SSH_HOST}:${REMOTE_WORK}/stage1_runs/${STAGE1_RUN_NAME}/" "${LOCAL_DIR}/"
"${RSYNC_BASE[@]}" "root@${SSH_HOST}:${REMOTE_WORK}/stage1_train.stdout.log" "${LOCAL_DIR}/stage1_train.stdout.log" || true

echo "[stage1_pull] pulled artifacts to ${LOCAL_DIR}"
