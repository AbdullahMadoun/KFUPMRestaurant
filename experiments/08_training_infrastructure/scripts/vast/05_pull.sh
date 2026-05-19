#!/usr/bin/env bash
# Pull logs + checkpoints back from the rented instance once training finishes.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[pull] FAIL: SSH_HOST not set." >&2
    exit 1
fi

LOCAL_LOGS="${REPO_DIR}/logs/${RUN_NAME}"
LOCAL_CKPTS="${REPO_DIR}/checkpoints/${RUN_NAME}"
LOCAL_OUTPUTS="${REPO_DIR}/outputs/${RUN_NAME}"
mkdir -p "${LOCAL_LOGS}" "${LOCAL_CKPTS}" "${LOCAL_OUTPUTS}"

RSYNC_BASE=(rsync -azh --progress -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")

# Logs first (small + always wanted)
echo "[pull] logs..."
"${RSYNC_BASE[@]}" "root@${SSH_HOST}:${REMOTE_LOGS}/${RUN_NAME}/" "${LOCAL_LOGS}/"

# Checkpoints (might be a few hundred MB)
echo "[pull] checkpoints..."
"${RSYNC_BASE[@]}" "root@${SSH_HOST}:${REMOTE_CKPTS}/${RUN_NAME}/" "${LOCAL_CKPTS}/" || true

# Outputs (reports + plots if generated)
echo "[pull] outputs..."
"${RSYNC_BASE[@]}" "root@${SSH_HOST}:${REMOTE_WORK}/outputs/${RUN_NAME}/" "${LOCAL_OUTPUTS}/" || true

# Stdout snapshot
echo "[pull] stdout..."
"${RSYNC_BASE[@]}" "root@${SSH_HOST}:${REMOTE_WORK}/train.stdout.log" "${LOCAL_LOGS}/train.stdout.log" || true

echo
echo "[pull] DONE."
echo "[pull] local artifacts:"
echo "         logs:        ${LOCAL_LOGS}"
echo "         checkpoints: ${LOCAL_CKPTS}"
echo "         outputs:     ${LOCAL_OUTPUTS}"
echo
echo "[pull] next:"
echo "       python scripts/compare_runs.py logs/trial-20260321-cleandata1/joint ${LOCAL_LOGS}/joint"
echo "       python scripts/registry_append.py ${LOCAL_LOGS}/joint"
echo "       bash scripts/vast/06_destroy.sh   # when you're done"
