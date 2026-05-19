#!/usr/bin/env bash
# Start remote TensorBoard and a local 127.0.0.1:6006 SSH tunnel.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[stage1_tb] FAIL: SSH_HOST not set." >&2
    exit 1
fi

export STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${RUN_NAME:-stage1-qwen7b}}"
REMOTE_TB_DIR="${REMOTE_WORK}/stage1_runs/${STAGE1_RUN_NAME}/tensorboard"
LOCAL_PORT="${STAGE1_TENSORBOARD_LOCAL_PORT:-6006}"
REMOTE_PORT="${STAGE1_TENSORBOARD_REMOTE_PORT:-6006}"

SSH_BASE=(ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"${SSH_HOST}")

"${SSH_BASE[@]}" "
    mkdir -p ${REMOTE_TB_DIR}
    tmux kill-session -t stage1-tensorboard 2>/dev/null || true
    tmux new-session -d -s stage1-tensorboard \"tensorboard --logdir ${REMOTE_WORK}/stage1_runs --host 127.0.0.1 --port ${REMOTE_PORT}\"
"

if [[ -f "${SCRIPT_DIR}/.stage1_tensorboard_tunnel.pid" ]]; then
    OLD_PID="$(cat "${SCRIPT_DIR}/.stage1_tensorboard_tunnel.pid" || true)"
    if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        kill "${OLD_PID}" 2>/dev/null || true
    fi
fi

ssh -N -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    root@"${SSH_HOST}" >/dev/null 2>&1 &
echo "$!" > "${SCRIPT_DIR}/.stage1_tensorboard_tunnel.pid"

echo "[stage1_tb] remote TensorBoard logdir: ${REMOTE_WORK}/stage1_runs"
echo "[stage1_tb] local URL: http://127.0.0.1:${LOCAL_PORT}"
