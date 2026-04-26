#!/usr/bin/env bash
# Convenience: open an interactive tmux session on the remote instance.
# Detach with Ctrl-b, then d (training keeps running).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[tmux] FAIL: SSH_HOST not set." >&2
    exit 1
fi

ssh -t -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    root@"${SSH_HOST}" "tmux attach -t ${TMUX_SESSION:-train}"
