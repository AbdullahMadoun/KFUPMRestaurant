#!/usr/bin/env bash
# Destroy the rented instance. Prompts for confirmation because this is
# irreversible — anything on the disk that wasn't pulled with 05_pull.sh is gone.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${INSTANCE_ID:-}" ]]; then
    echo "[destroy] no INSTANCE_ID in state. Nothing to do."
    exit 0
fi

echo "[destroy] target instance: ${INSTANCE_ID}"
echo "[destroy] this is IRREVERSIBLE — anything not pulled is gone."
echo
read -r -p "[destroy] type 'yes' to confirm: " confirm
if [[ "${confirm}" != "yes" ]]; then
    echo "[destroy] aborted."
    exit 1
fi

vastai destroy instance "${INSTANCE_ID}"

# Cancel the local auto-destroy watchdog if it's still running
PID_FILE="${SCRIPT_DIR}/.auto_destroy.pid"
if [[ -f "${PID_FILE}" ]]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        kill "${PID}" || true
        echo "[destroy] cancelled auto-destroy watchdog (pid ${PID})"
    fi
    rm -f "${PID_FILE}"
fi

# Wipe state so subsequent script invocations don't touch a dead instance
rm -f "${STATE_FILE}"
echo "[destroy] DONE. State cleared."
echo "[destroy] balance check:"
vastai show user 2>&1 | head -3
