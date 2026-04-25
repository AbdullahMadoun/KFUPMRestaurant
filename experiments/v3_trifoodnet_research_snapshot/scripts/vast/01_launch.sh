#!/usr/bin/env bash
# Launch a vast.ai instance using the chosen offer and an auto-destroy timer.
#
# Prints the instance id, SSH command, and saves them to .state for the next
# scripts to consume.
#
# Cost guard: schedules a `vastai destroy instance` locally via `at` (or a
# fallback nohup sleep) so even if you walk away the instance dies on time.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

# Auto-pick cheapest available 4090 if OFFER_ID is unset.
if [[ -z "${OFFER_ID:-}" ]]; then
    echo "[launch] OFFER_ID empty — picking cheapest 4090..."
    OFFER_ID="$(bash "${SCRIPT_DIR}/pick_offer.sh")"
    export OFFER_ID
fi

echo "[launch] offer:        ${OFFER_ID}"
echo "[launch] image:        ${DOCKER_IMAGE}"
echo "[launch] disk:         ${DISK_GB} GB"
echo "[launch] auto-destroy: ${AUTO_DESTROY_MINUTES} minutes from now"
echo

# ── Create the instance ───────────────────────────────────────────────────────
# --ssh keeps SSH access (instead of jupyter only).
# --direct gives a public ip:port (vs proxy).
# --onstart-cmd sets up tmux session early so the instance is ready when we ssh.
CREATE_OUTPUT=$(vastai create instance "${OFFER_ID}" \
    --image "${DOCKER_IMAGE}" \
    --disk "${DISK_GB}" \
    --ssh \
    --direct \
    --onstart-cmd "apt-get update >/dev/null 2>&1 || true; apt-get install -y -qq tmux rsync git jq >/dev/null 2>&1 || true; mkdir -p ${REMOTE_WORK}; touch ${REMOTE_WORK}/.ready" \
    2>&1)
echo "${CREATE_OUTPUT}"

# Output looks like: "Started. {'success': True, 'new_contract': 12345678}"
INSTANCE_ID=$(echo "${CREATE_OUTPUT}" | grep -oE "'new_contract': [0-9]+" | grep -oE "[0-9]+" || true)
if [[ -z "${INSTANCE_ID}" ]]; then
    echo "[launch] FAIL: could not parse instance id from output above" >&2
    exit 1
fi
export INSTANCE_ID
echo "[launch] instance id: ${INSTANCE_ID}"

# ── Wait for instance to come up + grab SSH details ──────────────────────────
echo "[launch] waiting for ssh details to populate (up to 5 min)..."
for i in $(seq 1 30); do
    sleep 10
    INFO=$(vastai show instance "${INSTANCE_ID}" --raw 2>&1 || true)
    SSH_HOST=$(echo "${INFO}" | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(d.get('ssh_host') or d.get('public_ipaddr') or '')
except Exception:
    pass
")
    SSH_PORT=$(echo "${INFO}" | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    p = d.get('ssh_port') or 22
    print(p)
except Exception:
    print(22)
")
    if [[ -n "${SSH_HOST}" ]]; then
        export SSH_HOST SSH_PORT
        echo "[launch] ssh ready: root@${SSH_HOST}:${SSH_PORT}"
        break
    fi
    echo "[launch]   poll ${i}/30 — not ready yet"
done

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[launch] FAIL: SSH host did not become available in 5 min" >&2
    exit 1
fi

# ── Persist state for the rest of the pipeline ───────────────────────────────
save_state
echo "[launch] saved instance id + ssh details to ${STATE_FILE}"

# ── Optional auto-destroy watchdog (set AUTO_DESTROY_MINUTES=0 to skip) ─────
# Skip the watchdog entirely for long-running training where the user wants
# manual control over termination. With AUTO_DESTROY_MINUTES=0, the instance
# stays up until `06_destroy.sh` is run.
if [[ "${AUTO_DESTROY_MINUTES}" -gt 0 ]]; then
    DESTROY_LOG="${SCRIPT_DIR}/.auto_destroy.log"
    nohup bash -c "sleep $((AUTO_DESTROY_MINUTES * 60)); vastai destroy instance ${INSTANCE_ID} >> '${DESTROY_LOG}' 2>&1" \
        >/dev/null 2>&1 &
    DESTROY_PID=$!
    echo "${DESTROY_PID}" > "${SCRIPT_DIR}/.auto_destroy.pid"
    echo "[launch] auto-destroy scheduled at +${AUTO_DESTROY_MINUTES} min (pid ${DESTROY_PID}, log: ${DESTROY_LOG})"
else
    echo "[launch] auto-destroy DISABLED (AUTO_DESTROY_MINUTES=0). Run 06_destroy.sh when done."
fi

echo
echo "[launch] DONE."
echo "[launch] next: bash scripts/vast/02_push.sh"
echo "[launch] manual ssh: $(ssh_cmd)"
