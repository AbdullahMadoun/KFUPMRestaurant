#!/usr/bin/env bash
# Launch a standard Vast container for Stage 1 Qwen-VL training.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

export STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-stage1-qwen7b-$(date -u +%Y%m%d-%H%M)}"
export RUN_NAME="${STAGE1_RUN_NAME}"
export STAGE1_MAX_DPH="${STAGE1_MAX_DPH:-2.00}"
export STAGE1_DISK_GB="${STAGE1_DISK_GB:-180}"
export STAGE1_DOCKER_IMAGE="${STAGE1_DOCKER_IMAGE:-pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel}"
export VASTAI_BIN="${VASTAI_BIN:-}"
if [[ -z "${VASTAI_BIN}" ]]; then
    if command -v vastai >/dev/null 2>&1; then
        VASTAI_BIN="$(command -v vastai)"
    elif command -v vastai.exe >/dev/null 2>&1; then
        VASTAI_BIN="$(command -v vastai.exe)"
    elif [[ -x /mnt/c/Python312/Scripts/vastai.exe ]]; then
        VASTAI_BIN="/mnt/c/Python312/Scripts/vastai.exe"
    elif [[ -x /c/Python312/Scripts/vastai.exe ]]; then
        VASTAI_BIN="/c/Python312/Scripts/vastai.exe"
    else
        echo "[stage1_launch] FAIL: vastai CLI not found. Set VASTAI_BIN=/path/to/vastai." >&2
        exit 1
    fi
fi

if [[ -z "${OFFER_ID:-}" ]]; then
    echo "[stage1_launch] OFFER_ID empty; selecting fastest expected offer under \$${STAGE1_MAX_DPH}/hr..."
    OFFER_ID="$(STAGE1_MAX_DPH="${STAGE1_MAX_DPH}" STAGE1_PICK_TABLE_FILE="${SCRIPT_DIR}/.stage1_offer_table.txt" bash "${SCRIPT_DIR}/stage1_pick_offer.sh")"
    export OFFER_ID
    cat "${SCRIPT_DIR}/.stage1_offer_table.txt"
fi

echo "[stage1_launch] offer: ${OFFER_ID}"
echo "[stage1_launch] image: ${STAGE1_DOCKER_IMAGE}"
echo "[stage1_launch] disk:  ${STAGE1_DISK_GB} GB"
echo "[stage1_launch] run:   ${STAGE1_RUN_NAME}"

CREATE_OUTPUT="$("${VASTAI_BIN}" create instance "${OFFER_ID}" \
    --image "${STAGE1_DOCKER_IMAGE}" \
    --disk "${STAGE1_DISK_GB}" \
    --ssh \
    --direct \
    --onstart-cmd "apt-get update >/dev/null 2>&1 || true; apt-get install -y -qq tmux rsync git jq openssh-client >/dev/null 2>&1 || true; mkdir -p ${REMOTE_WORK}; touch ${REMOTE_WORK}/.ready" \
    2>&1)"
echo "${CREATE_OUTPUT}"

INSTANCE_ID="$(echo "${CREATE_OUTPUT}" | grep -oE "'new_contract': [0-9]+" | grep -oE "[0-9]+" || true)"
if [[ -z "${INSTANCE_ID}" ]]; then
    echo "[stage1_launch] FAIL: could not parse instance id." >&2
    exit 1
fi
export INSTANCE_ID

echo "[stage1_launch] waiting for SSH details..."
for i in $(seq 1 45); do
    sleep 10
    INFO="$("${VASTAI_BIN}" show instance "${INSTANCE_ID}" --raw 2>/dev/null || true)"
    SSH_HOST="$(echo "${INFO}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read() or '{}'); print(d.get('ssh_host') or d.get('public_ipaddr') or '')" 2>/dev/null || true)"
    SSH_PORT="$(echo "${INFO}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read() or '{}'); print(d.get('ssh_port') or 22)" 2>/dev/null || echo 22)"
    if [[ -n "${SSH_HOST}" ]]; then
        export SSH_HOST SSH_PORT
        break
    fi
    echo "[stage1_launch] poll ${i}/45; SSH not ready"
done

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[stage1_launch] FAIL: SSH host did not become available." >&2
    exit 1
fi

save_state
echo "[stage1_launch] saved state to ${STATE_FILE}"
echo "[stage1_launch] SSH: $(ssh_cmd)"
echo "[stage1_launch] next: bash scripts/vast/stage1_02_push_code.sh"
