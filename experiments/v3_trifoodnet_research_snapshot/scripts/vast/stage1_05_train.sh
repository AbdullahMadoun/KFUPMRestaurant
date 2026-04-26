#!/usr/bin/env bash
# Start Stage 1 Qwen-VL training in a remote tmux session.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[stage1_train] FAIL: SSH_HOST not set." >&2
    exit 1
fi

export STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${RUN_NAME:-stage1-qwen7b}}"
export STAGE1_REFERENCE_POLICY="${STAGE1_REFERENCE_POLICY:-exclude}"
export STAGE1_EXPECTED_HASH="${STAGE1_EXPECTED_HASH:-${DATASET_EXPECTED_HASH:-}}"
export STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
export STAGE1_MODEL_ID="${STAGE1_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
export STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-4}"
export STAGE1_PROBE_BATCH="${STAGE1_PROBE_BATCH:-1}"
export STAGE1_MAX_PIXELS="${STAGE1_MAX_PIXELS:-1003520}"

SSH_BASE=(ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"${SSH_HOST}")

REMOTE_VRAM_MB="$("${SSH_BASE[@]}" "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1" | tr -dc '0-9' || true)"
REMOTE_VRAM_MB="${REMOTE_VRAM_MB:-0}"
HASH_ARGS=()
if [[ -n "${STAGE1_EXPECTED_HASH}" ]]; then
    HASH_ARGS=(--expected-hash "${STAGE1_EXPECTED_HASH}")
fi

if [[ "${STAGE1_PROBE_BATCH}" == "1" && -z "${STAGE1_PER_DEVICE_BATCH_SIZE:-}" ]]; then
    echo "[stage1_train] probing safe microbatch on remote GPU..."
    PROBE_JSON="${REMOTE_WORK}/stage1_runs/${STAGE1_RUN_NAME}/batch_probe.json"
    "${SSH_BASE[@]}" "
        set -e
        cd ${REMOTE_WORK}/code
        mkdir -p ${REMOTE_WORK}/stage1_runs/${STAGE1_RUN_NAME}
        source /root/.hf_env 2>/dev/null || true
        python -m stage1_kcfd.probe_batch_size \
          --export-root ${REMOTE_DATASET} \
          --model-id ${STAGE1_MODEL_ID} \
          --reference-policy ${STAGE1_REFERENCE_POLICY} \
          ${HASH_ARGS[*]} \
          --candidate-batches ${STAGE1_CANDIDATE_BATCHES:-4,2,1} \
          --max-pixels ${STAGE1_MAX_PIXELS} \
          --output-json ${PROBE_JSON}
    "
    STAGE1_PER_DEVICE_BATCH_SIZE="$("${SSH_BASE[@]}" "python - <<'PY'
import json
payload=json.load(open('${PROBE_JSON}', 'r', encoding='utf-8'))
print(payload['selected_per_device_batch_size'])
PY")"
    STAGE1_GRAD_ACCUM="$("${SSH_BASE[@]}" "python - <<'PY'
import json
payload=json.load(open('${PROBE_JSON}', 'r', encoding='utf-8'))
print(payload['recommended_gradient_accumulation_steps'])
PY")"
fi

if [[ -z "${STAGE1_PER_DEVICE_BATCH_SIZE:-}" ]]; then
    if [[ "${REMOTE_VRAM_MB}" -ge 78000 ]]; then
        STAGE1_PER_DEVICE_BATCH_SIZE=4
    elif [[ "${REMOTE_VRAM_MB}" -ge 47000 ]]; then
        STAGE1_PER_DEVICE_BATCH_SIZE=2
    else
        STAGE1_PER_DEVICE_BATCH_SIZE=1
    fi
fi
if [[ -z "${STAGE1_GRAD_ACCUM:-}" ]]; then
    STAGE1_GRAD_ACCUM=$(( (16 + STAGE1_PER_DEVICE_BATCH_SIZE - 1) / STAGE1_PER_DEVICE_BATCH_SIZE ))
fi

echo "[stage1_train] remote VRAM MB: ${REMOTE_VRAM_MB}"
echo "[stage1_train] microbatch:      ${STAGE1_PER_DEVICE_BATCH_SIZE}"
echo "[stage1_train] grad accum:      ${STAGE1_GRAD_ACCUM}"
echo "[stage1_train] effective batch: $((STAGE1_PER_DEVICE_BATCH_SIZE * STAGE1_GRAD_ACCUM))"
echo "[stage1_train] reference policy:${STAGE1_REFERENCE_POLICY}"

"${SSH_BASE[@]}" "
    set -e
    cd ${REMOTE_WORK}/code
    source /root/.hf_env 2>/dev/null || true
    tmux kill-session -t stage1-train 2>/dev/null || true
    tmux new-session -d -s stage1-train -x 220 -y 60 \"
        cd ${REMOTE_WORK}/code &&
        source /root/.hf_env 2>/dev/null || true;
        python train.py \
          --export-root ${REMOTE_DATASET} \
          --output-dir ${REMOTE_WORK}/stage1_runs \
          --run-name ${STAGE1_RUN_NAME} \
          --reference-policy ${STAGE1_REFERENCE_POLICY} \
          --model-id ${STAGE1_MODEL_ID} \
          --epochs ${STAGE1_EPOCHS} \
          --num-workers ${STAGE1_NUM_WORKERS} \
          --per-device-batch-size ${STAGE1_PER_DEVICE_BATCH_SIZE} \
          --gradient-accumulation-steps ${STAGE1_GRAD_ACCUM} \
          --max-pixels ${STAGE1_MAX_PIXELS} \
          ${HASH_ARGS[*]} \
          2>&1 | tee ${REMOTE_WORK}/stage1_train.stdout.log
    \"
    tmux ls
"

echo "[stage1_train] training started in remote tmux session stage1-train"
echo "[stage1_train] attach: TMUX_SESSION=stage1-train bash scripts/vast/99_attach_tmux.sh"
echo "[stage1_train] tensorboard: bash scripts/vast/stage1_04_tensorboard.sh"
