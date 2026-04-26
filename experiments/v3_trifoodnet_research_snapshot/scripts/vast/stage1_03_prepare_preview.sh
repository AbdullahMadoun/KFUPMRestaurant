#!/usr/bin/env bash
# Install Stage 1 deps, run preflight, render training examples, and pull them locally.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[stage1_prepare] FAIL: SSH_HOST not set." >&2
    exit 1
fi

export STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${RUN_NAME:-stage1-qwen7b}}"
export STAGE1_REFERENCE_POLICY="${STAGE1_REFERENCE_POLICY:-exclude}"
export STAGE1_SPLIT_SEED="${STAGE1_SPLIT_SEED:-420}"
export STAGE1_PREVIEW_SAMPLES="${STAGE1_PREVIEW_SAMPLES:-12}"
export STAGE1_PREVIEW_SEED="${STAGE1_PREVIEW_SEED:-20260426}"
export STAGE1_PREVIEW_DIR="${REMOTE_WORK}/stage1_runs/${STAGE1_RUN_NAME}/previews"
export STAGE1_EXPECTED_HASH="${STAGE1_EXPECTED_HASH:-${DATASET_EXPECTED_HASH:-}}"

SSH_BASE=(ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"${SSH_HOST}")
RSYNC_BASE=(rsync -azh --progress -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[stage1_prepare] writing HF_TOKEN from local environment to remote /root/.hf_env"
    "${SSH_BASE[@]}" "cat > /root/.hf_env <<'EOF'
export HF_TOKEN=${HF_TOKEN}
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
EOF
chmod 600 /root/.hf_env"
fi

echo "[stage1_prepare] installing Stage 1 dependencies..."
"${SSH_BASE[@]}" "
    set -e
    cd ${REMOTE_WORK}/code
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet --no-cache-dir -r requirements-stage1.txt
"

HASH_ARGS=()
if [[ -n "${STAGE1_EXPECTED_HASH}" ]]; then
    HASH_ARGS=(--expected-hash "${STAGE1_EXPECTED_HASH}")
fi

echo "[stage1_prepare] running preflight..."
"${SSH_BASE[@]}" "
    set -e
    cd ${REMOTE_WORK}/code
    python train.py \
      --export-root ${REMOTE_DATASET} \
      --output-dir ${REMOTE_WORK}/stage1_runs \
      --run-name ${STAGE1_RUN_NAME} \
      --preflight-only \
      ${HASH_ARGS[*]}
"

echo "[stage1_prepare] rendering training-data previews..."
"${SSH_BASE[@]}" "
    set -e
    cd ${REMOTE_WORK}/code
    python -m stage1_kcfd.visualize \
      --export-root ${REMOTE_DATASET} \
      --output-dir ${STAGE1_PREVIEW_DIR} \
      --split train \
      --max-samples ${STAGE1_PREVIEW_SAMPLES} \
      --reference-policy ${STAGE1_REFERENCE_POLICY} \
      --split-seed ${STAGE1_SPLIT_SEED} \
      --seed ${STAGE1_PREVIEW_SEED} \
      --selection class-diverse
"

LOCAL_PREVIEW_DIR="${REPO_DIR}/outputs/${STAGE1_RUN_NAME}/previews"
mkdir -p "${LOCAL_PREVIEW_DIR}"
"${RSYNC_BASE[@]}" "root@${SSH_HOST}:${STAGE1_PREVIEW_DIR}/" "${LOCAL_PREVIEW_DIR}/"

echo "[stage1_prepare] previews pulled to ${LOCAL_PREVIEW_DIR}"
echo "[stage1_prepare] inspect the PNGs before long training."
echo "[stage1_prepare] tensorboard: bash scripts/vast/stage1_04_tensorboard.sh"
echo "[stage1_prepare] train:       bash scripts/vast/stage1_05_train.sh"
