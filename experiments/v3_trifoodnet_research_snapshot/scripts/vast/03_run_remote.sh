#!/usr/bin/env bash
# Install pip deps and start the mini training inside a detached tmux session.
# After this script returns, training is running on the remote in the
# background — connect with 04_live_monitor.py or 99_attach_tmux.sh.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[run] FAIL: SSH_HOST not set. Run 01_launch.sh + 02_push.sh first." >&2
    exit 1
fi

SSH_BASE=(ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"${SSH_HOST}")

# ── Push HF_TOKEN from local .env so SAM3 + any other gated models can download
ENV_FILE="${REPO_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    HF_TOKEN_LOCAL="$(grep -E '^HF_TOKEN=' "${ENV_FILE}" | head -1 | cut -d= -f2-)"
    if [[ -n "${HF_TOKEN_LOCAL}" ]]; then
        echo "[run] pushing HF_TOKEN from .env to remote /root/.hf_env"
        "${SSH_BASE[@]}" "cat > /root/.hf_env <<'EOF'
export HF_TOKEN=${HF_TOKEN_LOCAL}
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN_LOCAL}
EOF
chmod 600 /root/.hf_env
grep -qxF 'source /root/.hf_env' /root/.bashrc 2>/dev/null || echo 'source /root/.hf_env' >> /root/.bashrc"
    fi
fi

# ── 1. Install pip deps not present in the base PyTorch image ─────────────────
# The PyTorch image already has torch+torchvision+CUDA. We add:
# - peft + bitsandbytes for Stage 1 LoRA + 4-bit
# - transformers + accelerate (Qwen + SAM)
# - PIL / Pillow already there
# - psutil / pyyaml for the logger + config
echo "[run] installing pip deps (cached after first run)..."
"${SSH_BASE[@]}" "
    set -e
    cd ${REMOTE_WORK}/code
    pip install --quiet --no-cache-dir \
        peft \
        bitsandbytes \
        'transformers>=4.56' \
        accelerate \
        psutil pyyaml omegaconf
    python3 -c "import transformers; assert hasattr(transformers, 'Sam3Model'), 'Sam3Model missing — transformers ${'$'}{transformers.__version__} too old'"
"

# ── 2. Quick local smoke before burning model-load time ──────────────────────
echo "[run] running CPU smoke on remote (no model load yet)..."
"${SSH_BASE[@]}" "
    cd ${REMOTE_WORK}/code
    python scripts/smoke_phase3.py --export-root ${REMOTE_DATASET}
" || { echo "[run] FAIL: smoke did not pass on remote — check the output above" >&2; exit 1; }

# ── 3. Launch training inside a detached tmux session ────────────────────────
# The tmux session 'train' survives ssh disconnects so the run continues even
# if the network blips or you close the terminal. Output is also tee'd to a
# log file the live_monitor can read.
TRAIN_OVERRIDES_STR="${TRAIN_OVERRIDES[*]}"
echo "[run] starting training in tmux session 'train'..."
echo "[run]   overrides: ${TRAIN_OVERRIDES_STR}"
"${SSH_BASE[@]}" "
    cd ${REMOTE_WORK}/code
    mkdir -p ${REMOTE_LOGS} ${REMOTE_CKPTS}
    # Symlink so cfg.paths.{logs,checkpoints,outputs} land in our work dir
    ln -sfn ${REMOTE_LOGS} ${REMOTE_WORK}/code/logs
    ln -sfn ${REMOTE_CKPTS} ${REMOTE_WORK}/code/checkpoints
    ln -sfn ${REMOTE_WORK}/outputs ${REMOTE_WORK}/code/outputs
    mkdir -p ${REMOTE_WORK}/outputs

    tmux kill-session -t train 2>/dev/null || true
    tmux new-session -d -s train -x 220 -y 50 \"
        cd ${REMOTE_WORK}/code &&
        source /root/.hf_env 2>/dev/null;
        python -m train_joint ${TRAIN_OVERRIDES_STR} 2>&1 | tee ${REMOTE_WORK}/train.stdout.log
    \"
    sleep 2
    tmux ls
"

echo
echo "[run] training launched in tmux session 'train' on the remote."
echo "[run] LOG dir on remote:  ${REMOTE_LOGS}/${RUN_NAME}/joint/"
echo "[run] STDOUT on remote:   ${REMOTE_WORK}/train.stdout.log"
echo "[run] RUN NAME (locally too): ${RUN_NAME}"
echo
echo "[run] now in another terminal:"
echo "       python scripts/vast/04_live_monitor.py"
echo "  or:  bash scripts/vast/99_attach_tmux.sh   # raw stdout"
