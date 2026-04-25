#!/usr/bin/env bash
# Push CODE ONLY to the rented instance (~5 MB, fast).
# Dataset comes from Google Drive via 00b_pull_dataset_from_drive.sh — runs
# entirely on the instance, no laptop bandwidth involved.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" || -z "${INSTANCE_ID:-}" ]]; then
    echo "[push] FAIL: SSH_HOST/INSTANCE_ID not set. Run 01_launch.sh first." >&2
    exit 1
fi

SSH_BASE=(ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"${SSH_HOST}")
RSYNC_BASE=(rsync -azh --progress -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")

echo "[push] waiting for instance to finish booting..."
for i in $(seq 1 30); do
    if "${SSH_BASE[@]}" "test -f ${REMOTE_WORK}/.ready" 2>/dev/null; then
        echo "[push] instance ready."
        break
    fi
    sleep 5
    echo "[push]   poll ${i}/30 — onstart still running"
done

# ── 1. Code: just the python + scripts + yaml. Everything else is noise. ─────
echo "[push] code → ${REMOTE_WORK}/code"
"${SSH_BASE[@]}" "mkdir -p ${REMOTE_WORK}/code"
"${RSYNC_BASE[@]}" \
    --include="*.py" --include="*.yaml" --include="*.yml" --include="*.json" \
    --include="*.md" --include="requirements.txt" \
    --include="scripts/" --include="scripts/vast/" --include="scripts/**/*.sh" \
    --include="tests/" --include="tests/**/*.py" \
    --include="*/" \
    --exclude="logs/" --exclude="checkpoints/" --exclude="outputs/" \
    --exclude="results/" --exclude="weights/" --exclude="snapshots/" \
    --exclude="__pycache__/" --exclude="*.pyc" --exclude=".pytest_cache/" \
    --exclude="*" \
    "${REPO_DIR}/" \
    "root@${SSH_HOST}:${REMOTE_WORK}/code/"

# ── 2. Verify code present ───────────────────────────────────────────────────
echo "[push] verifying remote code layout..."
"${SSH_BASE[@]}" "
    echo '  code files:' \$(find ${REMOTE_WORK}/code -name '*.py' | wc -l)
    echo '  has master_config.yaml:' \$(test -f ${REMOTE_WORK}/code/master_config.yaml && echo yes || echo NO)
    echo '  has dataset_v3_adapter.py:' \$(test -f ${REMOTE_WORK}/code/dataset_v3_adapter.py && echo yes || echo NO)
"

echo
echo "[push] DONE."
echo "[push] next:"
echo "         bash scripts/vast/00b_pull_dataset_from_drive.sh   # 1.6 GB → instance via Drive"
echo "         bash scripts/vast/03_run_remote.sh                  # install + train"
