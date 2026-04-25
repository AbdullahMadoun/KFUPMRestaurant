#!/usr/bin/env bash
# Shared state for the vast.ai mini-smoke pipeline.
# Source this BEFORE every other script:  source scripts/vast/00_state.sh
#
# Edit OFFER_ID after picking from `vastai search offers`.
# Edit INSTANCE_ID after `01_launch.sh` reports it.

set -euo pipefail

# ── Local paths ──────────────────────────────────────────────────────────────
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATASET_DIR="/Users/abdulrazzak/Resturant_Pipeline_Feb/results_v2/exports/v3_2026-04-24_61ac038c"
export STATE_FILE="${REPO_DIR}/scripts/vast/.state"

# ── Google Drive (dataset distribution) ──────────────────────────────────────
# After running bundle_dataset.sh and uploading the .tar to Drive, fill in
# DRIVE_FILE_ID below. The other two fields are produced by bundle_dataset.sh
# and printed at the end of its output.
export DRIVE_FILE_ID="${DRIVE_FILE_ID:-1brzFmBKCUGWkUwG9_1HkpTcnxuyJp1Rk}"  # v3_2026-04-24_61ac038c.tar
export DRIVE_TAR_SHA8="${DRIVE_TAR_SHA8:-6473cdab}"     # from bundle_dataset.sh output
export DATASET_EXPECTED_HASH="${DATASET_EXPECTED_HASH:-61ac038c}"  # manifest.json content_hash_sha8

# ── Vast.ai offer ─────────────────────────────────────────────────────────────
# Empty by default → 01_launch.sh calls pick_offer.sh and grabs the cheapest 4090.
# Override with `OFFER_ID=<id> bash 01_launch.sh` to force a specific instance.
# We avoid 5090 because stock pytorch lacks Blackwell sm_120 kernels — see
# README for the dependency cascade we hit. Use 4090 (sm_89) for guaranteed-clean.
export OFFER_ID="${OFFER_ID:-}"

# Image: PyTorch 2.7 + CUDA 12.8. CUDA 12.8 is required for sm_120 (Blackwell)
# wheels — last attempt with cu124 hit "no kernel image" on the 5090 because
# the older wheel set lacks Blackwell kernels. PyTorch 2.7 was the first stable
# release with proper sm_120 compatibility.
# For 4090 (sm_89), `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` also works.
export DOCKER_IMAGE="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"

# Reasonable disk: dataset (~3GB) + HF model cache (Qwen ~3GB + SAM ~1GB) +
# checkpoints (~2GB) + headroom. 50 is the floor; bumping to 80 to be safe.
export DISK_GB="${DISK_GB:-80}"

# Hard cap so we cannot leak credit. Auto-destroy via cron-on-instance + local
# scheduled vastai destroy as belt-and-suspenders. Default 240 min = 4 hrs gives
# room for a 10-epoch full-pass training run plus eval and report generation.
# Override per-launch:  AUTO_DESTROY_MINUTES=360 bash 01_launch.sh
export AUTO_DESTROY_MINUTES="${AUTO_DESTROY_MINUTES:-240}"

# ── Remote paths inside the container ─────────────────────────────────────────
export REMOTE_WORK="/root/work"
export REMOTE_DATASET="${REMOTE_WORK}/dataset_v3_export"
export REMOTE_LOGS="${REMOTE_WORK}/logs"
export REMOTE_CKPTS="${REMOTE_WORK}/checkpoints"

# ── Instance state, persisted between scripts ────────────────────────────────
load_state() {
    if [[ -f "$STATE_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$STATE_FILE"
    fi
}

# Load persisted state FIRST so a RUN_NAME written by 01_launch.sh wins over
# a fresh timestamp. Earlier this ran AFTER the RUN_NAME default block, which
# meant TRAIN_OVERRIDES locked in the new timestamp before the cached one
# could win — silently routing every later script to the wrong log path.
load_state || true

# ── Mini-run config overrides ────────────────────────────────────────────────
# Passed positionally to `python -m train_joint`. Each is "key=value".
# RUN_NAME comes from .state (if persisted by 01_launch.sh) or fresh timestamp.
if [[ -z "${RUN_NAME:-}" ]]; then
    export RUN_NAME="trial-$(date -u +%Y%m%d-%H%M)-real-1hr"
fi
# Real run: 10 full epochs, dev eval every epoch so we see the curve, save
# checkpoint every epoch (rotated, keep last 5). Tuned for "let it cook for
# at least an hour, extend if good, kill cleanly."
export TRAIN_OVERRIDES=(
    "run.name=${RUN_NAME}"
    "joint.training.epochs=10"
    "joint.training.max_batches_per_epoch=0"   # 0 = full pass (no cap)
    "joint.eval.interval=1"                     # eval each epoch
    "data.num_workers=4"
    "logging.save_every_n_epochs=1"             # checkpoint every epoch
    "logging.keep_last_n_ckpts=5"               # 5 epoch ckpts on disk + best/ + best_by_monitor/
    "logging.wandb=false"
    "hardware.compile=false"
    "data.integration.adapter.export_root=${REMOTE_DATASET}"
)

save_state() {
    {
        echo "INSTANCE_ID=${INSTANCE_ID:-}"
        echo "SSH_HOST=${SSH_HOST:-}"
        echo "SSH_PORT=${SSH_PORT:-22}"
        echo "RUN_NAME=${RUN_NAME:-}"
    } > "$STATE_FILE"
}

# Convenience: print the SSH command users need.
ssh_cmd() {
    load_state
    if [[ -z "${SSH_HOST:-}" ]]; then
        echo "(SSH_HOST not yet set — run 01_launch.sh first)"
        return 1
    fi
    echo "ssh -p ${SSH_PORT} root@${SSH_HOST}"
}

# Trailing load_state was removed — it would have clobbered RUN_NAME values
# computed in this session.
