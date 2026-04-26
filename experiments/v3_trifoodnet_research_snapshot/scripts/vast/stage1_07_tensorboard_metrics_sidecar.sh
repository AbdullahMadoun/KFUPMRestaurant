#!/usr/bin/env bash
# Start a non-invasive TensorBoard sidecar that logs system/progress metrics.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[stage1_tb_sidecar] FAIL: SSH_HOST not set." >&2
    exit 1
fi

export STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${RUN_NAME:-stage1-qwen7b}}"
export STAGE1_TB_SIDECAR_INTERVAL="${STAGE1_TB_SIDECAR_INTERVAL:-15}"

SSH_BASE=(ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"${SSH_HOST}")

"${SSH_BASE[@]}" "
    set -e
    tmux kill-session -t stage1-tb-sidecar 2>/dev/null || true
    tmux new-session -d -s stage1-tb-sidecar -x 160 -y 40 \"
        python - <<'PY'
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

run = Path('/root/work/stage1_runs/${STAGE1_RUN_NAME}')
logdir = run / 'tensorboard'
writer = SummaryWriter(log_dir=str(logdir))
report = json.load(open(run / 'preflight' / 'dataset_report.json', 'r', encoding='utf-8'))
trainer = json.load(open(run / 'trainer_config.json', 'r', encoding='utf-8'))
train_images = int(report['train_images'])
batch_size = int(trainer['per_device_batch_size'])
grad_accum = int(trainer['gradient_accumulation_steps'])
epochs = int(trainer['epochs'])
forward_batches = math.ceil(train_images / batch_size)
updates_per_epoch = math.ceil(forward_batches / grad_accum)
total_updates = updates_per_epoch * epochs

static_scalars = {
    'run/train_images': train_images,
    'run/val_images': int(report['val_images']),
    'run/test_images': int(report['test_images']),
    'run/train_items': int(report['train_items']),
    'run/val_items': int(report['val_items']),
    'run/per_device_batch_size': batch_size,
    'run/gradient_accumulation_steps': grad_accum,
    'run/effective_batch_size': batch_size * grad_accum,
    'run/forward_batches_per_epoch': forward_batches,
    'run/optimizer_updates_per_epoch': updates_per_epoch,
    'run/total_optimizer_updates': total_updates,
}
for tag, value in static_scalars.items():
    writer.add_scalar(tag, value, 0)
writer.add_text('run/summary', json.dumps(static_scalars, indent=2, sort_keys=True), 0)
writer.flush()

def latest_train_step():
    try:
        ea = EventAccumulator(str(logdir))
        ea.Reload()
        vals = ea.Scalars('train/loss')
        return int(vals[-1].step) if vals else 0
    except Exception:
        return 0

def gpu_metrics():
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits',
        ], text=True, timeout=5)
        row = next(csv.reader([out.strip().splitlines()[0]]))
        util, mem_used, mem_total, power, temp = [float(value.strip()) for value in row]
        return {
            'system/gpu_util_pct': util,
            'system/gpu_memory_used_gb': mem_used / 1024.0,
            'system/gpu_memory_total_gb': mem_total / 1024.0,
            'system/gpu_memory_used_pct': 100.0 * mem_used / max(mem_total, 1.0),
            'system/gpu_power_w': power,
            'system/gpu_temperature_c': temp,
        }
    except Exception:
        return {}

interval = int('${STAGE1_TB_SIDECAR_INTERVAL}')
while True:
    step = latest_train_step()
    metrics = gpu_metrics()
    usage = shutil.disk_usage('/root/work')
    metrics.update({
        'progress/current_optimizer_step': step,
        'progress/current_epoch_float': step / max(updates_per_epoch, 1),
        'progress/percent_complete': 100.0 * step / max(total_updates, 1),
        'system/work_disk_used_gb': (usage.total - usage.free) / (1024 ** 3),
        'system/work_disk_free_gb': usage.free / (1024 ** 3),
        'system/load_1m': os.getloadavg()[0],
    })
    for tag, value in metrics.items():
        writer.add_scalar(tag, float(value), step)
    writer.flush()
    time.sleep(interval)
PY
    \"
    tmux ls
"

echo "[stage1_tb_sidecar] started stage1-tb-sidecar for ${STAGE1_RUN_NAME}"
