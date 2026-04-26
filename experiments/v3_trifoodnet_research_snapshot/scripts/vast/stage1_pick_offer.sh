#!/usr/bin/env bash
# Pick the fastest expected single-GPU Vast offer for Stage 1 Qwen-VL training.
#
# Stdout: offer id only.
# Stderr: ranked table with the scoring inputs.

set -euo pipefail

MAX_DPH="${STAGE1_MAX_DPH:-2.00}"
MIN_VRAM_MB="${STAGE1_MIN_VRAM_MB:-32000}"
MIN_DISK_GB="${STAGE1_MIN_DISK_GB:-80}"
LIMIT="${STAGE1_OFFER_LIMIT:-12}"
VASTAI_BIN="${VASTAI_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
TABLE_FILE="${STAGE1_PICK_TABLE_FILE:-${PWD}/scripts/vast/.stage1_offer_table.txt}"

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
        echo "[stage1_pick_offer] FAIL: vastai CLI not found. Set VASTAI_BIN=/path/to/vastai." >&2
        exit 1
    fi
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python.exe >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python.exe)"
    elif [[ -x /mnt/c/Python312/python.exe ]]; then
        PYTHON_BIN="/mnt/c/Python312/python.exe"
    elif [[ -x /c/Python312/python.exe ]]; then
        PYTHON_BIN="/c/Python312/python.exe"
    else
        echo "[stage1_pick_offer] FAIL: python not found. Set PYTHON_BIN=/path/to/python." >&2
        exit 1
    fi
fi

QUERY="num_gpus=1 rentable=true dph_total<=${MAX_DPH}"
OFFERS_JSON="$("${VASTAI_BIN}" search offers --raw "${QUERY}" -o dph_total)"
OFFERS_FILE="$(mktemp)"
printf "%s" "${OFFERS_JSON}" > "${OFFERS_FILE}"
trap 'rm -f "${OFFERS_FILE}"' EXIT
mkdir -p "$(dirname "${TABLE_FILE}")"

OFFERS_FILE="${OFFERS_FILE}" TABLE_FILE="${TABLE_FILE}" MAX_DPH="${MAX_DPH}" MIN_VRAM_MB="${MIN_VRAM_MB}" MIN_DISK_GB="${MIN_DISK_GB}" LIMIT="${LIMIT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys

with open(os.environ["OFFERS_FILE"], "r", encoding="utf-8-sig") as handle:
    offers = json.load(handle)
max_dph = float(os.environ["MAX_DPH"])
min_vram_mb = float(os.environ["MIN_VRAM_MB"])
min_disk_gb = float(os.environ["MIN_DISK_GB"])
limit = int(os.environ["LIMIT"])
table_file = os.environ["TABLE_FILE"]


def number(row, key, default=0.0):
    try:
        return float(row.get(key) or default)
    except Exception:
        return float(default)


def vram_factor(vram_gb):
    # Bigger VRAM allows larger microbatches, which should reduce step count for
    # this Stage 1 SFT path. 80GB is best; 48GB-class is preferred over 32/40GB.
    if vram_gb >= 78:
        return 1.45
    if vram_gb >= 47.5:
        return 1.22
    if vram_gb >= 40:
        return 1.08
    if vram_gb >= 32:
        return 1.0
    return 0.50


def expected_speed_score(row):
    vram_gb = number(row, "gpu_total_ram", number(row, "gpu_ram")) / 1024.0
    dlperf = number(row, "dlperf")
    reliability = max(number(row, "reliability2", number(row, "reliability")), 0.85)
    disk = min(number(row, "disk_bw"), 10000.0)
    pcie = min(number(row, "pcie_bw"), 64.0)
    net = min(number(row, "inet_up"), number(row, "inet_down"), 4000.0)
    return dlperf * reliability * vram_factor(vram_gb) * (1.0 + disk / 100000.0) * (1.0 + pcie / 800.0) * (1.0 + net / 40000.0)


candidates = []
for offer in offers:
    if offer.get("rented"):
        continue
    if str(offer.get("verification")) != "verified":
        continue
    if number(offer, "dph_total", 999.0) > max_dph:
        continue
    if number(offer, "gpu_total_ram", number(offer, "gpu_ram")) < min_vram_mb:
        continue
    if number(offer, "disk_space") < min_disk_gb:
        continue
    candidates.append((expected_speed_score(offer), offer))

if not candidates:
    with open(table_file, "w", encoding="utf-8") as handle:
        handle.write(
            f"No verified rentable single-GPU offers under ${max_dph:.2f}/hr with "
            f">={min_vram_mb / 1024.0:.1f}GB VRAM and >={min_disk_gb:.0f}GB free disk.\n"
        )
    sys.exit(1)

candidates.sort(key=lambda pair: pair[0], reverse=True)
lines = ["[stage1_pick_offer] ranked fastest expected Stage 1 candidates:"]
lines.append(f"{'rank':<4} {'id':<10} {'gpu':<18} {'vram':>6} {'batch':>5} {'$/hr':>7} {'dlperf':>8} {'score':>8} {'rel':>7} {'net up/down':>13} {'disk':>8} {'loc'}")
for rank, (score, offer) in enumerate(candidates[:limit], start=1):
    vram_gb = number(offer, "gpu_total_ram", number(offer, "gpu_ram")) / 1024.0
    batch = 4 if vram_gb >= 78 else (2 if vram_gb >= 47.5 else 1)
    net = f"{number(offer, 'inet_up'):.0f}/{number(offer, 'inet_down'):.0f}"
    lines.append(
        f"{rank:<4} {str(offer.get('id')):<10} {str(offer.get('gpu_name', ''))[:18]:<18} "
        f"{vram_gb:6.1f} {batch:5d} {number(offer, 'dph_total'):7.3f} "
        f"{number(offer, 'dlperf'):8.1f} {score:8.1f} {number(offer, 'reliability2'):7.4f} "
        f"{net:>13} {number(offer, 'disk_bw'):8.0f} {offer.get('geolocation', '')}"
    )
with open(table_file, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")

print(candidates[0][1]["id"])
PY
