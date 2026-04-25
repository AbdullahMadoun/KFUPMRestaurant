#!/usr/bin/env bash
# Find the cheapest verified RTX 4090 with adequate network.
# 4090 (sm_89) is well-supported by stock pytorch 2.5 + bitsandbytes 0.43,
# avoiding the Blackwell sm_120 dependency cascade we hit on the 5090.
#
# Stdout: just the offer ID (so callers can `OFFER_ID=$(pick_offer.sh)`).
# Stderr: a formatted top-5 table for human eyeballs.

set -euo pipefail

GPU_NAME="${GPU_NAME:-RTX_4090}"
MIN_INET="${MIN_INET:-500}"
MIN_DISK="${MIN_DISK:-80}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.95}"

QUERY="gpu_name=${GPU_NAME} num_gpus=1 verified=true rentable=true inet_down>=${MIN_INET} inet_up>=${MIN_INET} disk_space>=${MIN_DISK} reliability>=${MIN_RELIABILITY}"

OFFERS=$(vastai search offers --raw "${QUERY}" -o 'dph_total' 2>&1)

echo "[pick_offer] top 5 candidates (cheapest first):" >&2
echo "${OFFERS}" | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
if not data:
    print('NO OFFERS', file=sys.stderr); sys.exit(1)
print(f'{\"ID\":<10} {\"\$/hr\":<8} {\"VRAM\":<6} {\"disk\":<8} {\"net up/down\":<14} {\"rel\":<6} {\"loc\":<22}', file=sys.stderr)
for o in data[:5]:
    print(f'{o[\"id\"]:<10} {o[\"dph_total\"]:<8.4f} {o[\"gpu_ram\"]/1024:<6.0f} {o[\"disk_space\"]:<8.0f} {o[\"inet_up\"]:<6.0f}/{o[\"inet_down\"]:<6.0f} {o[\"reliability2\"]:<6.4f} {o[\"geolocation\"]:<22}', file=sys.stderr)
print(data[0]['id'])
"
