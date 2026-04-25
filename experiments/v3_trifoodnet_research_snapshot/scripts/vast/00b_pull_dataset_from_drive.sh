#!/usr/bin/env bash
# Pull the dataset tarball from Google Drive on the rented vast.ai instance,
# extract it, and verify it matches the expected dataset hash.
#
# Replaces the laptop-rsync path: the instance pulls directly from Drive over
# its (much faster) data-center network, no laptop involvement.
#
# Required env (sourced from 00_state.sh):
#   DRIVE_FILE_ID            — the long id between '/d/' and '/view' in the share URL
#   DRIVE_TAR_SHA8           — sha8 of the tarball (from bundle_dataset.sh output)
#   DATASET_EXPECTED_HASH    — content_hash_sha8 from manifest.json (e.g. 61ac038c)
#   REMOTE_DATASET           — where to extract on the instance (default: /root/work/dataset_v3_export)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_state.sh
source "${SCRIPT_DIR}/00_state.sh"

if [[ -z "${SSH_HOST:-}" ]]; then
    echo "[drive] FAIL: SSH_HOST not set. Run 01_launch.sh first." >&2
    exit 1
fi
if [[ -z "${DRIVE_FILE_ID:-}" ]]; then
    echo "[drive] FAIL: DRIVE_FILE_ID not set. Add it to 00_state.sh after upload." >&2
    exit 1
fi
if [[ -z "${DRIVE_TAR_SHA8:-}" ]]; then
    echo "[drive] WARN: DRIVE_TAR_SHA8 not set — tarball integrity will not be verified." >&2
fi
if [[ -z "${DATASET_EXPECTED_HASH:-}" ]]; then
    echo "[drive] WARN: DATASET_EXPECTED_HASH not set — manifest hash will not be verified." >&2
fi

SSH_BASE=(ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"${SSH_HOST}")

REMOTE_TAR_PATH="/root/work/dataset.tar"

echo "[drive] file id:           ${DRIVE_FILE_ID}"
echo "[drive] expected tar sha8: ${DRIVE_TAR_SHA8:-<unverified>}"
echo "[drive] expected dataset:  ${DATASET_EXPECTED_HASH:-<unverified>}"
echo

# ── 1. Install gdown if missing (it's a tiny pure-python pkg) ────────────────
"${SSH_BASE[@]}" '
    set -e
    if ! command -v gdown >/dev/null 2>&1; then
        echo "[drive] installing gdown on remote..."
        pip install --quiet gdown 2>&1 | tail -2
    fi
    gdown --version 2>&1 | head -1
'

# ── 2. Download (gdown 6.x CLI: bare ID or URL as positional, no --id/--fuzzy) ──
echo "[drive] downloading via gdown (this is the long step — ~2-5 min for 1.6 GB)..."
"${SSH_BASE[@]}" "
    set -e
    mkdir -p ${REMOTE_DATASET%/*}
    rm -f ${REMOTE_TAR_PATH}
    gdown 'https://drive.google.com/uc?id=${DRIVE_FILE_ID}' -O ${REMOTE_TAR_PATH}
    ls -lh ${REMOTE_TAR_PATH}
"

# ── 3. Verify tar integrity ──────────────────────────────────────────────────
if [[ -n "${DRIVE_TAR_SHA8:-}" ]]; then
    echo "[drive] verifying tar sha8..."
    REMOTE_SHA=$("${SSH_BASE[@]}" "sha256sum ${REMOTE_TAR_PATH} | cut -c1-8")
    REMOTE_SHA=${REMOTE_SHA//[$'\r\n ']/}
    if [[ "${REMOTE_SHA}" != "${DRIVE_TAR_SHA8}" ]]; then
        echo "[drive] FAIL: tar sha8 mismatch — expected ${DRIVE_TAR_SHA8}, got ${REMOTE_SHA}" >&2
        echo "[drive]   the download is corrupted or pointing at the wrong file." >&2
        exit 1
    fi
    echo "[drive]   tar sha8 OK"
fi

# ── 4. Extract ────────────────────────────────────────────────────────────────
# Strategy: read the top-level dir name from the tar BEFORE extracting and
# BEFORE rm-ing the tar. Then extract into a staging dir, identify the single
# child, and rename to REMOTE_DATASET. This avoids the prior bug where the
# `rm -f` raced ahead of the rename detection.
echo "[drive] extracting..."
"${SSH_BASE[@]}" "
    set -e
    rm -rf ${REMOTE_DATASET}
    mkdir -p ${REMOTE_DATASET%/*}
    # Skip macOS xattr metadata entries (._foo) which sometimes lead the archive
    # depending on how it was created. These are files, not dirs, and would fail
    # the [[ -d ]] check below, leaving the actual extracted folder un-renamed.
    EXTRACTED_NAME=\$(tar -tf ${REMOTE_TAR_PATH} 2>/dev/null | grep -v '/\\._' | grep -v '^\\._' | head -1 | cut -d/ -f1)
    echo \"[drive]   tar top-level dir: \${EXTRACTED_NAME}\"
    cd ${REMOTE_DATASET%/*} && tar -xf ${REMOTE_TAR_PATH} 2>/dev/null
    # Strip macOS-specific xattr metadata files (._foo) — harmless but noisy.
    find ${REMOTE_DATASET%/*} -maxdepth 1 -name '._*' -delete 2>/dev/null || true
    if [[ -d \"${REMOTE_DATASET%/*}/\${EXTRACTED_NAME}\" ]]; then
        mv \"${REMOTE_DATASET%/*}/\${EXTRACTED_NAME}\" \"${REMOTE_DATASET}\"
    elif [[ ! -d \"${REMOTE_DATASET}\" ]]; then
        echo '[drive] ERROR: extraction produced unexpected layout' >&2
        ls -la ${REMOTE_DATASET%/*} >&2
        exit 1
    fi
    rm -f ${REMOTE_TAR_PATH}
"

# ── 5. Verify against manifest.json hash ─────────────────────────────────────
if [[ -n "${DATASET_EXPECTED_HASH:-}" ]]; then
    echo "[drive] verifying dataset content hash..."
    ACTUAL=$("${SSH_BASE[@]}" "grep content_hash_sha8 ${REMOTE_DATASET}/manifest.json | grep -oE '[0-9a-f]{8}'")
    ACTUAL=${ACTUAL//[$'\r\n ']/}
    if [[ "${ACTUAL}" != "${DATASET_EXPECTED_HASH}" ]]; then
        echo "[drive] FAIL: manifest content_hash_sha8 mismatch — expected ${DATASET_EXPECTED_HASH}, got ${ACTUAL}" >&2
        exit 1
    fi
    echo "[drive]   manifest hash OK"
fi

# ── 6. Layout summary ────────────────────────────────────────────────────────
"${SSH_BASE[@]}" "
    echo '[drive] dataset layout on remote:'
    echo '   images:    '\$(ls ${REMOTE_DATASET}/images | wc -l)
    echo '   crops:     '\$(ls ${REMOTE_DATASET}/crops | wc -l)' folders'
    echo '   masks:     '\$(ls ${REMOTE_DATASET}/masks | wc -l)' folders'
    echo '   reference: '\$(ls ${REMOTE_DATASET}/reference | wc -l)' folders'
    du -sh ${REMOTE_DATASET}
"

echo
echo "[drive] DONE. Dataset is at ${REMOTE_DATASET} on the instance."
echo "[drive] next: bash scripts/vast/03_run_remote.sh"
