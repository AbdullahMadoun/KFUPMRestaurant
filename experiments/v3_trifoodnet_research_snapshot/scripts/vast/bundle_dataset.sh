#!/usr/bin/env bash
# Bundle the v3 export into one tar file for Google Drive upload.
#
# Why a tar (not zip, not folder):
#   - JPEGs are already compressed; zip/gzip add ~0% size and waste CPU
#   - Folder upload to Drive = 13.8K small files = slow API churn
#   - One tar file = one upload, one download, deterministic hash
#
# Output: <export_root>.tar in the same parent dir, with .sha8 sibling for verify.

set -euo pipefail

DATASET_DIR="${1:-/Users/abdulrazzak/Resturant_Pipeline_Feb/results_v2/exports/v3_2026-04-24_61ac038c}"
PARENT_DIR="$(dirname "${DATASET_DIR}")"
EXPORT_NAME="$(basename "${DATASET_DIR}")"
TAR_PATH="${PARENT_DIR}/${EXPORT_NAME}.tar"

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "[bundle] FAIL: dataset not found at ${DATASET_DIR}" >&2
    exit 1
fi

echo "[bundle] source:    ${DATASET_DIR}"
echo "[bundle] output:    ${TAR_PATH}"
echo "[bundle] expected:  ~1.6 GB"
echo

# ── tar ──
# -c create, -f file, -C parent so the archive's top-level entry is the export name
tar -cf "${TAR_PATH}" -C "${PARENT_DIR}" "${EXPORT_NAME}"

# ── verify ──
TAR_SIZE=$(du -h "${TAR_PATH}" | cut -f1)
TAR_BYTES=$(stat -f%z "${TAR_PATH}" 2>/dev/null || stat -c%s "${TAR_PATH}")
TAR_SHA=$(shasum -a 256 "${TAR_PATH}" | cut -c1-8)
echo "${TAR_SHA}" > "${TAR_PATH}.sha8"

# Also copy the manifest's content_hash for round-trip checks
MANIFEST_HASH=$(grep content_hash_sha8 "${DATASET_DIR}/manifest.json" | grep -oE '[0-9a-f]{8}')

echo
echo "[bundle] DONE."
echo "[bundle]   path:           ${TAR_PATH}"
echo "[bundle]   size:           ${TAR_SIZE} (${TAR_BYTES} bytes)"
echo "[bundle]   tar sha8:       ${TAR_SHA}     (saved to ${TAR_PATH}.sha8)"
echo "[bundle]   dataset hash:   ${MANIFEST_HASH}  (from manifest.json)"
echo
echo "[bundle] NEXT — manually do these in Google Drive:"
echo "         1. Open https://drive.google.com"
echo "         2. New → File upload → select ${TAR_PATH}"
echo "         3. After upload completes: right-click → Share → 'Anyone with the link' → 'Viewer'"
echo "         4. Copy share link. The file id is the long string between '/d/' and '/view'"
echo "            (e.g. https://drive.google.com/file/d/<FILE_ID>/view)"
echo "         5. Paste the FILE_ID back to me."
