#!/usr/bin/env bash
# ============================================================================
# setup_server.sh — One-shot setup for Qwen2.5-VL-72B FP8 on Vast.ai
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export GIT_SSH_KEY="$(cat ~/.ssh/id_ed25519)"
#   bash setup_server.sh
#
# Idempotent: safe to re-run after a crash or partial setup.
# ============================================================================
set -euo pipefail

# ── Configurable defaults ───────────────────────────────────────────────────
SOURCE_DIR="${SOURCE_DIR:-/root/KFUPMRestaurant/data/Sampled_Images_All}"
WORK_DIR="${WORK_DIR:-/root}"
REPO_DIR="${WORK_DIR}/KFUPMRestaurant"
PIPELINE_DIR="${REPO_DIR}/experiments/v3_3stage_mvp"
SAM3_DIR="${WORK_DIR}/sam3"
VENV_DIR="${WORK_DIR}/venv"
N_IMAGES="${N_IMAGES:-500}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-8}"

# AWQ fallback: set USE_AWQ=1 to use the INT4-quantized model instead of FP8
USE_AWQ="${USE_AWQ:-0}"

echo "================================================================"
echo "  72B FP8 Setup Script for Vast.ai"
echo "================================================================"

# ── Step 1: System packages ─────────────────────────────────────────────────
echo ""
echo "[Step 1/10] System packages..."
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0 wget curl > /dev/null 2>&1
echo "  Done."

# ── Step 2: GPU detection + Blackwell check ─────────────────────────────────
echo ""
echo "[Step 2/10] GPU detection..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "  GPU: ${GPU_NAME}, VRAM: ${GPU_MEM} MiB"

# Check for Blackwell (SM 120) — requires vLLM source build
IS_BLACKWELL=0
if python3 -c "import torch; cc = torch.cuda.get_device_capability(); exit(0 if cc[0] >= 12 else 1)" 2>/dev/null; then
    IS_BLACKWELL=1
    echo "  Blackwell GPU detected (SM >= 120). Will build vLLM from source if needed."
fi

# ── Step 3: Python venv setup ───────────────────────────────────────────────
echo ""
echo "[Step 3/10] Python virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created venv at ${VENV_DIR}"
else
    echo "  Venv already exists."
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q

# ── Step 4: SSH key + repo clone ────────────────────────────────────────────
echo ""
echo "[Step 4/10] Repository setup..."
if [ -n "${GIT_SSH_KEY:-}" ]; then
    mkdir -p ~/.ssh
    echo "${GIT_SSH_KEY}" > ~/.ssh/id_ed25519
    chmod 600 ~/.ssh/id_ed25519
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null
    echo "  SSH key configured."
else
    echo "  WARNING: GIT_SSH_KEY not set. Assuming repo is already cloned."
fi

if [ ! -d "${REPO_DIR}" ]; then
    git clone git@github.com:AbdullahMadoun/KFUPMRestaurant.git "${REPO_DIR}"
    echo "  Cloned KFUPMRestaurant."
else
    echo "  Repo already exists. Pulling latest..."
    cd "${REPO_DIR}" && git pull --ff-only || true
fi

# ── Step 5: SAM3 setup ─────────────────────────────────────────────────────
echo ""
echo "[Step 5/10] SAM3 setup..."
if [ ! -d "${SAM3_DIR}" ]; then
    git clone https://github.com/facebookresearch/sam3.git "${SAM3_DIR}"
    echo "  Cloned sam3."
else
    echo "  sam3 already exists."
fi
pip install -e "${SAM3_DIR}" -q

# ── Step 6: vLLM install ───────────────────────────────────────────────────
echo ""
echo "[Step 6/10] vLLM install..."
if python3 -c "import vllm; print(f'vLLM {vllm.__version__} already installed')" 2>/dev/null; then
    echo "  vLLM already installed."
else
    if [ "${IS_BLACKWELL}" -eq 1 ]; then
        echo "  Blackwell GPU: building vLLM from source..."
        pip install -q cmake ninja packaging setuptools-scm
        VLLM_SRC="${WORK_DIR}/vllm_src"
        if [ ! -d "${VLLM_SRC}" ]; then
            git clone https://github.com/vllm-project/vllm.git "${VLLM_SRC}"
        fi
        cd "${VLLM_SRC}" && pip install -e . 2>&1 | tail -5
        cd "${WORK_DIR}"
    else
        echo "  Installing vLLM via pip..."
        pip install vllm -q
    fi
fi

# ── Step 7: Python dependencies ────────────────────────────────────────────
echo ""
echo "[Step 7/10] Python dependencies..."
pip install -r "${PIPELINE_DIR}/requirements.txt" -q
echo "  Done."

# ── Step 8: HuggingFace login + model pre-download ─────────────────────────
echo ""
echo "[Step 8/10] Model download..."
if [ -z "${HF_TOKEN:-}" ]; then
    echo "  WARNING: HF_TOKEN not set. Skipping HF login (model may already be cached)."
else
    pip install -q huggingface_hub
    huggingface-cli login --token "${HF_TOKEN}"
    echo "  HuggingFace login OK."
fi

if [ "${USE_AWQ}" -eq 1 ]; then
    MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    QUANTIZATION="awq"
    echo "  AWQ mode: downloading ${MODEL_NAME} (~38GB)..."
else
    MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct"
    QUANTIZATION="fp8"
    echo "  FP8 mode: downloading ${MODEL_NAME} (~144GB)..."
fi

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_NAME}', resume_download=True)
print('  Model download complete.')
"

# ── Step 9: Generate config JSON ────────────────────────────────────────────
echo ""
echo "[Step 9/10] Generating config_72b_fp8.json..."

CONFIG_FILE="${PIPELINE_DIR}/config_72b_fp8.json"

cat > "${CONFIG_FILE}" << 'INNER_EOF'
{
  "device": "cuda",
  "output_dir": "batch_results_72b_fp8",
  "sam3_repo_path": "SAM3_DIR_PLACEHOLDER",
  "vlm": {
    "model_name": "MODEL_NAME_PLACEHOLDER",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "enforce_eager": true,
    "quantization": "QUANTIZATION_PLACEHOLDER",
    "force_json": true,
    "allowed_local_media_path": "/root",
    "temperature": 0.2,
    "max_tokens": 768
  },
  "sam": {
    "model_path": "facebook/sam3",
    "confidence_threshold": 0.1,
    "fallback_thresholds": [0.05, 0.02, 0.01],
    "crop_padding": 5,
    "bpe_search_paths": [
      "SAM3_DIR_PLACEHOLDER/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    ]
  },
  "match": {
    "embedding_model": "google/siglip2-base-patch16-224",
    "embedding_dim": 768,
    "index_path": "menu.index",
    "metadata_path": "menu_meta.json",
    "top_k": 3,
    "similarity_threshold": 0.5,
    "text_weight": 0.3,
    "use_text_matching": true
  },
  "nms": {
    "max_objects": 8,
    "iou_threshold": 0.7
  },
  "viz": {
    "draw_boxes": true,
    "alpha": 0.7,
    "thickness": 3,
    "font_scale": 0.9,
    "font_thickness": 2,
    "show_match_label": true,
    "show_price": true,
    "show_confidence": true,
    "color_low": [0, 50],
    "color_high": [180, 255]
  }
}
INNER_EOF

# Patch placeholders with actual values
sed -i "s|SAM3_DIR_PLACEHOLDER|${SAM3_DIR}|g" "${CONFIG_FILE}"
sed -i "s|MODEL_NAME_PLACEHOLDER|${MODEL_NAME}|g" "${CONFIG_FILE}"
sed -i "s|QUANTIZATION_PLACEHOLDER|${QUANTIZATION}|g" "${CONFIG_FILE}"

echo "  Config written to ${CONFIG_FILE}"
echo "  Model: ${MODEL_NAME}"
echo "  Quantization: ${QUANTIZATION}"

# ── Step 10: Run batch ──────────────────────────────────────────────────────
echo ""
echo "[Step 10/10] Running batch inference (n=${N_IMAGES})..."
echo "================================================================"

cd "${PIPELINE_DIR}"
python3 run_batch.py \
    --source_dir "${SOURCE_DIR}" \
    --n "${N_IMAGES}" \
    --config config_72b_fp8.json \
    --output_dir batch_results_72b_fp8 \
    --vlm_batch_size "${VLM_BATCH_SIZE}" \
    --resume

echo ""
echo "================================================================"
echo "  DONE! Results in: ${PIPELINE_DIR}/batch_results_72b_fp8/"
echo "  Check batch_summary.json for success rate."
echo "================================================================"
