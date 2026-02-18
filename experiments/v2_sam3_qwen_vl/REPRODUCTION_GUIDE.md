# Professional Reproduction & Troubleshooting Guide

This guide provides an exhaustive deep-dive into the technical implementation, environment challenges, and reproduction steps for the SAM3 + Qwen-VL Food Segmentation Pipeline.

## 1. System Architecture & Methodology

The pipeline utilizes a **Modular Cascaded Inference** design:
1.  **Reasoning Layer (Qwen2.5-VL)**: Analyzes the image to determine *what* is present and *how many* instances exist. It outputs structured JSON.
2.  **Segmentation Layer (SAM3)**: Receives text prompts from the reasoning layer. It performs zero-shot segmentation to find the exact pixel boundaries.
3.  **Refinement Layer (Dynamic Thresholding & NMS)**: 
    -   Handles failures by lowering confidence thresholds dynamically.
    -   Removes redundant overlaps using Global Non-Maximum Suppression (NMS).
4.  **Visualization Layer (OpenCV)**: Renders high-opacity, thick-contoured masks with opaque labels.

---

## 2. Environment Setup (The "Hard Way")

If pip installs fail or you encounter version conflicts, follow this verified recipe:

### GPU Requirements
- **Minimum**: 16GB VRAM (with aggressive memory management).
- **Recommended**: 24GB VRAM (NVIDIA 3090/4090/A100).

### Step-by-Step CLI
```bash
# 1. Create a clean environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with specific CUDA version (example for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install vLLM (Critical: Ensure it matches your CUDA version)
pip install vllm==0.6.3

# 4. Clone and Install SAM3 in Editable Mode
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..

# 5. Download SAM3 Assets
# Ensure 'bpe_simple_vocab_16e6.txt.gz' is in sam3/assets/
```

---

## 3. Critical Troubleshooting (Known Issues & Solutions)

### Issue A: Out of Memory (OOM) on GPU
**Symptoms**: `RuntimeError: CUDA out of memory` or vLLM initialization failure.
**Solution**: 
- Adjust `gpu_memory_utilization` in `qwen_food_prompter.py`. We found **0.4** is the sweet spot for 24GB cards when sharing with SAM3.
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as an environment variable.
- For 16GB cards, set `gpu_memory_utilization` to **0.25** and ensure SAM3 is not loaded until Qwen finishes.

### Issue B: vLLM Multi-Processing Error
**Symptoms**: `We must use the 'spawn' multiprocessing start method.`
**Solution**: This happens if CUDA is initialized before vLLM starts the engine. Our `main.py` is structured to avoid this, but if it occurs, ensure no `torch.cuda` calls happen before `LLM()` is instantiated.

### Issue C: `copyMakeBorder` or Visualization Crash
**Symptoms**: `OpenCV Error: (-215:Assertion failed) top >= 0`
**Solution**: This usually occurs when SAM3 returns a mask with extra singleton dimensions (e.g., `(1, 1, H, W)`). Our `SAM3Segmenter` includes a `.squeeze()` call to ensure masks are always 2D before passing to OpenCV.

### Issue D: SAM3 `ModuleNotFoundError`
**Symptoms**: `No module named 'sam3'`
**Solution**: Ensure you are running from the parent directory of the `sam3` folder and that `sys.path` is correctly adjusted (done automatically in `main.py`).

---

## 4. Experimental Record (The 6 Runs)

| Run | Key Innovation | Threshold | Result Quality |
| :--- | :--- | :--- | :--- |
| **1** | Baseline | 0.1 | Good, but missed counts. |
| **2** | Visual Descriptions | 0.1 | Improved boundary alignment. |
| **3** | Instance Counting | 0.01 | Extreme recall, some noise. |
| **4** | Global NMS | 0.2 | Clean, professional, precise. |
| **5** | Bold Viz | 0.1 | Clear, thick boundaries (4px). |
| **6** | Dynamic Threshold | **Auto** | Guaranteed detection per image. |

---

## 5. Directory Structure for Reproduction
- `code/`: Contains the 4 core scripts.
- `showcase/`: Pre-rendered galleries from all experimental phases.
- `requirements.txt`: Frozen dependency list.
- `REPRODUCTION_GUIDE.md`: This document.
