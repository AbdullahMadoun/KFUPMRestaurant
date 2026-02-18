# 🍴 Intelligent Food Segmentation Pipeline (Core Distribution)

This package contains the fully optimized, production-grade food segmentation pipeline developed through 6 iterative experimental runs. It uses **Qwen2.5-VL-3B** for reasoning and **SAM3** for precise pixel-level masking.

---

## 🛠 1. Exhaustive Installation Guide

### Hardware Requirements
- **GPU**: NVIDIA 24GB VRAM (3090/4090/A100) is necessary for high-resolution images or concurrent model usage.
- **CPU**: 16GB+ RAM.

### Step 1: Python Environment
We recommend using Conda or a Virtual Environment:
```bash
python3 -m venv food_seg_env
source food_seg_env/bin/activate
pip install -r requirements.txt
```

### Step 2: SAM3 Core Installation
SAM3 is a Meta AI research model. Install it as an editable package:
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e . && cd ..
```

### Step 3: Gated Model Access (Crucial)
1.  Go to [Hugging Face](https://huggingface.co/) and create an Account.
2.  Accepted the terms for `Qwen/Qwen2.5-VL-3B-Instruct` and optionally SAM3.
3.  Login via CLI:
    ```bash
    huggingface-cli login
    ```

---

## 🔬 2. The Experimental Journey (Showcase)

This project was built through rigorous testing. See **`WALKTHROUGH_ZIP.md`** for gallery comparisons of the following runs:
1.  **Baseline**: Proved the feasibility of semantic text prompts.
2.  **Visual Descriptions**: Validated that color/shape prompts improve boundary precision.
3.  **High Recall**: Pushed the model to find every item (Threshold 0.01).
4.  **NMS & Cleaning**: Added Global Non-Maximum Suppression to remove "ghost" results.
5.  **Bold Visualization**: Optimized the output for human review (4px thick contours, 0.7 opacity).
6.  **Final Polish**: Implemented **Dynamic Threshold Logic** (guaranteeing masks) and fixed cross-platform visualization bugs.

---

## ⚙️ 3. Technical Deep-Dive & Troubleshooting

### vLLM & Memory Management
The `qwen_food_prompter.py` script uses `vLLM` for accelerated generation. 
- **OOM Fix**: If you see `CUDA out of memory`, decrease `gpu_memory_utilization` from 0.4 to 0.25 in `qwen_food_prompter.py`.
- **Eager Mode**: We use `enforce_eager=True` to bypass potentially slow or unstable graph compilations during batch runs.

### SAM3 Integration
- **Dynamic Thresholding**: In `sam3_segmenter.py`, the `segment` method now includes a retry loop. If no results are found at the user's threshold, it retries at 0.05, 0.02, and 0.01.
- **Global NMS**: Prevents "The same item getting two labels". It sorts all candidate masks by score across ALL generated prompts and keeps only the top 5 non-overlapping ones.

### Image Processing (OpenCV Fixes)
- **Mask Squeezing**: We resolved the common `copyMakeBorder` error by ensuring all masks are squeezed to 2D before contour extraction.
- **Bold Visuals**: Contours are drawn with `cv2.drawContours` with a custom thickness (default 4px) to make them "pop" against busy backgrounds.

---

## 📖 4. Detailed Usage Guide

Run the pipeline on a folder of images:
```bash
python3 main.py "/path/to/my/images" \
  --output_dir "./segmented_images" \
  --threshold 0.1 \
  --alpha 0.4 \
  --thickness 4 \
  --skip_boxes
```

### Script Roles:
- `main.py`: Entry point. Manages paths, metrics, and orchestrates Qwen -> SAM3.
- `qwen_food_prompter.py`: Connects to vLLM.
- `sam3_segmenter.py`: Connects to SAM3 model.
- `visualizer.py`: Creates the final artistic overlays.

---
**Developed by Antigravity**
