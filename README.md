# 🍴 Intelligent Food Analysis & Segmentation Showcase

This repository documents the evolution of specialized food recognition technology, transitioning from few-shot classification to advanced multimodal reasoning.

[![Main Showcase](assets/v2/run6_final/segmented_mixed3.jpg)](experiments/v2_sam3_qwen_vl/)

---

## 🏆 The Evolution of the Pipeline

We present two distinct generations of our food analysis engine. While both achieve high-quality results, the transition to V2 represents a paradigm shift in how AI understands culinary scenes.

| Feature | **V1: Hybrid Classification** | **V2: Multimodal Reasoning** |
| :--- | :--- | :--- |
| **Logic** | Few-shot comparison against refs | Zero-shot visual reasoning |
| **Models** | FoodSAM + PictSure | **Qwen2.5-VL + SAM3** |
| **Context** | Required manually provided context images | Understands scene context natively |
| **Thresholding** | Fixed confidence scores | **Dynamic Auto-Retry Logic** |
| **Precision** | Standard segmentation | **Global NMS** (Overlapping detection fix) |

---

## 🔬 Experiment V2: SAM3 + Qwen-VL (State of the Art)

The current production-grade pipeline uses **Qwen2.5-VL-3B** for reasoning and **SAM3** for precise pixel-level masking. It was developed through 6 rigorous experimental iterations to reach peak performance.

### 🖼️ The 6-Run Innovation Gallery
We meticulously refined the pipeline through these major phases:

| Run 1: Baseline | Run 2: Visual Descriptions | Run 3: Instance Counting |
| :---: | :---: | :---: |
| ![R1](assets/v2/run1_baseline/segmented_mixed_1.jpg) | ![R2](assets/v2/run2_visual/segmented_mixed1.jpg) | ![R3](assets/v2/run3_refined/segmented_mixed1.jpg) |
| *Semantic Prompts* | *Color/Shape Prompts* | *High Recall (0.01)* |

| Run 4: Precision (NMS) | Run 5: Aesthetic Polish | Run 6: Final (Dynamic) |
| :---: | :---: | :---: |
| ![R4](assets/v2/run4_nms/segmented_mixed1.jpg) | ![R5](assets/v2/run5_bold/segmented_mixed1.jpg) | ![R6](assets/v2/run6_final/segmented_mixed3.jpg) |
| *No Ghosting* | *Bold Visuals (4px)* | *Guaranteed Detection* |

👉 **[Explore V2 Code & Guide](./experiments/v2_sam3_qwen_vl/)**

---

## 🥗 Experiment V1: FoodSAM + PictSure (Legacy)

Our first successful iteration utilized a hybrid approach combining semantic experts with a few-shot learner. While effective, it required high-quality reference "context" images to function.

<p align="center">
  <img src="assets/v1/results/mixed_1_hybrid_vis.jpg" width="45%" />
  <img src="assets/v1/results/mixed_2.jpg" width="45%" />
</p>

👉 **[Explore V1 Legacy Code](./experiments/v1_hybrid_foodsam_pictsure/)**

---

## 🛠️ Quick Start (V2)

1. **Setup Environment**:
   ```bash
   pip install -r experiments/v2_sam3_qwen_vl/requirements.txt
   ```
2. **Run Inference**:
   ```bash
   python experiments/v2_sam3_qwen_vl/main.py "/path/to/images" --output_dir "./results"
   ```

Check the [V2 Reproduction Guide](experiments/v2_sam3_qwen_vl/REPRODUCTION_GUIDE.md) for full hardware requirements and troubleshooting.

---
**Developed by Antigravity**
