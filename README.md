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

### 🖼️ The Innovation Journey: 6-Run Analysis

We meticulously refined the pipeline through these major phases. Click to expand each run and view the evolution.

<details>
<summary><b>🔬 Run 1: Baseline (Semantic Identifying)</b></summary>
*Standard food naming prompts.*
<p align="center">
  <img src="assets/v2/run1_baseline/segmented_CHICKEN.jpg" width="24%" alt="R1 Chicken" />
  <img src="assets/v2/run1_baseline/segmented_FISH.jpg" width="24%" alt="R1 Fish" />
  <img src="assets/v2/run1_baseline/segmented_mixed_1.jpg" width="24%" alt="R1 Mixed 1" />
  <img src="assets/v2/run1_baseline/segmented_mixed_2.jpg" width="24%" alt="R1 Mixed 2" />
</p>
</details>

<details>
<summary><b>🎨 Run 2: Visual Descriptions</b></summary>
*Color and Shape based prompting for improved boundary alignment.*
<p align="center">
  <img src="assets/v2/run2_visual/segmented_CHICKEN.jpg" width="24%" alt="R2 Chicken" />
  <img src="assets/v2/run2_visual/segmented_FISH.jpg" width="24%" alt="R2 Fish" />
  <img src="assets/v2/run2_visual/segmented_mixed1.jpg" width="24%" alt="R2 Mixed 1" />
  <img src="assets/v2/run2_visual/segmented_mixed2.jpg" width="24%" alt="R2 Mixed 2" />
</p>
</details>

<details>
<summary><b>📊 Run 3: High Recall (Instance Counting)</b></summary>
*Extremely low threshold (0.01) to find every potential item.*
<p align="center">
  <img src="assets/v2/run3_refined/segmented_CHICKEN.jpg" width="24%" alt="R3 Chicken" />
  <img src="assets/v2/run3_refined/segmented_FISH.jpg" width="24%" alt="R3 Fish" />
  <img src="assets/v2/run3_refined/segmented_mixed1.jpg" width="24%" alt="R3 Mixed 1" />
  <img src="assets/v2/run3_refined/segmented_mixed2.jpg" width="24%" alt="R3 Mixed 2" />
</p>
</details>

<details>
<summary><b>🎯 Run 4: Precision Refinement (Global NMS)</b></summary>
*Eliminating "Ghost" results and overlapping detections using standard NMS.*
<p align="center">
  <img src="assets/v2/run4_nms/segmented_CHICKEN.jpg" width="24%" alt="R4 Chicken" />
  <img src="assets/v2/run4_nms/segmented_FISH.jpg" width="24%" alt="R4 Fish" />
  <img src="assets/v2/run4_nms/segmented_mixed1.jpg" width="24%" alt="R4 Mixed 1" />
  <img src="assets/v2/run4_nms/segmented_mixed2.jpg" width="24%" alt="R4 Mixed 2" />
</p>
</details>

<details>
<summary><b>✨ Run 5: Bold Visuals & Aesthetic Polish</b></summary>
*Thicker 4px boundaries and high-opacity overlays for human review.*
<p align="center">
  <img src="assets/v2/run5_bold/segmented_CHICKEN.jpg" width="24%" alt="R5 Chicken" />
  <img src="assets/v2/run5_bold/segmented_FISH.jpg" width="24%" alt="R5 Fish" />
  <img src="assets/v2/run5_bold/segmented_mixed1.jpg" width="24%" alt="R5 Mixed 1" />
  <img src="assets/v2/run5_bold/segmented_mixed2.jpg" width="24%" alt="R5 Mixed 2" />
</p>
</details>

<details open>
<summary><b>🚀 Run 6: Final (Dynamic Auto-Retry Logic)</b></summary>
*The production version. Guaranteed detection via intelligent threshold retries.*
<p align="center">
  <img src="assets/v2/run6_final/segmented_CHICKEN.jpg" width="24%" alt="R6 Chicken" />
  <img src="assets/v2/run6_final/segmented_FISH.jpg" width="24%" alt="R6 Fish" />
  <img src="assets/v2/run6_final/segmented_mixed1.jpg" width="24%" alt="R6 Mixed 1" />
  <img src="assets/v2/run6_final/segmented_mixed3.jpg" width="24%" alt="R6 Mixed 2" />
</p>
</details>

👉 **[Explore V2 Code & Reproduction Guide](./experiments/v2_sam3_qwen_vl/)**

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
