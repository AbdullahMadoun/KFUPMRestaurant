# 🍴 Intelligent Food Analysis, Segmentation, and TriFoodNet Research

This repository now serves two purposes at once:

- a public showcase of the earlier V1 and V2 food-analysis pipelines
- the main public home of the V3 TriFoodNet research track

The V3 track is now the primary focus. It preserves the training-oriented code,
retained experiment evidence, faculty-review documentation, and representative
`batch8` source samples behind the latest multi-stage system.

[![TriFoodNet Research Track](assets/v3/batch8_samples/Cluster_161_frame_frame_091147_00/visualization.jpg)](experiments/v3_trifoodnet_research_snapshot/)

---

## 🏆 The Evolution of the Pipeline

This repository now presents three complementary tracks:

- `V1`: an early hybrid FoodSAM + PictSure pipeline
- `V2`: a polished public Qwen2.5-VL + SAM3 inference showcase
- `V3`: a research-facing TriFoodNet snapshot with training code, reports, and retained experiments

Together they show the progression from a practical prototype to a more serious
research system with preserved empirical evidence.

| Feature | **V1: Hybrid Classification** | **V2: Multimodal Reasoning** | **V3: Research Snapshot** |
| :--- | :--- | :--- | :--- |
| **Primary goal** | early functional prototype | public inference showcase | faculty-facing training and evaluation package |
| **Models** | FoodSAM + PictSure | **Qwen2.5-VL + SAM3** | **Qwen2.5-VL + SAM + PictSure-style classifier** |
| **Context** | manual reference context | scene-level reasoning | staged end-to-end research workflow |
| **Evidence style** | example outputs | polished visual runs | logs, reports, metrics, validation, docs |
| **Best use** | lineage | demo | technical assessment |

---

## 🧪 Main Track: V3 TriFoodNet Research Snapshot

The main active direction in this repository is the V3 TriFoodNet research
snapshot under `experiments/v3_trifoodnet_research_snapshot/`. This is the part
of the repo intended for technical review, thesis discussion, and future model
development.

It keeps:

- the end-to-end three-stage training code
- retained multi-run logs and reports
- a faculty-facing review path
- representative `batch8` source inputs and visual outputs
- architecture and loss diagrams

### Current Research Position

The retained best run still used a partially trainable SAM path:

- Stage 2 image encoder: frozen
- Stage 2 prompt encoder: frozen
- Stage 2 mask decoder: trainable

The next planned simplification is to freeze SAM completely and treat it as a
fixed segmentation module while focusing optimization on Stage 1 and Stage 3.
The reason is not that this is already proven as the final answer; it is the
current research hypothesis for reducing training complexity and isolating where
the largest remaining gains are likely to come from.

In other words:

- preserve SAM as a stable segmentation component
- reduce Stage 2 optimization burden in joint training
- push the learning effort toward grounding and masked-item recognition

### Why Freeze SAM Further

Based on the retained snapshot:

- the best run already froze most of SAM and only left the mask decoder trainable
- the overall training stack is complex, and the project already has strong
  evidence that Stage 1 and Stage 3 are the more strategic places to keep
  iterating
- freezing SAM fully is the cleanest next experiment if the goal is to simplify
  the training path without discarding the three-stage design

### Research Snapshot Links

- **Main V3 README:** [TriFoodNet research snapshot](./experiments/v3_trifoodnet_research_snapshot/README.md)
- **Faculty review guide:** [Review path](./experiments/v3_trifoodnet_research_snapshot/docs/FACULTY_REVIEW_GUIDE.md)
- **All-trials report:** [Retained cross-run comparison](./experiments/v3_trifoodnet_research_snapshot/outputs/all_trials_report_20260321/index.md)
- **Best-run summary:** [Cleandata1 summary](./experiments/v3_trifoodnet_research_snapshot/outputs/trial-20260321-cleandata1/report_metrics/RESULTS_SUMMARY.md)

---

## 🔬 Experiment V2: SAM3 + Qwen-VL Showcase

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

## 📚 V3 Snapshot Overview

This repository now also includes a separate research-facing track under
`experiments/v3_trifoodnet_research_snapshot/`. Unlike the public V2 inference
showcase, this track preserves the actual training-oriented code, retained run
history, validation notes, and faculty-review documentation for the later
three-stage pipeline:

- Stage 1: Qwen2.5-VL grounding
- Stage 2: SAM-based segmentation
- Stage 3: masked-item classification with a PictSure-style model

<p align="center">
  <img src="assets/v3/diagrams/trifoodnet_inference_pipeline.png" width="34%" alt="TriFoodNet inference pipeline" />
  <img src="assets/v3/diagrams/trifoodnet_multitask_loss.png" width="60%" alt="TriFoodNet multitask loss" />
</p>

### Latest Retained Training Snapshot

| Item | Value |
| :--- | :--- |
| Strongest retained run | `trial-20260321-cleandata1` |
| Best retained checkpoint | `epoch_038` by `joint/combined = 1.9375961198969618` |
| Final retained epoch | `epoch_040` |
| Final dev Stage 1 recall@0.5 | `0.8636363636363636` |
| Final dev Stage 2 mIoU | `0.5733921838932934` |
| Final dev Stage 3 accuracy | `0.5` |
| Runs compared in retained report | `15` |

### Batch8 Input Package

The V3 MVP used the `batch_results_v8_500` package from the `v3_3stage_mvp`
workspace. That package contains:

- `500` total images
- `467` successful processed cases
- raw image cases with `original.jpg`, `visualization.jpg`, item crops, and masks

Representative public samples are included under:

- `assets/v3/batch8_samples/`

<p align="center">
  <img src="assets/v3/batch8_samples/Cluster_0_frame_frame_025403_00/original.jpg" width="22%" alt="Batch8 original sample 1" />
  <img src="assets/v3/batch8_samples/Cluster_0_frame_frame_025403_00/visualization.jpg" width="22%" alt="Batch8 visualization sample 1" />
  <img src="assets/v3/batch8_samples/Cluster_161_frame_frame_091147_00/original.jpg" width="22%" alt="Batch8 original sample 2" />
  <img src="assets/v3/batch8_samples/Cluster_161_frame_frame_091147_00/visualization.jpg" width="22%" alt="Batch8 visualization sample 2" />
</p>

### Review-Oriented Entry Points

- **Faculty-facing overview:** [V3 snapshot README](./experiments/v3_trifoodnet_research_snapshot/README.md)
- **Supervisor review path:** [Faculty review guide](./experiments/v3_trifoodnet_research_snapshot/docs/FACULTY_REVIEW_GUIDE.md)
- **Batch8 source note:** [Batch8 dataset note](./experiments/v3_trifoodnet_research_snapshot/BATCH8_DATASET_NOTE.md)
- **Cross-run comparison:** [All-trials report](./experiments/v3_trifoodnet_research_snapshot/outputs/all_trials_report_20260321/index.md)
- **Best-run summary:** [Results summary](./experiments/v3_trifoodnet_research_snapshot/outputs/trial-20260321-cleandata1/report_metrics/RESULTS_SUMMARY.md)

👉 **[Explore V3 Research Snapshot](./experiments/v3_trifoodnet_research_snapshot/)**

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
