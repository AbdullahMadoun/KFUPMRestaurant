# Paper ↔ Code Mapping

This document maps every section of the paper **"TahamajaNet: A 72B→7B Distillation Pipeline"** to the exact files in this repository.

---

## §3 — Data Engine: The Teacher Pipeline

| Paper Concept | Code Location |
|---|---|
| Stage 1: Qwen2.5-VL-72B-AWQ labeling | `experiments/05_72b_teacher_evaluation/` — Full 72B batch results on 809 images |
| Structured JSON prompt (Figure 2) | `experiments/04_three_stage_batch_prototype/stage1_vlm.py` — prompt template |
| Prompt engineering (splitting/exclusion rules) | `experiments/06_vlm_grounded_prompting/` — prompt variant comparison |
| Stage 2: SAM3 mask refinement | `src/tahamajanet/stage2_sam.py` — confidence cascade (0.10→0.05→0.02→0.01) |
| Stage 3: DINOv2+SigLIP clustering + HDBSCAN | `src/tahamajanet/dataset_integration.py` — dataset construction pipeline |
| v3.2 dataset (4,629 items, 32 classes) | `experiments/08_training_infrastructure/DATASET.md` — dataset specification |
| Human cluster curation workflow | `experiments/03_trifoodnet_joint_training/` — cluster viewer tools |

## §4 — Student Pipeline: TahamajaNet Inference

| Paper Concept | Code Location |
|---|---|
| Stage 1: Distilled Qwen2.5-VL-7B (bbox-only) | `src/tahamajanet/stage1_qwen.py` |
| Bbox-only JSON schema | `src/tahamajanet/stage1_kcfd/schema.py` |
| LoRA config (r=16, α=32) | `src/tahamajanet/stage1_kcfd/model.py` |
| Stage 2: SAM3 (frozen, shared) | `src/tahamajanet/stage2_sam.py` |
| Stage 3: DINOv2 max-similarity retrieval | `src/tahamajanet/stage3_vector_db.py` |
| Full 3-stage pipeline orchestration | `src/tahamajanet/pipeline.py` |
| Training setup (Table 3) | `src/tahamajanet/stage1_kcfd/config.py` + `src/tahamajanet/master_config.yaml` |

## §5 — Cardinality-Aware Fine-Tuning

| Paper Concept | Code Location |
|---|---|
| Structural count loss |$\hat{n} - n$| | `src/tahamajanet/stage1_kcfd/loss.py` |
| Count loss weight λ=5.0 | `src/tahamajanet/stage1_kcfd/config.py` |
| Cardinality regression observation | `src/tahamajanet/losses.py` (joint training losses, pre-kcfd) |
| Training loop with count-aware eval | `src/tahamajanet/stage1_kcfd/trainer.py` |

## §6 — Dish-Correct Metric

| Paper Concept | Code Location |
|---|---|
| Count-exact, class set-IoU, multiset-match, dish-correct | `src/tahamajanet/metrics.py` |
| Evaluation harness (596 held-out images) | `src/tahamajanet/eval_harness.py` |
| Per-cardinality breakdown | `src/tahamajanet/stage1_kcfd/eval.py` |

## §7 — Experiments

| Paper Concept | Code Location |
|---|---|
| Base 7B vs Distilled 7B comparison | `src/tahamajanet/eval_harness.py` |
| Ablation: max-sim vs mean-pooled | `experiments/09_dinov2_knn_classifier/VECTOR_DB.md` |
| Batch inference across 500+ images | `experiments/04_three_stage_batch_prototype/run_batch.py` |
| Qualitative examples | `experiments/04_three_stage_batch_prototype/batch_results_v8_500/` |

---

## Experiment Timeline (Chronological)

```
01 FoodSAM+PictSure ──→ 02 SAM3+Qwen ──→ 10 Config Refactor
                                              │
                              06 Grounded Prompting
                                              │
                              04 Three-Stage Batch Prototype ──→ 05 72B Teacher Eval
                                              │
                              03 TriFoodNet Joint Training
                                              │
                              08 Training Infrastructure
                                              │
                              07 Stage 1 Distillation ──→ 09 DINOv2 k-NN
                                              │
                                    ┌─────────┘
                                    ▼
                           FINAL PAPER RESULTS
                           (86.4% dish-correct)
```
