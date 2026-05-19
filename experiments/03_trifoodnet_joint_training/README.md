# Experiment 03: TriFoodNet Joint Multi-Task Training

**Branch:** `main` (V3 content)  
**Timeline:** Core research phase  
**Status:** Foundation for final paper results

## Approach
TriFoodNet is a jointly-trained three-stage pipeline where:
- **Stage 1** (Qwen2.5-VL) detects food items via bounding boxes
- **Stage 2** (SAM) refines boxes into pixel-accurate masks (mask decoder trainable)
- **Stage 3** classifies masked crops using a specialist classifier (initially ICL-based)

The key hypothesis was joint end-to-end training across all three stages with a multi-task loss.

## Key Files
- `pipeline.py` — Full 3-stage inference pipeline
- `stage1_qwen.py` — Stage 1 VLM-based detection
- `stage2_sam.py` — Stage 2 SAM segmentation
- `stage3_icl.py` — Stage 3 in-context learning classifier (later replaced by DINOv2 k-NN in Experiment 09)
- `train_joint.py` — Joint training loop across all stages
- `losses.py` — Multi-task loss definitions
- `metrics.py` — Evaluation metrics
- `eval_harness.py` — Evaluation harness for dev/test sets
- `master_config.yaml` — Training configuration
- `dataset_integration.py` / `dataset_v3_adapter.py` — Dataset loading and preprocessing

## Outcome
Achieved Stage 1 recall@0.5 of 0.864, Stage 2 mIoU of 0.573, and Stage 3 accuracy of 0.5 on the best retained run (`trial-20260321-cleandata1`). Identified the cardinality regression problem: bbox IoU improves during training while exact count accuracy degrades.

## Relevance to Final Paper
This is the primary codebase from which the paper's pipeline evolved. The cardinality regression observation (§5) came from analyzing training curves in this experiment. The joint training approach was later simplified: SAM was fully frozen and Stage 3 was replaced with DINOv2 k-NN retrieval.
