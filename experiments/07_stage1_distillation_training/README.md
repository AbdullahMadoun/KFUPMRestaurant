# Experiment 07: Stage 1 Distillation Training (Cardinality-Aware Fine-Tuning)

**Branch:** `stage1-qwen-training-code`  
**Timeline:** Final training phase  
**Status:** **This is the code that produced the paper's results**

## Approach
Fine-tuning of `Qwen2.5-VL-7B-Instruct` on the v3.2 dataset using:
- **LoRA** (r=16, α=32) on all attention projections (q, k, v, o)
- **Unfrozen vision encoder** (~691M additional parameters)
- **Bbox-only JSON schema** (class-agnostic detection)
- **Cardinality-aware structural loss** penalizing |predicted_count - true_count|

Training setup:
- Hardware: Single H100 SXM 80GB at bf16
- Optimizer: AdamW, 2e-5 peak LR with cosine decay
- Effective batch size: 16 (4 × grad_accum 4)
- Epochs: 10 planned, ~25 min/epoch
- Count loss weight λ = 5.0

## Key Files
- `stage1_kcfd/` — Complete Stage 1 training module:
  - `model.py` — Model wrapper with LoRA + vision encoder unfreezing
  - `dataset.py` — v3.2 dataset loader for bbox-only training
  - `loss.py` — **Cardinality-aware structural loss** (the paper's key contribution)
  - `trainer.py` — Training loop with count-aware evaluation
  - `eval.py` — Evaluation with count-exact and multiset-match metrics
  - `schema.py` — JSON schema validation for VLM outputs
  - `bbox_audit.py` / `bbox_canonical.py` — Bounding box quality checks
  - `config.py` — Training hyperparameters
  - `qwen_io.py` — Qwen model I/O utilities
  - `probe_batch_size.py` — GPU memory probing
  - `visualize.py` — Training visualization
- `train.py` — Standalone training entry point
- `requirements-stage1.txt` — Stage 1 specific dependencies
- `scripts/vast/stage1_*.sh` — 8 deployment scripts for remote training on vast.ai
- `STAGE1_KCFD.md` — Detailed training documentation
- `STAGE1_REMOTE_RUN.md` — Remote deployment guide

## Outcome
The distilled 7B model with cardinality-aware loss achieved:
- **86.4% dish-correct** (vs 39.4% baseline)
- **90.3% count-exact** (vs 46.3% baseline)
- **0.0% zero-detection rate** (vs 37.8% baseline)
- Per-item recall: 45.4% → 90.6%

## Relevance to Final Paper
This is the most critical experiment. It directly implements:
- §4.1 (Distilled Qwen2.5-VL-7B student)
- §4.4 (Training setup — Table 3 hyperparameters)
- §5 (Cardinality-Aware Fine-Tuning — the structural count loss)
- §7 (All experimental results)
