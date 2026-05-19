# Experiment 04: Three-Stage Batch Inference Prototype

**Branch:** `3-stage-MVP`  
**Timeline:** Prototyping phase — iterative prompt and model size testing  
**Status:** Superseded; results informed final prompt design

## Approach
This experiment focused on prototyping the three-stage inference pipeline at scale. The key goal was to run batch inference across hundreds of tray images, testing different VLM sizes and prompt strategies. Nine distinct batch result sets were generated:

| Run | Description |
|-----|-------------|
| `batch_results` | Initial baseline |
| `batch_results_v2_test` / `v2_test2` | Early prompt iterations |
| `batch_results_v3_test` | Improved prompt with splitting rules |
| `batch_results_v4_test` | Stricter JSON schema |
| `batch_results_v5_test` | Tightened bbox constraints |
| `batch_results_v6_test` | Reverted to Qwen2.5-3B baseline |
| `batch_results_v7_test` | Qwen2.5-VL-7B with unique descriptions |
| `batch_results_v8_500` | Final 500-image run with 7B + optimized prompt (467 success) |

## Key Files
- `stage1_vlm.py` — VLM bounding box generation (tested 3B and 7B)
- `stage2_sam.py` — SAM3 mask refinement
- `stage3_match.py` — Early vector-matching classifier
- `vector_store.py` — Vector database for crop embeddings
- `build_index.py` — Reference bank construction
- `run_batch.py` — Batch inference runner
- `main.py` — Single-image inference entry point
- `config.py` / `ptypes.py` / `logger.py` / `visualizer.py` — Infrastructure

## Outcome
Demonstrated that **Qwen2.5-VL-7B** with the optimized prompt achieved 93.4% parse success rate (467/500 images). The 3B model was too weak; the 7B struck the right balance. Batch results v8 became the image source for V3 training data.

## Relevance to Final Paper
This experiment produced the prompt template described in §3.1 (Stage 1 Teacher VLM) and validated the model size choice (7B student). The batch_results_v8 images directly fed into the v3.2 dataset construction.
