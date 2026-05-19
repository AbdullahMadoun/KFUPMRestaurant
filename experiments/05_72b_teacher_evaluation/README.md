# Experiment 05: 72B Teacher Model Evaluation

**Branch:** `florence2-test`  
**Timeline:** Data engine development  
**Status:** Directly produced the teacher labels for the final v3.2 dataset

## Approach
Despite the branch name (`florence2-test`), this experiment's primary contribution was evaluating the **Qwen2.5-VL-72B-AWQ** teacher model in full batch mode. The 72B model was served via vLLM with AWQ-Marlin quantization and run on all 809 source images.

Key configuration:
- Model: `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` with AWQ-Marlin backend
- FP8 KV-cache + prefix caching
- Structured JSON prompt with splitting/exclusion rules

## Key Files
- `batch_results_72b_fp8/` — Full 72B inference results on 809 images
- `viewer_72b.html` — Interactive HTML viewer for inspecting 72B results (with name field support)
- `setup_server.sh` — vLLM server setup for 72B model
- `exports.txt` — Environment exports for GPU configuration
- All `stage1_vlm.py`, `stage2_sam.py`, etc. — Shared pipeline code

## Outcome
The 72B teacher achieved **100% JSON-parse success** (809/809 images) and produced the raw bounding box + name labels that were subsequently refined via SAM3 and DINOv2+SigLIP clustering into the v3.2 dataset.

## Relevance to Final Paper
This is the exact teacher pipeline described in §3 (Data Engine). The 72B results are the starting point for the entire distillation pipeline. The paper reports: "72B teacher succeeds on 4,999/4,999 images (zero JSON-parse failures) in 5,778s of wall-clock time."
