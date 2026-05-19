# Experiment 02: SAM3 + Qwen-VL Detection

**Branch:** `main` (V2 content)  
**Timeline:** Mid project  
**Status:** Superseded by V3

## Approach
Replaced FoodSAM with **SAM3** for segmentation and introduced **Qwen-VL** as the vision-language model for initial food detection. This was the first integration of a VLM into the pipeline.

## Key Files
- `main.py` — Entry point for V2 inference
- `qwen_food_prompter.py` — Prompt engineering for Qwen-VL food detection
- `sam3_segmenter.py` — SAM3 mask generation from VLM-emitted bounding boxes
- `config.py` — Pipeline configuration
- `pipeline_types.py` — Type definitions for inter-stage data flow
- `visualizer.py` — Result visualization

## Outcome
Proved that Qwen-VL could reliably detect food items and emit bounding boxes, and that SAM3 could refine those boxes into high-quality masks. This became the core architectural pattern (VLM → SAM → classifier) used in all subsequent work.

## Relevance to Final Paper
Directly established the three-stage decomposition described in §3 (Data Engine) and §4 (Student Pipeline). The V2 pipeline is the ancestor of the final TahamajaNet architecture.
