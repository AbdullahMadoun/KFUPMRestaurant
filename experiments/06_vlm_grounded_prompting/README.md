# Experiment 06: VLM Grounded Prompting

**Branch:** `experiment/v3-grounded-prompting`  
**Timeline:** Prompt engineering research  
**Status:** Results incorporated into final prompt design

## Approach
Systematic exploration of VLM grounding strategies — specifically, whether providing both text descriptions AND bounding box coordinates in the prompt improves detection quality compared to text-only or bbox-only prompting.

The experiment tested multiple prompt variants:
- Text-only prompting (just describe what you see)
- Bbox-only prompting (just emit coordinates)
- Grounded prompting (text + bbox jointly)
- Comparison prompts with different levels of specificity

## Key Files
- `test_grounded.py` / `test_grounded_images.py` — Grounded prompting evaluation
- `test_vlm_grounding.py` / `test_vlm_grounding_batch.py` — Batch grounding tests
- `test_prompt_comparison.py` — Head-to-head prompt variant comparison
- `test_prompt_v3.py` — V3 prompt iteration
- `grounding_outputs/` — Saved outputs from each prompt strategy
- `qwen_food_prompter.py` — Prompt template engine
- `sam3_segmenter.py` — SAM3 integration

## Outcome
Grounded prompting (text+bbox) produced marginally better localization but the bbox-only schema ultimately won because it simplified the supervision target and decoupled detection from classification — a key architectural insight.

## Relevance to Final Paper
This experiment directly informed the design decision described in §4.1: "We deliberately strip class names and descriptions from the supervision target. This forces Stage 1 to specialise in what VLMs do well — locating cohesive food regions."
