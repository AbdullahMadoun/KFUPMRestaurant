# Experiment 10: Config-Driven Pipeline Refactor

**Branch:** `refactor`  
**Timeline:** Mid project (between V2 and V3)  
**Status:** Architecture patterns carried forward into V3

## Approach
Refactored the V2 pipeline from hardcoded scripts into a config-driven architecture. Introduced:
- Externalized configuration files for all pipeline parameters
- Reproducible run scripts (`reproduce_runs.sh`)
- Modular pipeline types and structured data flow
- Multi-experiment gallery support in README

## Key Files
- `config.py` — Config-driven pipeline parameters
- `main.py` — Config-aware entry point
- `qwen_food_prompter.py` — Prompt template engine
- `sam3_segmenter.py` — SAM3 integration
- `pipeline_types.py` — Structured type definitions
- `reproduce_runs.sh` — Reproducibility script
- `REPRODUCTION_GUIDE.md` — Step-by-step reproduction instructions
- `V2_DETAILS.md` — Detailed V2 architecture documentation

## Relevance to Final Paper
The config-driven patterns developed here were carried forward into the V3 `master_config.yaml` system used in the final training pipeline.
