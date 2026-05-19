# Experiment 08: Training Infrastructure & Handoff

**Branch:** `training-only`  
**Timeline:** Training deployment and infrastructure  
**Status:** Supporting infrastructure for Experiments 07 and 09

## Approach
This branch focused on the training infrastructure required to run experiments on remote GPU instances (vast.ai). Key contributions:
- Handoff documentation for resuming training across sessions
- Dataset specification and split management
- Optimized vast.ai instance defaults (5090 profile, batched eval)
- Fully frozen SAM3 + PictSure ViT backbone configuration
- Batched Stage 1 evaluation for ~5x dev eval speedup

## Key Files
- `HANDOFF.md` — Complete handoff guide for resuming training on new instances
- `DATASET.md` — Dataset specification, split definitions, and curation notes
- `master_config.yaml` — Master training configuration with all hyperparameters
- `scripts/vast/` — Deployment scripts for vast.ai instances

## Relevance to Final Paper
Documents the infrastructure behind the compute costs reported in §4.5 (Total Resource Cost) and the dataset splits described in §3.4 (Final Dataset v3.2).
