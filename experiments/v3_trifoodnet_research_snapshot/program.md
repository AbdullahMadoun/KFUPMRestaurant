# TriFoodNet Research Program

This file is the standing research brief for agents working on this repository.
The objective is to improve the project into a publishable, reproducible food
classification and pricing system while keeping the codebase easy to modify.

## Mission

Build a clean research stack around three modes:

1. Official pretrained PictSure baseline from Hugging Face.
2. Local CLIP retrieval baseline for ablations.
3. TriFoodNet three-stage pipeline for trainable experiments.

## Rules

1. Preserve the official PictSure baseline as a fixed external reference.
2. Do not commit secrets, tokens, or local paths.
3. Prefer small, reviewable changes over large rewrites.
4. Every model-facing change should either improve reproducibility,
   measurement, or baseline quality.
5. If an experiment changes the evaluation protocol, document that explicitly.

## Priorities

### Priority 0: Research hygiene

- Keep configs loadable.
- Keep imports stable.
- Keep docs aligned with the actual snapshot.
- Make baseline scripts runnable on CPU before optimizing for GPU.

### Priority 1: Baselines

- Official PictSure inference path must stay simple and documented.
- Local `pictsure_baseline.py` remains available for apples-to-apples ablations.
- Report top-1 accuracy, top-k accuracy, latency, and onboarding cost for new classes.

### Priority 2: Reproducibility

- Every experiment should have:
  - a config snapshot
  - a result summary
  - a short rationale
  - an explicit dataset split

### Priority 3: Trainable improvements

- Improve Stage 3 first because it is the cleanest comparison surface.
- Only expand joint training after data loaders, metrics, and checkpointing are stable.
- Avoid changing Stage 1, Stage 2, and Stage 3 simultaneously.

## Suggested Experiment Loop

1. Start from the official PictSure baseline.
2. Evaluate on a fixed held-out split and record metrics.
3. Compare against the local Stage 3 ICL classifier.
4. Isolate one hypothesis at a time:
   - reference sampling
   - prototype aggregation
   - cross-attention depth
   - augmentation
   - calibration
5. Keep the best result only if the metric gain is real and reproducible.

## Deliverables

The repository should converge toward:

- a documented baseline workflow
- one-command evaluation scripts
- explicit experiment records
- publishable tables and plots
- generated multi-run experiment reports
- a stable package layout
