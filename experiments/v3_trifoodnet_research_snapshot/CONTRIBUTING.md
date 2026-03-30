# Contributing

## Scope

This repository is a research snapshot, not a polished framework release.
Changes should improve reproducibility, readability, and experiment discipline
without obscuring the retained training evidence.

## Setup

```bash
pip install -e ./pictsure_library
pip install -e ".[research,dev]"
```

Use `master_config.yaml` as the main runtime config and override dataset paths
locally before running training or validation.

## Preferred Workflow

1. Keep code changes small and tied to one purpose.
2. Update docs when behavior, paths, or checkpoint semantics change.
3. Prefer adding validation or tests when changing data contracts or metrics.
4. Keep experiment outputs out of commits unless they are intentionally curated
   artifacts under `logs/`, `outputs/`, or `weights/`.

## Testing

Run the lightweight checks that fit your change:

```bash
python validate_pipeline_contracts.py --help
python -m pytest tests/test_allocation.py
python -m pytest tests/test_dataset.py
```

Heavier training or model-loading paths require the external dataset, model
access, and a compatible GPU environment.

## Artifacts And Git Hygiene

- `weights/best_checkpoint.tar` is tracked through Git LFS.
- Restored checkpoint payloads under `checkpoints/.../epoch_*` and large files
  created in `checkpoints/.../joint/best/` are ignored on purpose.
- Do not commit tokens, machine-specific paths, or ad hoc local dataset copies.

## Documentation

The fastest navigation path is:

1. `README.md`
2. `docs/README.md`
3. `docs/REPOSITORY_MAP.md`

If a change affects onboarding, update those files first.
