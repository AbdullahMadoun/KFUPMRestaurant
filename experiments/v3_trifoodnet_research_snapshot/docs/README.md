# Documentation Hub

This repo keeps most source files at the root, plus retained experiment
artifacts in `logs/`, `outputs/`, and `weights/`. Use this page as the entry
point instead of scanning the root directory manually.

## Start Here

- `README.md`: public project overview and quick start
- `BATCH8_DATASET_NOTE.md`: source and sample note for the `batch_results_v8_500` package
- `docs/FACULTY_REVIEW_GUIDE.md`: review path for a supervisor or committee member
- `RESUME_GUIDE.md`: best starting point for resuming work
- `TRAINING_GUIDE.md`: training path and checkpoint semantics
- `DATA_PIPELINE.md`: dataset contract and pointer-image behavior
- `EVAL_GUIDE.md`: evaluation and report interpretation

## Repository Navigation

- `docs/REPOSITORY_MAP.md`: grouped file and directory map
- `docs/GITHUB_PUBLISHING.md`: first-push checklist, Git LFS, and artifact policy
- `ARCHITECTURE.md`: architecture summary
- `CONFIGS_GUIDE.md`: detailed configuration reference

## Code Entry Points

- `train_joint.py`: joint training
- `run_single_inference.py`: single-image inference
- `run_dev_inference.py`: dev inference workflow
- `validate_pipeline_contracts.py`: strongest validation entry point
- `experiment_report.py`: report generation
- `benchmark_runtime.py`: runtime benchmarking

## Core Modules

- `pipeline.py`: end-to-end pipeline orchestration
- `dataset_integration.py`: manifest-driven dataset loading
- `stage1_qwen.py`: grounding stage
- `stage2_sam.py`: segmentation stage
- `stage3_icl.py`: classification stage
- `losses.py` and `metrics.py`: optimization and evaluation logic

## Artifacts

- `logs/`: retained training logs per run
- `outputs/`: markdown, CSV, JSON, SVG, and visualization outputs
- `weights/`: checkpoint provenance for the strongest retained run
- `checkpoints/`: lightweight checkpoint metadata plus restore targets

## Historical Notes

- `PROGRESS.md`: longer project chronology
- `EXPERIMENTS_INDEX.md`: run inventory and best-run summary
- `VALIDATION_REPORT.md`: packaged validation evidence
- `review-findings.md`: prior review notes

## External Dependencies

This snapshot does not include the reviewed export dataset or
`Sampled_Images_All/`. Read `RESUME_GUIDE.md` and `DATA_PIPELINE.md` before
trying to train or validate against real data.
