# Repository Map

This map groups the snapshot by function so a GitHub reader can find code,
artifacts, and operational docs quickly.

## Top-Level Directories

- `checkpoints/`
  Lightweight checkpoint tree plus restore targets for the retained best run.
- `documentation/experiments/`
  Research notes on architecture, hardware, loss design, and dataset alignment.
- `docs/`
  GitHub-facing navigation and publishing docs.
- `logs/`
  Historical run logs, config snapshots, metrics, and summaries.
- `outputs/`
  Generated run reports, all-trials comparisons, plots, and visualization JSON.
- `pictsure_library/`
  Bundled local PictSure package used by the Stage 3 path.
- `results/`
  Small retained inference result artifacts.
- `tests/`
  Lightweight validation and regression tests.
- `weights/`
  Packaged best-checkpoint tarball and provenance notes.

## Main Entry Points

- `train_joint.py`
  Main joint training loop for the three-stage system.
- `run_single_inference.py`
  Single-image inference CLI.
- `run_dev_inference.py`
  Dev-set inference workflow.
- `validate_pipeline_contracts.py`
  End-to-end validation helper for data and model contracts.
- `experiment_report.py`
  Converts logs into reports and plots.
- `benchmark_runtime.py`
  Benchmarks training-related runtime paths.

## Core Runtime Modules

- `pipeline.py`
  Connects Stage 1, Stage 2, and Stage 3 into one pipeline.
- `dataset_integration.py`
  Reviewed-export loader, split rebuilding, pointer resolution, and collator logic.
- `item_processing.py`
  Shared image and crop utilities.
- `config_loader.py`
  Loads and merges config values.
- `losses.py`
  Joint and stage-specific losses.
- `metrics.py`
  Box, mask, and classification metrics.

## Stage Implementations

- `stage1_qwen.py`
  Grounding with Qwen2.5-VL and LoRA support.
- `stage2_sam.py`
  Box-prompted segmentation with SAM.
- `stage3_icl.py`
  Few-shot style masked-crop classification.
- `pictsure_official.py`
  Adapter to public upstream PictSure checkpoints.
- `pictsure_baseline.py`
  Local baseline path retained for ablations.

## Operational Documentation

- `README.md`
  Public overview and quick start.
- `RESUME_GUIDE.md`
  Best first stop for resuming the project.
- `TRAINING_GUIDE.md`
  Training semantics and checkpoint notes.
- `DATA_PIPELINE.md`
  Expected dataset layout and pointer-image behavior.
- `EVAL_GUIDE.md`
  Evaluation expectations and report usage.
- `CONFIGS_GUIDE.md`
  Deep config reference.
- `VALIDATION_REPORT.md`
  Evidence that the packaged snapshot was checked.

## Artifact-Oriented Files

- `weights/best_checkpoint.tar`
  Packaged best checkpoint from `trial-20260321-cleandata1`.
- `weights/CHECKPOINT_PROVENANCE.md`
  Explains where the packaged checkpoint came from.
- `outputs/all_trials_report_20260321/`
  Cross-run comparison report.
- `outputs/trial-20260321-cleandata1/`
  Best-run reports and visualizations.
- `logs/trial-20260321-cleandata1/joint/`
  Best-run raw logging artifacts.

## Helpers With Provenance Value

- `restore_best_checkpoint.sh`
  Bash helper for expanding the packaged checkpoint into `checkpoints/`.
- `restore_best_checkpoint.ps1`
  PowerShell equivalent for Windows users.
- `fix_dataset.py`
  Legacy helper retained for provenance; not the primary recommended entry point.
- `bundle_deployment.py`
  Snapshot-generation helper retained for provenance.
- `create_safe_zip.py`
  Archive helper retained for provenance.
