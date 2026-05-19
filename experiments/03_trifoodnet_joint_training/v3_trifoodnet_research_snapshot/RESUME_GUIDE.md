# Resume Guide

## 1. Project Summary

TriFoodNet is a three-stage research pipeline for cafeteria-tray food understanding. It grounds food items with Qwen2.5-VL, segments them with SAM3, and classifies the resulting masked crops with a PictSure-based few-shot model. The strongest copied experiment in this snapshot is `trial-20260321-cleandata1`, selected by the same `joint/combined` checkpoint score used in the original training loop.

## 2. Repo Structure

```text
  AGENT_KICKSTART.md
  ARCHITECTURE.md
  COLD_START_GUIDE.md
  CONFIGS_GUIDE.md
  DATA_PIPELINE.md
  ENV_SNAPSHOT.txt
  EVAL_GUIDE.md
  EXPERIMENTS_INDEX.md
  EXPERIMENT_REPORTING.md
  GIT_STATE.txt
  INTEGRATION_DATASET_GUIDE.md
  MANIFEST.in
  MANIFEST.md
  PROGRESS.md
  README.md
  RESUME_GUIDE.md
  SNAPSHOT_META.json
  TRAINING_GUIDE.md
  VALIDATION_REPORT.md
  benchmark_runtime.py
  bundle_deployment.py
  check_trainable.py
  config_loader.py
  create_safe_zip.py
  data_correctness_report.json
  dataset_integration.py
  experiment_logging.py
  experiment_report.py
  fix_dataset.py
  images_zip_pointer_patch_report.json
  item_processing.py
  losses.py
  master_config.yaml
  metrics.py
  pictsure_baseline.py
  pictsure_official.py
  pipeline.py
  pictsure_library/
    LICENSE
    README.md
    pyproject.toml
    PictSure/
  post_training_artifacts.py
  program.md
  pyproject.toml
  reupload_validation_report.json
  review-findings.md
  run_dev_inference.py
  run_isolated_inference.py
  run_official_pictsure.py
  run_single_inference.py
  restore_best_checkpoint.sh
  stage1_qwen.py
  stage2_sam.py
  stage3_icl.py
  test_pictsure_alignment.py
  test_pictsure_lora.py
  train_joint.py
  train_stage3_hf.py
  validate_dataset.py
  validate_pipeline_contracts.py
  validate_results.json
  validation_report.json
  validation_report_data_only.json
  verify_split.py
  visualize_val_predictions.py
  weights/
    CHECKPOINT_PROVENANCE.md
    best_checkpoint.tar
  checkpoints/
    trial-20260321-cleandata1/
      joint/
        config_snapshot.yaml
        best/
          stage1_lora/
            README.md
            adapter_config.json
  documentation/
    experiments/
      arch_overview.md
      dataset_alignment.md
      hardware_optimization.md
      loss_functions.md
      training_metrics.md
  results/
    dev/
      inference_results.json
  ...
```

## 3. Environment Setup

```bash
conda create -n trifoodnet-snapshot python=3.12
conda activate trifoodnet-snapshot
pip install -e ./pictsure_library
pip install -e .
pip install ".[research,dev]"
```

If you want to use the upstream public adapter instead of the bundled local PictSure copy, also run `pip install ".[official-pictsure]"`.

## 4. Data Setup

This snapshot does not bundle the reviewed export dataset or `Sampled_Images_All/`. To reproduce evaluation/inference, obtain the reviewed export batch root used by the project and place it at a location of your choice, then override:

- `data.integration.batch_root`
- `data.integration.export_root` when needed
- `data.integration.repo_root` to an absolute directory that exposes `Sampled_Images_All/` for pointer-backed images

Expected structure is documented in `DATA_PIPELINE.md`. Do not rely on the default `repo_root: "."` unless you are running from the snapshot root and have already mounted the pointer-image assets relative to that working directory. The primary CLIs do not expose arbitrary config override flags, so the intended path is to edit `master_config.yaml` in place or copy it and pass that copy through `--config` where supported.

## 5. Resume Training

```bash
./restore_best_checkpoint.sh
```

This restores the winning `epoch_038` checkpoint and mirrors it into `joint/best/` for the snapshot tooling. There is no native optimizer-state resume path for the saved joint checkpoints. `train_joint.py` only warm-starts from per-stage checkpoints and the packaged joint checkpoint is weight-only. Use it for evaluation, inference, or manual continuation work after wiring those weights back into a custom restart path.

## 6. Run Evaluation

```bash
python validate_pipeline_contracts.py   --config ./master_config.yaml   --run-name trial-20260321-cleandata1   --split dev   --max-images 5   --output ./validation_report.json
```

## 7. Run Inference

```bash
python run_single_inference.py --help
```

The single-image inference path lives in `run_single_inference.py`; after `./restore_best_checkpoint.sh`, update its config overrides to point at your dataset roots and then run it from the snapshot root.

## 8. Key Files Map

- Change model architecture -> `stage1_qwen.py`, `stage2_sam.py`, `stage3_icl.py`, `pipeline.py`
- Change loss function -> `losses.py`
- Change data augmentation / loading -> `dataset_integration.py`, `item_processing.py`
- Change evaluation metrics -> `metrics.py`, `validate_pipeline_contracts.py`, `experiment_report.py`
- Change experiment logging -> `experiment_logging.py`
- Change reporting artifacts -> `post_training_artifacts.py`, `experiment_report.py`

## 9. Known Issues And TODOs

- No TODO/FIXME/HACK markers were found in kept source files.
- External data is still required; the snapshot preserves code, configs, logs, and the best checkpoint, but not the reviewed export dataset or `Sampled_Images_All/`.
- `train_joint.py` can only warm-start from weights. It does not restore optimizer, scheduler, or scaler state from the retained joint checkpoint.
- Legacy helpers such as `create_safe_zip.py`, `bundle_deployment.py`, and `fix_dataset.py` are retained for provenance but contain stale path assumptions and should not be treated as primary entry points.

## 10. Experiment History Summary

See `EXPERIMENTS_INDEX.md` for the full run table. The best retained run is `trial-20260321-cleandata1` with `joint/combined = 1.9375961198969618`.
