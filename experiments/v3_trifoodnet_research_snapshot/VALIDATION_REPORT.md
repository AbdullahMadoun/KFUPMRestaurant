# Validation Report

## Check 1 — Import Coverage
PASS — imported 28 import-safe modules from inside the snapshot.
Skipped side-effectful executable modules (covered by source docs and, where applicable, entry-point checks):
- `check_trainable.py`
- `run_dev_inference.py`
- `run_single_inference.py`
- `tests/test_item_processing.py`
- `tests/test_pipeline_contracts.py`
- `tests/test_sam3_allocation.py`
- `tests/test_stage2_sam.py`
- `verify_split.py`
- `visualize_val_predictions.py`

## Check 2 — No Orphan References
PASS — no unresolved literal file references were detected.

## Check 3 — Entry Point Dry Run
- `benchmark_runtime.py`: PASS — usage: benchmark_runtime.py [-h] {stage3-train,joint-train} ...
- `experiment_report.py`: PASS — usage: experiment_report.py [-h] [--logs-root LOGS_ROOT] [--run-dir RUN_DIR]
- `run_official_pictsure.py`: PASS — usage: run_official_pictsure.py [-h] --image IMAGE --reference-library
- `run_single_inference.py`: PASS — usage: run_single_inference.py [-h] --image_id IMAGE_ID
- `validate_pipeline_contracts.py`: PASS — usage: validate_pipeline_contracts.py [-h] [--config CONFIG]

## Check 4 — Best Checkpoint Integrity
PASS — `best_checkpoint.tar` SHA256 matches `c57a883a4fe3e760abe42a784cd17dafc958fa557505ee5afbdad58ca7185b40`.

## Check 5 — Documentation Coverage Score
PASS — 37/37 kept Python source files contain the snapshot header block (100.00%).

## Check 6 — Config Completeness
PASS — all 87 discovered config references map to at least one key in the copied configs.

## Agent Reports

### agent-arch
Status: PASS — `ARCHITECTURE.md` now covers all snapshot `ARCH` files, the Mermaid/data flow matches the active code path, and the packaged checkpoint parameter counts still confirm.

- No discrepancies found in the recheck.
- `ARCHITECTURE.md` now documents all six `ARCH` files: `stage1_qwen.py`, `stage2_sam.py`, `stage3_icl.py`, `pipeline.py`, `pictsure_official.py`, and `pictsure_baseline.py`.
- The Stage 3 support/reference input and Stage 1 `box`-only guarantee are now documented correctly.
- Packaged checkpoint counts re-confirmed from `weights/best_checkpoint.tar`: Stage 1 `7,372,800`, Stage 2 `2,298,881`, Stage 3 `129,531,654`, total `139,203,335`.

Confidence: High.

### agent-train
Status: PASS — the training docs now correctly distinguish the live `joint/best` alias from the packaged `epoch_038` winner, and the best-run hyperparameters still match the copied config artifacts.

- `master_config.yaml` now defaults to `trial-20260321-cleandata1` and matches the documented best-run setup.
- `TRAINING_GUIDE.md` now aligns with `train_joint.py`: the trainer tracks `joint/combined` separately, while the live `joint/best` save path is driven by the configured early-stop monitor (`dev/loss_total` for this run).
- `restore_best_checkpoint.sh` is consistent with the snapshot tooling: it restores the packaged `epoch_038` payload and mirrors it into `checkpoints/trial-20260321-cleandata1/joint/best/`.
- Joint checkpoints remain weight-only warm starts rather than true optimizer-state resumes.

Confidence: High.

### agent-data
Status: PASS with note — `DATA_PIPELINE.md` now matches the active joint dataset path, seed-driven split rebuilding, `(224, 224)` Stage 3 crop tensorization, and the full collator outputs.

- Active data paths align with config and loader construction: `batch_root=/root/dataset`, reviewed export roots under `_review`, and `repo_root="."` for pointer resolution.
- The preprocessing order now accurately reflects the active joint dataset path: `images_manifest.jsonl`, `stage3_item_classification.jsonl`, and `classes.json` drive the joint dataset flow.
- The split section now matches implementation: `enforce_supported_class_contract()` rebuilds splits with the configured seed, `stratified_split()` is seed-driven, and legacy `val` inputs normalize to `dev`.
- `JointBatchCollator` documentation now includes `masks` and `episode_class_counts`.
- Note: `validate_dataset.py` remains a legacy helper with a hard-coded Windows batch root, but the active data-pipeline docs no longer misstate the runtime path.

Confidence: High.

### agent-eval
Status: PASS — `EVAL_GUIDE.md` and `EXPERIMENTS_INDEX.md` now align with the reporting code, copied logs, and checkpoint provenance.

- The documented metric set is consistent with `metrics.py`, `experiment_report.py`, and the copied run artifacts.
- The corrected best-metric table in `EVAL_GUIDE.md` now matches `outputs/all_trials_report_20260321/summary.json` and `summary.csv`, including `trial-20260321-cleandata1 min_dev_loss_total = 3.1939461330572763` and `trial-20260321-full3 min_dev_loss_total = 4.562163972854615`.
- `EVAL_GUIDE.md` now uses `./restore_best_checkpoint.sh`, and that helper restores the packaged winner into both `epoch_038` and the snapshot-local `joint/best` path expected by validation/inference code.
- `EXPERIMENTS_INDEX.md`, `CHECKPOINT_PROVENANCE.md`, and `weights/best_checkpoint.tar` all agree on the winner: `trial-20260321-cleandata1`, `joint/combined = 1.9375961198969618`, checkpoint `epoch_038`, SHA256 `c57a883a4fe3e760abe42a784cd17dafc958fa557505ee5afbdad58ca7185b40`.

Confidence: High.

### agent-util
Status: PASS — UTIL/INFRA is aligned enough for cold-start use from the unpacked snapshot on disk, with only minor legacy-helper debt remaining.

- `stage3_icl.py` now resolves the snapshot-local `./pictsure_library`, so Stage 3 no longer relies on the external editable `PictSure` install captured in `ENV_SNAPSHOT.txt`.
- The bundled `pictsure_library` metadata is internally cleaner: its `pyproject.toml` no longer declares the missing CLI entry point, and `MANIFEST.in` now includes both `restore_best_checkpoint.sh` and the full `pictsure_library` tree.
- `pyproject.toml` and `ENV_SNAPSHOT.txt` are broadly compatible with the recorded environment; `torchvision` is now declared and the `research` extra now covers `bitsandbytes`, `peft`, `psutil`, and `wandb`.
- Remaining legacy/helper debt is documented rather than blocking: `bundle_deployment.py` and `create_safe_zip.py` are still unused Windows-specific packaging helpers, and `benchmark_runtime.py` remains marked stale against the current collator signature.

Confidence: High.

### agent-test
Status: PASS — the latest on-disk snapshot state passes the full discovered pytest surface.

- Re-verified the updated files on disk before finalizing: `tests/test_pipeline_contracts.py` now uses `trial-20260321-cleandata1`, `tests/test_dataset.py` now resolves batch root from snapshot config or `TRIFOODNET_BATCH_ROOT`, and `run_single_inference.py` only enables 4-bit when configured on CUDA.
- Commands run:
  - `pytest --collect-only -q`
  - `pytest tests -q`
  - `pytest test_pictsure_lora.py test_pictsure_alignment.py -q`
  - `pytest -q`
- Latest full-suite result from the snapshot root: `pytest -q` -> `15 passed, 19 warnings in 184.75s (0:03:04)`.
- Warnings are non-fatal and come from upstream/runtime layers (`DeprecationWarning` and `UserWarning` from upstream PictSure/Torch internals).

Confidence: High.
Changed Files: None.

### agent-resume
Status: PASS with documented prerequisite — a cold engineer can follow the snapshot’s documented resume/eval path as-is once the reviewed export dataset and any needed `Sampled_Images_All/` pointer assets are supplied.

- The primary resume path is internally consistent: `RESUME_GUIDE.md`, `restore_best_checkpoint.sh`, `master_config.yaml`, `validate_pipeline_contracts.py`, and `run_single_inference.py` all align on `trial-20260321-cleandata1` and `checkpoints/<run>/joint/best`.
- Snapshot-local PictSure recovery is now in place: `stage3_icl.py` falls back to the bundled `pictsure_library`, and `RESUME_GUIDE.md` documents installing it first.
- The previous inference/test rough edges are cleaned up: `run_single_inference.py` only enables 4-bit when configured and on CUDA, `tests/test_pipeline_contracts.py` now uses `trial-20260321-cleandata1`, and `tests/test_dataset.py` now follows snapshot config or `TRIFOODNET_BATCH_ROOT`.
- No snapshot-internal broken step remains in the documented restore/eval flow. The one explicit external prerequisite is the reviewed export dataset plus any required `Sampled_Images_All/` pointer assets, which the guide now calls out directly.
- “Resume training” remains weight restore rather than true optimizer-state continuation, and that limitation is documented rather than ambiguous.

Confidence: High.

SNAPSHOT STATUS: READY ✅ | Coverage: 100.00% | Issues: 0
