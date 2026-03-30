# Data Pipeline

## External Dataset Source

The active config points at an external reviewed export batch root:

- Original batch root: `/root/dataset`
- Export root default: `<batch_root>/_review/dataset`
- Annotation root default: `<batch_root>/_review/annotations`

The repository itself does not contain the full raw dataset. It expects the reviewed export contract described in `INTEGRATION_DATASET_GUIDE.md`.

## Expected Directory Structure

```text
<batch_root>/
  _review/
    annotations/
    dataset/
      images_manifest.jsonl
      stage1_item_detection.jsonl
      stage1_qwen_detection.jsonl
      stage2_sam_segmentation.jsonl
      stage3_item_classification.jsonl
      classes.json
      summary.json
```

Some `image_path` entries can be pointer files instead of raw image bytes. `dataset_integration.py` resolves those pointers against `repo_root` and `Sampled_Images_All/`.

## Preprocessing And Resolution Order

1. Resolve `batch_root`, `export_root`, `repo_root`, and annotation paths with `build_export_paths()`.
2. Load `images_manifest.jsonl`, `stage3_item_classification.jsonl`, and `classes.json` for the active joint dataset path; the Stage 1 and Stage 2 manifests remain auxiliary audit artifacts.
3. Resolve source images with pointer-file support when `original.jpg` is a UTF-8 pointer.
4. Filter to active, non-excluded, labeled items.
5. Rebuild supported train/dev/test splits through `enforce_supported_class_contract()` using the configured seed and class-frequency heuristics.
6. Remove classes/items that do not meet the minimum-support contract for episodic Stage 3 training; this improves coverage but does not guarantee every retained class appears in every split.
7. Resize and pad full images and Stage 2 masks to `cfg.data.image_size` (best run: `640`).
8. Convert Stage 3 support/query crops to `(224, 224)` tensors through `pil_images_to_tensor()` before episodic batching.

## DataLoader Settings In The Best Run

- `num_workers`: `8`
- `pin_memory`: `True`
- `persistent_workers`: enabled when `num_workers > 0`
- `prefetch_factor`: `2` when workers are enabled
- Train loader shuffle: `True`
- Train-eval/dev loader shuffle: `False`
- Collator: `JointBatchCollator`

## Custom Collation

`JointBatchCollator` prepares:

- Stage 1 prompt text and processor inputs.
- Stage 2 image tensors and prompt boxes.
- Stage 2 target masks for supervised segmentation loss.
- Stage 3 support/query episode tensors and labels.
- Episode class-count metadata used for diagnostics.
- Per-image metadata used by logging, evaluation, and report generation.

## Split Logic

`stratified_split()` constructs deterministic seed-driven train/dev/test splits and then:

- maximizes class coverage per split with class-frequency heuristics,
- strengthens training support for rare classes,
- filters unsupported classes that cannot satisfy the support contract,
- preserves removed/excluded class handling for Stage 3.

`JointFoodDataset` always rebuilds these split decisions through `enforce_supported_class_contract()`. Legacy `val` inputs normalize to `dev`.
