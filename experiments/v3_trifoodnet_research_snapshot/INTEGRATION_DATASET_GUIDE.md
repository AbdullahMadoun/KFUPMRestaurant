# Integration Dataset Guide

This document describes the exported review dataset as an integration contract for training, evaluation, and agent pipelines.

The goal is to let downstream systems consume the reviewed data without needing to understand the review app internals.

## Canonical Roots

Batch root:

`experiments/v3_3stage_mvp/batch_results_v8_500/`

Review root:

`experiments/v3_3stage_mvp/batch_results_v8_500/_review/`

Export root:

`experiments/v3_3stage_mvp/batch_results_v8_500/_review/dataset/`

Per-image annotations:

`experiments/v3_3stage_mvp/batch_results_v8_500/_review/annotations/`

## What Downstream Systems Should Read

Preferred integration surface:

- `dataset/images_manifest.jsonl`
- `dataset/stage1_item_detection.jsonl`
- `dataset/stage1_qwen_detection.jsonl`
- `dataset/stage2_sam_segmentation.jsonl`
- `dataset/stage3_item_classification.jsonl`
- `dataset/classes.json`
- `dataset/summary.json`

These files are the exported contract.

Downstream systems should not rely on:

- `review.sqlite3`
- internal autosave behavior
- review UI field layout
- direct interpretation of temporary frontend state

## Export Inclusion Rule

Only images explicitly marked `use_for_export: true` are included in the dataset manifests.

That means:

- `images_manifest.jsonl` contains only export-selected images
- stage manifests contain only rows from export-selected images
- annotation JSON files still exist per image, but the manifests are the actual training/eval source

If a training run should use only the approved export set, read the manifest files, not the full annotations directory.

## File Inventory

### `dataset/images_manifest.jsonl`

One JSON object per export-selected image.

Use this when:

- you want one record per image
- you want to build a custom loader
- you want all reviewed item metadata grouped under one image

Each row includes:

- image identity
- source image path
- image width and height
- image-level review status
- image-level notes
- `use_for_export`
- a full `items` array

This is the highest-level integration file.

### `dataset/stage1_item_detection.jsonl`

One JSON object per active, non-excluded item that belongs to an export-selected image and has a reviewed `bbox`.

Use this for:

- box detection
- grounding
- region proposal training

Primary target:

- `bbox`

### `dataset/stage1_qwen_detection.jsonl`

This is a compatibility duplicate of `stage1_item_detection.jsonl`.

Use it only if an older pipeline expects that filename.

### `dataset/stage2_sam_segmentation.jsonl`

One JSON object per active, non-excluded item that belongs to an export-selected image and has a non-empty reviewed mask.

Use this for:

- segmentation training
- segmentation evaluation
- prompt-box plus mask workflows

Primary fields:

- `bbox`
- `mask_path`
- `crop_path`

### `dataset/stage3_item_classification.jsonl`

One JSON object per active, non-excluded item that belongs to an export-selected image and has a final class label.

Use this for:

- crop classification
- retrieval-style item recognition
- masked item recognition

Primary fields:

- `crop_path`
- `final_class`
- `class_id`

### `dataset/classes.json`

The canonical exported class library.

Each class entry includes:

- `name`
- `source`
- `usage_count`

Use this if a downstream system needs:

- a stable label vocabulary
- class-to-index mapping
- monitoring of class coverage

### `dataset/summary.json`

High-level counts for the current export set.

This is useful for:

- dataset sanity checks
- CI validation
- training job metadata

Important fields:

- `available_images`
- `selected_images`
- `total_images`
- `stage1_items`
- `stage2_items`
- `stage3_items`
- `classes`

`total_images` is the number of images actually exported.

## Per-Image Annotation Files

Each processed image has:

`_review/annotations/<image_id>.json`

These files are useful for:

- inspection
- debugging
- manual audit
- image-by-image downstream tooling

They are not the preferred entry point for bulk training.

Use them when:

- you already know a specific `image_id`
- you want complete item lineage for one image
- you want to compare active, merged, and excluded items

## Path Resolution Rules

Most exported paths are relative paths, not absolute paths.

Resolve them against the batch root:

`experiments/v3_3stage_mvp/batch_results_v8_500/`

Examples:

- `image_path = images/<image_id>/original.jpg`
- `mask_path = _review/masks/<image_id>/item_000_mask.png`
- `crop_path = _review/crops/<image_id>/item_000.png`

### Important Source Image Detail

`image_path` may point to `original.jpg`, and in this dataset that file can be a pointer file rather than raw image bytes.

Integration code should support this rule:

1. Try to open the file as an image.
2. If it is not a valid image, read it as UTF-8 text.
3. Treat the text as a pointer to the real source image.
4. If the pointer includes `Sampled_Images_All/`, resolve it relative to the repo root.
5. Otherwise, fall back to matching the filename inside `Sampled_Images_All/`.

This matches the review app behavior and avoids failures when the source image is stored indirectly.

## Common Row Fields

The stage manifest rows reuse a common set of fields.

Most important fields:

- `image_id`: unique image key
- `image_path`: relative path to the original image entry
- `image_width`, `image_height`: source image dimensions
- `item_index`: item id within the image
- `source_item_indices`: lineage after merges
- `bbox`: canonical reviewed training box
- `qwen_bbox`: helper box kept for audit/debug compatibility
- `sam_bbox`: reviewed mask-derived box
- `mask_path`: relative path to the reviewed full-image binary mask
- `crop_path`: relative path to the reviewed RGBA crop
- `final_class`: final manual class label
- `class_id`: exported class index
- `qwen_status`, `sam_status`, `classification_status`: review metadata
- `review_status`: image-level review status
- `use_for_export`: export selection flag
- `notes`: item-level notes

## Meaning Of Item State

### `active`

If `active` is `true`, the item is a trainable candidate.

If `active` is `false`, the item was merged into another item and should not be trained independently.

### `excluded`

If `excluded` is `true`, the item is intentionally removed from stage manifests.

Use cases:

- false positives
- non-food detections
- duplicate fragments
- unwanted utensils or plate artifacts

### `source_item_indices`

This preserves lineage when items are merged.

Use it if:

- you need auditability
- you want to trace merged detections back to source segments
- you want to debug label provenance

## Training Interpretation By Stage

### Stage 1

Read:

- `dataset/stage1_item_detection.jsonl`

Target:

- `bbox`

Ignore:

- original Qwen text

### Stage 2

Read:

- `dataset/stage2_sam_segmentation.jsonl`

Target:

- `mask_path`

Prompt or region box:

- `bbox`

### Stage 3

Read:

- `dataset/stage3_item_classification.jsonl`

Input:

- `crop_path`

Target:

- `final_class`

Optional extra signal:

- `mask_path`

## Recommended Integration Strategy

### For training jobs

Use the stage manifests directly.

That gives you:

- a fixed row schema
- only export-approved images
- only active and non-excluded items
- already materialized masks and crops

### For analytics and QA

Use:

- `dataset/summary.json`
- `dataset/classes.json`
- `dataset/images_manifest.jsonl`

### For image-level tooling

Use:

- `_review/annotations/<image_id>.json`

## Stability Expectations

The review app rewrites the exported manifests after successful saves.

That means:

- the export root is the latest dataset state
- counts and rows can change while review is ongoing
- long-running training jobs should snapshot or copy the export set before starting

If reproducibility matters, freeze:

- the manifest files
- the masks
- the crops
- the classes file
- the summary file

## Minimal Loader Contract

If an agent only needs one rule, use this:

- read rows from the stage manifest that matches the task
- resolve all relative paths against the batch root
- treat `bbox` as the canonical training box
- ignore inactive and excluded items unless doing audit/debug work
- use only export-selected images, which is already enforced by the manifests

## Practical Mapping

If your downstream component needs this kind of input, use this file:

- object detection or grounding: `stage1_item_detection.jsonl`
- segmentation: `stage2_sam_segmentation.jsonl`
- crop classification: `stage3_item_classification.jsonl`
- custom multi-task pipeline: `images_manifest.jsonl`
- vocabulary sync: `classes.json`
- dataset health check: `summary.json`

## What Not To Assume

Do not assume:

- every processed image is exported
- `image_path` is always a raw image file
- `qwen_bbox` is the training target
- merged source items should still be trained
- excluded items should appear in stage manifests

The safest contract is the exported manifest set plus the path resolution rules above.
