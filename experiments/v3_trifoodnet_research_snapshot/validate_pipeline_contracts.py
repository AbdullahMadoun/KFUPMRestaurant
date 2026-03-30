# =============================================================================
# FILE: validate_pipeline_contracts.py
# CATEGORY: EVAL
# PURPOSE: Deep contract validation across the dataset, split logic, and full pipeline.
# DEPENDENCIES: config_loader.py, dataset_integration.py, item_processing.py, metrics.py, pipeline.py, post_training_artifacts.py, stage1_qwen.py, stage2_sam.py, stage3_icl.py, train_joint.py
# USED BY: tests/test_pipeline_contracts.py
# KEY CLASSES/FUNCTIONS: parse_args, _dev_ratio, _dataset_kwargs, validate_class_mapping, validate_split_contract, validate_episode_contract, validate_annotation_contract, validate_stage3_support_capacity, validate_stage3_crop_contract, validate_image_source_contract, build_pipeline, _stage2_pixel_tensor
# LAST MODIFIED: 2026-03-21T20:27:54.923060+00:00
# SNAPSHOT NOTES: contains hardcoded absolute paths that must be updated for a new environment; snapshot defaults now point at local config/output paths and the retained best run
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import torch

from config_loader import load_config
from dataset_integration import (
    JointFoodDataset,
    build_class_name_index,
    build_export_paths,
    choose_item_label,
    enforce_supported_class_contract,
    filter_active_items,
    load_masked_item_image,
    resolve_image_asset,
    read_json,
    read_jsonl,
)
from item_processing import masked_crop_from_pil
from metrics import mask_iou
from post_training_artifacts import build_stage3_reference_library
from stage1_qwen import QwenGrounder
from stage2_sam import SAM3Segmenter
from stage3_icl import FoodClassifier
from pipeline import TriFoodNet
from train_joint import _build_bnb_config, _resolve_amp_dtype, _resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Rigorous data and pipeline contract validation.")
    parser.add_argument("--config", default="./master_config.yaml")
    parser.add_argument("--run-name", default="trial-20260321-cleandata1")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--disable-nms", action="store_true", default=True)
    parser.add_argument("--full-data-scan", action="store_true", default=True)
    parser.add_argument("--output", default="./validation_report.json")
    return parser.parse_args()


def _dev_ratio(cfg) -> float:
    integration = cfg.data.integration
    return float(getattr(integration, "dev_ratio", getattr(integration, "val_ratio", 0.1)))


def _dataset_kwargs(cfg, split: str, *, n_way: Optional[int] = None, k_shot: Optional[int] = None, query_per_class: int = 1):
    integration = cfg.data.integration
    return dict(
        batch_root=integration.batch_root,
        export_root=(integration.export_root or None),
        repo_root=(integration.repo_root or None),
        split=split,
        episode_support_split="train",
        image_size=cfg.data.image_size,
        train_ratio=integration.train_ratio,
        val_ratio=_dev_ratio(cfg),
        test_ratio=integration.test_ratio,
        split_seed=integration.split_seed,
        n_way=cfg.stage3.eval.n_way if n_way is None else n_way,
        k_shot=cfg.stage3.eval.k_shot if k_shot is None else k_shot,
        query_per_class=query_per_class,
    )


def validate_class_mapping(export_root: Path) -> Dict[str, object]:
    classes = read_json(export_root / "classes.json")
    stage3_rows = read_jsonl(export_root / "stage3_item_classification.jsonl")
    class_names, class_name_to_id = build_class_name_index(export_root, classes=classes, stage3_rows=stage3_rows)

    name_to_ids: Dict[str, set[int]] = {}
    for row in stage3_rows:
        class_name = choose_item_label(row)
        class_id = int(row.get("class_id", -1))
        if class_id < 0 or not class_name:
            continue
        name_to_ids.setdefault(class_name, set()).add(class_id)

    ambiguous = {name: sorted(ids) for name, ids in name_to_ids.items() if len(ids) > 1}
    missing_ids = [
        {"class_name": name, "class_id": ids[0]}
        for name, ids in sorted((name, sorted(ids)) for name, ids in name_to_ids.items())
        if ids[0] >= len(class_names) or class_names[ids[0]] != name
    ]

    if ambiguous:
        raise AssertionError(f"Ambiguous class_id mapping detected: {ambiguous}")
    if missing_ids:
        raise AssertionError(f"Class-name index mismatch detected: {missing_ids[:5]}")

    return {
        "num_stage3_classes": len(name_to_ids),
        "indexed_class_slots": len(class_names),
        "max_class_id": max(class_name_to_id.values()) if class_name_to_id else -1,
    }


def validate_split_contract(cfg) -> Dict[str, object]:
    splits = {
        name: JointFoodDataset(
            **_dataset_kwargs(cfg, name),
        )
        for name in ("train", "dev", "test")
    }

    train_labels = {
        choose_item_label(row)
        for row in splits["train"].stage3_rows
    }
    dev_labels = {
        choose_item_label(row)
        for row in splits["dev"].stage3_rows
    }
    test_labels = {
        choose_item_label(row)
        for row in splits["test"].stage3_rows
    }

    unseen_in_dev = sorted(label for label in dev_labels if label not in train_labels)
    unseen_in_test = sorted(label for label in test_labels if label not in train_labels)
    if unseen_in_dev or unseen_in_test:
        raise AssertionError(
            f"Held-out labels remain in eval splits. dev_unseen={unseen_in_dev}, test_unseen={unseen_in_test}"
        )

    removed = set(splits["train"].removed_classes)
    leaked_removed = sorted(
        {
            choose_item_label(row)
            for dataset in splits.values()
            for row in dataset.stage3_rows
            if choose_item_label(row) in removed
        }
    )
    if leaked_removed:
        raise AssertionError(f"Removed classes leaked into retained stage3 rows: {leaked_removed}")

    return {
        "train_stage3_labels": len(train_labels),
        "dev_stage3_labels": len(dev_labels),
        "test_stage3_labels": len(test_labels),
        "supported_classes": list(splits["train"].supported_classes),
        "removed_classes": list(splits["train"].removed_classes),
    }


def validate_episode_contract(cfg) -> Dict[str, object]:
    dataset = JointFoodDataset(
        **_dataset_kwargs(
            cfg,
            "train",
            n_way=cfg.stage3.episode.n_way,
            k_shot=cfg.stage3.episode.k_shot,
            query_per_class=cfg.stage3.episode.query_per_class,
        ),
    )

    query_total = 0
    missing_support = 0
    invalid_labels = 0
    support_sizes = set()

    for index in range(len(dataset)):
        example = dataset[index]
        for query_label, episode in zip(example["stage3_query_labels"], example["stage3_support_episodes"]):
            if query_label < 0:
                invalid_labels += 1
                continue
            if episode is None:
                missing_support += 1
                continue
            query_total += 1
            support_labels = set(int(label) for label in episode["support_labels"])
            support_sizes.add(len(episode["support_labels"]))
            if int(query_label) not in support_labels:
                missing_support += 1

    if missing_support:
        raise AssertionError(f"Found {missing_support} Stage 3 queries missing their class in support episodes.")

    return {
        "queries_checked": query_total,
        "invalid_query_labels": invalid_labels,
        "support_sizes": sorted(support_sizes),
    }


def validate_annotation_contract(cfg) -> Dict[str, object]:
    image_size = int(cfg.data.image_size)
    expected_support_size = int(cfg.stage3.eval.n_way) * int(cfg.stage3.eval.k_shot)
    split_reports: Dict[str, Dict[str, object]] = {}

    for split in ("train", "dev", "test"):
        dataset = JointFoodDataset(**_dataset_kwargs(cfg, split))
        split_report: Dict[str, object] = {
            "images": len(dataset),
            "items": 0,
            "avg_items_per_image": 0.0,
            "max_items_per_image": 0,
            "images_with_multiple_items": 0,
            "items_with_masks": 0,
            "blank_masked_crops": 0,
            "item_histogram": {"1": 0, "2": 0, "3+": 0},
        }
        total_items = 0

        for index in range(len(dataset)):
            sample = dataset[index]
            pil_image = sample["pil_image"]
            stage1_items = list(sample.get("stage1_items", []))
            boxes = sample.get("boxes")
            masks = list(sample.get("masks", []))
            query_labels = list(sample.get("stage3_query_labels", []))
            support_episodes = list(sample.get("stage3_support_episodes", []))

            if pil_image.size != (image_size, image_size):
                raise AssertionError(f"{split} sample {sample['image_id']} resized image is {pil_image.size}, expected {(image_size, image_size)}")

            item_count = len(stage1_items)
            if item_count <= 0:
                raise AssertionError(f"{split} sample {sample['image_id']} contains zero active labeled items after filtering.")
            total_items += item_count
            split_report["max_items_per_image"] = max(int(split_report["max_items_per_image"]), item_count)
            if item_count > 1:
                split_report["images_with_multiple_items"] = int(split_report["images_with_multiple_items"]) + 1
            if item_count <= 1:
                split_report["item_histogram"]["1"] += 1
            elif item_count == 2:
                split_report["item_histogram"]["2"] += 1
            else:
                split_report["item_histogram"]["3+"] += 1

            tensor_box_count = int(boxes.shape[0]) if isinstance(boxes, torch.Tensor) else len(boxes or [])
            expected_lengths = [item_count, tensor_box_count, len(masks), len(query_labels), len(support_episodes)]
            if len(set(expected_lengths)) != 1:
                raise AssertionError(
                    f"{split} sample {sample['image_id']} has mismatched item lengths: "
                    f"items={item_count}, boxes={tensor_box_count}, masks={len(masks)}, "
                    f"query_labels={len(query_labels)}, support_episodes={len(support_episodes)}"
                )

            for item_index, item in enumerate(stage1_items):
                label = str(item["label"])
                expected_class_id = int(dataset.class_name_to_id.get(label, -1))
                actual_class_id = int(query_labels[item_index])
                if expected_class_id < 0:
                    raise AssertionError(f"{split} sample {sample['image_id']} item {item_index} has unknown label '{label}'")
                if actual_class_id != expected_class_id:
                    raise AssertionError(
                        f"{split} sample {sample['image_id']} item {item_index} label-id mismatch: "
                        f"label={label}, expected={expected_class_id}, actual={actual_class_id}"
                    )

                box = boxes[item_index].tolist() if isinstance(boxes, torch.Tensor) else list(item["box"])
                if len(box) != 4:
                    raise AssertionError(f"{split} sample {sample['image_id']} item {item_index} has malformed box: {box}")
                x1, y1, x2, y2 = [float(v) for v in box]
                if not all(torch.isfinite(torch.tensor([x1, y1, x2, y2]))):
                    raise AssertionError(f"{split} sample {sample['image_id']} item {item_index} box has non-finite values: {box}")
                if not (0.0 <= x1 < x2 <= image_size and 0.0 <= y1 < y2 <= image_size):
                    raise AssertionError(f"{split} sample {sample['image_id']} item {item_index} box out of bounds: {box}")

                mask = masks[item_index]
                if mask is not None:
                    split_report["items_with_masks"] = int(split_report["items_with_masks"]) + 1
                    if tuple(mask.shape) != (image_size, image_size):
                        raise AssertionError(
                            f"{split} sample {sample['image_id']} item {item_index} mask shape {tuple(mask.shape)} "
                            f"!= {(image_size, image_size)}"
                        )
                    if not torch.isfinite(mask).all():
                        raise AssertionError(f"{split} sample {sample['image_id']} item {item_index} mask contains non-finite values")
                    if float(mask.sum().item()) <= 0.0:
                        raise AssertionError(f"{split} sample {sample['image_id']} item {item_index} mask is empty")
                    crop = masked_crop_from_pil(pil_image, mask, box)
                    if crop.convert("L").getbbox() is None:
                        split_report["blank_masked_crops"] = int(split_report["blank_masked_crops"]) + 1

                episode = support_episodes[item_index]
                if episode is None:
                    raise AssertionError(f"{split} sample {sample['image_id']} item {item_index} is missing a Stage 3 support episode")
                support_labels = [int(v) for v in episode["support_labels"]]
                if actual_class_id not in support_labels:
                    raise AssertionError(
                        f"{split} sample {sample['image_id']} item {item_index} support episode "
                        f"does not contain query class {actual_class_id}"
                    )
                if len(support_labels) != expected_support_size:
                    raise AssertionError(
                        f"{split} sample {sample['image_id']} item {item_index} support size {len(support_labels)} "
                        f"!= expected {expected_support_size}"
                    )

        split_report["items"] = total_items
        split_report["avg_items_per_image"] = total_items / max(len(dataset), 1)
        if int(split_report["blank_masked_crops"]) > 0:
            raise AssertionError(f"{split} contains {split_report['blank_masked_crops']} blank masked crops after resizing.")
        split_reports[split] = split_report

    return split_reports


def validate_stage3_support_capacity(cfg) -> Dict[str, object]:
    dataset = JointFoodDataset(**_dataset_kwargs(cfg, "train"))
    stage3_summary = dataset.stage3_split_summary
    train_counts = stage3_summary["train"]["class_item_counts"]
    dev_counts = stage3_summary["dev"]["class_item_counts"]
    test_counts = stage3_summary["test"]["class_item_counts"]
    k_shot = int(cfg.stage3.eval.k_shot)

    classes_missing_from_dev = sorted(label for label in train_counts if dev_counts.get(label, 0) <= 0)
    classes_missing_from_test = sorted(label for label in train_counts if test_counts.get(label, 0) <= 0)
    duplicated_support_risk = {
        label: int(count)
        for label, count in sorted(train_counts.items())
        if int(count) < k_shot
    }
    one_per_split_classes = sorted(
        label
        for label, train_count in train_counts.items()
        if int(train_count) == 1 and int(dev_counts.get(label, 0)) == 1 and int(test_counts.get(label, 0)) == 1
    )

    return {
        "train_class_item_counts": train_counts,
        "dev_class_item_counts": dev_counts,
        "test_class_item_counts": test_counts,
        "classes_missing_from_dev": classes_missing_from_dev,
        "classes_missing_from_test": classes_missing_from_test,
        "classes_with_train_support_below_k_shot": duplicated_support_risk,
        "one_item_per_split_classes": one_per_split_classes,
    }


def validate_stage3_crop_contract(cfg) -> Dict[str, object]:
    integration = cfg.data.integration
    export_paths = build_export_paths(
        integration.batch_root,
        export_root=(integration.export_root or None),
        repo_root=(integration.repo_root or None),
    )
    image_rows = read_jsonl(export_paths.export_root / "images_manifest.jsonl")
    stage3_rows = read_jsonl(export_paths.export_root / "stage3_item_classification.jsonl")
    labeled_images = [
        row for row in image_rows
        if any(item.get("classification_status") == "labeled" for item in row.get("items", []))
    ]
    contract = enforce_supported_class_contract(
        labeled_images,
        stage3_rows,
        train_ratio=integration.train_ratio,
        val_ratio=_dev_ratio(cfg),
        test_ratio=integration.test_ratio,
        seed=integration.split_seed,
    )

    blank_crops = 0
    rows_with_mask = 0
    crop_sizes = []
    for row in contract["stage3_rows"]:
        crop = load_masked_item_image(row, export_paths.batch_root, export_paths.repo_root)
        crop_sizes.append((int(crop.width), int(crop.height)))
        if row.get("mask_path"):
            rows_with_mask += 1
            if crop.convert("L").getbbox() is None:
                blank_crops += 1

    if blank_crops:
        raise AssertionError(f"Stage 3 crop contract produced {blank_crops} blank masked crops.")

    avg_width = sum(width for width, _ in crop_sizes) / max(len(crop_sizes), 1)
    avg_height = sum(height for _, height in crop_sizes) / max(len(crop_sizes), 1)
    return {
        "rows_checked": len(contract["stage3_rows"]),
        "rows_with_mask_path": rows_with_mask,
        "avg_crop_width": avg_width,
        "avg_crop_height": avg_height,
    }


def validate_image_source_contract(cfg) -> Dict[str, object]:
    integration = cfg.data.integration
    export_paths = build_export_paths(
        integration.batch_root,
        export_root=(integration.export_root or None),
        repo_root=(integration.repo_root or None),
    )
    image_rows = read_jsonl(export_paths.export_root / "images_manifest.jsonl")
    source_counts: Dict[str, int] = {}
    sample_rows: List[Dict[str, str]] = []

    for row in image_rows:
        _, metadata = resolve_image_asset(
            row["image_path"],
            export_paths.batch_root,
            repo_root=export_paths.repo_root,
            allow_visualization_fallback=True,
        )
        source_kind = metadata["source_kind"]
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
        if len(sample_rows) < 10:
            sample_rows.append(
                {
                    "image_id": str(row["image_id"]),
                    "requested_path": metadata["requested_path"],
                    "resolved_path": metadata["resolved_path"],
                    "source_kind": source_kind,
                }
            )

    return {
        "rows_checked": len(image_rows),
        "source_counts": source_counts,
        "sample_rows": sample_rows,
    }


def build_pipeline(cfg) -> TriFoodNet:
    device = _resolve_device(getattr(cfg.hardware, "device", "auto"))
    amp_dtype = _resolve_amp_dtype(cfg.hardware, device)
    bnb_config = _build_bnb_config(cfg.hardware, device)
    export_paths = build_export_paths(
        cfg.data.integration.batch_root,
        export_root=(cfg.data.integration.export_root or None),
        repo_root=(cfg.data.integration.repo_root or None),
    )
    class_records = read_json(export_paths.export_root / "classes.json")
    class_names, class_name_to_id = build_class_name_index(
        export_paths.export_root,
        classes=class_records if isinstance(class_records, list) else None,
    )

    c1, c2, c3 = cfg.stage1, cfg.stage2, cfg.stage3
    stage1 = QwenGrounder(
        model_name=c1.model_name,
        lora_r=c1.lora.r,
        lora_alpha=c1.lora.alpha,
        lora_dropout=c1.lora.dropout,
        lora_target_modules=c1.lora.target_modules,
        use_rslora=c1.lora.get("use_rslora", False),
        device=device,
        gradient_checkpointing=cfg.hardware.gradient_checkpointing,
        quantization_config=bnb_config,
    )
    stage2 = SAM3Segmenter(
        model_name=c2.model_name,
        freeze_image_encoder=c2.freeze.image_encoder,
        freeze_prompt_encoder=c2.freeze.prompt_encoder,
        device=device,
        gradient_checkpointing=cfg.hardware.gradient_checkpointing,
        quantization_config=bnb_config,
        torch_dtype=amp_dtype,
        bce_weight=float(getattr(c2.loss, "bce_weight", 1.0)),
        dice_weight=float(getattr(c2.loss, "dice_weight", 1.0)),
    )
    stage3 = FoodClassifier(
        clip_model=c3.clip_model,
        num_layers=c3.transformer.num_layers,
        num_heads=c3.transformer.num_heads,
        ff_dim=c3.transformer.ff_dim,
        dropout=c3.transformer.dropout,
        lora_cfg=getattr(c3, "lora", None),
        num_classes=max(
            int(cfg.data.num_classes),
            len(class_names),
            (max(class_name_to_id.values()) + 1) if class_name_to_id else 0,
        ),
        class_names=class_names,
        train_embedding=bool(getattr(c3, "train_embedding", True)),
        inference_n_way=int(c3.eval.n_way),
        inference_k_shot=int(c3.eval.k_shot),
    ).to(device)
    pipeline = TriFoodNet(stage1, stage2, stage3)
    pipeline.load(str(Path(cfg.paths.checkpoints) / cfg.run.name / "joint" / "best"))
    ref_images, ref_labels, _ = build_stage3_reference_library(cfg)
    pipeline.stage3.set_support_set(ref_images, ref_labels)
    return pipeline


def _stage2_pixel_tensor(pil_image, device: torch.device) -> torch.Tensor:
    img_np = np.array(pil_image)
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0).to(device)


@torch.inference_mode()
def validate_teacher_forced_stage2(
    cfg,
    split: str,
    max_images: int,
    pipeline: Optional[TriFoodNet] = None,
) -> Dict[str, object]:
    dataset = JointFoodDataset(**_dataset_kwargs(cfg, split))
    pipeline = pipeline or build_pipeline(cfg)
    device = next(pipeline.stage2.parameters()).device

    total_gt_masks = 0
    total_pred_masks = 0
    aligned_iou_sum = 0.0
    checked_images = 0
    samples = []

    for index in range(min(max_images, len(dataset))):
        sample = dataset[index]
        if sample["boxes"].numel() == 0:
            continue
        pixel_values = _stage2_pixel_tensor(sample["pil_image"], device)
        gt_boxes = sample["boxes"].to(device)
        output = pipeline.stage2.predict(
            pixel_values,
            [gt_boxes],
            nms_iou_threshold=0.0,
            score_threshold=0.0,
        )
        pred_masks = list(output["pred_masks"][0])
        gt_masks = [mask for mask in sample.get("masks", []) if mask is not None]
        if not gt_masks:
            continue

        per_image_ious = []
        for item_index, gt_mask in enumerate(gt_masks):
            total_gt_masks += 1
            if item_index < len(pred_masks):
                total_pred_masks += 1
                iou = mask_iou(pred_masks[item_index], gt_mask)
                aligned_iou_sum += iou
                per_image_ious.append(iou)

        checked_images += 1
        samples.append(
            {
                "image_id": str(sample["image_id"]),
                "gt_items": len(gt_masks),
                "pred_items": len(pred_masks),
                "aligned_mean_iou": sum(per_image_ious) / max(len(per_image_ious), 1),
            }
        )

    if max_images > 0 and checked_images == 0:
        raise AssertionError("Teacher-forced Stage 2 validation ran on zero samples.")

    return {
        "checked_images": checked_images,
        "aligned_item_mean_iou": aligned_iou_sum / max(total_gt_masks, 1),
        "pred_masks_per_gt_mask": total_pred_masks / max(total_gt_masks, 1),
        "samples": samples,
    }


@torch.inference_mode()
def validate_teacher_forced_stage3(
    cfg,
    split: str,
    max_images: int,
    pipeline: Optional[TriFoodNet] = None,
) -> Dict[str, object]:
    dataset = JointFoodDataset(**_dataset_kwargs(cfg, split))
    pipeline = pipeline or build_pipeline(cfg)
    stage3_device = next(pipeline.stage3.parameters()).device

    total_items = 0
    total_correct = 0
    samples = []

    for index in range(min(max_images, len(dataset))):
        sample = dataset[index]
        item_results = []
        for item_index, item in enumerate(sample.get("stage1_items", [])):
            mask = sample["masks"][item_index] if item_index < len(sample["masks"]) else None
            box = item["box"]
            crop = (
                masked_crop_from_pil(sample["pil_image"], mask, box)
                if mask is not None
                else sample["pil_image"].crop(tuple(int(round(v)) for v in box))
            )
            prediction = pipeline.stage3.classify(
                crop,
                device=stage3_device.type,
                top_k=1,
            )
            pred_label = prediction[0][0] if prediction else "unknown"
            correct = int(pred_label == item["label"])
            total_items += 1
            total_correct += correct
            item_results.append(
                {
                    "gt_label": item["label"],
                    "pred_label": pred_label,
                    "correct": correct,
                }
            )
        samples.append(
            {
                "image_id": str(sample["image_id"]),
                "items": item_results,
            }
        )

    if max_images > 0 and not samples:
        raise AssertionError("Teacher-forced Stage 3 validation ran on zero samples.")

    return {
        "checked_images": len(samples),
        "checked_items": total_items,
        "top1_acc": total_correct / max(total_items, 1),
        "samples": samples,
    }


@torch.inference_mode()
def validate_runtime(
    cfg,
    split: str,
    max_images: int,
    disable_nms: bool,
    pipeline: Optional[TriFoodNet] = None,
) -> Dict[str, object]:
    dataset = JointFoodDataset(**_dataset_kwargs(cfg, split))
    pipeline = pipeline or build_pipeline(cfg)
    nms_iou = 0.0 if disable_nms else float(cfg.stage2.nms.iou_threshold)
    score_threshold = 0.0 if disable_nms else float(cfg.stage2.nms.score_threshold)

    samples = []
    total_masks_present = 0
    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_correct = 0
    total_mask_iou = 0.0
    for index in range(min(max_images, len(dataset))):
        sample = dataset[index]
        output = pipeline.run(
            pil_image=sample["pil_image"],
            image_id=str(sample["image_id"]),
            nms_iou_threshold=nms_iou,
            score_threshold=score_threshold,
            top_k_classes=1,
        )
        pred_masks_present = sum(1 for item in output.items if item.mask is not None)
        total_masks_present += pred_masks_present
        pred_boxes = [item.box for item in output.items]
        gt_items = list(sample.get("stage1_items", []))
        gt_boxes = [item["box"] for item in gt_items]
        matches = []
        if pred_boxes or gt_boxes:
            from metrics import greedy_box_matches
            matches = greedy_box_matches(pred_boxes, gt_boxes, threshold=float(cfg.stage1.eval.iou_threshold))
        total_gt += len(gt_items)
        total_pred += len(output.items)
        total_matches += len(matches)
        for match in matches:
            pred_item = output.items[match.pred_index]
            gt_item = gt_items[match.gt_index]
            gt_mask = sample["masks"][match.gt_index] if match.gt_index < len(sample["masks"]) else None
            if pred_item.label == gt_item["label"]:
                total_correct += 1
            if pred_item.mask is not None and gt_mask is not None:
                total_mask_iou += mask_iou(pred_item.mask, gt_mask)
        samples.append(
            {
                "image_id": str(sample["image_id"]),
                "pred_items": len(output.items),
                "gt_items": len(gt_items),
                "matched_items": len(matches),
                "pred_masks_present": pred_masks_present,
                "pred_labels": [item.label for item in output.items],
                "gt_labels": [item["label"] for item in gt_items],
            }
        )

    if max_images > 0 and not samples:
        raise AssertionError("Runtime validation ran on zero samples.")

    return {
        "checked_images": len(samples),
        "total_masks_present": total_masks_present,
        "stage1_recall@0.5": total_matches / max(total_gt, 1),
        "stage1_precision@0.5": total_matches / max(total_pred, 1),
        "stage2_mIoU": total_mask_iou / max(total_gt, 1),
        "stage3_acc": total_correct / max(total_gt, 1),
        "stage3_matched_acc": total_correct / max(total_matches, 1),
        "samples": samples,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=[f"run.name={args.run_name}"])
    runtime_pipeline = build_pipeline(cfg) if args.max_images > 0 else None
    report = {
        "run_name": args.run_name,
        "class_mapping": validate_class_mapping(
            build_export_paths(
                cfg.data.integration.batch_root,
                export_root=(cfg.data.integration.export_root or None),
                repo_root=(cfg.data.integration.repo_root or None),
            ).export_root
        ),
        "split_contract": validate_split_contract(cfg),
        "episode_contract": validate_episode_contract(cfg),
        "annotation_contract": validate_annotation_contract(cfg) if args.full_data_scan else None,
        "stage3_support_capacity": validate_stage3_support_capacity(cfg) if args.full_data_scan else None,
        "stage3_crop_contract": validate_stage3_crop_contract(cfg) if args.full_data_scan else None,
        "image_source_contract": validate_image_source_contract(cfg) if args.full_data_scan else None,
        "runtime": (
            validate_runtime(
                cfg,
                split=args.split,
                max_images=args.max_images,
                disable_nms=args.disable_nms,
                pipeline=runtime_pipeline,
            )
            if args.max_images > 0
            else None
        ),
        "teacher_forced_stage2": (
            validate_teacher_forced_stage2(
                cfg,
                split=args.split,
                max_images=args.max_images,
                pipeline=runtime_pipeline,
            )
            if args.max_images > 0
            else None
        ),
        "teacher_forced_stage3": (
            validate_teacher_forced_stage3(
                cfg,
                split=args.split,
                max_images=args.max_images,
                pipeline=runtime_pipeline,
            )
            if args.max_images > 0
            else None
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
