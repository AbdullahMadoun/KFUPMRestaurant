# =============================================================================
# FILE: post_training_artifacts.py
# CATEGORY: UTIL
# PURPOSE: Split summaries, visualizations, and report generation after training.
# DEPENDENCIES: dataset_integration.py, experiment_report.py, metrics.py
# USED BY: train_joint.py, validate_pipeline_contracts.py
# KEY CLASSES/FUNCTIONS: _dev_ratio, build_stage3_reference_library, generate_split_summary, _mask_to_pil, _draw_label, _visualize_prediction, _visualize_ground_truth, _compose_visual_pair, _compose_prediction_only, _save_contact_sheet, generate_split_visualizations, generate_training_report
# LAST MODIFIED: 2026-03-21T21:21:10.758024+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from dataset_integration import (
    JointFoodDataset,
    build_class_name_index,
    build_export_paths,
    load_masked_item_image,
    normalize_split_name,
    read_json,
)
from dataset_v3_adapter import V3ExportAdapter, adapter_from_config
from experiment_report import generate_report
from metrics import greedy_box_matches


def _build_dataset_from_cfg(cfg, *, split: str, episode_support_split: str, n_way: int,
                             k_shot: int, query_per_class: int) -> JointFoodDataset:
    """Construct a JointFoodDataset honoring data.integration.adapter when set.

    All three callers in this module share the same dataset args, so we keep a
    single helper to avoid drift between train_joint.py and the report scripts.
    """
    integration = cfg.data.integration
    adapter_cfg = getattr(integration, "adapter", None)
    use_adapter = bool(adapter_cfg is not None and getattr(adapter_cfg, "kind", ""))
    adapter = adapter_from_config(integration) if use_adapter else None
    return JointFoodDataset(
        batch_root=(None if use_adapter else integration.batch_root),
        export_root=(None if use_adapter else (integration.export_root or None)),
        repo_root=(integration.repo_root or None),
        split=split,
        episode_support_split=episode_support_split,
        image_size=cfg.data.image_size,
        train_ratio=integration.train_ratio,
        val_ratio=_dev_ratio(integration),
        test_ratio=integration.test_ratio,
        split_seed=integration.split_seed,
        n_way=n_way,
        k_shot=k_shot,
        query_per_class=query_per_class,
        adapter=adapter,
    )


PALETTE = [
    (239, 68, 68, 100),
    (59, 130, 246, 100),
    (34, 197, 94, 100),
    (245, 158, 11, 100),
    (168, 85, 247, 100),
    (236, 72, 153, 100),
    (20, 184, 166, 100),
    (244, 114, 182, 100),
]


def _dev_ratio(integration_cfg) -> float:
    return float(getattr(integration_cfg, "dev_ratio", getattr(integration_cfg, "val_ratio", 0.1)))


def build_stage3_reference_library(cfg) -> tuple[List[Image.Image], List[int], Dict[str, object]]:
    integration = cfg.data.integration
    dataset = _build_dataset_from_cfg(
        cfg,
        split="train",
        episode_support_split="train",
        n_way=cfg.stage3.eval.n_way,
        k_shot=cfg.stage3.eval.k_shot,
        query_per_class=1,
    )

    # Adapter mode emits the legacy class records in-memory on the dataset; in
    # legacy mode we still read classes.json from disk.
    class_records = list(getattr(dataset, "classes", None) or [])
    if not class_records:
        class_records = read_json(dataset.paths.export_root / "classes.json")
    class_names, _ = build_class_name_index(
        dataset.paths.export_root,
        classes=class_records if isinstance(class_records, list) else None,
        stage3_rows=dataset.stage3_rows,
    )
    min_images = int(getattr(cfg.stage3.reference_library, "min_images_per_class", 1))
    max_images = int(getattr(cfg.stage3.reference_library, "max_images_per_class", 1))

    support_images: List[Image.Image] = []
    support_labels: List[int] = []
    counts_by_class: Dict[int, int] = {}

    for row in dataset.stage3_rows:
        class_id = int(row["class_id"])
        if counts_by_class.get(class_id, 0) >= max_images:
            continue
        support_images.append(load_masked_item_image(row, dataset.paths.batch_root, dataset.paths.repo_root))
        support_labels.append(class_id)
        counts_by_class[class_id] = counts_by_class.get(class_id, 0) + 1

    available_class_ids = sorted({int(row["class_id"]) for row in dataset.stage3_rows})
    expected_class_ids = sorted(getattr(dataset, "supported_class_ids", available_class_ids))
    insufficient_classes = [
        {"class_id": class_id, "class_name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"}
        for class_id in expected_class_ids
        if counts_by_class.get(class_id, 0) < min_images
    ]
    if insufficient_classes:
        raise RuntimeError(
            "Training split does not provide the minimum number of Stage 3 support examples for all classes: "
            + ", ".join(entry["class_name"] for entry in insufficient_classes)
        )
    missing_configured_classes = [
        {
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
        }
        for class_id in expected_class_ids
        if class_id not in available_class_ids
    ]

    return support_images, support_labels, {
        "num_support_images": len(support_images),
        "max_images_per_class": max_images,
        "min_images_per_class": min_images,
        "counts_by_class_id": counts_by_class,
        "class_names": class_names,
        "available_class_ids": available_class_ids,
        "missing_configured_classes": missing_configured_classes,
    }


def generate_split_summary(cfg, output_dir: str | Path) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _build_dataset_from_cfg(
        cfg,
        split="train",
        episode_support_split="train",
        n_way=cfg.stage3.eval.n_way,
        k_shot=cfg.stage3.eval.k_shot,
        query_per_class=1,
    )
    payload = {
        "run_name": cfg.run.name,
        "split_summary": dataset.split_summary,
        "stage3_split_summary": getattr(dataset, "stage3_split_summary", {}),
        "supported_classes": getattr(dataset, "supported_classes", []),
        "removed_classes": getattr(dataset, "removed_classes", []),
        "supported_class_ids": getattr(dataset, "supported_class_ids", []),
    }
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return payload


def _mask_to_pil(mask: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    mask_np = (mask.detach().cpu().numpy() > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np, mode="L")
    if mask_img.size != size:
        mask_img = mask_img.resize(size, resample=Image.Resampling.NEAREST)
    return mask_img


def _draw_label(draw: ImageDraw.ImageDraw, box: Sequence[int], text: str, color: Sequence[int]):
    text_pos = (box[0], max(0, box[1] - 18))
    text_bbox = draw.textbbox(text_pos, text)
    draw.rectangle(text_bbox, fill=(0, 0, 0, 190))
    draw.text(text_pos, text, fill=(255, 255, 255))
    draw.rectangle(box, outline=tuple(color[:3]), width=3)


def _visualize_prediction(pil_image: Image.Image, output) -> Image.Image:
    canvas = pil_image.convert("RGBA")
    for idx, item in enumerate(output.items):
        color = PALETTE[idx % len(PALETTE)]
        if item.mask is not None:
            mask_img = _mask_to_pil(item.mask, canvas.size)
            color_layer = Image.new("RGBA", canvas.size, color)
            canvas = Image.composite(color_layer, canvas, mask_img)

        draw = ImageDraw.Draw(canvas)
        box = [int(round(v)) for v in item.box]
        label = f"{idx + 1}. {item.label} {item.confidence:.2f}"
        _draw_label(draw, box, label, color)

    footer_lines = [
        f"image_id: {output.image_id}",
        f"predicted_items: {len(output.items)}",
    ]
    source_kind = getattr(output, "image_source_kind", None)
    if source_kind:
        footer_lines.append(f"image_source: {source_kind}")
    footer_height = 28 * len(footer_lines) + 12
    framed = Image.new("RGBA", (canvas.width, canvas.height + footer_height), (17, 24, 39, 255))
    framed.paste(canvas, (0, 0))
    footer_draw = ImageDraw.Draw(framed)
    y = canvas.height + 8
    for line in footer_lines:
        footer_draw.text((12, y), line, fill=(255, 255, 255))
        y += 28
    return framed.convert("RGB")


def _visualize_ground_truth(
    pil_image: Image.Image,
    image_id: str,
    gt_items: Sequence[dict],
    gt_masks: Sequence[torch.Tensor | None],
) -> Image.Image:
    canvas = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    for idx, item in enumerate(gt_items):
        color = PALETTE[idx % len(PALETTE)]
        mask = gt_masks[idx] if idx < len(gt_masks) else None
        if mask is not None:
            mask_img = _mask_to_pil(mask, canvas.size)
            color_layer = Image.new("RGBA", canvas.size, color)
            canvas = Image.composite(color_layer, canvas, mask_img)
            draw = ImageDraw.Draw(canvas)

        box = [int(round(v)) for v in item["box"]]
        label = f"{idx + 1}. {item['label']}"
        _draw_label(draw, box, label, color)

    footer_lines = [
        f"image_id: {image_id}",
        f"ground_truth_items: {len(gt_items)}",
    ]
    footer_height = 28 * len(footer_lines) + 12
    framed = Image.new("RGBA", (canvas.width, canvas.height + footer_height), (15, 23, 42, 255))
    framed.paste(canvas, (0, 0))
    footer_draw = ImageDraw.Draw(framed)
    y = canvas.height + 8
    for line in footer_lines:
        footer_draw.text((12, y), line, fill=(255, 255, 255))
        y += 28
    return framed.convert("RGB")


def _compose_visual_pair(predicted: Image.Image, ground_truth: Image.Image) -> Image.Image:
    width = predicted.width + ground_truth.width
    height = max(predicted.height, ground_truth.height) + 44
    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "Predictions", fill=(15, 23, 42))
    draw.text((predicted.width + 16, 12), "Ground Truth", fill=(15, 23, 42))
    canvas.paste(predicted, (0, 44))
    canvas.paste(ground_truth, (predicted.width, 44))
    return canvas


def _compose_prediction_only(predicted: Image.Image) -> Image.Image:
    height = predicted.height + 44
    canvas = Image.new("RGB", (predicted.width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "Predictions", fill=(15, 23, 42))
    canvas.paste(predicted, (0, 44))
    return canvas


def _save_contact_sheet(image_paths: Iterable[Path], output_path: Path, columns: int = 2, thumb_size: tuple[int, int] = (640, 420)):
    image_paths = list(image_paths)
    if not image_paths:
        return

    thumbs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(image, thumb_size)
        tile = Image.new("RGB", thumb_size, (245, 245, 245))
        offset = ((thumb_size[0] - thumb.width) // 2, (thumb_size[1] - thumb.height) // 2)
        tile.paste(thumb, offset)
        thumbs.append(tile)

    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (thumb_size[0] * columns, thumb_size[1] * rows), (255, 255, 255))
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * thumb_size[0]
        y = (idx // columns) * thumb_size[1]
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def generate_split_visualizations(
    pipeline,
    cfg,
    output_dir: str | Path,
    *,
    split: str = "dev",
    max_images: int | None = None,
    top_k_classes: int = 1,
    include_ground_truth: bool = True,
) -> Dict[str, object]:
    split = normalize_split_name(split)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    support_images, support_labels, support_stats = build_stage3_reference_library(cfg)
    pipeline.stage3.set_support_set(support_images, support_labels)
    pipeline.eval()

    dataset = _build_dataset_from_cfg(
        cfg,
        split=split,
        episode_support_split="train",
        n_way=cfg.stage3.eval.n_way,
        k_shot=cfg.stage3.eval.k_shot,
        query_per_class=1,
    )

    limit = len(dataset) if max_images is None else min(max_images, len(dataset))
    written_images: List[Path] = []
    results: List[Dict[str, object]] = []
    latency_totals: Dict[str, float] = {"stage1": 0.0, "stage2": 0.0, "stage3": 0.0, "total": 0.0}

    with torch.no_grad():
        for index in range(limit):
            sample = dataset[index]
            output = pipeline.run(
                pil_image=sample["pil_image"],
                image_id=str(sample["image_id"]),
                nms_iou_threshold=cfg.stage2.nms.iou_threshold,
                score_threshold=cfg.stage2.nms.score_threshold,
                top_k_classes=top_k_classes,
            )
            setattr(output, "image_source_kind", sample.get("image_source_kind"))

            pred_visual = _visualize_prediction(sample["pil_image"], output)
            if include_ground_truth:
                gt_visual = _visualize_ground_truth(
                    sample["pil_image"],
                    str(sample["image_id"]),
                    sample.get("stage1_items", []),
                    sample.get("masks", []),
                )
                visual = _compose_visual_pair(pred_visual, gt_visual)
            else:
                visual = _compose_prediction_only(pred_visual)
            image_path = images_dir / f"{sample['image_id']}_predictions.png"
            visual.save(image_path)
            written_images.append(image_path)

            for key, value in output.latency_ms.items():
                latency_totals[key] = latency_totals.get(key, 0.0) + float(value)

            gt_items = [
                {
                    "label": item["label"],
                    "box": [float(v) for v in item["box"]],
                }
                for item in sample.get("stage1_items", [])
            ]
            predicted_items = [
                {
                    "label": item.label,
                    "confidence": float(item.confidence),
                    "box": [float(v) for v in item.box],
                    "price": float(item.price),
                    "has_mask": item.mask is not None,
                }
                for item in output.items
            ]
            matches = greedy_box_matches(
                [item["box"] for item in predicted_items],
                [item["box"] for item in gt_items],
                threshold=float(cfg.stage1.eval.iou_threshold),
            )
            match_details = [
                {
                    "pred_index": match.pred_index,
                    "gt_index": match.gt_index,
                    "iou": float(match.iou),
                    "pred_label": predicted_items[match.pred_index]["label"],
                    "gt_label": gt_items[match.gt_index]["label"],
                    "label_correct": predicted_items[match.pred_index]["label"] == gt_items[match.gt_index]["label"],
                }
                for match in matches
            ]

            results.append(
                {
                    "image_id": str(sample["image_id"]),
                    "notes": sample.get("notes"),
                    "review_status": sample.get("review_status"),
                    "image_path": sample.get("image_path"),
                    "resolved_image_path": sample.get("resolved_image_path"),
                    "image_source_kind": sample.get("image_source_kind"),
                    "ground_truth_items": gt_items,
                    "predicted_items": predicted_items,
                    "matches": match_details,
                    "latency_ms": {key: float(value) for key, value in output.latency_ms.items()},
                    "visual_path": str(image_path.relative_to(output_dir)),
                }
            )

    averages = {
        key: (value / max(len(results), 1))
        for key, value in latency_totals.items()
    }
    summary = {
        "run_name": cfg.run.name,
        "split": split,
        "num_images": len(results),
        "include_ground_truth": bool(include_ground_truth),
        "support_library": support_stats,
        "average_latency_ms": averages,
        "images": results,
    }
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    _save_contact_sheet(written_images, output_dir / "contact_sheet.jpg")

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Dev Visualizations</title></head><body>",
        f"<h1>Run: {cfg.run.name}</h1>",
        f"<p>Split: {split} | Images: {len(results)} | Support images: {support_stats['num_support_images']}</p>",
        (
            "<p>Each panel shows model predictions on the left and ground truth on the right.</p>"
            if include_ground_truth
            else "<p>Each panel shows model predictions only. Ground truth is preserved in the metadata below.</p>"
        ),
        "<ul>",
    ]
    for result in results:
        html_lines.append(
            f"<li><h3>{result['image_id']}</h3>"
            f"<img src='{result['visual_path']}' style='max-width:900px;width:100%;height:auto'>"
            f"<p>Image source: {result.get('image_source_kind')}<br>Resolved path: {result.get('resolved_image_path')}</p>"
            f"<h4>Ground Truth</h4><pre>{json.dumps(result['ground_truth_items'], indent=2)}</pre>"
            f"<h4>Predictions</h4><pre>{json.dumps(result['predicted_items'], indent=2)}</pre>"
            f"<h4>Matches</h4><pre>{json.dumps(result['matches'], indent=2)}</pre></li>"
        )
    html_lines.extend(["</ul>", "</body></html>"])
    (output_dir / "index.html").write_text("\n".join(html_lines), encoding="utf-8")
    return summary


def generate_training_report(cfg, logs_dir: str | Path, output_dir: str | Path) -> Dict[str, object]:
    logs_dir = Path(logs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_report(
        run_dirs=[logs_dir],
        output_dir=output_dir,
        title=f"TriFoodNet Train/Dev Report: {cfg.run.name}",
    )
    return {
        "logs_dir": str(logs_dir),
        "report_dir": str(output_dir),
        "index_path": str(output_dir / "index.md"),
    }
