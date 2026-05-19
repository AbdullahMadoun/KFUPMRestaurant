# =============================================================================
# FILE: visualize_val_predictions.py
# CATEGORY: EVAL
# PURPOSE: Visualization script for validation predictions and qualitative inspection.
# DEPENDENCIES: config_loader.py, dataset_integration.py, item_processing.py, pipeline.py, run_dev_inference.py, stage1_qwen.py, stage2_sam.py, stage3_icl.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: parse_args, _resolve_amp_dtype, _build_pipeline, _build_support_set, _mask_to_pil, _draw_label, _visualize_prediction, _predict_from_gt_boxes, _save_contact_sheet, main
# LAST MODIFIED: 2026-03-21T20:24:18.595544+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

sys.path.append(os.getcwd())

from config_loader import load_config
from dataset_integration import (
    JointFoodDataset,
    build_class_name_index,
    build_export_paths,
    load_masked_item_image,
    read_json,
)
from item_processing import masked_crop_from_pil
from pipeline import TriFoodNet
from run_dev_inference import build_bnb_config, resolve_device
from stage1_qwen import QwenGrounder
from stage2_sam import SAM3Segmenter
from stage3_icl import FoodClassifier


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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize validation-set mask and class predictions.")
    parser.add_argument("--run-name", default="trial-20260321-full3")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--use-gt-boxes",
        action="store_true",
        help="Use GT boxes as SAM prompts. Default is full end-to-end Stage 1 -> Stage 2 -> Stage 3 inference.",
    )
    parser.add_argument("--use-stage1", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to outputs/val_visuals/<run_name>/<split>",
    )
    return parser.parse_args()


def _resolve_amp_dtype(hardware_cfg, device: torch.device):
    if device.type != "cuda":
        return None
    if getattr(hardware_cfg, "bf16", False):
        return torch.bfloat16
    if getattr(hardware_cfg, "fp16", False):
        return torch.float16
    return None


def _build_pipeline(cfg, device: torch.device) -> TriFoodNet:
    c1 = cfg.stage1
    c2 = cfg.stage2
    c3 = cfg.stage3
    amp_dtype = _resolve_amp_dtype(cfg.hardware, device)
    bnb_config = build_bnb_config(cfg, device)

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

    stage1 = QwenGrounder(
        model_name=c1.model_name,
        lora_r=c1.lora.r,
        lora_alpha=c1.lora.alpha,
        lora_dropout=c1.lora.dropout,
        lora_target_modules=c1.lora.target_modules,
        use_rslora=c1.lora.get("use_rslora", False),
        gradient_checkpointing=cfg.hardware.gradient_checkpointing,
        quantization_config=bnb_config,
    )
    stage2 = SAM3Segmenter(
        model_name=c2.model_name,
        freeze_image_encoder=c2.freeze.image_encoder,
        freeze_prompt_encoder=c2.freeze.prompt_encoder,
        gradient_checkpointing=cfg.hardware.gradient_checkpointing,
        quantization_config=bnb_config,
        torch_dtype=amp_dtype,
    )
    stage3 = FoodClassifier(
        clip_model=c3.clip_model,
        num_layers=c3.transformer.num_layers,
        num_heads=c3.transformer.num_heads,
        ff_dim=c3.transformer.ff_dim,
        dropout=c3.transformer.dropout,
        lora_cfg=getattr(c3, "lora", None),
        num_classes=max(int(cfg.data.num_classes), len(class_names), (max(class_name_to_id.values()) + 1) if class_name_to_id else 0),
        class_names=class_names,
        train_embedding=bool(getattr(c3, "train_embedding", True)),
        inference_n_way=int(c3.eval.n_way),
        inference_k_shot=int(c3.eval.k_shot),
    ).to(device)

    pipeline = TriFoodNet(stage1, stage2, stage3)
    checkpoint_dir = Path(cfg.paths.checkpoints) / cfg.run.name / "joint" / "best"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
    pipeline.load(str(checkpoint_dir))
    pipeline.eval()
    return pipeline


def _build_support_set(pipeline: TriFoodNet, cfg) -> int:
    train_dataset = JointFoodDataset(
        batch_root=cfg.data.integration.batch_root,
        export_root=(cfg.data.integration.export_root or None),
        repo_root=(cfg.data.integration.repo_root or None),
        split="train",
        image_size=cfg.data.image_size,
        train_ratio=cfg.data.integration.train_ratio,
        val_ratio=cfg.data.integration.val_ratio,
        test_ratio=cfg.data.integration.test_ratio,
        split_seed=cfg.data.integration.split_seed,
        n_way=cfg.stage3.episode.n_way,
        k_shot=cfg.stage3.episode.k_shot,
        query_per_class=cfg.stage3.episode.query_per_class,
    )

    support_images: List[Image.Image] = []
    support_labels: List[int] = []
    seen_class_ids: set[int] = set()

    for row in train_dataset.stage3_rows:
        class_id = int(row["class_id"])
        if class_id in seen_class_ids:
            continue
        support_images.append(load_masked_item_image(row, train_dataset.paths.batch_root, train_dataset.paths.repo_root))
        support_labels.append(class_id)
        seen_class_ids.add(class_id)
        if len(seen_class_ids) >= int(cfg.data.num_classes):
            break

    if not support_images:
        raise RuntimeError("Failed to build a support set from the training split.")

    pipeline.stage3.set_support_set(support_images, support_labels)
    return len(support_images)


def _mask_to_pil(mask: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    mask_np = (mask.detach().cpu().numpy() > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np, mode="L")
    if mask_img.size != size:
        mask_img = mask_img.resize(size, resample=Image.Resampling.NEAREST)
    return mask_img


def _draw_label(draw: ImageDraw.ImageDraw, box: list[int], text: str, color: tuple[int, int, int, int]):
    text_pos = (box[0], max(0, box[1] - 18))
    text_bbox = draw.textbbox(text_pos, text)
    draw.rectangle(text_bbox, fill=(0, 0, 0, 190))
    draw.text(text_pos, text, fill=(255, 255, 255))
    draw.rectangle(box, outline=color[:3], width=3)


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
    footer_height = 28 * len(footer_lines) + 12
    framed = Image.new("RGBA", (canvas.width, canvas.height + footer_height), (17, 24, 39, 255))
    framed.paste(canvas, (0, 0))
    footer_draw = ImageDraw.Draw(framed)
    y = canvas.height + 8
    for line in footer_lines:
        footer_draw.text((12, y), line, fill=(255, 255, 255))
        y += 28
    return framed.convert("RGB")


def _predict_from_gt_boxes(pipeline: TriFoodNet, sample: dict, score_threshold: float, nms_iou_threshold: float):
    stage2_device = next(pipeline.stage2.parameters()).device
    image_tensor = sample["image_tensor"].unsqueeze(0).to(stage2_device)
    boxes = sample["boxes"].to(stage2_device)

    stage2_output = pipeline.stage2.predict(
        image_tensor,
        [boxes],
        nms_iou_threshold=nms_iou_threshold,
        score_threshold=score_threshold,
    )
    masks = stage2_output["pred_masks"][0]
    kept_indices = stage2_output.get("kept_indices", [[]])[0]

    items = []
    detection_mask_pairs = []
    if masks and kept_indices:
        for det_idx, mask in zip(kept_indices, masks):
            if det_idx < len(sample["stage1_items"]):
                detection_mask_pairs.append((sample["stage1_items"][det_idx], mask))
    if not detection_mask_pairs:
        detection_mask_pairs = [(det, None) for det in sample["stage1_items"]]

    for det, mask in detection_mask_pairs:
        box = det["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = (
            masked_crop_from_pil(sample["pil_image"], mask, box)
            if mask is not None
            else sample["pil_image"].crop((x1, y1, x2, y2))
        )
        preds = pipeline.stage3.classify(crop, device=stage2_device.type, top_k=1)
        label, confidence = preds[0] if preds else ("unknown", 0.0)
        items.append(
            {
                "box": [float(v) for v in box],
                "mask": mask,
                "label": label,
                "confidence": float(confidence),
                "price": 0.0,
            }
        )

    return {
        "image_id": str(sample["image_id"]),
        "items": items,
        "latency_ms": {},
    }


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


def main():
    args = parse_args()
    cfg = load_config("master_config.yaml", overrides=[f"run.name={args.run_name}"])
    device = resolve_device(getattr(cfg.hardware, "device", "auto"))

    output_dir = Path(args.output_dir or Path("outputs") / "val_visuals" / args.run_name / args.split)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(cfg, device)
    support_count = _build_support_set(pipeline, cfg)

    dataset = JointFoodDataset(
        batch_root=cfg.data.integration.batch_root,
        export_root=(cfg.data.integration.export_root or None),
        repo_root=(cfg.data.integration.repo_root or None),
        split=args.split,
        image_size=cfg.data.image_size,
        train_ratio=cfg.data.integration.train_ratio,
        val_ratio=float(getattr(cfg.data.integration, "dev_ratio", getattr(cfg.data.integration, "val_ratio", 0.1))),
        test_ratio=cfg.data.integration.test_ratio,
        split_seed=cfg.data.integration.split_seed,
        n_way=cfg.stage3.eval.n_way,
        k_shot=cfg.stage3.eval.k_shot,
        query_per_class=1,
    )

    limit = args.max_images if args.max_images is not None else len(dataset)
    limit = min(limit, len(dataset))
    results = []
    written_images: List[Path] = []

    with torch.no_grad():
        for index in range(limit):
            sample = dataset[index]
            use_stage1 = bool(args.use_stage1 or not args.use_gt_boxes)
            if use_stage1:
                output = pipeline.run(
                    pil_image=sample["pil_image"],
                    image_id=str(sample["image_id"]),
                )
                image_id = output.image_id
                output_items = [
                    {
                        "label": item.label,
                        "confidence": float(item.confidence),
                        "box": [float(v) for v in item.box],
                        "price": float(item.price),
                        "mask": item.mask,
                    }
                    for item in output.items
                ]
                latency_ms = output.latency_ms
            else:
                output = _predict_from_gt_boxes(
                    pipeline,
                    sample,
                    score_threshold=cfg.stage2.nms.score_threshold,
                    nms_iou_threshold=cfg.stage2.nms.iou_threshold,
                )
                image_id = output["image_id"]
                output_items = output["items"]
                latency_ms = output["latency_ms"]

            visual_output = type("VisualOutput", (), {"image_id": image_id, "items": []})()
            for item in output_items:
                visual_output.items.append(
                    type(
                        "VisualItem",
                        (),
                        {
                            "box": item["box"],
                            "mask": item["mask"],
                            "label": item["label"],
                            "confidence": item["confidence"],
                        },
                    )()
                )

            visual = _visualize_prediction(sample["pil_image"], visual_output)
            image_path = images_dir / f"{sample['image_id']}_predictions.png"
            visual.save(image_path)
            written_images.append(image_path)

            results.append(
                {
                    "image_id": image_id,
                    "notes": sample.get("notes"),
                    "review_status": sample.get("review_status"),
                    "image_path": sample.get("image_path"),
                    "resolved_image_path": sample.get("resolved_image_path"),
                    "image_source_kind": sample.get("image_source_kind"),
                    "prompt_source": "stage1" if use_stage1 else "gt_boxes",
                    "latency_ms": latency_ms,
                    "items": [
                        {
                            "label": item["label"],
                            "confidence": item["confidence"],
                            "box": item["box"],
                            "price": item["price"],
                            "has_mask": item["mask"] is not None,
                        }
                        for item in output_items
                    ],
                    "visual_path": str(image_path.relative_to(output_dir)),
                }
            )

    summary = {
        "run_name": args.run_name,
        "split": args.split,
        "support_examples": support_count,
        "num_images": len(results),
        "output_dir": str(output_dir),
        "images": results,
    }

    with open(output_dir / "predictions.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    _save_contact_sheet(written_images, output_dir / "contact_sheet.jpg")

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Validation Visuals</title></head><body>",
        f"<h1>Validation Visuals: {args.run_name}</h1>",
        f"<p>Split: {args.split} | Images: {len(results)} | Support examples: {support_count}</p>",
        "<p>Default mode is live end-to-end inference. GT-box prompting only happens when --use-gt-boxes is set.</p>",
        "<ul>",
    ]
    for result in results:
        html_lines.append(
            f"<li><h3>{result['image_id']}</h3>"
            f"<img src='{result['visual_path']}' style='max-width:900px;width:100%;height:auto'>"
            f"<p>Prompt source: {result['prompt_source']}<br>Image source: {result.get('image_source_kind')}<br>Resolved path: {result.get('resolved_image_path')}</p>"
            f"<pre>{json.dumps(result['items'], indent=2)}</pre></li>"
        )
    html_lines.extend(["</ul>", "</body></html>"])
    (output_dir / "index.html").write_text("\n".join(html_lines))

    print(f"Saved validation visuals to {output_dir}")
    print(f"Saved contact sheet to {output_dir / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
