from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont

from .bbox_canonical import get_bbox_scale_factor, load_image_at_v3_resolution, scale_bbox
from .config import CANONICAL_STAGE1_SPLIT_SEED, STAGE1_PROMPT, Stage1Config
from .dataset import Stage1KCFDDataset
from .eval import generate_text
from .schema import Stage1Item, Stage1Target, parse_prediction


GT_COLOR = (36, 148, 70)
PRED_COLOR = (220, 64, 48)
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        return ImageFont.load_default()


def _clamp_box(box: Sequence[float], size: tuple[int, int]) -> list[float]:
    width, height = size
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def _item_label(item: Stage1Item | dict, prefix: str = "") -> str:
    if isinstance(item, Stage1Item):
        name = item.name
        descriptor = item.descriptor
    else:
        name = str(item.get("name", "item"))
        descriptor = str(item.get("descriptor") or item.get("description") or item.get("vlm_description") or "")
    label = f"{prefix}{name}".strip()
    if descriptor:
        label = f"{label}: {descriptor}"
    return label[:96]


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_items(
    image: Image.Image,
    items: Iterable[Stage1Item | dict],
    *,
    color: tuple[int, int, int],
    prefix: str = "",
    width: int = 3,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for item in items:
        bbox = item.bbox if isinstance(item, Stage1Item) else item.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = _clamp_box(bbox, canvas.size)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = _item_label(item, prefix=prefix)
        text_w, text_h = _text_size(draw, label, font)
        label_x = int(max(0, min(x1, canvas.size[0] - text_w - 6)))
        label_y = int(max(0, y1 - text_h - 6))
        draw.rectangle([label_x, label_y, label_x + text_w + 6, label_y + text_h + 5], fill=TEXT_BG)
        draw.text((label_x + 3, label_y + 2), label, fill=TEXT_FG, font=font)
    return canvas


def _target_from_example(example: dict) -> Stage1Target:
    return Stage1Target(items=list(example["items"]))


def _scaled_gt_items(example: dict, export_root: Path) -> List[Stage1Item]:
    raw_items = example["raw_items"]
    if not raw_items:
        return []
    sx, sy = get_bbox_scale_factor(raw_items[0], export_root)
    scaled: List[Stage1Item] = []
    for item in example["items"]:
        scaled.append(Stage1Item(name=item.name, bbox=scale_bbox(item.bbox, sx, sy), descriptor=item.descriptor))
    return scaled


def _sample_labels(sample) -> set[str]:
    labels = {
        str(row.get("class_slug") or row.get("class_display_name") or row.get("name") or "food")
        for row in sample.items
    }
    labels.discard("")
    return labels


def select_preview_indices(
    dataset: Stage1KCFDDataset,
    *,
    max_samples: int,
    seed: int,
    mode: str,
) -> List[int]:
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if mode == "random":
        return indices[:max_samples]
    if mode == "first":
        return list(range(min(n, max_samples)))
    if mode != "class-diverse":
        raise ValueError("preview selection must be one of: first, random, class-diverse")

    selected: List[int] = []
    seen: set[str] = set()
    remaining = indices[:]
    while remaining and len(selected) < max_samples:
        best_pos = 0
        best_score = None
        for pos, idx in enumerate(remaining):
            labels = _sample_labels(dataset.samples[idx])
            new_labels = labels - seen
            score = (len(new_labels), len(labels), len(dataset.samples[idx].items), -pos)
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
        idx = remaining.pop(best_pos)
        selected.append(idx)
        seen.update(_sample_labels(dataset.samples[idx]))
    return selected


def save_training_previews(
    dataset: Stage1KCFDDataset,
    output_dir: str | Path,
    *,
    max_samples: int = 5,
    include_full_resolution: bool = True,
    seed: int = 1337,
    selection: str = "class-diverse",
) -> List[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    indices = select_preview_indices(dataset, max_samples=max_samples, seed=seed, mode=selection)
    selected_manifest = []
    for out_idx, dataset_idx in enumerate(indices):
        example = dataset[dataset_idx]
        image_id = str(example["image_id"])
        labels = sorted(_sample_labels(dataset.samples[dataset_idx]))
        selected_manifest.append({"dataset_index": dataset_idx, "image_id": image_id, "classes": labels})
        mask_native = draw_items(example["image"], example["items"], color=GT_COLOR, prefix="GT ")
        mask_path = output / f"{out_idx:02d}_{image_id}_mask_native_gt.png"
        mask_native.save(mask_path)
        paths.append(mask_path)

        if include_full_resolution and example["raw_items"]:
            full_image = load_image_at_v3_resolution(example["raw_items"][0], dataset.export_root)
            full = draw_items(full_image, _scaled_gt_items(example, dataset.export_root), color=GT_COLOR, prefix="GT ")
            full_path = output / f"{out_idx:02d}_{image_id}_full_res_gt.png"
            full.save(full_path)
            paths.append(full_path)

    manifest = {
        "split": dataset.config.split,
        "split_seed": dataset.config.split_seed,
        "reference_policy": dataset.config.reference_policy,
        "selection": selection,
        "selection_seed": seed,
        "seed": seed,
        "count": len(paths),
        "selected": selected_manifest,
        "distinct_classes": sorted({label for row in selected_manifest for label in row["classes"]}),
        "files": [str(path) for path in paths],
        "coordinate_note": "mask_native_gt uses model-input coordinates; full_res_gt scales boxes to original image size.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


class Stage1Visualizer:
    def __init__(self, *, prompt: str = STAGE1_PROMPT):
        self.prompt = prompt

    def generate_validation_panel(
        self,
        model,
        processor,
        dataset: Stage1KCFDDataset,
        output_dir: str | Path,
        *,
        max_samples: int = 8,
        max_new_tokens: int = 512,
    ) -> List[Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        count = min(len(dataset), max_samples)
        for idx in range(count):
            example = dataset[idx]
            text = generate_text(model, processor, example["image"], self.prompt, max_new_tokens=max_new_tokens)
            valid, pred, error = parse_prediction(text)
            image = draw_items(example["image"], example["items"], color=GT_COLOR, prefix="GT ")
            image = draw_items(image, pred.items, color=PRED_COLOR, prefix="PRED ")
            path = output / f"{idx:02d}_{example['image_id']}_pred_vs_gt.png"
            image.save(path)
            sidecar = {
                "image_id": example["image_id"],
                "valid": valid,
                "parse_error": error,
                "prediction": text,
                "target": example["target"],
            }
            path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
            paths.append(path)
        return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage 1 v3 training-data box previews.")
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--reference-policy", choices=["pause", "exclude", "train", "include"], default="exclude")
    parser.add_argument("--split-seed", type=int, default=CANONICAL_STAGE1_SPLIT_SEED)
    parser.add_argument("--seed", type=int, default=1337, help="Preview selection seed, not the train/val/test split seed.")
    parser.add_argument("--selection", choices=["first", "random", "class-diverse"], default="class-diverse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Stage1Config(
        export_root=Path(args.export_root),
        split=args.split,
        reference_policy=args.reference_policy,
        split_seed=args.split_seed,
    )
    dataset = Stage1KCFDDataset(config)
    paths = save_training_previews(
        dataset,
        args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        selection=args.selection,
    )
    print(json.dumps({"files": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
