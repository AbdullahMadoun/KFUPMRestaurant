# =============================================================================
# FILE: dataset_integration.py
# CATEGORY: DATA
# PURPOSE: Manifest-driven dataset integration, split construction, asset resolution, and collators.
# DEPENDENCIES: item_processing.py
# USED BY: benchmark_runtime.py, post_training_artifacts.py, run_dev_inference.py, run_isolated_inference.py, run_single_inference.py, tests/test_pipeline_contracts.py, train_joint.py, train_stage3_hf.py, validate_pipeline_contracts.py, verify_split.py, visualize_val_predictions.py
# KEY CLASSES/FUNCTIONS: read_json, read_jsonl, ExportPaths, build_export_paths, normalize_split_name, stratified_split, summarize_split_mapping, summarize_stage3_split_mapping, class_names_present_in_all_splits, labeled_active_class_names, _allocate_split_sizes, _desired_train_image_count
# LAST MODIFIED: 2026-03-21T20:28:19.235893+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
Manifest-based dataset integration for the reviewed export contract.

This module follows `INTEGRATION_DATASET_GUIDE.md` and is designed to work
before the actual dataset is present. It provides:

- path resolution against the batch root
- pointer-file image resolution
- deterministic hash-based splits
- stage-specific manifest datasets
- a joint dataset + collator for end-to-end training
- optional export snapshotting for reproducible experiments
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from item_processing import masked_crop_from_pil, pil_images_to_tensor


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@dataclass
class ExportPaths:
    batch_root: Path
    export_root: Path
    repo_root: Path
    annotations_root: Path


def build_export_paths(
    batch_root: str | Path,
    export_root: Optional[str | Path] = None,
    repo_root: Optional[str | Path] = None,
) -> ExportPaths:
    batch_root = Path(batch_root).resolve()
    export_root = Path(export_root).resolve() if export_root else (batch_root / "_review" / "dataset")
    repo_root = Path(repo_root).resolve() if repo_root else batch_root.parent.parent.parent
    annotations_root = batch_root / "_review" / "annotations"
    return ExportPaths(
        batch_root=batch_root,
        export_root=export_root,
        repo_root=repo_root,
        annotations_root=annotations_root,
    )


def normalize_split_name(split: str | None) -> str:
    normalized = str(split or "train").strip().lower()
    aliases = {
        "train": "train",
        "dev": "dev",
        "val": "dev",
        "valid": "dev",
        "validation": "dev",
        "test": "test",
    }
    return aliases.get(normalized, normalized)


def stratified_split(
    items: List[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 1337,
) -> Dict[str, str]:
    rng = random.Random(seed)
    split_names = ("train", "dev", "test")
    target_sizes = _allocate_split_sizes(len(items), train_ratio, val_ratio, test_ratio)
    per_class_counts = _per_class_image_counts(items)

    image_infos = []
    for row in items:
        labels = sorted({choose_item_label(item) for item in filter_active_items(row.get("items", []))})
        if not labels:
            continue
        image_infos.append(
            {
                "image_id": str(row["image_id"]),
                "labels": labels,
                "hardness": _image_hardness(row, per_class_counts),
            }
        )

    label_to_indices: Dict[str, List[int]] = {}
    for index, info in enumerate(image_infos):
        for label in info["labels"]:
            label_to_indices.setdefault(label, []).append(index)
    desired_train_counts = {
        label: _desired_train_image_count(len(indices))
        for label, indices in label_to_indices.items()
    }

    assignments: Dict[int, str] = {}
    split_counts = {name: 0 for name in split_names}
    split_label_counts = {name: {} for name in split_names}
    split_hardness = {name: 0.0 for name in split_names}

    def assign(index: int, split: str):
        if index in assignments:
            return
        assignments[index] = split
        split_counts[split] += 1
        split_hardness[split] += float(image_infos[index]["hardness"])
        for label in image_infos[index]["labels"]:
            counts = split_label_counts[split]
            counts[label] = counts.get(label, 0) + 1

    def coverage_sort_key(split: str, index: int):
        info = image_infos[index]
        uncovered_gain = sum(
            1 for candidate_label in info["labels"] if split_label_counts[split].get(candidate_label, 0) == 0
        )
        rarity_gain = sum(1.0 / max(len(label_to_indices[candidate_label]), 1) for candidate_label in info["labels"])
        if split == "train":
            return (
                -uncovered_gain,
                info["hardness"],
                -rarity_gain,
                info["image_id"],
            )
        return (
            -uncovered_gain,
            -info["hardness"],
            -len(info["labels"]),
            -rarity_gain,
            info["image_id"],
        )

    def ensure_split_coverage(split: str):
        for label in sorted(label_to_indices, key=lambda name: (len(label_to_indices[name]), name)):
            if split_label_counts[split].get(label, 0) > 0 or split_counts[split] >= target_sizes[split]:
                continue
            candidates = [idx for idx in label_to_indices[label] if idx not in assignments]
            if not candidates:
                continue
            candidates.sort(key=lambda idx: coverage_sort_key(split, idx))
            assign(candidates[0], split)

    # First minimize unseen classes by maximizing class coverage in each split.
    ensure_split_coverage("train")
    ensure_split_coverage("dev")
    ensure_split_coverage("test")

    # Then strengthen train support for the genuinely rare classes where
    # an extra train example is available without touching held-out coverage.
    for label in sorted(label_to_indices, key=lambda name: (len(label_to_indices[name]), name)):
        desired_count = desired_train_counts.get(label, 1)
        while (
            split_label_counts["train"].get(label, 0) < desired_count
            and split_counts["train"] < target_sizes["train"]
        ):
            candidates = [idx for idx in label_to_indices[label] if idx not in assignments]
            if not candidates:
                break
            candidates.sort(
                key=lambda idx: (
                    -sum(
                        1
                        for candidate_label in image_infos[idx]["labels"]
                        if split_label_counts["train"].get(candidate_label, 0)
                        < desired_train_counts.get(candidate_label, 1)
                    ),
                    image_infos[idx]["hardness"],
                    image_infos[idx]["image_id"],
                )
            )
            assign(candidates[0], "train")

    remaining_indices = [idx for idx in range(len(image_infos)) if idx not in assignments]
    remaining_indices.sort(
        key=lambda idx: (
            -image_infos[idx]["hardness"],
            len(image_infos[idx]["labels"]),
            image_infos[idx]["image_id"],
        )
    )

    def best_eval_split(index: int) -> Optional[str]:
        candidates = [name for name in ("dev", "test") if split_counts[name] < target_sizes[name]]
        if not candidates:
            return None
        info = image_infos[index]
        scored = []
        for split in candidates:
            unseen_bonus = sum(1 for label in info["labels"] if split_label_counts[split].get(label, 0) == 0)
            remaining_capacity = target_sizes[split] - split_counts[split]
            hardness_balance = -split_hardness[split]
            scored.append((unseen_bonus, remaining_capacity, hardness_balance, split == "dev", split))
        scored.sort(reverse=True)
        return scored[0][-1]

    for index in remaining_indices:
        eval_split = best_eval_split(index)
        if eval_split is not None:
            assign(index, eval_split)
        elif split_counts["train"] < target_sizes["train"]:
            assign(index, "train")

    # Fill any residual capacity deterministically.
    for index in range(len(image_infos)):
        if index in assignments:
            continue
        for split in split_names:
            if split_counts[split] < target_sizes[split]:
                assign(index, split)
                break

    return {
        image_infos[index]["image_id"]: split
        for index, split in assignments.items()
    }


def summarize_split_mapping(items: Sequence[dict], split_mapping: Dict[str, str]) -> Dict[str, dict]:
    summary = {
        "train": {"images": 0, "active_items": 0, "class_image_counts": {}},
        "dev": {"images": 0, "active_items": 0, "class_image_counts": {}},
        "test": {"images": 0, "active_items": 0, "class_image_counts": {}},
    }

    for row in items:
        split = normalize_split_name(split_mapping.get(str(row["image_id"]), "train"))
        bucket = summary.setdefault(split, {"images": 0, "active_items": 0, "class_image_counts": {}})
        bucket["images"] += 1
        active_items = filter_active_items(row.get("items", []))
        bucket["active_items"] += len(active_items)
        seen_labels = {choose_item_label(item) for item in active_items}
        for label in seen_labels:
            counts = bucket["class_image_counts"]
            counts[label] = counts.get(label, 0) + 1
    return summary


def summarize_stage3_split_mapping(stage3_rows: Sequence[dict], split_mapping: Dict[str, str]) -> Dict[str, dict]:
    summary = {
        "train": {"items": 0, "class_item_counts": {}},
        "dev": {"items": 0, "class_item_counts": {}},
        "test": {"items": 0, "class_item_counts": {}},
    }
    for row in stage3_rows:
        split = normalize_split_name(split_mapping.get(str(row["image_id"]), "train"))
        bucket = summary.setdefault(split, {"items": 0, "class_item_counts": {}})
        bucket["items"] += 1
        label = choose_item_label(row)
        counts = bucket["class_item_counts"]
        counts[label] = counts.get(label, 0) + 1
    return summary


def class_names_present_in_all_splits(summary: Dict[str, dict], count_key: str) -> set[str]:
    split_sets = []
    for split in ("train", "dev", "test"):
        bucket = summary.get(split, {})
        counts = bucket.get(count_key, {})
        split_sets.append({name for name, value in counts.items() if int(value) > 0})
    if not split_sets:
        return set()
    return set.intersection(*split_sets)


def labeled_active_class_names(row: dict) -> set[str]:
    names = set()
    for item in filter_active_items(row.get("items", [])):
        if item.get("classification_status") != "labeled":
            continue
        names.add(choose_item_label(item))
    return names


def _allocate_split_sizes(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    ratios = {
        "train": float(train_ratio),
        "dev": float(val_ratio),
        "test": float(test_ratio),
    }
    ratio_sum = sum(ratios.values())
    raw = {name: total * (value / max(ratio_sum, 1e-9)) for name, value in ratios.items()}
    sizes = {name: int(raw[name]) for name in ratios}
    remainder = total - sum(sizes.values())
    for name, _ in sorted(((name, raw[name] - sizes[name]) for name in ratios), key=lambda item: item[1], reverse=True):
        if remainder <= 0:
            break
        sizes[name] += 1
        remainder -= 1
    return sizes


def _desired_train_image_count(total_images: int) -> int:
    if total_images <= 1:
        return total_images
    if total_images == 2:
        return 1
    if total_images <= 4:
        return 2
    return 1


def _per_class_image_counts(rows: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for label in {choose_item_label(item) for item in filter_active_items(row.get("items", []))}:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _image_hardness(row: dict, per_class_counts: Dict[str, int]) -> float:
    items = filter_active_items(row.get("items", []))
    if not items:
        return 0.0
    seen_item_count = float(len(items))
    class_diversity = float(len({choose_item_label(item) for item in items}))
    rarity_tiebreak = sum(1.0 / max(per_class_counts.get(choose_item_label(item), 1), 1) for item in items)
    # Harder images are primarily those with more supported items.
    return (100.0 * seen_item_count) + (10.0 * class_diversity) + rarity_tiebreak


def _clone_row_with_allowed_items(row: dict, allowed_classes: set[str]) -> Optional[dict]:
    filtered_items = [
        dict(item)
        for item in filter_active_items(row.get("items", []))
        if item.get("classification_status") == "labeled" and choose_item_label(item) in allowed_classes
    ]
    if not filtered_items:
        return None
    cloned = dict(row)
    cloned["items"] = filtered_items
    return cloned


def filter_rows_to_allowed_classes(
    image_rows: Sequence[dict],
    stage3_rows: Sequence[dict],
    allowed_classes: set[str],
) -> tuple[List[dict], List[dict], set[str]]:
    filtered_images: List[dict] = []
    retained_image_ids: set[str] = set()
    for row in image_rows:
        filtered_row = _clone_row_with_allowed_items(row, allowed_classes)
        if filtered_row is None:
            continue
        filtered_images.append(filtered_row)
        retained_image_ids.add(str(filtered_row["image_id"]))

    filtered_stage3 = [
        row
        for row in stage3_rows
        if choose_item_label(row) in allowed_classes and str(row["image_id"]) in retained_image_ids
    ]
    return filtered_images, filtered_stage3, retained_image_ids


def enforce_supported_class_contract(
    image_rows: Sequence[dict],
    stage3_rows: Sequence[dict],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, object]:
    working_images = [row for row in image_rows if labeled_active_class_names(row)]
    working_stage3 = list(stage3_rows)
    image_classes = {label for row in working_images for label in labeled_active_class_names(row)}
    stage3_classes = {choose_item_label(row) for row in working_stage3}

    image_class_counts = _per_class_image_counts(working_images)
    stage3_item_counts: Dict[str, int] = {}
    for row in working_stage3:
        label = choose_item_label(row)
        stage3_item_counts[label] = stage3_item_counts.get(label, 0) + 1

    min_total_images_per_class = 3
    min_total_items_per_class = 3
    allowed_classes = {
        label
        for label, image_count in image_class_counts.items()
        if image_count >= min_total_images_per_class and stage3_item_counts.get(label, 0) >= min_total_items_per_class
    }

    filtered_images, filtered_stage3, retained_image_ids = filter_rows_to_allowed_classes(
        working_images,
        working_stage3,
        allowed_classes,
    )
    split_mapping = stratified_split(
        filtered_images,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    image_summary = summarize_split_mapping(filtered_images, split_mapping)
    stage3_summary = summarize_stage3_split_mapping(filtered_stage3, split_mapping)
    final_supported_classes = sorted(allowed_classes)

    return {
        "image_rows": filtered_images,
        "stage3_rows": filtered_stage3,
        "retained_image_ids": retained_image_ids,
        "split_mapping": split_mapping,
        "split_summary": image_summary,
        "stage3_split_summary": stage3_summary,
        "supported_classes": final_supported_classes,
        "removed_classes": sorted((image_classes | stage3_classes) - set(final_supported_classes)),
    }


def resolve_relative_path(path_value: str | Path, batch_root: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(batch_root) / path


def resolve_pointer_path(pointer_text: str, repo_root: Path) -> Optional[Path]:
    pointer = pointer_text.strip().strip('"').strip("'")
    if not pointer:
        return None

    pointer_path = Path(pointer)
    if pointer_path.is_absolute() and pointer_path.exists():
        return pointer_path

    normalized = pointer.replace("\\", "/")
    if "Sampled_Images_All/" in normalized:
        suffix = normalized.split("Sampled_Images_All/", 1)[1]
        candidate = repo_root / "Sampled_Images_All" / suffix
        if candidate.exists():
            return candidate

    candidate = repo_root / normalized
    if candidate.exists():
        return candidate

    fallback_root = repo_root / "Sampled_Images_All"
    if fallback_root.exists():
        matches = list(fallback_root.rglob(Path(normalized).name))
        if matches:
            return matches[0]

    return None


def load_image_with_pointer_support(
    image_path: str | Path,
    batch_root: str | Path,
    repo_root: Optional[str | Path] = None,
    *,
    allow_visualization_fallback: bool = False,
) -> Image.Image:
    image, _ = load_image_with_metadata(
        image_path,
        batch_root,
        repo_root=repo_root,
        allow_visualization_fallback=allow_visualization_fallback,
    )
    return image


def _is_loadable_image(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.stem.lower() == "dummy":
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def resolve_image_asset(
    image_path: str | Path,
    batch_root: str | Path,
    repo_root: Optional[str | Path] = None,
    *,
    allow_visualization_fallback: bool = False,
) -> Tuple[Path, Dict[str, str]]:
    batch_root = Path(batch_root).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else batch_root.parent.parent.parent
    candidate = resolve_relative_path(image_path, batch_root)

    if _is_loadable_image(candidate):
        return candidate, {
            "requested_path": str(candidate),
            "resolved_path": str(candidate),
            "source_kind": "direct_image",
        }

    pointer_text = ""
    try:
        pointer_text = candidate.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        pointer_text = ""

    if pointer_text:
        resolved = resolve_pointer_path(pointer_text, repo_root)
        if resolved is not None and _is_loadable_image(resolved):
            return resolved, {
                "requested_path": str(candidate),
                "resolved_path": str(resolved),
                "source_kind": "pointer_resolved",
            }

    if candidate.name.lower() == "original.jpg":
        image_id = candidate.parent.name
        sampled_root = repo_root / "Sampled_Images_All"
        for extension in (".jpg", ".jpeg", ".png"):
            sampled_candidate = sampled_root / f"{image_id}{extension}"
            if _is_loadable_image(sampled_candidate):
                return sampled_candidate, {
                    "requested_path": str(candidate),
                    "resolved_path": str(sampled_candidate),
                    "source_kind": "sampled_images_lookup",
                }

    viz_candidate = candidate.parent / "visualization.jpg"
    if allow_visualization_fallback and _is_loadable_image(viz_candidate):
        return viz_candidate, {
            "requested_path": str(candidate),
            "resolved_path": str(viz_candidate),
            "source_kind": "visualization_fallback",
        }

    if pointer_text:
        raise FileNotFoundError(
            f"Could not resolve image pointer '{pointer_text}' from {candidate}"
        )
    raise FileNotFoundError(f"Could not load image asset from {candidate}")


def load_image_with_metadata(
    image_path: str | Path,
    batch_root: str | Path,
    repo_root: Optional[str | Path] = None,
    *,
    allow_visualization_fallback: bool = False,
) -> Tuple[Image.Image, Dict[str, str]]:
    resolved_path, metadata = resolve_image_asset(
        image_path,
        batch_root,
        repo_root=repo_root,
        allow_visualization_fallback=allow_visualization_fallback,
    )
    with Image.open(resolved_path) as image:
        image.load()
        return image.convert("RGB"), metadata


def load_mask(mask_path: str | Path, batch_root: str | Path) -> Image.Image:
    with Image.open(resolve_relative_path(mask_path, batch_root)) as mask:
        return mask.convert("L")


def load_masked_item_image(
    row: dict,
    batch_root: str | Path,
    repo_root: Optional[str | Path] = None,
) -> Image.Image:
    bbox = row.get("sam_bbox") or row.get("bbox") or row.get("qwen_bbox")

    if row.get("mask_path"):
        try:
            image = load_image_with_pointer_support(
                row["image_path"],
                batch_root,
                repo_root,
                allow_visualization_fallback=False,
            )
            mask = load_mask(row["mask_path"], batch_root)
            return masked_crop_from_pil(image, mask, bbox=bbox)
        except FileNotFoundError:
            pass

    if row.get("crop_path"):
        return load_image_with_pointer_support(row["crop_path"], batch_root, repo_root)

    if bbox:
        image = load_image_with_pointer_support(
            row["image_path"],
            batch_root,
            repo_root,
            allow_visualization_fallback=False,
        )
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        return image.crop((x1, y1, x2, y2))

    return load_image_with_pointer_support(
        row["image_path"],
        batch_root,
        repo_root,
        allow_visualization_fallback=False,
    )


def resize_and_pad_image(
    image: Image.Image,
    target_size: int,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, float, float, int, int]:
    width, height = image.size
    scale = min(target_size / max(width, 1), target_size / max(height, 1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), fill)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, scale, pad_x, pad_y


def resize_and_pad_mask(
    mask: Image.Image,
    target_size: int,
    scale_x: float,
    scale_y: float,
    pad_x: int,
    pad_y: int,
) -> Image.Image:
    width, height = mask.size
    new_w = max(1, int(round(width * scale_x)))
    new_h = max(1, int(round(height * scale_y)))
    resized = mask.resize((new_w, new_h), Image.NEAREST)
    canvas = Image.new("L", (target_size, target_size), 0)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def scale_bbox(box: Sequence[float], scale_x: float, scale_y: float, pad_x: int, pad_y: int) -> List[float]:
    x1, y1, x2, y2 = box
    return [
        x1 * scale_x + pad_x,
        y1 * scale_y + pad_y,
        x2 * scale_x + pad_x,
        y2 * scale_y + pad_y,
    ]


def choose_item_label(item: dict) -> str:
    for key in ("final_class", "label", "coarse_label", "class_name"):
        value = item.get(key)
        if value:
            return str(value)
    return "food_item"


def build_class_name_index(
    export_root: str | Path,
    *,
    classes: Optional[Sequence[dict]] = None,
    stage3_rows: Optional[Sequence[dict]] = None,
) -> tuple[List[str], Dict[str, int]]:
    export_root = Path(export_root)
    class_records = list(classes) if classes is not None else read_json(export_root / "classes.json")
    stage3_records = (
        list(stage3_rows)
        if stage3_rows is not None
        else read_jsonl(export_root / "stage3_item_classification.jsonl")
    )

    names_by_id: Dict[int, str] = {}
    for row in stage3_records:
        class_name = choose_item_label(row)
        class_id = row.get("class_id")
        if not class_name or class_id is None:
            continue
        names_by_id[int(class_id)] = str(class_name)

    inferred_offset = 1 if names_by_id and min(names_by_id) >= 1 else 0
    for index, entry in enumerate(class_records):
        class_name = entry.get("name")
        if not class_name:
            continue
        class_id = entry.get("class_id")
        if class_id is None:
            class_id = index + inferred_offset
        names_by_id.setdefault(int(class_id), str(class_name))

    if not names_by_id:
        return [], {}

    max_class_id = max(names_by_id)
    class_names = [f"class_{class_id}" for class_id in range(max_class_id + 1)]
    for class_id, class_name in names_by_id.items():
        class_names[class_id] = class_name
    class_name_to_id = {class_name: class_id for class_id, class_name in names_by_id.items()}
    return class_names, class_name_to_id


def filter_active_items(items: Sequence[dict]) -> List[dict]:
    filtered = []
    for item in items:
        if not item.get("bbox"):
            continue
        if item.get("active", True) is False:
            continue
        if item.get("excluded", False):
            continue
        filtered.append(item)
    return filtered


class Stage1ManifestDataset(Dataset):
    def __init__(self, batch_root: str | Path, export_root: Optional[str | Path] = None):
        self.paths = build_export_paths(batch_root, export_root=export_root)
        self.rows = read_jsonl(self.paths.export_root / "stage1_item_detection.jsonl")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = load_image_with_pointer_support(row["image_path"], self.paths.batch_root, self.paths.repo_root)
        return {"row": row, "image": image}


class Stage2ManifestDataset(Dataset):
    def __init__(self, batch_root: str | Path, export_root: Optional[str | Path] = None):
        self.paths = build_export_paths(batch_root, export_root=export_root)
        self.rows = read_jsonl(self.paths.export_root / "stage2_sam_segmentation.jsonl")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = load_image_with_pointer_support(row["image_path"], self.paths.batch_root, self.paths.repo_root)
        mask = load_mask(row["mask_path"], self.paths.batch_root)
        return {"row": row, "image": image, "mask": mask}


class Stage3ManifestDataset(Dataset):
    def __init__(self, batch_root: str | Path, export_root: Optional[str | Path] = None):
        self.paths = build_export_paths(batch_root, export_root=export_root)
        self.rows = read_jsonl(self.paths.export_root / "stage3_item_classification.jsonl")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        crop = load_masked_item_image(row, self.paths.batch_root, self.paths.repo_root)
        return {"row": row, "crop": crop}


class Stage3EpisodeLibrary:
    def __init__(
        self,
        support_rows: Sequence[dict],
        query_rows: Sequence[dict],
        batch_root: str | Path,
        repo_root: str | Path,
        num_classes: int,
        n_way: int,
        k_shot: int,
        query_per_class: int,
    ):
        self.support_rows = list(support_rows)
        self.query_rows = list(query_rows)
        self.batch_root = Path(batch_root)
        self.repo_root = Path(repo_root)
        self.num_classes = int(num_classes)
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
        self.support_by_class: Dict[str, List[dict]] = {}
        self.query_by_class: Dict[str, List[dict]] = {}
        self.class_counts_by_id: Dict[int, int] = {}
        self._crop_cache: Dict[str, Image.Image] = {}
        for row in self.support_rows:
            class_name = choose_item_label(row)
            if not class_name:
                continue
            self.support_by_class.setdefault(str(class_name), []).append(row)
            class_id = int(row.get("class_id", -1))
            if class_id >= 0:
                self.class_counts_by_id[class_id] = self.class_counts_by_id.get(class_id, 0) + 1
        for row in self.query_rows:
            class_name = choose_item_label(row)
            if not class_name:
                continue
            self.query_by_class.setdefault(str(class_name), []).append(row)

        shared_classes = set(self.support_by_class) & set(self.query_by_class)
        self.class_names = sorted(
            name for name in shared_classes
            if len(self.support_by_class.get(name, [])) >= 1 and len(self.query_by_class.get(name, [])) >= 1
        )
        self.n_way = min(self.n_way, len(self.class_names))
        if self.n_way <= 0:
            raise ValueError("No shared classes available for episodic sampling.")

    def _row_cache_key(self, row: dict) -> str:
        return json.dumps(
            {
                "image_id": row.get("image_id"),
                "class_id": row.get("class_id"),
                "mask_path": row.get("mask_path"),
                "crop_path": row.get("crop_path"),
                "sam_bbox": row.get("sam_bbox"),
                "bbox": row.get("bbox"),
                "qwen_bbox": row.get("qwen_bbox"),
            },
            sort_keys=True,
        )

    def _load_cached_crop(self, row: dict) -> Image.Image:
        cache_key = self._row_cache_key(row)
        cached = self._crop_cache.get(cache_key)
        if cached is None:
            cached = load_masked_item_image(row, self.batch_root, self.repo_root).copy()
            self._crop_cache[cache_key] = cached
        return cached.copy()

    def sample_support_episode(
        self,
        seed_key: str,
        required_class_names: Optional[Sequence[str]] = None,
    ) -> dict:
        seed_int = int(hashlib.sha1(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed_int)
        required = []
        for class_name in required_class_names or ():
            normalized = str(class_name)
            if normalized in self.class_names and normalized not in required:
                required.append(normalized)

        target_way = min(max(self.n_way, len(required)), len(self.class_names))
        remaining = [name for name in self.class_names if name not in required]
        selected = list(required)
        if len(selected) < target_way:
            selected.extend(rng.sample(remaining, target_way - len(selected)))

        support_images: List[Image.Image] = []
        support_labels: List[int] = []
        class_count_vector = [1.0] * max(self.num_classes, 1)

        for class_name in selected:
            support_candidates = list(self.support_by_class[class_name])
            if len(support_candidates) < self.k_shot:
                support_rows = [rng.choice(support_candidates) for _ in range(self.k_shot)]
            else:
                support_rows = rng.sample(support_candidates, self.k_shot)

            class_id = int(support_rows[0]["class_id"])
            class_count_vector[class_id] = float(self.class_counts_by_id.get(class_id, len(support_candidates)))

            for row in support_rows:
                support_images.append(self._load_cached_crop(row))
                support_labels.append(int(row["class_id"]))

        return {
            "support_images": support_images,
            "support_labels": support_labels,
            "class_names": selected,
            "class_counts": class_count_vector,
        }

    def sample_episode(self, seed_key: str) -> dict:
        support_episode = self.sample_support_episode(seed_key)
        seed_int = int(hashlib.sha1(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed_int)
        selected = support_episode["class_names"]
        query_images: List[Image.Image] = []
        query_labels: List[int] = []

        for class_name in selected:
            query_candidates = list(self.query_by_class[class_name])
            if len(query_candidates) < self.query_per_class:
                query_rows = [rng.choice(query_candidates) for _ in range(self.query_per_class)]
            else:
                query_rows = rng.sample(query_candidates, self.query_per_class)
            for row in query_rows:
                query_images.append(self._load_cached_crop(row))
                query_labels.append(int(row["class_id"]))

        return {
            "support_images": support_episode["support_images"],
            "query_images": query_images,
            "support_labels": support_episode["support_labels"],
            "query_labels": query_labels,
            "class_names": selected,
            "class_counts": support_episode["class_counts"],
        }


# --- Snapshot note: Primary dataset used for joint training, evaluation, and episodic Stage 3 support generation. ---
class JointFoodDataset(Dataset):
    """
    Image-level dataset based on `images_manifest.jsonl` plus Stage 3 episodic rows.
    """

    def __init__(
        self,
        batch_root: str | Path,
        export_root: Optional[str | Path] = None,
        repo_root: Optional[str | Path] = None,
        split: str = "train",
        image_size: int = 640,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 1337,
        n_way: int = 5,
        k_shot: int = 5,
        query_per_class: int = 1,
        episode_support_split: Optional[str] = None,
    ):
        self.paths = build_export_paths(batch_root, export_root=export_root, repo_root=repo_root)
        self.split = normalize_split_name(split)
        self.episode_support_split = normalize_split_name(episode_support_split or self.split)
        self.image_size = image_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed

        image_rows = read_jsonl(self.paths.export_root / "images_manifest.jsonl")
        all_stage3_rows = read_jsonl(self.paths.export_root / "stage3_item_classification.jsonl")
        labeled_images = [
            row for row in image_rows
            if any(item.get("classification_status") == "labeled" for item in row.get("items", []))
        ]
        contract = enforce_supported_class_contract(
            labeled_images,
            all_stage3_rows,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.split_seed,
        )
        self.split_mapping = contract["split_mapping"]
        self.split_summary = contract["split_summary"]
        self.stage3_split_summary = contract["stage3_split_summary"]
        self.supported_classes = contract["supported_classes"]
        self.removed_classes = contract["removed_classes"]
        self.supported_image_ids = contract["retained_image_ids"]

        self.image_rows = [
            row for row in contract["image_rows"]
            if normalize_split_name(self.split_mapping.get(str(row["image_id"]))) == self.split
            and row.get("use_for_export", True)
        ]
        self._image_id_to_split = self.split_mapping
        self.classes = read_json(self.paths.export_root / "classes.json")
        self.class_names, self.class_name_to_id = build_class_name_index(
            self.paths.export_root,
            classes=self.classes,
            stage3_rows=all_stage3_rows,
        )
        self.stage3_rows = [
            row for row in contract["stage3_rows"]
            if normalize_split_name(self._image_id_to_split.get(str(row["image_id"]))) == self.split
        ]
        self.stage3_support_rows = [
            row for row in contract["stage3_rows"]
            if normalize_split_name(self._image_id_to_split.get(str(row["image_id"]))) == self.episode_support_split
        ]
        self.supported_class_ids = sorted({int(row["class_id"]) for row in contract["stage3_rows"]})
        self.stage3_library = Stage3EpisodeLibrary(
            support_rows=self.stage3_support_rows,
            query_rows=self.stage3_rows,
            batch_root=self.paths.batch_root,
            repo_root=self.paths.repo_root,
            num_classes=max(len(self.class_names), len(self.classes)),
            n_way=n_way,
            k_shot=k_shot,
            query_per_class=query_per_class,
        )
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class

    def _row_split(self, row: dict) -> str:
        return normalize_split_name(self.split_mapping.get(str(row["image_id"]), "train"))

    def __len__(self):
        return len(self.image_rows)

    def __getitem__(self, index: int):
        row = self.image_rows[index]
        image, image_source = load_image_with_metadata(
            row["image_path"],
            self.paths.batch_root,
            self.paths.repo_root,
        )
        resized, scale_x, scale_y, pad_x, pad_y = resize_and_pad_image(image, self.image_size)

        items = filter_active_items(row.get("items", []))
        stage1_items: List[dict] = []
        stage3_query_labels: List[int] = []
        stage3_support_episodes: List[Optional[dict]] = []
        boxes: List[List[float]] = []
        masks: List[Optional[torch.Tensor]] = []
        support_cache: Dict[str, Optional[dict]] = {}

        for item in items:
            bbox = item.get("bbox")
            if not bbox:
                continue
            label_name = choose_item_label(item)
            scaled_box = scale_bbox(bbox, scale_x, scale_y, pad_x, pad_y)
            boxes.append(scaled_box)
            stage1_items.append({"box": scaled_box, "label": label_name})
            class_id = int(self.class_name_to_id.get(label_name, -1))
            stage3_query_labels.append(class_id)

            support_episode = support_cache.get(label_name)
            if (
                support_episode is None
                and label_name not in support_cache
                and class_id >= 0
                and label_name in self.stage3_library.class_names
            ):
                support_episode = self.stage3_library.sample_support_episode(
                    f"{self.split}:{row['image_id']}:{index}:{label_name}",
                    required_class_names=[label_name],
                )
                support_cache[label_name] = support_episode
            stage3_support_episodes.append(support_episode)

            mask_tensor: Optional[torch.Tensor] = None
            if item.get("mask_path"):
                mask_img = load_mask(item["mask_path"], self.paths.batch_root)
                mask_img = resize_and_pad_mask(mask_img, self.image_size, scale_x, scale_y, pad_x, pad_y)
                mask_tensor = (TF.pil_to_tensor(mask_img).float() / 255.0).squeeze(0)
            masks.append(mask_tensor)

        image_tensor = TF.pil_to_tensor(resized).float() / 255.0

        return {
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "resolved_image_path": image_source["resolved_path"],
            "image_source_kind": image_source["source_kind"],
            "review_status": row.get("review_status"),
            "notes": row.get("notes"),
            "pil_image": resized,
            "image_tensor": image_tensor,
            "stage1_items": stage1_items,
            "stage3_query_labels": stage3_query_labels,
            "stage3_support_episodes": stage3_support_episodes,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "masks": masks,
        }


# --- Snapshot note: Batch collation logic that prepares Stage 1 supervision and Stage 3 episodes. ---
class JointBatchCollator:
    def __init__(
        self,
        stage1_processor,
        stage1_prompt: str,
        stage1_label_pad_token_id: int = -100,
    ):
        self.stage1_processor = stage1_processor
        self.stage1_prompt = stage1_prompt
        self.stage1_label_pad_token_id = stage1_label_pad_token_id

    def __call__(self, examples: Sequence[dict]) -> dict:
        pil_images = [example["pil_image"] for example in examples]
        image_tensors = torch.stack([example["image_tensor"] for example in examples], dim=0)
        boxes = [example["boxes"] for example in examples]
        masks = [example["masks"] for example in examples]

        stage1_batch = self._collate_stage1(examples, pil_images)
        stage3_batch = self._collate_stage3(examples)

        batch = {
            "image_ids": [example["image_id"] for example in examples],
            "review_status": [example["review_status"] for example in examples],
            "notes": [example["notes"] for example in examples],
            "pil_images": list(pil_images),
            "stage1_items": [example["stage1_items"] for example in examples],
            "images": image_tensors,
            "boxes": boxes,
            "masks": masks,
            **stage1_batch,
            **stage3_batch,
        }
        return batch

    def _collate_stage1(self, examples: Sequence[dict], pil_images: Sequence[Image.Image]) -> dict:
        prompt_messages = []
        full_messages = []
        for example, image in zip(examples, pil_images):
            target = json.dumps(example["stage1_items"], ensure_ascii=True)
            user_message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.stage1_prompt},
                ],
            }]
            full_message = user_message + [{
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            }]
            prompt_messages.append(
                self.stage1_processor.apply_chat_template(
                    user_message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_messages.append(
                self.stage1_processor.apply_chat_template(
                    full_message,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        prompt_inputs = self.stage1_processor(
            text=prompt_messages,
            images=list(pil_images),
            return_tensors="pt",
            padding=True,
        )
        full_inputs = self.stage1_processor(
            text=full_messages,
            images=list(pil_images),
            return_tensors="pt",
            padding=True,
        )

        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[row_index, :prompt_length] = self.stage1_label_pad_token_id
        labels[full_inputs["attention_mask"] == 0] = self.stage1_label_pad_token_id

        return {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs["attention_mask"],
            "pixel_values": full_inputs.get("pixel_values"),
            "image_grid_thw": full_inputs.get("image_grid_thw"),
            "video_grid_thw": full_inputs.get("video_grid_thw"),
            "s1_labels": labels,
        }

    def _collate_stage3(self, examples: Sequence[dict]) -> dict:
        if not examples:
            return {}

        flat_support: List[Image.Image] = []
        flat_query: List[Image.Image] = []
        flat_support_labels: List[int] = []
        query_labels: List[int] = []
        class_counts: List[List[int]] = []
        support_sizes: List[int] = []

        for example in examples:
            query_class_ids = list(example.get("stage3_query_labels", []))
            support_episodes = list(example.get("stage3_support_episodes", []))
            stage1_items = list(example.get("stage1_items", []))
            stage1_masks = list(example.get("masks", []))
            pil_image = example["pil_image"]

            if not support_episodes or not stage1_items:
                continue

            for query_index, class_id in enumerate(query_class_ids):
                if class_id < 0 or query_index >= len(stage1_items) or query_index >= len(support_episodes):
                    continue
                episode = support_episodes[query_index]
                if not episode:
                    continue
                support_images = episode["support_images"]
                support_labels = episode["support_labels"]
                if not support_images or class_id not in support_labels:
                    continue
                box = stage1_items[query_index]["box"]
                mask = stage1_masks[query_index] if query_index < len(stage1_masks) else None
                crop = (
                    masked_crop_from_pil(pil_image, mask, box)
                    if mask is not None
                    else pil_image.crop(tuple(int(round(v)) for v in box))
                )
                flat_support.extend(support_images)
                flat_support_labels.extend(support_labels)
                flat_query.append(crop)
                query_labels.append(int(class_id))
                class_counts.append(list(episode["class_counts"]))
                support_sizes.append(len(support_images))

        if not support_sizes:
            return {}

        batch_size = len(query_labels)
        support_per_episode = support_sizes[0]
        support_images = pil_images_to_tensor(flat_support).reshape(batch_size, support_per_episode, -1, 224, 224)
        query_images = pil_images_to_tensor(flat_query).reshape(batch_size, 1, -1, 224, 224)
        support_labels_tensor = torch.tensor(flat_support_labels, dtype=torch.long).reshape(batch_size, support_per_episode)
        query_labels_tensor = torch.tensor(query_labels, dtype=torch.long)

        return {
            "support_images": support_images,
            "query_images": query_images,
            "support_labels": support_labels_tensor,
            "query_labels": query_labels_tensor,
            "episode_class_counts": torch.tensor(class_counts, dtype=torch.float32),
        }


def snapshot_export_contract(
    batch_root: str | Path,
    snapshot_root: str | Path,
    export_root: Optional[str | Path] = None,
    include_assets: bool = True,
):
    """
    Create a reproducible copy of the exported manifests and, optionally, all
    referenced images/masks/crops used by the manifests.
    """
    paths = build_export_paths(batch_root, export_root=export_root)
    snapshot_root = Path(snapshot_root).resolve()
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_export_root = snapshot_root / "dataset"
    snapshot_export_root.mkdir(parents=True, exist_ok=True)

    manifest_names = [
        "images_manifest.jsonl",
        "stage1_item_detection.jsonl",
        "stage1_qwen_detection.jsonl",
        "stage2_sam_segmentation.jsonl",
        "stage3_item_classification.jsonl",
        "classes.json",
        "summary.json",
    ]

    for name in manifest_names:
        src = paths.export_root / name
        if src.exists():
            shutil.copy2(src, snapshot_export_root / name)

    if not include_assets:
        return

    referenced_paths = set()
    for manifest_name in manifest_names:
        if not manifest_name.endswith(".jsonl"):
            continue
        manifest_path = paths.export_root / manifest_name
        if not manifest_path.exists():
            continue
        for row in read_jsonl(manifest_path):
            for key in ("image_path", "mask_path", "crop_path"):
                if row.get(key):
                    referenced_paths.add(row[key])
            for item in row.get("items", []):
                for key in ("image_path", "mask_path", "crop_path"):
                    if item.get(key):
                        referenced_paths.add(item[key])

    for rel_path in sorted(referenced_paths):
        source = resolve_relative_path(rel_path, paths.batch_root)
        if not source.exists():
            continue
        destination = snapshot_root / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if str(rel_path).endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            try:
                with Image.open(source) as image:
                    image.load()
                shutil.copy2(source, destination)
                continue
            except (UnidentifiedImageError, OSError):
                pointer_text = source.read_text(encoding="utf-8").strip()
                resolved = resolve_pointer_path(pointer_text, paths.repo_root)
                if resolved is not None and resolved.exists():
                    shutil.copy2(resolved, destination)
                    continue
        shutil.copy2(source, destination)
