from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .bbox_canonical import load_image_at_mask_resolution
from .config import STAGE1_PROMPT, Stage1Config, Stage1KCFDConfig
from .qwen_io import build_assistant_conversation, build_user_conversation, processor_batch
from .schema import Stage1Item, Stage1Target, normalize_text, target_to_json, target_to_payload


SPLIT_METHOD = "stage1_image_level_stratified_class_item_balance_v2"
BBOX_MINOR_OUT_OF_FRAME_TOLERANCE_PX = 2.0


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_stage1_export(export_root: str | Path, *, expected_version: str | None = "v3", expected_hash: str | None = None) -> dict:
    root = Path(export_root)
    manifest_path = root / "manifest.json"
    items_path = root / "items.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing v3 manifest: {manifest_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"missing v3 items file: {items_path}")
    manifest = read_json(manifest_path)
    version = str(manifest.get("version", ""))
    dataset_hash = str(manifest.get("content_hash_sha8", "") or "")
    if expected_version and version != expected_version:
        raise ValueError(f"dataset version mismatch: manifest={version!r}, expected={expected_version!r}")
    if expected_hash and dataset_hash != expected_hash:
        raise ValueError(f"dataset hash mismatch: manifest={dataset_hash!r}, expected={expected_hash!r}")
    return manifest


@dataclass(frozen=True)
class Stage1ImageSample:
    image_id: str
    image_path: str
    items: List[dict]


def _is_reference(row: dict) -> bool:
    return bool(row.get("is_reference"))


def _name_for_item(row: dict) -> str:
    return normalize_text(row.get("name"))


def _descriptor_for_item(row: dict) -> str:
    return normalize_text(row.get("description") or row.get("vlm_description"))


def _bbox_frame_stats(root: Path, rows: Sequence[dict]) -> Dict[str, int]:
    stats = {
        "bbox_checked_count": 0,
        "invalid_bbox_count": 0,
        "bbox_nonpositive_count": 0,
        "bbox_out_of_frame_count": 0,
        "bbox_out_of_frame_minor_count": 0,
        "bbox_out_of_frame_major_count": 0,
        "bbox_frame_from_mask_count": 0,
        "bbox_frame_from_image_count": 0,
        "image_mask_size_mismatch_count": 0,
    }
    size_cache: Dict[str, tuple[int, int] | None] = {}

    def image_size(path: Path) -> tuple[int, int] | None:
        key = str(path)
        if key not in size_cache:
            if not path.exists():
                size_cache[key] = None
            else:
                with Image.open(path) as image:
                    size_cache[key] = image.size
        return size_cache[key]

    for row in rows:
        raw_bbox = row.get("bbox")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            stats["invalid_bbox_count"] += 1
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in raw_bbox]
        except Exception:
            stats["invalid_bbox_count"] += 1
            continue
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            stats["invalid_bbox_count"] += 1
            continue

        stats["bbox_checked_count"] += 1
        if x2 <= x1 or y2 <= y1:
            stats["bbox_nonpositive_count"] += 1
            continue

        image_dims = image_size(root / str(row.get("image_path", "")))
        mask_dims = image_size(root / str(row.get("mask_path", "")))
        frame_dims = mask_dims or image_dims
        if mask_dims is not None:
            stats["bbox_frame_from_mask_count"] += 1
        elif image_dims is not None:
            stats["bbox_frame_from_image_count"] += 1
        if image_dims is not None and mask_dims is not None and image_dims != mask_dims:
            stats["image_mask_size_mismatch_count"] += 1
        if frame_dims is None:
            continue
        width, height = frame_dims
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            stats["bbox_out_of_frame_count"] += 1
            max_excess = max(-x1, -y1, x2 - width, y2 - height)
            if max_excess <= BBOX_MINOR_OUT_OF_FRAME_TOLERANCE_PX:
                stats["bbox_out_of_frame_minor_count"] += 1
            else:
                stats["bbox_out_of_frame_major_count"] += 1
    return stats


def _bbox_center(row: dict) -> tuple[float, float, int]:
    bbox = row.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((y1 + y2) / 2.0, (x1 + x2) / 2.0, int(row.get("src_item_index", 0) or 0))


def _clamp_bbox_to_frame(bbox: Sequence[Any], frame_size: tuple[int, int] | None) -> List[float]:
    values = [float(v) for v in bbox]
    if frame_size is None:
        return values
    width, height = frame_size
    x1, y1, x2, y2 = values
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def _format_bbox_for_json(bbox: Sequence[float]) -> List[int | float]:
    return [
        int(round(float(v))) if abs(float(v) - round(float(v))) < 1e-6 else round(float(v), 3)
        for v in bbox
    ]


def _target_from_items(items: Sequence[dict], *, frame_size: tuple[int, int] | None = None) -> Stage1Target:
    target_items: List[Stage1Item] = []
    for row in sorted(items, key=_bbox_center):
        name = _name_for_item(row)
        descriptor = _descriptor_for_item(row)
        if not name or not descriptor:
            continue
        bbox = _clamp_bbox_to_frame(row["bbox"], frame_size)
        target_items.append(
            Stage1Item(
                name=name,
                bbox=_format_bbox_for_json(bbox),
                descriptor=descriptor,
            )
        )
    return Stage1Target(items=target_items)


def preflight_stage1_kcfd_export(
    export_root: str | Path,
    *,
    expected_version: str | None = None,
    expected_hash: str | None = None,
) -> Dict[str, int | str]:
    root = Path(export_root)
    manifest = validate_stage1_export(root, expected_version=expected_version, expected_hash=expected_hash)
    rows = read_jsonl(root / "items.jsonl")
    image_ids = {str(row["src_image_id"]) for row in rows}
    reference_rows = [row for row in rows if _is_reference(row)]
    report = {
        "export_root": str(root),
        "dataset_version": str(manifest.get("version", "")),
        "dataset_hash": str(manifest.get("content_hash_sha8", "")),
        "total_items": len(rows),
        "total_images": len(image_ids),
        "reference_items": len(reference_rows),
        "reference_count": len(reference_rows),
        "reference_images": len({str(row["src_image_id"]) for row in reference_rows}),
        "missing_name_count": sum(1 for row in rows if not _name_for_item(row)),
        "missing_descriptor_count": sum(1 for row in rows if not _descriptor_for_item(row)),
        "missing_mask_count": sum(1 for row in rows if not (root / str(row.get("mask_path", ""))).exists()),
        "missing_image_count": sum(1 for image_id in image_ids if not any((root / row["image_path"]).exists() for row in rows if str(row["src_image_id"]) == image_id)),
        "bbox_coordinate_mode": "mask_native",
    }
    report.update(_bbox_frame_stats(root, rows))
    return report


def incomplete_export_counts(stats: Dict[str, Any]) -> Dict[str, int]:
    bad_keys = [
        "missing_image_count",
        "missing_mask_count",
        "missing_name_count",
        "missing_descriptor_count",
        "invalid_bbox_count",
        "bbox_nonpositive_count",
        "bbox_out_of_frame_major_count",
    ]
    return {key: int(stats.get(key, 0) or 0) for key in bad_keys if int(stats.get(key, 0) or 0) > 0}


def _group_items(rows: Sequence[dict], reference_policy: str) -> List[Stage1ImageSample]:
    grouped: Dict[str, List[dict]] = {}
    reference_image_ids = {str(row["src_image_id"]) for row in rows if _is_reference(row)}
    if reference_policy == "pause" and reference_image_ids:
        raise ValueError(
            "reference_policy='pause' cannot construct a training dataset when reference items exist. "
            "Use preflight first, then choose reference_policy='exclude', 'train', or 'include'."
        )
    for row in rows:
        image_id = str(row["src_image_id"])
        if reference_policy == "exclude" and image_id in reference_image_ids:
            continue
        grouped.setdefault(image_id, []).append(row)
    samples: List[Stage1ImageSample] = []
    for image_id, items in sorted(grouped.items()):
        if not items:
            continue
        first = sorted(items, key=lambda item: int(item.get("src_item_index", 0) or 0))[0]
        samples.append(Stage1ImageSample(image_id=image_id, image_path=str(first["image_path"]), items=list(items)))
    return samples


def _split_labels_from_rows(samples: Sequence[Stage1ImageSample]) -> Dict[str, set[str]]:
    labels: Dict[str, set[str]] = {}
    for sample in samples:
        labels[sample.image_id] = {
            str(row.get("class_slug") or row.get("class_display_name") or row.get("name") or "food")
            for row in sample.items
        }
        labels[sample.image_id].discard("")
    return labels


def _sample_item_counts(samples: Sequence[Stage1ImageSample]) -> Dict[str, int]:
    return {sample.image_id: len(sample.items) for sample in samples}


def _allocate_split_sizes(n: int, config: Stage1Config) -> Dict[str, int]:
    ratio_sum = config.train_ratio + config.val_ratio + config.test_ratio
    raw = {
        "train": n * config.train_ratio / ratio_sum,
        "dev": n * config.val_ratio / ratio_sum,
        "test": n * config.test_ratio / ratio_sum,
    }
    sizes = {key: int(raw[key]) for key in raw}
    remainder = n - sum(sizes.values())
    fractions = sorted(((raw[key] - sizes[key], key) for key in sizes), reverse=True)
    for _, key in fractions[:remainder]:
        sizes[key] += 1
    return sizes


def _compute_stratified_image_split(samples: Sequence[Stage1ImageSample], config: Stage1Config) -> Dict[str, str]:
    labels_by_image = _split_labels_from_rows(samples)
    item_counts = _sample_item_counts(samples)
    ids = sorted(labels_by_image)
    rng = random.Random(config.split_seed)
    rng.shuffle(ids)
    target_sizes = _allocate_split_sizes(len(ids), config)
    total_items = sum(item_counts.values())
    ratio_sum = config.train_ratio + config.val_ratio + config.test_ratio
    target_items = {
        "train": total_items * config.train_ratio / ratio_sum,
        "dev": total_items * config.val_ratio / ratio_sum,
        "test": total_items * config.test_ratio / ratio_sum,
    }
    split_counts = {"train": 0, "dev": 0, "test": 0}
    split_item_counts = {"train": 0, "dev": 0, "test": 0}
    split_label_counts: Dict[str, Dict[str, int]] = {"train": {}, "dev": {}, "test": {}}
    assignments: Dict[str, str] = {}
    label_to_images: Dict[str, List[str]] = {}
    for image_id, labels in labels_by_image.items():
        for label in labels:
            label_to_images.setdefault(label, []).append(image_id)

    def assign(image_id: str, split: str) -> None:
        if image_id in assignments:
            return
        assignments[image_id] = split
        split_counts[split] += 1
        split_item_counts[split] += item_counts.get(image_id, 0)
        for label in labels_by_image.get(image_id, set()):
            split_label_counts[split][label] = split_label_counts[split].get(label, 0) + 1

    target_label_counts = {
        split: {
            label: len(images) * (
                config.train_ratio if split == "train" else config.val_ratio if split == "dev" else config.test_ratio
            ) / ratio_sum
            for label, images in label_to_images.items()
        }
        for split in ("train", "dev", "test")
    }

    for split in ("train", "dev", "test"):
        if target_sizes[split] <= 0:
            continue
        for label in sorted(label_to_images, key=lambda value: (len(label_to_images[value]), value)):
            if split_counts[split] >= target_sizes[split]:
                break
            if split_label_counts[split].get(label, 0) > 0:
                continue
            candidates = [image_id for image_id in sorted(label_to_images[label]) if image_id not in assignments]
            if candidates:
                assign(candidates[0], split)

    remaining = [image_id for image_id in ids if image_id not in assignments]
    remaining.sort(key=lambda image_id: (
        -sum(1.0 / max(len(label_to_images[label]), 1) for label in labels_by_image.get(image_id, set())),
        -len(labels_by_image.get(image_id, set())),
        -item_counts.get(image_id, 0),
        image_id,
    ))

    def split_score(image_id: str, split: str) -> tuple[float, float, float, float, str]:
        size_deficit = (target_sizes[split] - split_counts[split]) / max(target_sizes[split], 1)
        item_deficit = (target_items[split] - split_item_counts[split]) / max(target_items[split], 1.0)
        label_deficit = 0.0
        for label in labels_by_image.get(image_id, set()):
            desired = target_label_counts[split].get(label, 0.0)
            current = split_label_counts[split].get(label, 0)
            label_deficit += max(0.0, desired - current)
        eval_preference = 0.1 if split in {"dev", "test"} else 0.0
        return (label_deficit, item_deficit, size_deficit, eval_preference, split)

    for image_id in remaining:
        if image_id in assignments:
            continue
        candidates = [split for split in ("train", "dev", "test") if split_counts[split] < target_sizes[split]]
        if candidates:
            assign(image_id, max(candidates, key=lambda split: split_score(image_id, split)))
        else:
            assign(image_id, "train")
    return assignments


def _summarize_splits(samples: Sequence[Stage1ImageSample], split_mapping: Dict[str, str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        split: {"images": 0, "items": 0, "class_image_counts": {}, "class_item_counts": {}}
        for split in ("train", "dev", "test")
    }
    for sample in samples:
        split = split_mapping.get(sample.image_id, "train")
        bucket = summary.setdefault(split, {"images": 0, "items": 0, "class_image_counts": {}, "class_item_counts": {}})
        bucket["images"] += 1
        bucket["items"] += len(sample.items)
        seen = set()
        for row in sample.items:
            label = str(row.get("class_slug") or row.get("class_display_name") or row.get("name") or "food")
            bucket["class_item_counts"][label] = bucket["class_item_counts"].get(label, 0) + 1
            seen.add(label)
        for label in seen:
            bucket["class_image_counts"][label] = bucket["class_image_counts"].get(label, 0) + 1
    return summary


def _default_splits_path(config: Stage1Config, manifest: dict) -> Path:
    if config.splits_path is not None:
        return config.splits_path
    dataset_hash = str(manifest.get("content_hash_sha8", "") or config.export_root.name)
    filename = f"stage1_splits_{dataset_hash}_{config.split_seed}_{config.train_ratio:.4f}_{config.val_ratio:.4f}_{config.test_ratio:.4f}_{config.reference_policy}.json"
    primary = config.export_root / filename
    try:
        probe = config.export_root / ".stage1_splits_writable_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return primary
    except OSError:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
        return cache_root / "trifoodnet" / "stage1_splits" / filename


def _load_or_compute_split_ids(samples: Sequence[Stage1ImageSample], config: Stage1Config, manifest: dict) -> Dict[str, str]:
    path = _default_splits_path(config, manifest)
    dataset_hash = str(manifest.get("content_hash_sha8", "") or "")
    expected = {
        "dataset_hash": dataset_hash,
        "dataset_version": str(manifest.get("version", "")),
        "seed": config.split_seed,
        "ratios": {"train": config.train_ratio, "dev": config.val_ratio, "test": config.test_ratio},
        "reference_policy": config.reference_policy,
        "method": SPLIT_METHOD,
    }
    current_ids = {sample.image_id for sample in samples}
    if path.exists():
        payload = read_json(path)
        mapping = {str(key): str(value) for key, value in (payload.get("split_mapping") or {}).items()}
        ratios = payload.get("ratios") or {}
        ratios_match = all(abs(float(ratios.get(key, -1)) - expected["ratios"][key]) < 1e-9 for key in expected["ratios"])
        metadata_match = (
            str(payload.get("dataset_hash", "")) == expected["dataset_hash"]
            and str(payload.get("dataset_version", "")) == expected["dataset_version"]
            and int(payload.get("seed", -1)) == expected["seed"]
            and str(payload.get("reference_policy", "")) == expected["reference_policy"]
            and str(payload.get("method", "")) == SPLIT_METHOD
            and ratios_match
        )
        if metadata_match and current_ids.issubset(set(mapping)) and all(value in {"train", "dev", "test"} for value in mapping.values()):
            return mapping

    mapping = _compute_stratified_image_split(samples, config)
    payload = {
        **expected,
        "method": SPLIT_METHOD,
        "n_train": sum(1 for value in mapping.values() if value == "train"),
        "n_dev": sum(1 for value in mapping.values() if value == "dev"),
        "n_test": sum(1 for value in mapping.values() if value == "test"),
        "split_mapping": mapping,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return mapping


def _split_image_ids(samples: Sequence[Stage1ImageSample], config: Stage1Config, manifest: dict) -> Dict[str, str]:
    return _load_or_compute_split_ids(samples, config, manifest)


class Stage1KCFDDataset(Dataset):
    def __init__(self, config: Stage1KCFDConfig):
        self.config = config
        self.export_root = Path(config.export_root)
        self.manifest = validate_stage1_export(
            self.export_root,
            expected_version=config.expected_version,
            expected_hash=config.expected_hash,
        )
        self.rows = read_jsonl(self.export_root / "items.jsonl")
        all_samples = _group_items(self.rows, config.reference_policy)
        stats = preflight_stage1_kcfd_export(
            self.export_root,
            expected_version=config.expected_version,
            expected_hash=config.expected_hash,
        )
        bad = incomplete_export_counts(stats)
        if bad and not config.allow_incomplete_export:
            raise ValueError(f"export has incomplete Stage 1 training data: {bad}")
        split_mapping = _split_image_ids(all_samples, config, self.manifest)
        if config.reference_policy == "train":
            ref_images = {str(row["src_image_id"]) for row in self.rows if _is_reference(row)}
            for image_id in ref_images:
                if image_id in split_mapping:
                    split_mapping[image_id] = "train"
        requested = "dev" if config.split in {"dev", "val"} else config.split
        self.samples = [sample for sample in all_samples if split_mapping.get(sample.image_id) == requested]
        if requested == "train" and config.train_max_images > 0:
            self.samples = self.samples[:config.train_max_images]
        self.split_mapping = split_mapping
        self.all_samples = all_samples
        self.image_ids = [sample.image_id for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        first = sorted(sample.items, key=lambda item: int(item.get("src_item_index", 0) or 0))[0]
        image = load_image_at_mask_resolution(first, self.export_root)
        target = _target_from_items(sample.items, frame_size=image.size)
        return {
            "image": image,
            "image_id": sample.image_id,
            "src_image_id": sample.image_id,
            "image_path": str(self.export_root / sample.image_path),
            "raw_items": sample.items,
            "target": target_to_payload(target),
            "target_json": target_to_json(target),
            "items": target.items,
        }


class Stage1Collator:
    def __init__(self, processor, prompt: str = STAGE1_PROMPT):
        self.processor = processor
        self.prompt = prompt

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images = [example["image"] for example in examples]
        prompt_conversations = []
        full_conversations = []
        for example in examples:
            prompt_conversations.append(build_user_conversation(example["image"], self.prompt))
            full_conversations.append(build_assistant_conversation(example["image"], self.prompt, example["target_json"]))
        prompt_inputs = processor_batch(
            self.processor,
            prompt_conversations,
            add_generation_prompt=True,
            fallback_images=images,
        )
        full_inputs = processor_batch(
            self.processor,
            full_conversations,
            add_generation_prompt=False,
            fallback_images=images,
        )
        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        full_attention = full_inputs["attention_mask"]
        for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
            non_pad = torch.nonzero(full_attention[row_index], as_tuple=False).flatten()
            start = int(non_pad[0].item()) if len(non_pad) else 0
            labels[row_index, start:start + int(prompt_length)] = -100
        labels[full_inputs["attention_mask"] == 0] = -100
        full_inputs["labels"] = labels
        full_inputs["metadata"] = [{"image_id": example["image_id"], "target_json": example["target_json"]} for example in examples]
        return dict(full_inputs)


@dataclass
class Stage1DatasetBundle:
    train: Stage1KCFDDataset
    val: Stage1KCFDDataset
    test: Stage1KCFDDataset
    report: Dict[str, Any]
    serializable_config: Dict[str, Any]

    def make_collator(self, processor) -> Stage1Collator:
        return Stage1Collator(processor, prompt=str(self.serializable_config.get("prompt", STAGE1_PROMPT)))


def build_datasets_from_config(config: Stage1Config) -> Stage1DatasetBundle:
    train_cfg = Stage1KCFDConfig(**{**asdict(config), "split": "train"})
    val_cfg = Stage1KCFDConfig(**{**asdict(config), "split": "val"})
    test_cfg = Stage1KCFDConfig(**{**asdict(config), "split": "test"})
    train = Stage1KCFDDataset(train_cfg)
    val = Stage1KCFDDataset(val_cfg)
    test = Stage1KCFDDataset(test_cfg)
    report = dict(
        preflight_stage1_kcfd_export(
            config.export_root,
            expected_version=config.expected_version,
            expected_hash=config.expected_hash,
        )
    )
    report.update({
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
        "train_items": sum(len(sample.items) for sample in train.samples),
        "val_items": sum(len(sample.items) for sample in val.samples),
        "test_items": sum(len(sample.items) for sample in test.samples),
        "split_leakage": int(bool(set(train.image_ids) & set(val.image_ids) or set(train.image_ids) & set(test.image_ids) or set(val.image_ids) & set(test.image_ids))),
        "split_method": SPLIT_METHOD,
        "split_summary": _summarize_splits(train.all_samples, train.split_mapping),
        "reference_policy_applied": config.reference_policy,
    })
    serializable = asdict(config)
    for key in ("export_root", "output_dir", "splits_path"):
        if serializable.get(key) is not None:
            serializable[key] = str(serializable[key])
    return Stage1DatasetBundle(
        train=train,
        val=val,
        test=test,
        report=report,
        serializable_config=serializable,
    )
