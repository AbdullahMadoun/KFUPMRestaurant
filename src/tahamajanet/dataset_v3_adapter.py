# =============================================================================
# FILE: dataset_v3_adapter.py
# CATEGORY: DATA
# PURPOSE: Read the v3_export dataset format (manifest.json + items.jsonl + classes.json)
#          and emit dicts in the legacy shape JointFoodDataset historically consumed.
# DEPENDENCIES: dataset_integration.py (uses stratified_split, summarize_*)
# USED BY: dataset_integration.py (JointFoodDataset accepts an adapter), train_joint.py
# KEY CLASSES/FUNCTIONS: AdapterData, V3ExportAdapter
# =============================================================================
"""
v3_export → legacy-contract translator.

Schema produced by the v3 export (see Resturant_Pipeline_Feb/results_v2/exports/):
    manifest.json   { version, content_hash_sha8, counts, schema, ... }
    items.jsonl     one row per (src_image_id, src_item_index) with:
                      sample_id, class_id, compact_id, class_slug, class_display_name,
                      src_image_id, src_item_index, name, description,
                      vlm_label, vlm_canonical_label, vlm_description,
                      bbox, sam_score, image_width, image_height,
                      centrality_rank, is_reference,
                      image_path, crop_path, mask_path
    classes.json    [ { id, compact_id, slug, display_name, size, name_distribution } ]
    images/         flat <src_image_id>.jpg
    crops/          <src_image_id>/item_NNN.jpg
    masks/          <src_image_id>/item_NNN_mask.png
    reference/      <NNN_slug>/{ ref_001.jpg, ..., _centroid.jpg }

Legacy contract consumed by JointFoodDataset:
    image_rows      [ { image_id, image_path, image_width, image_height,
                        items: [ { bbox, mask_path, crop_path, classification_status,
                                   class_id, item_index, final_class, ... }, ... ],
                        use_for_export: True, review_status, ... } ]
    stage3_rows     [ { image_id, class_id, mask_path, crop_path, final_class, bbox, ... } ]
    classes         [ { name, source, usage_count } ]   (legacy-shaped)
    splits          { sample_id_or_image_id: "train"|"dev"|"test" }

Splits are computed on first read via stratified_split and persisted to
<export_root>/splits.json so subsequent runs reuse the same assignment.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AdapterData:
    """Output produced by a dataset adapter."""

    image_rows: List[dict]
    stage3_rows: List[dict]
    classes: List[dict]
    split_mapping: Dict[str, str]
    split_summary: Dict[str, dict]
    stage3_split_summary: Dict[str, dict]
    supported_classes: List[str]
    removed_classes: List[str]
    retained_image_ids: List[str]
    dataset_version: str
    dataset_hash: str
    export_root: Path
    classes_compact_ids: List[int] = field(default_factory=list)


class V3ExportAdapter:
    """Adapter for the v3_export dataset layout.

    Parameters
    ----------
    export_root : path to the export directory containing manifest.json, items.jsonl, classes.json
    expected_version : if given, raise on manifest.json["version"] mismatch
    expected_hash : if given, raise on manifest.json["content_hash_sha8"] mismatch
    splits_path : where to read/write splits.json (default: <export_root>/splits.json)
    train_ratio, dev_ratio, test_ratio : split ratios (recomputed only if splits_path is missing)
    split_seed : RNG seed for first-time split computation
    use_image_id_for_split : True → splits keyed by image (default), False → by sample_id
    """

    SUPPORTED_VERSIONS = {"v1", "v2", "v3"}

    def __init__(
        self,
        export_root: str | Path,
        *,
        expected_version: Optional[str] = None,
        expected_hash: Optional[str] = None,
        splits_path: Optional[str | Path] = None,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 1337,
    ):
        self.export_root = Path(export_root).resolve()
        if not self.export_root.is_dir():
            raise FileNotFoundError(f"V3 export root does not exist: {self.export_root}")

        self.manifest_path = self.export_root / "manifest.json"
        self.items_path = self.export_root / "items.jsonl"
        self.classes_path = self.export_root / "classes.json"
        for required in (self.manifest_path, self.items_path, self.classes_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"V3 export missing required file: {required.relative_to(self.export_root.parent)}"
                )

        # Default splits.json location: a writable user-cache dir keyed by
        # dataset hash, so a read-only export dir doesn't break the adapter
        # and concurrent runs of different dataset versions don't fight over
        # the same file. Override with `splits_path=...` to pin elsewhere.
        if splits_path:
            self.splits_path = Path(splits_path)
        else:
            cache_root = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
            cache_dir = cache_root / "trifoodnet" / "splits"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Hash filename = export's content_hash_sha8 if available, else "default".
            # We don't have the manifest read yet, so postpone the suffix; for now
            # the default still uses export_root if writable, and falls back below.
            primary = self.export_root / "splits.json"
            try:
                # Probe write — if export root isn't writable, use the cache.
                probe = self.export_root / ".splits_writable_probe"
                probe.write_text("")
                probe.unlink()
                self.splits_path = primary
            except OSError:
                self.splits_path = cache_dir / f"{self.export_root.name}.splits.json"
        self.train_ratio = float(train_ratio)
        self.dev_ratio = float(dev_ratio)
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)

        self._manifest: Optional[dict] = None
        self._classes: Optional[List[dict]] = None
        self._items: Optional[List[dict]] = None
        self._cached: Optional[AdapterData] = None

        # Validation that does not require reading large files yet.
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self._manifest = json.load(f)

        version = str(self._manifest.get("version", ""))
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported v3 export version {version!r}; adapter supports {sorted(self.SUPPORTED_VERSIONS)}"
            )
        if expected_version and version != expected_version:
            raise ValueError(
                f"Dataset version mismatch: manifest reports {version!r}, expected {expected_version!r}"
            )

        manifest_hash = str(self._manifest.get("content_hash_sha8", "") or "")
        if expected_hash and manifest_hash != expected_hash:
            raise ValueError(
                f"Dataset hash mismatch: manifest reports {manifest_hash!r}, expected {expected_hash!r}"
            )
        self.dataset_version = version
        self.dataset_hash = manifest_hash

    # ── public API ────────────────────────────────────────────────────────────

    def load(self) -> AdapterData:
        """Materialize the adapter contract. Cached after first call."""
        if self._cached is not None:
            return self._cached

        items = self._read_items()
        classes = self._read_classes()
        image_rows, stage3_rows = self._build_legacy_rows(items, classes)
        split_mapping = self._load_or_compute_splits(image_rows)
        split_summary = self._summarize_image_splits(image_rows, split_mapping)
        stage3_split_summary = self._summarize_stage3_splits(stage3_rows, split_mapping)

        # Sort classes by compact_id so build_class_name_index sees them in
        # contiguous order (the legacy code uses list-index as class_id when
        # the record does not carry one explicitly).
        # NOTE field-name asymmetry in the v3 export:
        #   items.jsonl rows  → "class_slug" / "class_display_name"
        #   classes.json rows → "slug" / "display_name"
        classes_by_compact = sorted(classes, key=lambda c: int(c["compact_id"]))
        legacy_classes = [
            {
                "name": entry["slug"],
                "source": "v3_export",
                "usage_count": int(entry["size"]),
                "class_id": int(entry["compact_id"]),
                "display_name": entry.get("display_name"),
                "legacy_id": int(entry["id"]),
            }
            for entry in classes_by_compact
        ]
        compact_ids = [int(entry["compact_id"]) for entry in classes_by_compact]

        self._cached = AdapterData(
            image_rows=image_rows,
            stage3_rows=stage3_rows,
            classes=legacy_classes,
            split_mapping=split_mapping,
            split_summary=split_summary,
            stage3_split_summary=stage3_split_summary,
            supported_classes=sorted({c["slug"] for c in classes}),
            removed_classes=[],
            retained_image_ids=sorted({row["image_id"] for row in image_rows}),
            dataset_version=self.dataset_version,
            dataset_hash=self.dataset_hash,
            export_root=self.export_root,
            classes_compact_ids=compact_ids,
        )
        return self._cached

    # ── internal: file readers ────────────────────────────────────────────────

    def _read_items(self) -> List[dict]:
        if self._items is not None:
            return self._items
        rows: List[dict] = []
        with self.items_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        self._items = rows
        return rows

    def _read_classes(self) -> List[dict]:
        if self._classes is not None:
            return self._classes
        with self.classes_path.open("r", encoding="utf-8") as f:
            self._classes = json.load(f)
        return self._classes

    # ── internal: schema translation ─────────────────────────────────────────

    def _build_legacy_rows(self, items: List[dict], classes: List[dict]) -> tuple[List[dict], List[dict]]:
        """Group items by src_image_id and emit (image_rows, stage3_rows) in legacy shape."""
        # class_id used downstream is the contiguous compact_id (matches num_classes=32)
        compact_id_by_legacy: Dict[int, int] = {
            int(c["id"]): int(c["compact_id"]) for c in classes
        }

        rows_by_image: Dict[str, List[dict]] = {}
        for raw in items:
            image_id = str(raw["src_image_id"])
            rows_by_image.setdefault(image_id, []).append(raw)

        image_rows: List[dict] = []
        stage3_rows: List[dict] = []

        for image_id, raw_items in sorted(rows_by_image.items()):
            # canonical sort by item index so item_NNN ordering is deterministic
            raw_items_sorted = sorted(raw_items, key=lambda r: int(r.get("src_item_index", 0)))
            first = raw_items_sorted[0]
            image_path = str(first["image_path"])
            image_width = int(first.get("image_width") or 0) or None
            image_height = int(first.get("image_height") or 0) or None

            translated_items: List[dict] = []
            for raw in raw_items_sorted:
                compact_class_id = compact_id_by_legacy.get(
                    int(raw["class_id"]), int(raw.get("compact_id", -1))
                )
                if compact_class_id < 0:
                    continue
                class_slug = str(raw.get("class_slug") or raw.get("name") or "food_item")
                bbox = list(raw.get("bbox") or [])
                if len(bbox) != 4:
                    continue
                item = {
                    # ── required by legacy callers ──
                    "bbox": [float(v) for v in bbox],
                    "mask_path": raw.get("mask_path"),
                    "crop_path": raw.get("crop_path"),
                    "classification_status": "labeled",
                    "class_id": int(compact_class_id),
                    "item_index": int(raw.get("src_item_index", 0)),
                    "final_class": class_slug,
                    "label": class_slug,
                    "active": True,
                    "excluded": False,
                    "use_for_export": True,
                    # ── v3 provenance preserved verbatim ──
                    "sample_id": raw.get("sample_id"),
                    "class_slug": class_slug,
                    "class_display_name": raw.get("class_display_name"),
                    "vlm_label": raw.get("vlm_label"),
                    "vlm_canonical_label": raw.get("vlm_canonical_label"),
                    "vlm_description": raw.get("vlm_description"),
                    "sam_score": raw.get("sam_score"),
                    "centrality_rank": raw.get("centrality_rank"),
                    "is_reference": raw.get("is_reference"),
                    "legacy_class_id": int(raw["class_id"]),
                    "image_id": image_id,
                    "src_item_index": int(raw.get("src_item_index", 0)),
                }
                translated_items.append(item)
                stage3_rows.append({
                    "image_id": image_id,
                    "image_path": image_path,           # required by load_masked_item_image
                    "class_id": int(compact_class_id),
                    "final_class": class_slug,
                    "label": class_slug,
                    "mask_path": raw.get("mask_path"),
                    "crop_path": raw.get("crop_path"),
                    "bbox": [float(v) for v in bbox],
                    "sam_bbox": [float(v) for v in bbox],  # alias used as bbox fallback
                    "sample_id": raw.get("sample_id"),
                    "centrality_rank": raw.get("centrality_rank"),
                    "is_reference": raw.get("is_reference"),
                    "src_item_index": int(raw.get("src_item_index", 0)),
                })

            if not translated_items:
                continue
            image_row = {
                "image_id": image_id,
                "image_path": image_path,
                "image_width": image_width,
                "image_height": image_height,
                "items": translated_items,
                "use_for_export": True,
                "review_status": "approved",
                "notes": None,
            }
            image_rows.append(image_row)

        return image_rows, stage3_rows

    # ── internal: splits (compute once, persist alongside dataset) ───────────

    def _load_or_compute_splits(self, image_rows: List[dict]) -> Dict[str, str]:
        if self.splits_path.exists():
            with self.splits_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            # Validate the cached splits match THIS dataset version + ratios + seed.
            # Otherwise we'd silently use a stale split for a new dataset, leaking
            # train into dev/test or mis-stratifying.
            cached_hash = str(payload.get("dataset_hash", ""))
            cached_seed = int(payload.get("seed", -1))
            cached_ratios = payload.get("ratios") or {}
            ratios_match = (
                abs(float(cached_ratios.get("train", -1)) - self.train_ratio) < 1e-6
                and abs(float(cached_ratios.get("dev", -1)) - self.dev_ratio) < 1e-6
                and abs(float(cached_ratios.get("test", -1)) - self.test_ratio) < 1e-6
            )
            if (
                cached_hash == self.dataset_hash
                and cached_seed == self.split_seed
                and ratios_match
            ):
                mapping = payload.get("split_mapping") or {}
                if mapping:
                    # Coverage check: every image_id we're about to load must be in
                    # the cached mapping. Otherwise downstream code calling
                    # split_mapping.get(id, "train") silently labels missing items
                    # as train — a leak vector for any image added/renamed after
                    # the cache was written.
                    cached_image_ids = {str(k) for k in mapping.keys()}
                    current_image_ids = {str(row["image_id"]) for row in image_rows}
                    # Value validation: every cached split label must be one of the
                    # three accepted names. A typo or corruption like "trian" would
                    # otherwise be passed through normalize_split_name as-is and
                    # silently exclude rows from train/dev/test entirely.
                    valid_split_names = {"train", "dev", "test"}
                    values_valid = all(
                        str(v) in valid_split_names for v in mapping.values()
                    )
                    if (
                        current_image_ids.issubset(cached_image_ids)
                        and values_valid
                    ):
                        return {str(k): str(v) for k, v in mapping.items()}
                    # else: image_rows contains IDs the cache doesn't cover OR the
                    # cached file has bogus split values → fall through to recompute.
            # Validation failed — fall through to recompute. Don't warn loudly
            # because this is normal when a dataset version bumps; the new
            # mapping will just overwrite the cache.

        mapping = self._compute_stratified_split(image_rows)
        payload = {
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "seed": self.split_seed,
            "ratios": {
                "train": self.train_ratio,
                "dev": self.dev_ratio,
                "test": self.test_ratio,
            },
            "method": "stratified_by_class_slug_v1",
            "n_train": sum(1 for v in mapping.values() if v == "train"),
            "n_dev": sum(1 for v in mapping.values() if v == "dev"),
            "n_test": sum(1 for v in mapping.values() if v == "test"),
            "split_mapping": mapping,
        }
        try:
            with self.splits_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except OSError as exc:
            raise OSError(
                f"Failed to write splits.json to {self.splits_path}. "
                "Pass splits_path=<writable path> to the adapter if the export dir is read-only."
            ) from exc
        return mapping

    def _compute_stratified_split(self, image_rows: List[dict]) -> Dict[str, str]:
        """Image-level stratified split. Class coverage per split is greedily maximized.

        We deliberately keep the implementation here (rather than reusing
        stratified_split from dataset_integration) so the adapter can survive
        v3-format-specific decisions: it does NOT prune low-support classes,
        because the v3 export already decided which classes are kept.
        """
        rng = random.Random(self.split_seed)

        # Image → set of class_slugs present in its items
        image_classes: Dict[str, set[str]] = {}
        for row in image_rows:
            slugs = {str(item.get("class_slug") or item.get("final_class")) for item in row.get("items", [])}
            slugs.discard("None")
            slugs.discard("")
            if slugs:
                image_classes[str(row["image_id"])] = slugs

        # Class → list of image_ids that contain it
        class_images: Dict[str, List[str]] = {}
        for img, slugs in image_classes.items():
            for slug in slugs:
                class_images.setdefault(slug, []).append(img)

        # Stable image ordering for determinism, then shuffled by seed
        all_images = sorted(image_classes.keys())
        rng.shuffle(all_images)

        target_total = len(all_images)
        ratios = {"train": self.train_ratio, "dev": self.dev_ratio, "test": self.test_ratio}
        ratio_sum = sum(ratios.values()) or 1.0
        target_sizes = {name: int(round(target_total * (ratios[name] / ratio_sum))) for name in ratios}
        # Top up so sizes sum to exactly target_total
        sized = sum(target_sizes.values())
        if sized < target_total:
            target_sizes["train"] += target_total - sized
        elif sized > target_total:
            target_sizes["train"] -= sized - target_total

        assignments: Dict[str, str] = {}
        split_class_counts: Dict[str, Dict[str, int]] = {"train": {}, "dev": {}, "test": {}}
        split_counts: Dict[str, int] = {"train": 0, "dev": 0, "test": 0}

        # Phase A: ensure every class appears in every split where target_size > 0,
        # in order of class rarity (rarest first).
        sorted_classes = sorted(class_images.keys(), key=lambda c: len(class_images[c]))
        for split_name in ("train", "dev", "test"):
            if target_sizes[split_name] <= 0:
                continue
            for class_slug in sorted_classes:
                if split_class_counts[split_name].get(class_slug, 0) > 0:
                    continue
                if split_counts[split_name] >= target_sizes[split_name]:
                    break
                # find unassigned image containing this class
                for img in class_images[class_slug]:
                    if img in assignments:
                        continue
                    assignments[img] = split_name
                    split_counts[split_name] += 1
                    for slug in image_classes[img]:
                        split_class_counts[split_name][slug] = split_class_counts[split_name].get(slug, 0) + 1
                    break

        # Phase B: fill remaining capacity in shuffled order
        for img in all_images:
            if img in assignments:
                continue
            for split_name in ("train", "dev", "test"):
                if split_counts[split_name] < target_sizes[split_name]:
                    assignments[img] = split_name
                    split_counts[split_name] += 1
                    break
            else:
                # all targets met → spill into train
                assignments[img] = "train"
                split_counts["train"] += 1

        return assignments

    # ── internal: summaries (mirror legacy enforce_supported_class_contract output) ──

    def _summarize_image_splits(self, image_rows, split_mapping):
        summary = {"train": _empty_image_split(), "dev": _empty_image_split(), "test": _empty_image_split()}
        for row in image_rows:
            split = split_mapping.get(str(row["image_id"]), "train")
            bucket = summary.setdefault(split, _empty_image_split())
            bucket["images"] += 1
            bucket["active_items"] += len(row.get("items", []))
            seen = set()
            for item in row.get("items", []):
                seen.add(str(item.get("class_slug") or item.get("final_class")))
            for slug in seen:
                bucket["class_image_counts"][slug] = bucket["class_image_counts"].get(slug, 0) + 1
        return summary

    def _summarize_stage3_splits(self, stage3_rows, split_mapping):
        summary = {"train": _empty_stage3_split(), "dev": _empty_stage3_split(), "test": _empty_stage3_split()}
        for row in stage3_rows:
            split = split_mapping.get(str(row["image_id"]), "train")
            bucket = summary.setdefault(split, _empty_stage3_split())
            bucket["items"] += 1
            slug = str(row.get("class_slug") or row.get("final_class") or row.get("label"))
            bucket["class_item_counts"][slug] = bucket["class_item_counts"].get(slug, 0) + 1
        return summary


def _empty_image_split():
    return {"images": 0, "active_items": 0, "class_image_counts": {}}


def _empty_stage3_split():
    return {"items": 0, "class_item_counts": {}}


def adapter_from_config(integration_cfg) -> V3ExportAdapter:
    """Construct a V3ExportAdapter from a config block.

    Expected fields on integration_cfg.adapter:
        kind:            "v3_export"   (only supported value for now)
        export_root:     path to the v3 export directory
        expected_version (optional)
        expected_hash    (optional)
        splits_path      (optional)
    Falls back to integration_cfg.train_ratio/dev_ratio/test_ratio/split_seed.
    """
    adapter_cfg = getattr(integration_cfg, "adapter", None)
    if adapter_cfg is None:
        raise ValueError(
            "data.integration.adapter is not configured. Set it to a v3_export block "
            "or pass adapter=None to JointFoodDataset for legacy mode."
        )
    kind = str(getattr(adapter_cfg, "kind", "v3_export"))
    if kind != "v3_export":
        raise ValueError(f"Unsupported adapter kind: {kind!r}")

    export_root = getattr(adapter_cfg, "export_root", None)
    if not export_root:
        env_root = os.environ.get("TRIFOODNET_DATASET_DIR")
        if env_root:
            export_root = env_root
        else:
            raise ValueError(
                "data.integration.adapter.export_root is empty and TRIFOODNET_DATASET_DIR "
                "env var is not set. Either set the env var, override the config field on "
                "the CLI (data.integration.adapter.export_root=/path/to/v3_export), or edit "
                "master_config.yaml. See CONFIG_GUIDE.md."
            )

    return V3ExportAdapter(
        export_root=export_root,
        expected_version=getattr(adapter_cfg, "expected_version", None) or None,
        expected_hash=getattr(adapter_cfg, "expected_hash", None) or None,
        splits_path=getattr(adapter_cfg, "splits_path", None) or None,
        train_ratio=float(getattr(integration_cfg, "train_ratio", 0.8)),
        dev_ratio=float(getattr(integration_cfg, "dev_ratio", getattr(integration_cfg, "val_ratio", 0.1))),
        test_ratio=float(getattr(integration_cfg, "test_ratio", 0.1)),
        split_seed=int(getattr(integration_cfg, "split_seed", 1337)),
    )
