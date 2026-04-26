from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tiny_mismatch_export(tmp_path: Path) -> Path:
    root = tmp_path / "v3_export"
    (root / "images").mkdir(parents=True)
    (root / "masks" / "img_001").mkdir(parents=True)
    Image.new("RGB", (160, 120), "white").save(root / "images" / "img_001.jpg")
    mask = Image.new("L", (80, 80), 0)
    for y in range(10, 30):
        for x in range(10, 30):
            mask.putpixel((x, y), 255)
    mask.save(root / "masks" / "img_001" / "item_000_mask.png")
    (root / "manifest.json").write_text(
        json.dumps({"version": "v3", "content_hash_sha8": "bboxfx"}, sort_keys=True),
        encoding="utf-8",
    )
    row = {
        "sample_id": "img_001__000",
        "src_image_id": "img_001",
        "src_item_index": 0,
        "class_slug": "rice",
        "class_display_name": "rice",
        "name": "rice",
        "description": "white rice mound with visible grains",
        "bbox": [10, 10, 30, 30],
        "is_reference": False,
        "image_width": 160,
        "image_height": 120,
        "image_path": "images/img_001.jpg",
        "mask_path": "masks/img_001/item_000_mask.png",
    }
    (root / "items.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return root


def test_bbox_audit_reports_mask_native_contract_and_wrong_fullres_delta(tiny_mismatch_export: Path, tmp_path: Path):
    bbox_audit = importlib.import_module("stage1_kcfd.bbox_audit")
    rows = [
        json.loads(line)
        for line in (tiny_mismatch_export / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    report = bbox_audit.compute_bbox_audit(rows, tiny_mismatch_export, reference_policy="include")
    manifest = bbox_audit.write_bbox_audit(
        tiny_mismatch_export,
        tmp_path / "audit",
        reference_policy="include",
        max_samples=2,
        seed=7,
        expected_hash="bboxfx",
    )

    assert report["coordinate_contract"].startswith("Training/eval: load image at mask resolution")
    assert report["image_mask_size_mismatch_count"] == 1
    assert report["raw_bbox_vs_mask_tight_px"]["median"] < report["wrong_raw_bbox_drawn_on_fullres_px"]["median"]
    assert (tmp_path / "audit" / "manifest.json").exists()
    assert manifest["files"]
    assert all(Path(path).exists() for path in manifest["files"])
