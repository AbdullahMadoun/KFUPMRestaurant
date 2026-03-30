# =============================================================================
# FILE: tests/test_dataset.py
# CATEGORY: TEST
# PURPOSE: Snapshot-retained source file for test_dataset.py.
# DEPENDENCIES: config_loader.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: get_tight_bbox, is_valid_label, is_close, test_images_manifest_constraints, test_stage_manifests_constraints
# LAST MODIFIED: 2026-03-21T07:21:11.280774+00:00
# SNAPSHOT NOTES: batch root now follows the snapshot config or TRIFOODNET_BATCH_ROOT override
# =============================================================================
import json
import os
from pathlib import Path
from PIL import Image
from config_loader import load_config

SNAPSHOT_ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(str(SNAPSHOT_ROOT / "master_config.yaml"))
BATCH_ROOT = Path(os.environ.get("TRIFOODNET_BATCH_ROOT", str(CFG.data.integration.batch_root)))
EXPORT_ROOT = BATCH_ROOT / "_review" / "dataset"

def get_tight_bbox(mask_path, batch_root):
    full_mask_path = batch_root / mask_path
    if not full_mask_path.exists():
        return None
    try:
        with Image.open(full_mask_path) as mask_img:
            mask_img = mask_img.convert("L")
            bbox = mask_img.getbbox()
            if bbox is not None:
                return list(bbox)
    except Exception as e:
        pass
    return None

def is_valid_label(item):
    return any(item.get(k) for k in ("final_class", "label", "coarse_label", "class_name"))

def is_close(b1, b2, tol=2.0):
    if b1 is None or b2 is None: return False
    return all(abs(c1 - c2) <= tol for c1, c2 in zip(b1, b2))

def test_images_manifest_constraints():
    manifest_path = EXPORT_ROOT / "images_manifest.jsonl"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            # Constraint 1: Only selected_images (use_for_export) are present
            assert row.get("use_for_export", False) is True, "Found image not marked for export"
            
            for item in row.get("items", []):
                if not item.get("active", True) or item.get("excluded", False):
                    continue
                
                # Constraint 2: All active items must have a label
                assert is_valid_label(item), "Active item found without a valid label"
                
                # Constraint 3: qwen_bbox equals tight bbox from SAM mask
                mask_path = item.get("mask_path")
                if mask_path:
                    tight_bbox = get_tight_bbox(mask_path, BATCH_ROOT)
                    if tight_bbox:
                        qwen_bbox = item.get("qwen_bbox")
                        assert is_close(qwen_bbox, tight_bbox), f"qwen_bbox {qwen_bbox} != tight_bbox {tight_bbox} for mask {mask_path}"

def test_stage_manifests_constraints():
    for stage_file in ["stage1_item_detection.jsonl", "stage2_sam_segmentation.jsonl", "stage3_item_classification.jsonl"]:
        path = EXPORT_ROOT / stage_file
        if not path.exists(): continue
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                
                assert is_valid_label(row), f"{stage_file} row missing label"
                
                mask_path = row.get("mask_path")
                if mask_path:
                    tight_bbox = get_tight_bbox(mask_path, BATCH_ROOT)
                    if tight_bbox:
                        qwen_bbox = row.get("qwen_bbox")
                        assert is_close(qwen_bbox, tight_bbox), f"{stage_file} qwen_bbox {qwen_bbox} mismatch tight_bbox {tight_bbox}"

if __name__ == "__main__":
    print("Running images_manifest.jsonl constraints...")
    test_images_manifest_constraints()
    print("Running stage manifests constraints...")
    test_stage_manifests_constraints()
    print("All tests passed successfully!")
