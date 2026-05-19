# =============================================================================
# FILE: fix_dataset.py
# CATEGORY: DATA
# PURPOSE: Dataset repair helper for earlier audit recovery work.
# DEPENDENCIES: None
# USED BY: None
# KEY CLASSES/FUNCTIONS: is_valid_label, get_tight_bbox, fix_dataset
# LAST MODIFIED: 2026-03-21T09:22:38+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import json
from pathlib import Path
from PIL import Image
import shutil

def is_valid_label(item):
    return any(item.get(k) for k in ("final_class", "label", "coarse_label", "class_name"))

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
        print(f"Error open mask {full_mask_path}: {e}")
    return None

def fix_dataset():
    batch_root = Path(r"d:\downloads\Restaurant_dataset\repo\experiments\v3_3stage_mvp\batch_results_v8_500")
    export_root = batch_root / "_review" / "dataset"
    
    # Backup original jsonls
    for f in export_root.glob("*.jsonl"):
        backup = f.with_suffix(".jsonl.bak")
        if not backup.exists():
            shutil.copy2(f, backup)

    # 1. Fix images_manifest.jsonl
    images_manifest_path = export_root / "images_manifest.jsonl"
    fixed_images = []
    with open(export_root / "images_manifest.jsonl.bak", 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            # Exclude images that are not for export (the prompt says "use only selected_images")
            if not row.get("use_for_export", False):
                continue
            
            for item in row.get("items", []):
                # Only keep items with labels active
                if not is_valid_label(item):
                    item["excluded"] = True
                    item["active"] = False
                
                # Fix qwen_bbox to match tight SAM mask bbox
                if item.get("active", True) and not item.get("excluded", False):
                    mask_path = item.get("mask_path")
                    if mask_path:
                        tight_bbox = get_tight_bbox(mask_path, batch_root)
                        if tight_bbox:
                            item["qwen_bbox"] = tight_bbox
            fixed_images.append(row)
            
    with open(images_manifest_path, 'w', encoding='utf-8') as f:
        for row in fixed_images:
            f.write(json.dumps(row) + "\n")

    # 2. Fix stage manifests (stage1, stage1_qwen, stage2, stage3)
    for stage_file in ["stage1_item_detection.jsonl", "stage1_qwen_detection.jsonl", "stage2_sam_segmentation.jsonl", "stage3_item_classification.jsonl"]:
        path = export_root / stage_file
        if not path.exists(): continue
        
        fixed_rows = []
        with open(export_root / f"{stage_file}.bak", 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                
                if not is_valid_label(row):
                    continue
                
                mask_path = row.get("mask_path")
                if mask_path:
                    tight_bbox = get_tight_bbox(mask_path, batch_root)
                    if tight_bbox:
                        row["qwen_bbox"] = tight_bbox
                
                fixed_rows.append(row)
                
        with open(path, 'w', encoding='utf-8') as f:
            for row in fixed_rows:
                f.write(json.dumps(row) + "\n")
                
    print("Dataset fixed successfully.")

if __name__ == "__main__":
    fix_dataset()
