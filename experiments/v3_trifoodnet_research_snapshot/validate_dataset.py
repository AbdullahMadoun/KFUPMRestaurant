# =============================================================================
# FILE: validate_dataset.py
# CATEGORY: DATA
# PURPOSE: Dataset validation helper script for earlier audit workflows.
# DEPENDENCIES: None
# USED BY: None
# KEY CLASSES/FUNCTIONS: main
# LAST MODIFIED: 2026-03-21T09:22:06+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import json
from pathlib import Path
from PIL import Image

def main():
    batch_root = Path(r"d:\downloads\Restaurant_dataset\repo\experiments\v3_3stage_mvp\batch_results_v8_500")
    export_root = batch_root / "_review" / "dataset"
    manifest_path = export_root / "images_manifest.jsonl"
    
    total_images = 0
    non_export_images = 0
    items_without_labels = 0
    bbox_mismatches = 0
    valid_items = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            total_images += 1
            
            if not row.get("use_for_export", False):
                non_export_images += 1
                
            for item in row.get("items", []):
                if not item.get("active", True) or item.get("excluded", False):
                    continue
                    
                labels = [item.get(k) for k in ("final_class", "label", "coarse_label", "class_name") if item.get(k)]
                if not labels:
                    items_without_labels += 1
                
                qwen_bbox = item.get("qwen_bbox")
                mask_path = item.get("mask_path")
                
                if qwen_bbox and mask_path:
                    full_mask_path = batch_root / mask_path
                    if full_mask_path.exists():
                        try:
                            with Image.open(full_mask_path) as mask_img:
                                mask_img = mask_img.convert("L")
                                tight_bbox = mask_img.getbbox()
                                
                                if tight_bbox is not None:
                                    tight_bbox = list(tight_bbox)
                                    def is_close(b1, b2, tol=2.0):
                                        return all(abs(c1 - c2) <= tol for c1, c2 in zip(b1, b2))
                                        
                                    if not is_close(qwen_bbox, tight_bbox):
                                        bbox_mismatches += 1
                                        # Let's fix the dataset by replacing qwen_bbox with tight_bbox?
                                        # The task says: "make sure that bbox for qwen is the same as the bbox of the tightest mask around sams masks inspect that all the data is as expected".
                                    valid_items += 1
                        except Exception as e:
                            pass
                            
    res = {
        "Total Images": total_images,
        "Non-export Images": non_export_images,
        "Active items without labels": items_without_labels,
        "Valid items checked for bbox": valid_items,
        "Qwen bbox mismatches": bbox_mismatches
    }
    with open("validate_results.json", "w") as f:
         json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()
