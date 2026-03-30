# =============================================================================
# FILE: run_isolated_inference.py
# CATEGORY: INFER
# PURPOSE: Inference helper used for isolated environment checks.
# DEPENDENCIES: config_loader.py, dataset_integration.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: main
# LAST MODIFIED: 2026-03-21T10:43:57.570929+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================

import os
import sys
import torch
import json
import subprocess
from pathlib import Path
from config_loader import load_config
from dataset_integration import JointFoodDataset

def main():
    cfg = load_config("master_config.yaml")
    dataset = JointFoodDataset(
        batch_root=cfg.data.integration.batch_root,
        export_root=cfg.data.integration.export_root,
        split="val",
        image_size=cfg.data.image_size,
        split_seed=cfg.data.integration.split_seed,
    )
    
    image_ids = [dataset[i]["image_id"] for i in range(len(dataset))]
    print(f"Isolated Inference: {len(image_ids)} images to process.")
    
    for i, img_id in enumerate(image_ids):
        print(f"\n>>> [{i+1}/{len(image_ids)}] STARTING ISOLATED PROCESS FOR {img_id}...")
        try:
            # Use a separate script run_single_inference.py
            cmd = ["python3", "run_single_inference.py", "--image_id", img_id]
            subprocess.run(cmd, check=True)
            print(f">>> [{i+1}/{len(image_ids)}] SUCCESS.")
        except Exception as e:
            print(f">>> [{i+1}/{len(image_ids)}] FAILED: {e}")

if __name__ == "__main__":
    main()
