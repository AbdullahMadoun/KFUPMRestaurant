# =============================================================================
# FILE: verify_split.py
# CATEGORY: DATA
# PURPOSE: Split verification helper for the reviewed export contract.
# DEPENDENCIES: config_loader.py, dataset_integration.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: None
# LAST MODIFIED: 2026-03-21T09:45:32.863934+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
from dataset_integration import JointFoodDataset
from config_loader import load_config
import torch

cfg = load_config("master_config.yaml")
root = cfg.data.integration.batch_root
export = cfg.data.integration.export_root

print("--- Split Verification ---")
for seed in [1337, 420]:
    ds = JointFoodDataset(batch_root=root, export_root=export, split="val", split_seed=seed)
    ids = [ds[i]["image_id"] for i in range(min(5, len(ds)))]
    print(f"Seed {seed} first 5 IDs: {ids}")
