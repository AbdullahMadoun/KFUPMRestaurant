# =============================================================================
# FILE: check_trainable.py
# CATEGORY: UTIL
# PURPOSE: Quick helper to inspect which parameters remain trainable.
# DEPENDENCIES: config_loader.py, pipeline.py, stage1_qwen.py, stage2_sam.py, stage3_icl.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: None
# LAST MODIFIED: 2026-03-21T09:36:19.170676+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import torch
from pipeline import TriFoodNet
from stage1_qwen import QwenGrounder
from stage2_sam import SAM3Segmenter
from stage3_icl import FoodClassifier
from config_loader import load_config
from transformers import BitsAndBytesConfig

cfg = load_config("master_config.yaml")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

s1 = QwenGrounder(model_name=cfg.stage1.model_name, lora_r=16, quantization_config=bnb_config)
s2 = SAM3Segmenter(model_name=cfg.stage2.model_name, quantization_config=bnb_config)
s3 = FoodClassifier(clip_model=cfg.stage3.clip_model, lora_cfg={"enabled": True, "r": 16})
pipeline = TriFoodNet(s1, s2, s3)

print("--- Trainability Report ---")
for i, stage in enumerate([s1, s2, s3], 1):
    trainable = sum(p.numel() for p in stage.parameters() if p.requires_grad)
    total = sum(p.numel() for p in stage.parameters())
    print(f"Stage {i}: {trainable:,} trainable / {total:,} total parameters")
