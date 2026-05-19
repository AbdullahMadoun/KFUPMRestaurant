# =============================================================================
# FILE: test_pictsure_lora.py
# CATEGORY: TEST
# PURPOSE: Research test script for PictSure LoRA behavior.
# DEPENDENCIES: config_loader.py, stage3_icl.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: test_pictsure_lora_optimization
# LAST MODIFIED: 2026-03-21T07:52:33.086110+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import torch
from pathlib import Path
from stage3_icl import FoodClassifier
from config_loader import load_config

def test_pictsure_lora_optimization():
    print("Testing PictSure LoRA Optimization...")
    cfg = load_config(str(Path(__file__).resolve().parent / "master_config.yaml"))
    c3 = cfg.stage3
    
    # Instantiate with LoRA enabled from config
    device = "cpu"
    classifier = FoodClassifier(
        clip_model=c3.clip_model,
        lora_cfg=c3.lora,
        train_embedding=bool(getattr(c3, "train_embedding", True)),
    ).to(device)
    
    # 1. Verify PeftModel wrap
    try:
        from peft import PeftModel
        is_lora = isinstance(classifier.icl, PeftModel)
        print(f"Is ICL Transformer a PeftModel? {is_lora}")
        assert is_lora, "ICL Transformer should be wrapped in PeftModel!"
    except ImportError:
        print("PEFT not installed, skipping PeftModel check.")
        return

    # 2. Verify trainable parameters
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in classifier.parameters() if not p.requires_grad)
    
    # The embedding backbone follows the current config's train_embedding flag.
    backbone_trainable = sum(p.numel() for p in classifier.pictsure_model.embedding.parameters() if p.requires_grad)
    print(f"Vision Backbone trainable parameters: {backbone_trainable}")
    if bool(getattr(c3, "train_embedding", True)):
        assert backbone_trainable > 0, "Vision backbone should remain trainable when train_embedding=true."
    else:
        assert backbone_trainable == 0, "Vision backbone should be frozen when train_embedding=false."
    
    # Transformer should have LoRA params
    print(f"Total Trainable parameters (should be LoRA + Projections): {trainable_params}")
    assert trainable_params > 0, "There should be some trainable parameters (LoRA)!"
    
    # Check if target modules exist in the PeftModel
    target_modules = c3.lora.target_modules
    found_lora = False
    for n, m in classifier.icl.named_modules():
        if any(t in n for t in target_modules) and "lora_" in n:
            found_lora = True
            break
    print(f"Found LoRA adapters in target modules {target_modules}? {found_lora}")
    assert found_lora, f"LoRA adapters not found in target modules {target_modules}!"

    print("LoRA Optimization Verification SUCCESSFUL!")

if __name__ == "__main__":
    test_pictsure_lora_optimization()
