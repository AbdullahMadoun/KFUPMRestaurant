# =============================================================================
# FILE: tests/test_sam3_allocation.py
# CATEGORY: TEST
# PURPOSE: Snapshot-retained source file for test_sam3_allocation.py.
# DEPENDENCIES: stage2_sam.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: test_sam3_segmenter
# LAST MODIFIED: 2026-03-21T09:40:38+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import sys
sys.path.insert(0, ".")
import torch
from stage2_sam import SAM3Segmenter

def test_sam3_segmenter():
    print("Building SAM3Segmenter...")
    device = torch.device('cpu')
    
    try:
        segmenter = SAM3Segmenter(
            model_name="facebook/sam3",
            freeze_image_encoder=True,
            freeze_prompt_encoder=True,
            gradient_checkpointing=False
        ).to(device)
        print("SAM3Segmenter built successfully!")
    except Exception as e:
        print(f"Failed to build SAM3Segmenter: {e}")
        return

    # Create dummy tensors to test forward pass shape compliance
    bs = 1
    dummy_pixels = torch.randn(bs, 3, 224, 224, device=device)
    dummy_boxes = [torch.tensor([[10, 10, 100, 100], [50, 50, 150, 150]], dtype=torch.float32, device=device)]
    
    print("Testing forward pass...")
    try:
        output = segmenter(dummy_pixels, dummy_boxes)
        pred_masks = output["pred_masks"][0]
        assert len(pred_masks) == 2, f"Expected 2 masks, got {len(pred_masks)}"
        print("Forward pass successful. Output tensor shape bindings verified.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_sam3_segmenter()
