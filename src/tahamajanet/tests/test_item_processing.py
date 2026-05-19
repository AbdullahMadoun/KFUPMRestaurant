# =============================================================================
# FILE: tests/test_item_processing.py
# CATEGORY: TEST
# PURPOSE: Snapshot-retained source file for test_item_processing.py.
# DEPENDENCIES: item_processing.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: test_pil_images_to_tensor_resizes_and_batches, test_masked_crop_from_pil_applies_mask_inside_bbox
# LAST MODIFIED: 2026-03-21T11:04:49.818270+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import sys
from pathlib import Path

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from item_processing import masked_crop_from_pil, pil_images_to_tensor


def test_pil_images_to_tensor_resizes_and_batches():
    image_a = Image.new("RGB", (32, 16), (255, 0, 0))
    image_b = Image.new("RGB", (16, 32), (0, 255, 0))

    batched = pil_images_to_tensor([image_a, image_b], size=(24, 24))

    assert batched.shape == (2, 3, 24, 24)
    assert batched.dtype == torch.float32


def test_masked_crop_from_pil_applies_mask_inside_bbox():
    image = Image.new("RGB", (4, 4), (255, 255, 255))
    mask = torch.zeros(4, 4)
    mask[1:3, 1:3] = 1.0

    cropped = masked_crop_from_pil(image, mask, bbox=(1, 1, 3, 3))

    assert cropped.size == (2, 2)
    pixels = [cropped.getpixel((x, y)) for y in range(cropped.height) for x in range(cropped.width)]
    assert all(pixel == (255, 255, 255) for pixel in pixels)
