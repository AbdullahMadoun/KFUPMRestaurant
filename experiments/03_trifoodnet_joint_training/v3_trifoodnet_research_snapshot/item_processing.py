# =============================================================================
# FILE: item_processing.py
# CATEGORY: DATA
# PURPOSE: Image and mask utilities shared by dataset loading and inference.
# DEPENDENCIES: None
# USED BY: dataset_integration.py, pipeline.py, stage3_icl.py, tests/test_item_processing.py, validate_pipeline_contracts.py, visualize_val_predictions.py
# KEY CLASSES/FUNCTIONS: pil_images_to_tensor, resize_mask_tensor, masked_crop_from_pil
# LAST MODIFIED: 2026-03-21T14:21:46.552691+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def pil_images_to_tensor(
    images: Sequence[Image.Image],
    size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    tensors = []
    for image in images:
        resized = image.convert("RGB").resize(size, Image.BICUBIC)
        tensors.append(TF.pil_to_tensor(resized).float() / 255.0)
    if not tensors:
        return torch.empty(0, 3, *size, dtype=torch.float32)
    return torch.stack(tensors, dim=0)


def resize_mask_tensor(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask tensor, got shape {tuple(mask.shape)}")
    if tuple(mask.shape) == tuple(size):
        return mask
    resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)


def masked_crop_from_pil(
    image: Image.Image,
    mask: torch.Tensor | Image.Image,
    bbox: Optional[Sequence[float]] = None,
    threshold: float = 0.5,
) -> Image.Image:
    image = image.convert("RGB")
    image_tensor = TF.pil_to_tensor(image)

    if isinstance(mask, Image.Image):
        mask_tensor = TF.pil_to_tensor(mask.convert("L")).float().squeeze(0) / 255.0
    elif isinstance(mask, torch.Tensor):
        mask_tensor = mask.detach().float().cpu()
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)!r}")

    mask_tensor = resize_mask_tensor(mask_tensor, (image.height, image.width))
    mask_binary = (mask_tensor >= threshold).to(image_tensor.dtype)
    masked_tensor = image_tensor * mask_binary.unsqueeze(0)
    masked_image = TF.to_pil_image(masked_tensor)

    if bbox is None:
        return masked_image

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(x1, image.width))
    x2 = max(x1 + 1, min(x2, image.width))
    y1 = max(0, min(y1, image.height))
    y2 = max(y1 + 1, min(y2, image.height))
    return masked_image.crop((x1, y1, x2, y2))
