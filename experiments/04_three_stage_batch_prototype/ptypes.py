"""Typed data structures for the 3-stage pipeline: Describe → Segment → Match."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import torch


@dataclass
class VisualItem:
    """Stage 1 output: what the VLM sees."""
    description: str        # "golden brown crispy surface with ridged texture"
    bbox: List[float]       # [x1, y1, x2, y2] pixel coords


@dataclass
class Detection:
    """Single SAM3 detection."""
    label: str              # the visual description used as prompt
    mask: torch.Tensor      # (H, W)
    box: torch.Tensor       # (4,) xyxy
    score: float


@dataclass
class SegmentedItem:
    """Stage 2 output: a segmented + cropped food item."""
    description: str        # visual description from Stage 1
    mask: np.ndarray        # (H, W) binary mask
    bbox: np.ndarray        # (4,) xyxy
    crop: np.ndarray        # (crop_H, crop_W, 3) BGR masked crop
    score: float            # SAM3 confidence


@dataclass
class MenuEntry:
    """Single item in the reference menu database."""
    name: str               # "Chicken"
    category: str           # "protein" | "carb" | "salad" | "side" | "drink" | "soup"
    price: float            # 15.0 SAR
    text_description: str   # "Grilled chicken pieces with golden brown skin"


@dataclass
class MatchResult:
    """Stage 3 output: identified menu item."""
    segmented: SegmentedItem
    menu_item: str                      # "Chicken"
    category: str                       # "protein"
    price: float                        # 15.0
    confidence: float                   # cosine similarity (0-1)
    top_k: List[Tuple[str, float]]      # [("Chicken", 0.92), ("Meat", 0.71), ...]


@dataclass
class PlateResult:
    """Final pipeline output for one plate image."""
    image_path: str
    matches: List[MatchResult]
    total_price: float
    stage1_time: float
    stage2_time: float
    stage3_time: float


def apply_nms(detections: List[Detection], max_objects: int, iou_threshold: float) -> List[Detection]:
    """Greedy mask-IoU NMS: keep top-scoring non-overlapping detections."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    keep: List[Detection] = []

    for det in detections:
        if len(keep) >= max_objects:
            break
        discard = False
        for kept in keep:
            intersection = torch.logical_and(det.mask, kept.mask).sum().float()
            union = torch.logical_or(det.mask, kept.mask).sum().float()
            iou = (intersection / union).item() if union > 0 else 0.0
            if iou > iou_threshold:
                discard = True
                break
        if not discard:
            keep.append(det)

    return keep


def nms_segmented(items: List[SegmentedItem], max_objects: int, iou_threshold: float) -> List[SegmentedItem]:
    """Greedy mask-IoU NMS on SegmentedItem list (numpy masks)."""
    if not items:
        return []

    items = sorted(items, key=lambda s: s.score, reverse=True)
    keep: List[SegmentedItem] = []

    for item in items:
        if len(keep) >= max_objects:
            break
        discard = False
        mask_bool = item.mask.astype(bool)
        for kept in keep:
            kept_bool = kept.mask.astype(bool)
            intersection = np.logical_and(mask_bool, kept_bool).sum()
            union = np.logical_or(mask_bool, kept_bool).sum()
            iou = intersection / union if union > 0 else 0.0
            if iou > iou_threshold:
                discard = True
                break
        if not discard:
            keep.append(item)

    return keep
