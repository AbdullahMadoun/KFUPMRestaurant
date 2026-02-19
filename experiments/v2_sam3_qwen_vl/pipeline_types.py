"""Typed data structures and NMS for the segmentation pipeline."""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class Detection:
    """Single detection from SAM3."""
    label: str
    mask: torch.Tensor   # (H, W)
    box: torch.Tensor    # (4,)
    score: float


@dataclass
class SegmentationResult:
    """Grouped detections for a single label, ready for visualization."""
    label: str
    masks: np.ndarray    # (N, H, W)
    boxes: np.ndarray    # (N, 4)
    scores: np.ndarray   # (N,)


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


def group_detections(detections: List[Detection]) -> List[SegmentationResult]:
    """Group detections by label and convert tensors to numpy arrays."""
    if not detections:
        return []

    groups: dict = {}
    for det in detections:
        if det.label not in groups:
            groups[det.label] = {"masks": [], "boxes": [], "scores": []}
        groups[det.label]["masks"].append(det.mask.unsqueeze(0))
        groups[det.label]["boxes"].append(det.box.unsqueeze(0))
        groups[det.label]["scores"].append(det.score)

    results: List[SegmentationResult] = []
    for label, data in groups.items():
        results.append(SegmentationResult(
            label=label,
            masks=torch.cat(data["masks"]).cpu().numpy(),
            boxes=torch.cat(data["boxes"]).cpu().numpy(),
            scores=np.array(data["scores"]),
        ))
    return results
