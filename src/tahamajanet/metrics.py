# =============================================================================
# FILE: metrics.py
# CATEGORY: EVAL
# PURPOSE: Metric helpers for boxes, masks, episodic accuracy accumulation, and text tables.
# DEPENDENCIES: None
# USED BY: benchmark_runtime.py, post_training_artifacts.py, train_joint.py, validate_pipeline_contracts.py
# KEY CLASSES/FUNCTIONS: box_iou, recall_at_iou, BoxMatch, greedy_box_matches, mask_iou, mean_iou, EpisodicAccumulator, format_metrics_table
# LAST MODIFIED: 2026-03-21T14:23:04.287095+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
Evaluation metrics for TriFoodNet research runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import torch


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def recall_at_iou(
    pred_boxes_per_image: Sequence[Sequence[Sequence[float]]],
    gt_boxes_per_image: Sequence[Sequence[Sequence[float]]],
    threshold: float = 0.5,
) -> float:
    matched = 0
    total = 0
    for pred_boxes, gt_boxes in zip(pred_boxes_per_image, gt_boxes_per_image):
        pred_boxes = list(pred_boxes)
        gt_boxes = list(gt_boxes)
        total += len(gt_boxes)
        if not gt_boxes:
            continue
        used = set()
        for gt_index, gt_box in enumerate(gt_boxes):
            best_index = None
            best_iou = -1.0
            for pred_index, pred_box in enumerate(pred_boxes):
                if pred_index in used:
                    continue
                iou = box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_index = pred_index
            if best_index is not None and best_iou >= threshold:
                used.add(best_index)
                matched += 1
    return matched / max(total, 1)


@dataclass(frozen=True)
class BoxMatch:
    pred_index: int
    gt_index: int
    iou: float


def greedy_box_matches(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    threshold: float = 0.5,
) -> List[BoxMatch]:
    candidates: List[BoxMatch] = []
    for pred_index, pred_box in enumerate(pred_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            iou = box_iou(pred_box, gt_box)
            if iou >= threshold:
                candidates.append(BoxMatch(pred_index=pred_index, gt_index=gt_index, iou=iou))

    candidates.sort(key=lambda match: match.iou, reverse=True)
    used_preds = set()
    used_gts = set()
    matches: List[BoxMatch] = []
    for match in candidates:
        if match.pred_index in used_preds or match.gt_index in used_gts:
            continue
        used_preds.add(match.pred_index)
        used_gts.add(match.gt_index)
        matches.append(match)
    return matches


def mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_mask = pred_mask.detach().cpu()
    gt_mask = gt_mask.detach().cpu()
    # Handle resolution mismatch (e.g. SAM 3 output vs GT mask)
    if pred_mask.shape != gt_mask.shape:
        import torch.nn.functional as F
        # Resizing to GT mask size and using nearest neighbor to preserve boolean nature
        pred_mask = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0).float(),
            size=gt_mask.shape,
            mode="nearest"
        ).squeeze(0).squeeze(0).bool()

    pred = pred_mask.bool()
    gt = gt_mask.bool()
    inter = (pred & gt).float().sum().item()
    union = (pred | gt).float().sum().item()
    if union <= 0:
        return 0.0
    return inter / union


def mean_iou(
    pred_masks_per_image: Sequence[Sequence[torch.Tensor]],
    gt_masks_per_image: Sequence[Sequence[torch.Tensor]],
) -> float:
    per_image_scores: List[float] = []
    for pred_masks, gt_masks in zip(pred_masks_per_image, gt_masks_per_image):
        pred_masks = list(pred_masks)
        gt_masks = [mask for mask in gt_masks if mask is not None]
        if not pred_masks or not gt_masks:
            if not pred_masks and not gt_masks:
                per_image_scores.append(1.0)
            else:
                per_image_scores.append(0.0)
            continue

        remaining = set(range(len(pred_masks)))
        image_scores: List[float] = []
        for gt_mask in gt_masks:
            best_index = None
            best_score = -1.0
            for pred_index in remaining:
                score = mask_iou(pred_masks[pred_index], gt_mask)
                if score > best_score:
                    best_score = score
                    best_index = pred_index
            if best_index is not None:
                remaining.remove(best_index)
                image_scores.append(best_score)
        per_image_scores.append(sum(image_scores) / max(len(gt_masks), 1))
    return sum(per_image_scores) / max(len(per_image_scores), 1)


@dataclass
class EpisodicAccumulator:
    correct: int = 0
    total: int = 0
    loss_sum: float = 0.0
    updates: int = 0
    extras: Dict[str, float] = field(default_factory=dict)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float | None = None):
        predictions = logits.argmax(dim=-1)
        self.correct += int((predictions == labels).sum().item())
        self.total += int(labels.numel())
        if loss is not None:
            self.loss_sum += float(loss)
        self.updates += 1

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(self.updates, 1)

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        key = f"{prefix}/" if prefix else ""
        result = {
            f"{key}acc": self.accuracy,
            f"{key}correct": float(self.correct),
            f"{key}total": float(self.total),
        }
        if self.updates > 0:
            result[f"{key}loss"] = self.avg_loss
        return result


def format_metrics_table(metrics: Dict[str, float | int | str]) -> str:
    width = max((len(key) for key in metrics), default=0)
    lines = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        lines.append(f"{key:<{width}} : {rendered}")
    return "\n".join(lines)
