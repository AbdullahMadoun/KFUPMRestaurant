# =============================================================================
# FILE: losses.py
# CATEGORY: TRAIN
# PURPOSE: Stage-specific and joint loss definitions used by training and evaluation.
# DEPENDENCIES: None
# USED BY: benchmark_runtime.py, pipeline.py, tests/test_allocation.py, train_joint.py, train_stage3_hf.py
# KEY CLASSES/FUNCTIONS: giou_loss, Stage1Loss, dice_loss, Stage2Loss, Stage3Loss, _expand_class_adjustment, _sanitize_logits, JointLoss
# LAST MODIFIED: 2026-03-21T13:17:55.384683+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
trifoodnet/losses/losses.py
────────────────────────────
All loss functions in one place for easy modification.

Stage 1  : GIoU + CrossEntropy(coarse label)
Stage 2  : BCE + Dice
Stage 3  : CrossEntropy(episode class)
Joint    : weighted sum of the above
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Grounding loss
# ──────────────────────────────────────────────────────────────────────────────

def giou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Generalised IoU loss for axis-aligned boxes.
    pred_boxes, gt_boxes : [N, 4] in (x1, y1, x2, y2) format.
    Returns scalar mean loss.
    """
    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h

    # Areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    gt_area   = (gt_boxes[:, 2]   - gt_boxes[:, 0]).clamp(min=0)   * \
                (gt_boxes[:, 3]   - gt_boxes[:, 1]).clamp(min=0)
    union     = pred_area + gt_area - inter

    iou       = inter / (union + 1e-6)

    # Enclosing box
    enc_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enc_area - union) / (enc_area + 1e-6)
    return (1 - giou).mean()


class Stage1Loss(nn.Module):
    """
    L1 = lambda_giou * GIoU(pred_box, gt_box) + lambda_ce * CE(pred_label, gt_label)

    Note: Qwen is a generative model — box coordinates are decoded from text tokens.
    The HF model computes token-level CE loss internally when `labels` is passed.
    This auxiliary loss is used when you have extracted box logits from a detection head
    attached on top of Qwen's hidden states (optional extension).

    For the standard training loop, set use_auxiliary=False and rely on the
    autoregressive label-prediction loss from Qwen directly.
    """

    def __init__(self, lambda_giou: float = 1.0, lambda_ce: float = 0.5):
        super().__init__()
        self.lambda_giou = lambda_giou
        self.lambda_ce   = lambda_ce

    def forward(
        self,
        lm_loss:     torch.Tensor,               # autoregressive loss from HF model
        pred_boxes:  Optional[torch.Tensor] = None,  # [N, 4]
        gt_boxes:    Optional[torch.Tensor] = None,  # [N, 4]
        pred_labels: Optional[torch.Tensor] = None,  # [N, C] logits
        gt_labels:   Optional[torch.Tensor] = None,  # [N]
    ) -> Tuple[torch.Tensor, dict]:
        loss = lm_loss
        info = {"lm_loss": lm_loss.item()}

        if pred_boxes is not None and gt_boxes is not None and len(pred_boxes) > 0:
            g = giou_loss(pred_boxes, gt_boxes)
            loss = loss + self.lambda_giou * g
            info["giou_loss"] = g.item()

        if pred_labels is not None and gt_labels is not None:
            c = F.cross_entropy(pred_labels, gt_labels)
            loss = loss + self.lambda_ce * c
            info["ce_loss"] = c.item()

        info["total"] = loss.item()
        return loss, info


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Segmentation loss
# ──────────────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, gt: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)
    inter = (p * g).sum()
    return 1 - (2 * inter + smooth) / (p.sum() + g.sum() + smooth)


class Stage2Loss(nn.Module):
    """
    L2 = bce_weight * BCE(pred, gt) + dice_weight * Dice(pred, gt)
    Both terms act on per-pixel probabilities (after sigmoid).
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        pred_logits: torch.Tensor,   # [B, H, W]  raw logits
        gt_masks:    torch.Tensor,   # [B, H, W]  binary float
    ) -> Tuple[torch.Tensor, dict]:
        bce  = F.binary_cross_entropy_with_logits(pred_logits, gt_masks)
        dice = dice_loss(torch.sigmoid(pred_logits), gt_masks)
        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss, {"bce": bce.item(), "dice": dice.item(), "total": loss.item()}


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Episodic classification loss
# ──────────────────────────────────────────────────────────────────────────────

# --- Snapshot note: Current Stage 3 classification loss wrapper, including the cross-entropy path used by the best run. ---
class Stage3Loss(nn.Module):
    """
    L3 = CrossEntropy(logits, gt_class) over the query set of each episode.
    Supports simple long-tail variants for minority classes.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        kind: str = "cross_entropy",
        logit_adjust_tau: float = 1.0,
    ):
        super().__init__()
        self.kind = kind
        self.logit_adjust_tau = logit_adjust_tau
        self.label_smoothing = float(label_smoothing)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits:      torch.Tensor,   # [B*Q, N]
        gt_labels:   torch.Tensor,   # [B*Q]
        sample_per_class: Optional[torch.Tensor] = None,   # [B, N] or [N]
    ) -> Tuple[torch.Tensor, dict]:
        adjusted_logits = logits
        if self.kind == "balanced_softmax":
            if sample_per_class is None:
                raise ValueError("sample_per_class is required for balanced_softmax")
            adjusted_logits = logits + _expand_class_adjustment(logits, sample_per_class).log()
        elif self.kind == "logit_adjusted":
            if sample_per_class is None:
                raise ValueError("sample_per_class is required for logit_adjusted")
            priors = _expand_class_adjustment(logits, sample_per_class)
            priors = priors / priors.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            adjusted_logits = logits + self.logit_adjust_tau * priors.clamp_min(1e-12).log()
        elif self.kind != "cross_entropy":
            raise ValueError(f"Unsupported Stage3 loss kind: {self.kind}")

        had_nonfinite_logits = not torch.isfinite(adjusted_logits).all()
        adjusted_logits = _sanitize_logits(adjusted_logits)
        raw_logits = _sanitize_logits(logits)

        if self.label_smoothing > 0.0 and had_nonfinite_logits:
            loss = F.cross_entropy(adjusted_logits, gt_labels, label_smoothing=0.0)
        else:
            loss = self.ce(adjusted_logits, gt_labels)
        if not torch.isfinite(loss):
            loss = F.cross_entropy(_sanitize_logits(adjusted_logits.float()), gt_labels, label_smoothing=0.0)
        if not torch.isfinite(loss):
            loss = adjusted_logits.new_zeros(())
        acc  = (raw_logits.argmax(dim=-1) == gt_labels).float().mean()
        return loss, {"ce": loss.item(), "acc": acc.item(), "kind": self.kind}


def _expand_class_adjustment(logits: torch.Tensor, sample_per_class: torch.Tensor) -> torch.Tensor:
    sample_per_class = sample_per_class.to(logits.device, dtype=logits.dtype).clamp_min(1.0)
    if sample_per_class.dim() == 1:
        return sample_per_class.unsqueeze(0).expand(logits.shape[0], -1)
    if sample_per_class.dim() != 2:
        raise ValueError("sample_per_class must have shape [N] or [B, N]")
    if logits.shape[0] == sample_per_class.shape[0]:
        return sample_per_class
    if logits.shape[0] % sample_per_class.shape[0] != 0:
        raise ValueError(
            f"Cannot align logits batch {logits.shape[0]} with class counts {sample_per_class.shape}"
        )
    repeat = logits.shape[0] // sample_per_class.shape[0]
    return sample_per_class.repeat_interleave(repeat, dim=0)


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    finite_mask = torch.isfinite(logits)
    if finite_mask.all():
        return logits.clamp(min=-50.0, max=50.0)
    sanitized = logits.clone()
    finite_values = sanitized[finite_mask]
    fill_value = -1e4
    if finite_values.numel() > 0:
        fill_value = min(fill_value, finite_values.min().detach().float().item() - 100.0)
    sanitized = torch.nan_to_num(sanitized, nan=0.0, posinf=1e4, neginf=fill_value)
    return sanitized.clamp(min=-50.0, max=50.0)


# ──────────────────────────────────────────────────────────────────────────────
# Joint loss
# ──────────────────────────────────────────────────────────────────────────────

class JointLoss(nn.Module):
    """
    L_total = lambda1 * L1 + lambda2 * L2 + lambda3 * L3

    Weights are mutable at runtime — just update self.lambda1/2/3.
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.stage1_loss = Stage1Loss()
        self.stage2_loss = Stage2Loss()
        self.stage3_loss = Stage3Loss()

    def forward(
        self,
        loss1: torch.Tensor,
        loss2: torch.Tensor,
        loss3: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        total = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3
        return total, {
            "loss1": loss1.item(),
            "loss2": loss2.item(),
            "loss3": loss3.item(),
            "total": total.item(),
        }
