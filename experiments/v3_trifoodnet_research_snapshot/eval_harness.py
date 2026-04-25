# =============================================================================
# FILE: eval_harness.py
# CATEGORY: EVAL
# PURPOSE: Single canonical evaluation entry point used by training, scripts,
#          and ablations. Replaces the historical split between
#          ``evaluate_objective`` and ``evaluate_inference_dataset``.
# DEPENDENCIES: pipeline.py, metrics.py, losses.py
# USED BY: train_joint.py (and downstream ablation scripts)
# KEY CLASSES/FUNCTIONS: EvalMode, EvalReport, evaluate_split, COMBINED_FORMULA_VERSION
# =============================================================================
"""
One eval function, one schema, one combined formula.

Why this exists:
    The previous codebase had two eval functions that produced overlapping but
    inconsistent metric dicts:
      - evaluate_objective         : teacher-forced loss eval (gt boxes/masks)
      - evaluate_inference_dataset : end-to-end inference (real Qwen → SAM → ICL)
    The combined-metric formula lived inline in train_joint.py:478, so any
    ablation that wanted to compute it elsewhere had to copy the line — and
    that's how the early-stopping/best-checkpoint mismatch happened.

The harness:
    - takes a `mode` enum so callers explicitly choose teacher_forced vs end_to_end
      (oracle modes are reserved for the registry-driven ablations in Phase 4)
    - returns a schema-locked EvalReport so adding/removing a metric is a
      visible code change, not a silent dict drift
    - centralizes the `combined` definition

If you change the metric set or the combined formula, bump COMBINED_FORMULA_VERSION
so a downstream consumer can detect the change.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional

import torch


COMBINED_FORMULA_VERSION = 1
"""Bump when the `dev/combined` formula changes. Logged with every report."""


class EvalMode(str, Enum):
    """How the pipeline is invoked during evaluation."""

    TEACHER_FORCED = "teacher_forced"   # GT boxes routed into Stage 2/3 (decoupled metric)
    END_TO_END = "end_to_end"           # Stage 1 generates boxes → Stage 2 → Stage 3
    ORACLE_STAGE1 = "oracle_stage1"     # reserved for ablation campaigns
    ORACLE_STAGE2 = "oracle_stage2"
    ORACLE_STAGE3 = "oracle_stage3"


@dataclass
class EvalReport:
    """Schema-locked output of one evaluation pass."""

    split: str
    mode: str
    dataset_version: str
    dataset_hash: str
    n_images: int
    n_items: int
    n_pred_items: int
    n_matches: int
    n_nan_batches: int
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    combined: float = 0.0
    combined_formula_version: int = COMBINED_FORMULA_VERSION

    def flat_metrics(self, prefix: str) -> Dict[str, float]:
        """Flatten into a logger-friendly dict.

        Example: prefix="dev" produces keys like
        ``dev/stage1_recall@0.5``, ``dev/stage3_episode_acc``,
        ``dev/combined``, ``dev/n_images``.
        """
        out: Dict[str, float] = {}
        for k, v in self.metrics.items():
            out[f"{prefix}/{k}"] = float(v)
        for k, v in self.latency_ms.items():
            out[f"{prefix}/latency_{k}_ms"] = float(v)
        out[f"{prefix}/combined"] = float(self.combined)
        out[f"{prefix}/n_images"] = float(self.n_images)
        out[f"{prefix}/n_items"] = float(self.n_items)
        out[f"{prefix}/n_pred_items"] = float(self.n_pred_items)
        out[f"{prefix}/n_matches"] = float(self.n_matches)
        out[f"{prefix}/n_nan_batches"] = float(self.n_nan_batches)
        out[f"{prefix}/combined_formula_version"] = float(self.combined_formula_version)
        return out

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def combined_score(metrics: Dict[str, float]) -> float:
    """Single source of truth for the joint headline metric.

    formula v1:   stage1_recall@0.5 + stage2_mIoU + stage3_acc

    The components are end-to-end (after Stage 1 prediction). When called on a
    teacher-forced report, stage1_recall@0.5 is missing → contribution is 0,
    making the metric automatically less informative under teacher forcing.
    """
    return float(
        float(metrics.get("stage1_recall@0.5", 0.0))
        + float(metrics.get("stage2_mIoU", 0.0))
        + float(metrics.get("stage3_acc", 0.0))
    )


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation primitives
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_objective_loss(
    pipeline,
    eval_loader,
    *,
    device: torch.device,
    cfg,
    amp_dtype: Optional[torch.dtype],
    stage3_loss_fn,
) -> Dict[str, float]:
    """Teacher-forced loss eval. Returns averaged per-stage and total losses
    plus the episodic accuracy (computed in pipeline.forward via Stage3Loss).
    """
    from train_joint import _to_device, _accumulate_metrics  # avoid circular import

    pipeline.eval()
    loss_sums: Dict[str, float] = {}
    num_batches = 0
    correct = 0
    total = 0

    for batch in eval_loader:
        batch = _to_device(batch, device)
        ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with ctx:
            out = pipeline.forward(
                batch,
                use_gt_boxes=True,
                loss_weights=(
                    cfg.joint.loss_weights.lambda1,
                    cfg.joint.loss_weights.lambda2,
                    cfg.joint.loss_weights.lambda3,
                ),
                stage3_loss_fn=stage3_loss_fn,
                strict_finite=True,
            )
        m = out["metrics"]
        loss_sums = _accumulate_metrics(
            loss_sums,
            {
                "loss_stage1": float(out["loss_stage1"].item()),
                "loss_stage2": float(out["loss_stage2"].item()),
                "loss_stage3": float(out["loss_stage3"].item()),
                "loss_total": float(out["loss_total"].item()),
                "stage1_lm_loss": float(m["stage1/lm_loss"]),
                "stage2_bce_loss": float(m["stage2/bce_loss"]),
                "stage2_dice_loss": float(m["stage2/dice_loss"]),
                "stage3_ce_loss": float(m["stage3/ce_loss"]),
            },
        )
        n = int(batch["query_labels"].numel())
        correct += int(round(float(m["stage3/acc"]) * n))
        total += n
        num_batches += 1

    averaged = {key: value / max(num_batches, 1) for key, value in loss_sums.items()}
    averaged["stage3_episode_acc"] = correct / max(total, 1)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return averaged


@torch.no_grad()
def evaluate_inference_loop(
    pipeline,
    eval_dataset,
    *,
    cfg,
) -> tuple[Dict[str, float], Dict[str, float], int, int, int]:
    """End-to-end inference over a dataset split. Returns (metrics, latency, n_images, n_items, n_pred_items, n_matches)."""
    from metrics import greedy_box_matches, mask_iou  # avoid circular import

    pipeline.eval()
    threshold = float(cfg.stage1.eval.iou_threshold)
    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_class_correct = 0
    total_mask_iou = 0.0
    latency_sums: Dict[str, float] = {"stage1": 0.0, "stage2": 0.0, "stage3": 0.0, "total": 0.0}

    # Retrieval-vs-transformer diagnostic accumulators.
    # For each matched item we record:
    #   - retrieval_hit          : was the GT class even in the top-K candidates the
    #                              cosine retriever sent to the transformer?
    #   - given_recalled         : of items where retrieval surfaced the GT class, how
    #                              often did the transformer pick it correctly?
    #   - cosine_top1_correct    : was the rank-1 most-similar class (i.e., the answer
    #                              if we skipped the transformer entirely) the GT class?
    # The third one is a "transformer-removed" baseline. Comparing stage3_acc against
    # stage3_cosine_top1_acc tells us whether the transformer adds real lift over pure
    # cosine retrieval, or whether it's just rubber-stamping the retriever's pick.
    # Items whose pred_item carries no `candidate_classes` field (legacy callers) are
    # silently skipped.
    diag_total_with_candidates = 0
    diag_retrieval_hits = 0
    diag_given_recalled_correct = 0
    diag_cosine_top1_correct = 0

    for index in range(len(eval_dataset)):
        sample = eval_dataset[index]
        output = pipeline.run(
            pil_image=sample["pil_image"],
            image_id=str(sample["image_id"]),
            nms_iou_threshold=cfg.stage2.nms.iou_threshold,
            score_threshold=cfg.stage2.nms.score_threshold,
            top_k_classes=1,
        )
        pred_items = list(output.items)
        gt_items = list(sample.get("stage1_items", []))
        gt_masks = list(sample.get("masks", []))
        pred_boxes = [item.box for item in pred_items]
        gt_boxes = [item["box"] for item in gt_items]
        matches = greedy_box_matches(pred_boxes, gt_boxes, threshold=threshold)

        total_gt += len(gt_items)
        total_pred += len(pred_items)
        total_matches += len(matches)

        for match in matches:
            pred_item = pred_items[match.pred_index]
            gt_item = gt_items[match.gt_index]
            gt_mask = gt_masks[match.gt_index] if match.gt_index < len(gt_masks) else None
            if pred_item.mask is not None and gt_mask is not None:
                total_mask_iou += mask_iou(pred_item.mask, gt_mask)
            classified_correct = (pred_item.label == gt_item["label"])
            if classified_correct:
                total_class_correct += 1

            # Diagnostic: was GT class in the cosine retriever's top-K candidates?
            candidates = getattr(pred_item, "candidate_classes", None)
            if candidates is not None and len(candidates) > 0:
                diag_total_with_candidates += 1
                if gt_item["label"] in candidates:
                    diag_retrieval_hits += 1
                    if classified_correct:
                        diag_given_recalled_correct += 1
                # Cosine top-1 baseline: would picking the most-similar class
                # (without consulting the transformer) have been correct?
                # candidate_classes is ordered by descending similarity → [0] is rank-1.
                if candidates[0] == gt_item["label"]:
                    diag_cosine_top1_correct += 1

        for key, value in output.latency_ms.items():
            latency_sums[key] = latency_sums.get(key, 0.0) + float(value)

    n_images = max(len(eval_dataset), 1)
    metrics = {
        "stage1_recall@0.5": total_matches / max(total_gt, 1),
        "stage1_precision@0.5": total_matches / max(total_pred, 1),
        "stage2_mIoU": total_mask_iou / max(total_gt, 1),
        "stage3_acc": total_class_correct / max(total_gt, 1),
        "stage3_matched_acc": total_class_correct / max(total_matches, 1),
        "pred_items_per_image": total_pred / n_images,
    }
    # Diagnostic metrics emitted only when the inference path actually populated
    # candidate_classes (i.e., new pipeline; legacy callers will not have them).
    if diag_total_with_candidates > 0:
        metrics["stage3_retrieval_recall@K"] = diag_retrieval_hits / max(diag_total_with_candidates, 1)
        metrics["stage3_acc_given_retrieved"] = diag_given_recalled_correct / max(diag_retrieval_hits, 1)
        # Transformer-removed baseline: how often is the retriever's rank-1 already correct?
        metrics["stage3_cosine_top1_acc"] = diag_cosine_top1_correct / max(diag_total_with_candidates, 1)
        # Lift = how much does the transformer add over pure retrieval. Positive = good,
        # near-zero = transformer not earning its keep, negative = transformer hurts.
        # We compare on the same denominator (items where retrieval ran).
        cosine_top1_acc = diag_cosine_top1_correct / max(diag_total_with_candidates, 1)
        # Same denominator restriction for the transformer-side comparison: count
        # transformer-correct only over items where retrieval ran (so the comparison
        # is apples-to-apples). diag_given_recalled_correct already counts these
        # but only counts hits that landed in retrieval AND the transformer got right —
        # we need transformer-correct-on-items-with-candidates regardless of whether
        # retrieval surfaced GT. Recompute:
        # (this is a lower-bound for the transformer; if retrieval missed entirely
        # the transformer can't be right anyway)
        metrics["stage3_transformer_lift_over_top1"] = (
            (diag_given_recalled_correct - diag_cosine_top1_correct) / max(diag_total_with_candidates, 1)
        )
        metrics["n_with_candidates"] = float(diag_total_with_candidates)
        metrics["n_retrieval_hits"] = float(diag_retrieval_hits)
        metrics["n_cosine_top1_correct"] = float(diag_cosine_top1_correct)
    latency = {
        "stage1": latency_sums["stage1"] / n_images,
        "stage2": latency_sums["stage2"] / n_images,
        "stage3": latency_sums["stage3"] / n_images,
        "total": latency_sums["total"] / n_images,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, latency, n_images, total_gt, total_pred, total_matches


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_split(
    pipeline,
    *,
    dataset,
    loader,
    mode: EvalMode,
    cfg,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    stage3_loss_fn,
    split_name: str,
) -> EvalReport:
    """Run one full evaluation pass and return a schema-locked report.

    Parameters
    ----------
    dataset : the JointFoodDataset for the split (used by end-to-end inference)
    loader  : the DataLoader for the split (used by teacher-forced loss eval)
    mode    : EvalMode — TEACHER_FORCED, END_TO_END, or oracle variants

    Behavior:
        - TEACHER_FORCED: runs only the loss eval (cheap, used for early stopping)
        - END_TO_END:     runs both loss eval AND inference loop (full report)
        - ORACLE_*:       reserved for Phase 4 stage registry; not implemented yet
    """
    if mode in (EvalMode.ORACLE_STAGE1, EvalMode.ORACLE_STAGE2, EvalMode.ORACLE_STAGE3):
        raise NotImplementedError(
            f"Oracle eval mode {mode!r} is reserved for Phase 4 (stage registry). "
            "Implement after build_stage{1,2,3} factories land."
        )

    nan_counts_before = pipeline.get_nan_counts()
    metrics: Dict[str, float] = {}
    latency: Dict[str, float] = {}
    n_pred_items = 0
    n_matches = 0
    n_items = 0
    n_images = len(dataset) if dataset is not None else 0

    if mode in (EvalMode.TEACHER_FORCED, EvalMode.END_TO_END):
        loss_metrics = evaluate_objective_loss(
            pipeline, loader,
            device=device, cfg=cfg, amp_dtype=amp_dtype, stage3_loss_fn=stage3_loss_fn,
        )
        metrics.update(loss_metrics)

    if mode == EvalMode.END_TO_END:
        e2e_metrics, latency, n_images, n_items, n_pred_items, n_matches = evaluate_inference_loop(
            pipeline, dataset, cfg=cfg,
        )
        metrics.update(e2e_metrics)

    nan_counts_after = pipeline.get_nan_counts()
    nan_delta = sum(
        max(0, nan_counts_after.get(k, 0) - nan_counts_before.get(k, 0))
        for k in nan_counts_after
    )

    return EvalReport(
        split=split_name,
        mode=mode.value,
        dataset_version=getattr(dataset, "dataset_version", "unknown") if dataset is not None else "unknown",
        dataset_hash=getattr(dataset, "dataset_hash", "") if dataset is not None else "",
        n_images=n_images,
        n_items=n_items,
        n_pred_items=n_pred_items,
        n_matches=n_matches,
        n_nan_batches=nan_delta,
        metrics=metrics,
        latency_ms=latency,
        combined=combined_score(metrics),
    )
