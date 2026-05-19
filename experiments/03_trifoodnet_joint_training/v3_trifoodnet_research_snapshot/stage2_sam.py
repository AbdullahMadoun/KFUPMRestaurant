# =============================================================================
# FILE: stage2_sam.py
# CATEGORY: ARCH
# PURPOSE: Stage 2 box-prompted segmentation on top of the SAM3 model family.
# DEPENDENCIES: None
# USED BY: benchmark_runtime.py, check_trainable.py, pipeline.py, run_dev_inference.py, run_single_inference.py, tests/test_sam3_allocation.py, tests/test_stage2_sam.py, train_joint.py, validate_pipeline_contracts.py, visualize_val_predictions.py
# KEY CLASSES/FUNCTIONS: load_sam, SAM3Segmenter, global_nms, _scale_normalized_boxes, _box_iou_matrix, _match_queries_to_prompt_boxes, mask_iou, segmentation_loss, dice_loss
# LAST MODIFIED: 2026-03-21T14:59:00.532654+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
trifoodnet/models/stage2_sam.py
────────────────────────────────
Stage 2 — SAM3 box-prompted segmentation.

Frozen: image encoder (ViT-H) + prompt encoder.
Trainable: mask decoder only.

Accepts bounding box prompts from Stage 1 (or GT boxes during curriculum).
Applies global NMS to remove overlapping masks.
"""

from __future__ import annotations
from contextlib import nullcontext
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# SAM3 import shim
# ──────────────────────────────────────────────────────────────────────────────

def load_sam(
    model_name: str, 
    quantization_config: Optional[Any] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> Tuple[nn.Module, Any]:
    """
    Load SAM/SAM2/SAM3 from HF transformers.
    """
    from transformers import AutoModel, AutoProcessor
    
    hf_token = os.environ.get("HF_TOKEN")
    model_kwargs = {}
    processor_kwargs = {}
    if hf_token:
        model_kwargs["token"] = hf_token
        processor_kwargs["token"] = hf_token
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    
    # Check if we should use Sam3 or standard Sam
    if "sam3" in model_name.lower():
        from transformers import Sam3Model, Sam3Processor
        model     = Sam3Model.from_pretrained(model_name, **model_kwargs)
        processor = Sam3Processor.from_pretrained(model_name, **processor_kwargs)
    else:
        # Fallback to Auto (works for SAM and SAM2 in most modern transformers)
        model     = AutoModel.from_pretrained(model_name, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        
    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────────

# --- Snapshot note: Core Stage 2 module: prompt-conditioned SAM mask prediction with prompt/query matching. ---
class SAM3Segmenter(nn.Module):
    """
    SAM3 with only the mask decoder trainable.

    Parameters
    ----------
    model_name              : HF hub id or local path to SAM3 weights
    freeze_image_encoder    : freeze ViT-H image encoder (strongly recommended)
    freeze_prompt_encoder   : freeze prompt encoder
    gradient_checkpointing  : activation checkpointing on mask decoder
    """

    def __init__(
        self,
        model_name:             str = "facebook/sam3",
        freeze_image_encoder:   bool = True,
        freeze_prompt_encoder:  bool = True,
        gradient_checkpointing: bool = False,
        quantization_config:    Optional[Any] = None,
        torch_dtype:            Optional[torch.dtype] = None,
        device:                 Optional[str | torch.device] = None,
        bce_weight:             float = 1.0,
        dice_weight:            float = 1.0,
    ):
        super().__init__()
        resolved_device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = load_sam(
            model_name, 
            quantization_config=quantization_config,
            torch_dtype=torch_dtype
        )
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        if quantization_config is None:
            self.model.to(resolved_device)

        # Freeze / unfreeze components (skip if quantized, as Int4 cannot hold gradients)
        if quantization_config is None:
            self._apply_freeze(freeze_image_encoder, freeze_prompt_encoder)

        # Gradient checkpointing on decoder if supported
        if gradient_checkpointing and hasattr(self.model, "sam_mask_decoder"):
            pass

    def _model_device_dtype(self) -> Tuple[torch.device, Optional[torch.dtype]]:
        for parameter in self.model.parameters():
            if parameter.is_floating_point():
                return parameter.device, parameter.dtype
        for parameter in self.model.parameters():
            return parameter.device, None
        return torch.device("cpu"), None

    # ── freeze logic ──────────────────────────────────────────────────────────

    def _apply_freeze(self, freeze_enc: bool, freeze_prompt: bool):
        for p in self.model.parameters():
            p.requires_grad_(False)

        image_encoder_names = ["vision_encoder", "image_encoder", "sam_vision_encoder"]
        prompt_encoder_names = ["prompt_encoder", "sam_prompt_encoder"]
        decoder_names = ["sam_mask_decoder", "mask_decoder"]

        if not freeze_enc:
            for name in image_encoder_names:
                if hasattr(self.model, name):
                    for p in getattr(self.model, name).parameters():
                        p.requires_grad_(True)

        if not freeze_prompt:
            for name in prompt_encoder_names:
                if hasattr(self.model, name):
                    for p in getattr(self.model, name).parameters():
                        p.requires_grad_(True)

        for name in decoder_names:
            if hasattr(self.model, name):
                for p in getattr(self.model, name).parameters():
                    p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"[SAM3Segmenter] Trainable params: {trainable:,} / {total:,} "
              f"({100*trainable/total:.2f}%)")

    # ── forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,                        # [B, 3, H, W]
        input_boxes:  List[torch.Tensor],                  # list of [N_i, 4] per image
        gt_masks:     Optional[List[List[torch.Tensor]]] = None,  # for loss
    ) -> Dict:
        """
        Returns dict with:
          pred_masks  : List[List[Tensor]]  binary masks per item per image
          mask_logits : List[List[Tensor]]  raw mask logits per item per image
          iou_scores  : List[List[float]]
          loss        : Tensor (if gt_masks provided)
        """
        all_masks, all_logits, all_ious = [], [], []
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        n_items = 0
        loss_sums = {"bce": 0.0, "dice": 0.0, "total": 0.0}

        for b_idx in range(len(input_boxes)):
            boxes = input_boxes[b_idx]      # [N, 4]
            if boxes.shape[0] == 0:
                all_masks.append([])
                all_logits.append([])
                all_ious.append([])
                continue

            img_logits, img_masks, img_ious = self._forward_single(
                pixel_values[b_idx:b_idx+1], boxes
            )

            if gt_masks is not None and gt_masks[b_idx] is not None:
                for pred_logits, gt_m in zip(img_logits, gt_masks[b_idx]):
                    if gt_m is None:
                        continue
                    gt_mask = gt_m.to(pixel_values.device, dtype=pred_logits.dtype)
                    loss_item, loss_info = segmentation_loss(
                        pred_logits,
                        gt_mask,
                        bce_weight=self.bce_weight,
                        dice_weight=self.dice_weight,
                        return_details=True,
                    )
                    total_loss = total_loss + loss_item
                    n_items += 1
                    for key in loss_sums:
                        loss_sums[key] += float(loss_info[key])

            all_masks.append(img_masks)
            all_logits.append(img_logits)
            all_ious.append(img_ious)

        if n_items > 0:
            total_loss = total_loss / n_items

        return {
            "pred_masks": all_masks,
            "mask_logits": all_logits,
            "iou_scores": all_ious,
            "loss":       total_loss,
            "loss_info":  {key: value / max(n_items, 1) for key, value in loss_sums.items()},
        }

    def _forward_single(
        self,
        pixel_values: torch.Tensor,     # [1, 3, H, W]
        boxes: torch.Tensor,            # [N, 4]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """Run SAM on one image for N box prompts."""
        logits_list, masks_list, iou_list = [], [], []

        if self.processor is not None:
            device, model_dtype = self._model_device_dtype()
            inputs = self.processor(
                images=pixel_values,
                input_boxes=boxes.unsqueeze(0).tolist(),
                return_tensors="pt",
            )
            processed_inputs = {}
            for key, value in inputs.items():
                if not isinstance(value, torch.Tensor):
                    processed_inputs[key] = value
                    continue
                if value.is_floating_point() and model_dtype is not None:
                    processed_inputs[key] = value.to(device=device, dtype=model_dtype)
                else:
                    processed_inputs[key] = value.to(device=device)

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=model_dtype)
                if device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}
                else nullcontext()
            )
            with autocast_ctx:
                outputs = self.model(**processed_inputs, multimask_output=False)

            # HF Sam3Model is DETR-style and returns a fixed set of query masks.
            # We must decode those queries back to the input prompt boxes instead
            # of treating every query as an independent prompt output.
            pred_masks = outputs.pred_masks.squeeze(0)
            if pred_masks.ndim != 3:
                raise ValueError(f"Unexpected SAM3 pred_masks shape: {tuple(outputs.pred_masks.shape)}")
            target_size = tuple(int(v) for v in pixel_values.shape[-2:])
            if pred_masks.shape[-2:] != target_size:
                pred_masks = F.interpolate(
                    pred_masks.unsqueeze(1),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            
            pred_scores = None
            if hasattr(outputs, "pred_logits") and outputs.pred_logits is not None:
                try:
                    pred_scores = torch.sigmoid(outputs.pred_logits.reshape(-1).float())
                except Exception:
                    pred_scores = None
            if pred_scores is None and hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
                try:
                    pred_scores = outputs.iou_scores.reshape(-1).float()
                except Exception:
                    pred_scores = None
            if pred_scores is None or pred_scores.numel() != pred_masks.shape[0]:
                pred_scores = torch.ones(pred_masks.shape[0], device=pred_masks.device, dtype=torch.float32)

            if hasattr(outputs, "presence_logits") and outputs.presence_logits is not None:
                presence = torch.sigmoid(outputs.presence_logits.reshape(-1).float())
                if presence.numel() == 1:
                    pred_scores = pred_scores * presence[0]
                elif presence.numel() == pred_scores.numel():
                    pred_scores = pred_scores * presence

            pred_boxes = getattr(outputs, "pred_boxes", None)
            if pred_boxes is not None:
                pred_boxes = pred_boxes.squeeze(0).float()
            if pred_boxes is None or pred_boxes.ndim != 2 or pred_boxes.shape[0] != pred_masks.shape[0]:
                pred_boxes = torch.zeros(pred_masks.shape[0], 4, device=pred_masks.device, dtype=torch.float32)
                pred_boxes[:, 2] = float(target_size[1])
                pred_boxes[:, 3] = float(target_size[0])
            else:
                pred_boxes = _scale_normalized_boxes(
                    pred_boxes,
                    height=target_size[0],
                    width=target_size[1],
                )

            selected_query_indices = _match_queries_to_prompt_boxes(
                prompt_boxes=boxes.to(pred_boxes.device, dtype=torch.float32),
                query_boxes=pred_boxes,
                query_scores=pred_scores.to(pred_boxes.device),
            )

            for query_index in selected_query_indices:
                logits = pred_masks[int(query_index)]
                logits_list.append(logits)
                masks_list.append(logits > 0.0)
                iou_list.append(float(pred_scores[int(query_index)].item()))
        else:
            raise NotImplementedError("Requires SAM3Processor.")

        return logits_list, masks_list, iou_list

    # ── inference with NMS ────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict(
        self,
        pixel_values: torch.Tensor,
        input_boxes:  List[torch.Tensor],
        nms_iou_threshold: float = 0.5,
        score_threshold:   float = 0.5,
    ) -> Dict:
        output = self.forward(pixel_values, input_boxes)
        nms_masks, nms_ious, nms_indices = [], [], []
        for masks, ious in zip(output["pred_masks"], output["iou_scores"]):
            m, s, idx = global_nms(masks, ious, nms_iou_threshold, score_threshold)
            nms_masks.append(m)
            nms_ious.append(s)
            nms_indices.append(idx)
        return {"pred_masks": nms_masks, "iou_scores": nms_ious, "kept_indices": nms_indices}


# ──────────────────────────────────────────────────────────────────────────────
# Global NMS for mask deduplication
# ──────────────────────────────────────────────────────────────────────────────

def global_nms(
    masks: List[torch.Tensor],
    scores: List[float],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> Tuple[List[torch.Tensor], List[float], List[int]]:
    if not masks:
        return [], [], []

    pairs = [
        (idx, m, s)
        for idx, (m, s) in enumerate(zip(masks, scores))
        if s >= score_threshold
    ]
    if not pairs:
        # SAM score calibration can be poor during finetuning. For a prompt-based
        # segmenter we still prefer the best available masks over dropping every
        # prompt and forcing the pipeline into a box-only fallback.
        pairs = [
            (idx, m, s)
            for idx, (m, s) in enumerate(zip(masks, scores))
        ]
    if not pairs:
        return [], [], []

    if iou_threshold <= 0:
        return (
            [mask for _, mask, _ in pairs],
            [float(score) for _, _, score in pairs],
            [int(idx) for idx, _, _ in pairs],
        )

    pairs.sort(key=lambda x: x[2], reverse=True)
    kept_masks, kept_scores, kept_indices = [], [], []

    for idx, mask, score in pairs:
        suppressed = False
        for km in kept_masks:
            if mask_iou(mask, km) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept_masks.append(mask)
            kept_scores.append(score)
            kept_indices.append(idx)

    return kept_masks, kept_scores, kept_indices


def _scale_normalized_boxes(
    boxes: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    scaled = boxes.clone()
    scaled[:, 0::2] = scaled[:, 0::2] * float(width)
    scaled[:, 1::2] = scaled[:, 1::2] * float(height)
    scaled[:, 0::2] = scaled[:, 0::2].clamp(0.0, float(width))
    scaled[:, 1::2] = scaled[:, 1::2].clamp(0.0, float(height))
    return scaled


def _box_iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device if a.numel() else b.device)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0.0) * (a[:, 3] - a[:, 1]).clamp(min=0.0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0.0) * (b[:, 3] - b[:, 1]).clamp(min=0.0))[None, :]
    union = area_a + area_b - inter
    return torch.where(union > 0.0, inter / union, torch.zeros_like(inter))


def _match_queries_to_prompt_boxes(
    prompt_boxes: torch.Tensor,
    query_boxes: torch.Tensor,
    query_scores: torch.Tensor,
) -> List[int]:
    if prompt_boxes.numel() == 0 or query_boxes.numel() == 0:
        return []

    prompt_boxes = prompt_boxes.float()
    query_boxes = query_boxes.float()
    query_scores = query_scores.float()
    iou_matrix = _box_iou_matrix(prompt_boxes, query_boxes)
    combined = iou_matrix + (0.05 * query_scores.unsqueeze(0))

    candidates: List[Tuple[float, float, float, int, int]] = []
    for prompt_index in range(prompt_boxes.shape[0]):
        for query_index in range(query_boxes.shape[0]):
            candidates.append(
                (
                    float(combined[prompt_index, query_index].item()),
                    float(iou_matrix[prompt_index, query_index].item()),
                    float(query_scores[query_index].item()),
                    prompt_index,
                    query_index,
                )
            )
    candidates.sort(reverse=True)

    assignments: List[Optional[int]] = [None] * prompt_boxes.shape[0]
    used_queries = set()
    for _, _, _, prompt_index, query_index in candidates:
        if assignments[prompt_index] is not None or query_index in used_queries:
            continue
        assignments[prompt_index] = query_index
        used_queries.add(query_index)

    if any(query_index is None for query_index in assignments):
        score_order = torch.argsort(query_scores, descending=True).tolist()
        for prompt_index, query_index in enumerate(assignments):
            if query_index is not None:
                continue
            for fallback_query in score_order:
                if fallback_query not in used_queries:
                    assignments[prompt_index] = int(fallback_query)
                    used_queries.add(int(fallback_query))
                    break

    return [int(query_index) for query_index in assignments if query_index is not None]


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).float().sum()
    union = (a | b).float().sum()
    if union == 0:
        return 0.0
    return float(inter / union)


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation loss (BCE + Dice)
# ──────────────────────────────────────────────────────────────────────────────

def segmentation_loss(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    bce_weight:  float = 1.0,
    dice_weight: float = 1.0,
    return_details: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    # SAM 3 might produce different resolution than GT
    if pred.shape != gt.shape:
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode="bilinear").squeeze(0).squeeze(0)

    # Ensure same device
    pred = torch.nan_to_num(pred, nan=0.0, posinf=20.0, neginf=-20.0)
    gt = torch.nan_to_num(gt.to(pred.device), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    bce  = F.binary_cross_entropy_with_logits(pred, gt)
    dice = dice_loss(torch.sigmoid(pred), gt)
    total = bce_weight * bce + dice_weight * dice
    if not torch.isfinite(total):
        total = pred.new_zeros(())
        bce = pred.new_zeros(())
        dice = pred.new_zeros(())
    if return_details:
        return total, {"bce": float(bce.item()), "dice": float(dice.item()), "total": float(total.item())}
    return total


def dice_loss(pred: torch.Tensor, gt: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_flat = pred.contiguous().view(-1)
    gt_flat   = gt.contiguous().view(-1)
    intersection = (pred_flat * gt_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
