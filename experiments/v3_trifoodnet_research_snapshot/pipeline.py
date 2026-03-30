# =============================================================================
# FILE: pipeline.py
# CATEGORY: ARCH
# PURPOSE: End-to-end TriFoodNet orchestration for inference, checkpoint I/O, and joint loss wiring.
# DEPENDENCIES: item_processing.py, losses.py, stage1_qwen.py, stage2_sam.py, stage3_icl.py
# USED BY: benchmark_runtime.py, check_trainable.py, run_dev_inference.py, run_single_inference.py, train_joint.py, validate_pipeline_contracts.py, visualize_val_predictions.py
# KEY CLASSES/FUNCTIONS: PriceLookup, FoodItem, PipelineOutput, TriFoodNet, _ms, _finite_loss_or_zero, _trainable_state_dict, _load_compatible_state_dict
# LAST MODIFIED: 2026-03-21T14:36:20.391748+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
trifoodnet/models/pipeline.py
──────────────────────────────
TriFoodNet — Full three-stage end-to-end pipeline.

  Stage 1 (QwenGrounder)   → bounding boxes + coarse labels
  Stage 2 (SAMSegmenter)   → pixel masks per box
  Stage 3 (FoodClassifier or official PictSure wrapper)
                        → fine-grained class + price lookup

The pipeline supports:
  • Independent inference (call stage-by-stage)
  • End-to-end inference (call pipeline.run())
  • Joint training (pipeline.forward() returns all losses)
  • Curriculum learning for Stage 2 (GT boxes vs predicted boxes)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from item_processing import masked_crop_from_pil

try:
    from .stage1_qwen import QwenGrounder, parse_detections
    from .stage2_sam import SAM3Segmenter
    from .stage3_icl import FoodClassifier
except ImportError:
    from stage1_qwen import QwenGrounder, parse_detections
    from stage2_sam import SAM3Segmenter
    from stage3_icl import FoodClassifier
try:
    from .losses import Stage3Loss
except ImportError:
    from losses import Stage3Loss


# ──────────────────────────────────────────────────────────────────────────────
# Price lookup  (configurable at runtime)
# ──────────────────────────────────────────────────────────────────────────────

class PriceLookup:
    """Simple dict-backed price table.  Load from JSON or set programmatically."""

    def __init__(self, price_table: Optional[Dict[str, float]] = None):
        self._table: Dict[str, float] = price_table or {}

    def get_price(self, class_name: str, default: float = 0.0) -> float:
        return self._table.get(class_name, default)

    def load_json(self, path: str):
        import json
        with open(path) as f:
            self._table = json.load(f)

    def set_price(self, class_name: str, price: float):
        self._table[class_name] = price


# ──────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FoodItem:
    box:        List[float]
    mask:       Optional[torch.Tensor]   # [H, W] bool
    label:      str
    confidence: float
    price:      float = 0.0

@dataclass
class PipelineOutput:
    items:            List[FoodItem]
    total_price:      float
    latency_ms:       Dict[str, float]   # {"stage1": ..., "stage2": ..., "stage3": ..., "total": ...}
    raw_detections:   List[Dict]         # Stage 1 raw output
    image_id:         Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

# --- Snapshot note: Top-level orchestrator that chains the three stages for inference and joint loss computation. ---
class TriFoodNet(nn.Module):
    """
    Full TriFoodNet inference + training pipeline.

    Parameters
    ----------
    stage1       : QwenGrounder
    stage2       : SAM3Segmenter
    stage3       : FoodClassifier
    price_lookup : PriceLookup (optional)
    """

    def __init__(
        self,
        stage1:       QwenGrounder,
        stage2:       SAM3Segmenter,
        stage3:       FoodClassifier,
        price_lookup: Optional[PriceLookup] = None,
        debug:        bool = False,
    ):
        super().__init__()
        self.stage1       = stage1
        self.stage2       = stage2
        self.stage3       = stage3
        self.price_lookup = price_lookup or PriceLookup()
        self.debug        = bool(debug)

    # ── single-image inference ────────────────────────────────────────────────

    @torch.inference_mode()
    def run(
        self,
        pil_image:         Image.Image,
        nms_iou_threshold: float = 0.5,
        score_threshold:   float = 0.5,
        top_k_classes:     int   = 1,
        image_id:          Optional[str] = None,
    ) -> PipelineOutput:
        """
        Full end-to-end inference on one PIL dish image.
        Target < 2000ms total latency.
        """
        t0 = time.perf_counter()

        # ── Stage 1: Visual grounding ──────────────────────────────────────
        t1_start = time.perf_counter()
        detections_list = self.stage1.generate_detections([pil_image])
        detections = detections_list[0] if detections_list else []
        t1_end = time.perf_counter()

        if not detections:
            if self.debug:
                print("[Pipeline Debug] Zero detections from Stage 1.")
            return PipelineOutput(
                items=[], total_price=0.0,
                latency_ms={"stage1": _ms(t1_start, t1_end), "stage2": 0, "stage3": 0,
                            "total": _ms(t0, t1_end)},
                raw_detections=[],
                image_id=image_id,
            )

        # ── Stage 2: Segmentation ─────────────────────────────────────────
        t2_start = time.perf_counter()

        import numpy as np
        img_np = np.array(pil_image)
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(next(self.stage2.parameters()).device)

        boxes = torch.tensor(
            [d["box"] for d in detections], dtype=torch.float32
        ).to(img_tensor.device)

        seg_output = self.stage2.predict(
            img_tensor, [boxes],
            nms_iou_threshold=nms_iou_threshold,
            score_threshold=score_threshold,
        )
        masks = seg_output["pred_masks"][0]    # list of [H,W] bool tensors
        kept_indices = seg_output.get("kept_indices", [[]])[0]
        t2_end = time.perf_counter()

        # ── Stage 3: Classification ───────────────────────────────────────
        t3_start = time.perf_counter()
        items: List[FoodItem] = []

        detection_mask_pairs = []
        if masks and kept_indices:
            for det_idx, mask in zip(kept_indices, masks):
                if det_idx < len(detections):
                    detection_mask_pairs.append((detections[det_idx], mask.detach().cpu()))

        # Fallback: if SAM3 filtered everything out but Stage 1 had boxes, 
        # we MUST still classify the Stage 1 boxes (no-mask fallback).
        if not detection_mask_pairs and detections:
            if self.debug:
                print(f"[Pipeline Debug] SAM3 returned 0 kept_indices for {len(detections)} detections. Falling back to all boxes.")
            detection_mask_pairs = [(det, None) for det in detections]

        for det, mask in detection_mask_pairs:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            crop = (
                masked_crop_from_pil(pil_image, mask, det["box"])
                if mask is not None
                else pil_image.crop((x1, y1, x2, y2))
            )

            preds = self.stage3.classify(
                crop,
                device=img_tensor.device.type,
                top_k=top_k_classes,
            )
            best_class, confidence = preds[0] if preds else (det.get("label", "unknown"), 0.0)
            price = self.price_lookup.get_price(best_class)

            items.append(FoodItem(
                box=det["box"],
                mask=mask,
                label=best_class,
                confidence=confidence,
                price=price,
            ))

        t3_end = time.perf_counter()
        total_price = sum(it.price for it in items)

        return PipelineOutput(
            items=items,
            total_price=total_price,
            latency_ms={
                "stage1": _ms(t1_start, t1_end),
                "stage2": _ms(t2_start, t2_end),
                "stage3": _ms(t3_start, t3_end),
                "total":  _ms(t0, t3_end),
            },
            raw_detections=detections,
            image_id=image_id,
        )

    # ── joint training forward ────────────────────────────────────────────────

    def forward(
        self,
        batch:         dict,
        use_gt_boxes:  bool = True,
        loss_weights:  Tuple[float, float, float] = (1.0, 1.0, 1.0),
        stage3_loss_fn: Optional[Stage3Loss] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Joint training forward pass.

        Parameters
        ----------
        batch         : collated batch from FoodDataset("full")
        use_gt_boxes  : True  → use GT boxes as SAM prompts (curriculum epoch 1)
                        False → use Stage 1 predicted boxes
        loss_weights  : (lambda1, lambda2, lambda3)

        Returns
        -------
        dict with keys: loss_stage1, loss_stage2, loss_stage3, loss_total
        """
        lam1, lam2, lam3 = loss_weights
        loss_device = batch["images"].device if "images" in batch else next(self.stage2.parameters()).device

        # Stage 1 forward (returns HF loss)
        s1_out = self.stage1(
            input_ids      = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            pixel_values   = batch.get("pixel_values"),
            image_grid_thw = batch.get("image_grid_thw"),
            video_grid_thw = batch.get("video_grid_thw"),
            labels         = batch.get("s1_labels"),
        )
        loss1 = s1_out.loss if s1_out.loss is not None else torch.zeros((), device=loss_device)
        loss1 = _finite_loss_or_zero(loss1)
        stage1_metrics = {"stage1/lm_loss": float(loss1.detach().item())}

        # Teacher forcing uses GT boxes during training. Inference/eval can switch
        # to Qwen-generated boxes by setting use_gt_boxes=False.
        prompt_boxes = batch["boxes"] if use_gt_boxes else self._stage1_prompt_boxes(batch["pil_images"], batch["boxes"])
        s2_out = self.stage2(
            pixel_values = batch["images"],
            input_boxes  = prompt_boxes,
            gt_masks     = batch.get("masks"),
        )
        loss2 = _finite_loss_or_zero(s2_out["loss"])
        stage2_loss_info = s2_out.get("loss_info", {})
        stage2_metrics = {
            "stage2/bce_loss": float(stage2_loss_info.get("bce", 0.0)),
            "stage2/dice_loss": float(stage2_loss_info.get("dice", 0.0)),
            "stage2/loss": float(loss2.detach().item()),
        }

        # Stage 3 forward (episodic — only if episode data present)
        loss3 = torch.zeros((), device=loss_device)
        stage3_metrics = {
            "stage3/ce_loss": 0.0,
            "stage3/acc": 0.0,
        }
        if "support_images" in batch and "support_labels" in batch:
            logits = self.stage3(
                support_images = batch["support_images"],
                query_images   = batch["query_images"],
                support_labels = batch["support_labels"],
            )
            if stage3_loss_fn is not None:
                loss3, stage3_loss_info = stage3_loss_fn(
                    logits,
                    batch["query_labels"],
                    sample_per_class=batch.get("episode_class_counts"),
                )
                loss3 = _finite_loss_or_zero(loss3)
                stage3_metrics = {
                    "stage3/ce_loss": float(stage3_loss_info["ce"]),
                    "stage3/acc": float(stage3_loss_info["acc"]),
                }
            else:
                import torch.nn.functional as F
                loss3 = F.cross_entropy(logits, batch["query_labels"])
                loss3 = _finite_loss_or_zero(loss3)
                stage3_metrics = {
                    "stage3/ce_loss": float(loss3.detach().item()),
                    "stage3/acc": float((logits.argmax(dim=-1) == batch["query_labels"]).float().mean().item()),
                }

        loss_total = _finite_loss_or_zero(lam1 * loss1 + lam2 * loss2 + lam3 * loss3)
        metrics = {
            **stage1_metrics,
            **stage2_metrics,
            **stage3_metrics,
            "joint/loss": float(loss_total.detach().item()),
        }

        return {
            "loss_stage1": loss1,
            "loss_stage2": loss2,
            "loss_stage3": loss3,
            "loss_total":  loss_total,
            "metrics": metrics,
        }

    # ── serialization ─────────────────────────────────────────────────────────

    def save(self, checkpoint_dir: str):
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.stage1.save_lora(os.path.join(checkpoint_dir, "stage1_lora"))
        torch.save(
            _trainable_state_dict(self.stage2),
            os.path.join(checkpoint_dir, "stage2.pt"),
        )
        torch.save(
            _trainable_state_dict(self.stage3),
            os.path.join(checkpoint_dir, "stage3.pt"),
        )
        print(f"[TriFoodNet] Checkpoint saved to {checkpoint_dir}")

    def load(self, checkpoint_dir: str):
        import os
        # Load Stage 1 LoRA (safe for quantized base)
        s1_path = os.path.join(checkpoint_dir, "stage1_lora")
        if os.path.exists(s1_path):
            self.stage1.load_lora(s1_path)
            
        # Load Stage 2 (Selective)
        s2_path = os.path.join(checkpoint_dir, "stage2.pt")
        if os.path.exists(s2_path):
            try:
                s2_state = torch.load(s2_path, map_location="cpu")
                loaded_keys = _load_compatible_state_dict(self.stage2, s2_state)
                if loaded_keys == 0:
                    print("[TriFoodNet] Stage 2 checkpoint had no compatible weights to load.")
            except Exception as e:
                print(f"[TriFoodNet] Skipping Stage 2 weight load (likely quantization mismatch): {e}")
                
        # Load Stage 3 full trainable state when available. Fall back to the
        # older transformer-only checkpoint format for backward compatibility.
        s3_full_path = os.path.join(checkpoint_dir, "stage3.pt")
        if os.path.exists(s3_full_path):
            try:
                s3_state = torch.load(s3_full_path, map_location="cpu")
                loaded_keys = _load_compatible_state_dict(self.stage3, s3_state)
                if loaded_keys == 0:
                    print("[TriFoodNet] Stage 3 checkpoint had no compatible weights to load.")
            except Exception as e:
                print(f"[TriFoodNet] Skipping Stage 3 weight load: {e}")
        s3_path = os.path.join(checkpoint_dir, "stage3_icl.pt")
        if os.path.exists(s3_path):
            try:
                stage3_icl = getattr(self.stage3.icl, "_orig_mod", self.stage3.icl)
                s3_state = torch.load(s3_path, map_location="cpu")
                loaded_keys = _load_compatible_state_dict(stage3_icl, s3_state)
                if loaded_keys == 0:
                    print("[TriFoodNet] Stage 3 checkpoint had no compatible weights to load.")
            except Exception as e:
                print(f"[TriFoodNet] Skipping Stage 3 weight load: {e}")
        print(f"[TriFoodNet] Checkpoint load sequence complete from {checkpoint_dir}")

    def _stage1_prompt_boxes(self, pil_images: List[Image.Image], fallback_boxes: List[torch.Tensor]) -> List[torch.Tensor]:
        detections = self.stage1.generate_detections(list(pil_images))
        predicted_boxes: List[torch.Tensor] = []
        for dets, fallback in zip(detections, fallback_boxes):
            if dets:
                predicted_boxes.append(
                    torch.tensor([det["box"] for det in dets], dtype=torch.float32, device=fallback.device)
                )
            else:
                predicted_boxes.append(fallback)
        return predicted_boxes


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ms(t_start: float, t_end: float) -> float:
    return round((t_end - t_start) * 1000, 2)


def _finite_loss_or_zero(loss: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(loss).all():
        return loss
    return loss.detach().new_zeros(())


def _trainable_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
        if name in trainable_names
    }


def _load_compatible_state_dict(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
    current_state = module.state_dict()
    compatible = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in current_state and current_state[name].shape == tensor.shape
    }
    if compatible:
        module.load_state_dict(compatible, strict=False)
    return len(compatible)
