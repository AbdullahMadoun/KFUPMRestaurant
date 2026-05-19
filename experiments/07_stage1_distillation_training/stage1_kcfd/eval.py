from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .model import first_parameter_device, model_input_tensors, tensor_batch_to_device
from .qwen_io import build_user_conversation, processor_batch
from .schema import Stage1Item, Stage1Target, descriptor_word_count, parse_prediction, target_to_payload


def _boxes(items: Sequence[Stage1Item]) -> np.ndarray:
    return np.asarray([item.bbox for item in items], dtype=np.float64).reshape((-1, 4))


def _area(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    wh = np.clip(boxes[:, 2:4] - boxes[:, 0:2], a_min=0.0, a_max=None)
    return wh[:, 0] * wh[:, 1]


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.size == 0 or gt.size == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    lt = np.maximum(pred[:, None, :2], gt[None, :, :2])
    rb = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = _area(pred)[:, None] + _area(gt)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def giou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.size == 0 or gt.size == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    iou = iou_matrix(pred, gt)
    lt = np.minimum(pred[:, None, :2], gt[None, :, :2])
    rb = np.maximum(pred[:, None, 2:], gt[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    enclosing = wh[:, :, 0] * wh[:, :, 1]
    pred_area = _area(pred)[:, None]
    gt_area = _area(gt)[None, :]
    inter = iou * (pred_area + gt_area) / np.maximum(1.0 + iou, 1e-9)
    union = pred_area + gt_area - inter
    return iou - (enclosing - union) / np.maximum(enclosing, 1e-9)


def hungarian_matches(ious: np.ndarray) -> List[Tuple[int, int]]:
    if ious.size == 0:
        return []
    rows, cols = linear_sum_assignment(1.0 - ious)
    return [(int(row), int(col)) for row, col in zip(rows, cols)]


def _bucket_gt_count(n: int) -> str:
    if n <= 1:
        return "count_0_1"
    if n <= 3:
        return "count_2_3"
    if n <= 5:
        return "count_4_5"
    return "count_6_plus"


def _size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def evaluate_pair(pred: Stage1Target, gt: Stage1Target, valid: bool) -> Dict[str, float]:
    pred_boxes = _boxes(pred.items)
    gt_boxes = _boxes(gt.items)
    ious = iou_matrix(pred_boxes, gt_boxes)
    gious = giou_matrix(pred_boxes, gt_boxes)
    pairs = hungarian_matches(ious)
    n_pred = len(pred.items)
    n_gt = len(gt.items)
    out: Dict[str, float] = {
        "valid_json_rate": float(valid),
        "json_schema_accuracy": float(valid),
        "exact_count_accuracy": float(n_pred == n_gt),
        "count_mae": float(abs(n_pred - n_gt)),
        "count_rmse": float((n_pred - n_gt) ** 2),
        "count_bias": float(n_pred - n_gt),
        "overcount_rate": float(n_pred > n_gt),
        "undercount_rate": float(n_pred < n_gt),
        "zero_pred_rate": float(n_pred == 0),
        "bbox_validity_rate": float(valid),
        "descriptor_word_count_valid_rate": float(
            valid and all(5 <= descriptor_word_count(item.descriptor) <= 10 for item in pred.items)
        ),
        "empty_descriptor_rate": float(valid and any(not item.descriptor for item in pred.items)),
        "duplicate_name_rate": float(valid and len({item.name for item in pred.items}) < len(pred.items) and len(pred.items) > 0),
    }
    for threshold in (0.50, 0.75, 0.90):
        matched = [(p, g) for p, g in pairs if ious[p, g] >= threshold]
        precision = len(matched) / max(n_pred, 1)
        recall = len(matched) / max(n_gt, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        suffix = f"{threshold:.2f}"
        out[f"precision@{suffix}"] = float(precision)
        out[f"recall@{suffix}"] = float(recall)
        out[f"f1@{suffix}"] = float(f1)
        if abs(threshold - 0.50) < 1e-9:
            out["matched_precision@0.5"] = float(precision)
            out["matched_recall@0.5"] = float(recall)
            out["matched_f1@0.5"] = float(f1)
            out["exact_set_match@0.5"] = float(n_pred == n_gt and len(matched) == n_gt)
    good50 = [(p, g) for p, g in pairs if ious[p, g] >= 0.5]
    out["mean_matched_iou"] = float(np.mean([ious[p, g] for p, g in good50])) if good50 else 0.0
    out["mean_matched_giou"] = float(np.mean([gious[p, g] for p, g in good50])) if good50 else 0.0
    for k in (1, 3, 5, 10):
        capped_pred = pred_boxes[:k]
        capped_ious = iou_matrix(capped_pred, gt_boxes)
        capped_pairs = hungarian_matches(capped_ious)
        recalls = []
        for threshold in np.arange(0.50, 0.951, 0.05):
            recalls.append(sum(1 for p, g in capped_pairs if capped_ious[p, g] >= threshold) / max(n_gt, 1))
        out[f"AR@{k}"] = float(np.mean(recalls)) if recalls else 0.0
    out["AR@[.50:.95]"] = out["AR@10"]
    bucket = _bucket_gt_count(n_gt)
    out[f"{bucket}/exact_count_accuracy"] = out["exact_count_accuracy"]
    out[f"{bucket}/count_mae"] = out["count_mae"]
    if gt_boxes.size:
        areas = _area(gt_boxes)
        for name in ("small", "medium", "large"):
            selected = [idx for idx, area in enumerate(areas) if _size_bucket(float(area)) == name]
            if not selected:
                continue
            hit = 0
            for _, gt_idx in good50:
                if gt_idx in selected:
                    hit += 1
            out[f"size_{name}/recall@0.50"] = hit / max(len(selected), 1)
    return out


def average_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    result = {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}
    if "count_rmse" in result:
        result["count_rmse"] = float(math.sqrt(result["count_rmse"]))
    return result


@torch.no_grad()
def generate_text(model, processor, image, prompt: str, *, max_new_tokens: int) -> str:
    message = build_user_conversation(image, prompt)
    inputs = processor_batch(processor, [message], add_generation_prompt=True, fallback_images=[image])
    inputs = tensor_batch_to_device(dict(inputs), first_parameter_device(model))
    output_ids = model.generate(**model_input_tensors(inputs), max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


class Stage1Evaluator:
    def __init__(self, *, prompt: str, max_samples: Optional[int] = None):
        self.prompt = prompt
        self.max_samples = max_samples

    def evaluate(self, model, processor, dataset, *, output_path: Path | None = None, max_new_tokens: int = 512) -> Dict[str, float]:
        model.eval()
        rows: List[Dict[str, float]] = []
        details: List[Dict[str, Any]] = []
        n = len(dataset) if self.max_samples is None else min(len(dataset), self.max_samples)
        for idx in range(n):
            example = dataset[idx]
            text = generate_text(model, processor, example["image"], self.prompt, max_new_tokens=max_new_tokens)
            valid, pred, error = parse_prediction(text)
            gt_payload = example["target"]
            gt = Stage1Target(items=[Stage1Item(name=row["name"], bbox=row["bbox"], descriptor=row["descriptor"]) for row in gt_payload["items"]])
            metrics = evaluate_pair(pred, gt, valid)
            rows.append(metrics)
            details.append({
                "image_id": example["image_id"],
                "prediction": text,
                "target": gt_payload,
                "valid": valid,
                "parse_error": error,
                "metrics": metrics,
            })
        summary = average_metrics(rows)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump({"metrics": summary, "examples": details}, handle, indent=2, sort_keys=True)
        return summary
