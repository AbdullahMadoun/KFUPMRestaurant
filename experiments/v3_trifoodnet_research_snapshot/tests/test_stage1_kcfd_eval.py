from __future__ import annotations

import math

import numpy as np

from stage1_kcfd.eval import average_metrics, evaluate_pair, iou_matrix
from stage1_kcfd.schema import Stage1Item, Stage1Target, parse_prediction


def _target(*boxes: list[int]) -> Stage1Target:
    return Stage1Target(
        items=[
            Stage1Item(name=f"item-{idx}", bbox=box, descriptor="visible textured food region")
            for idx, box in enumerate(boxes, start=1)
        ]
    )


def test_iou_matrix_is_class_agnostic_and_spatial_only():
    ious = iou_matrix(
        pred=np.asarray([[0, 0, 10, 10], [30, 30, 40, 40]]),
        gt=np.asarray([[0, 0, 10, 10], [5, 5, 15, 15]]),
    )

    assert ious.shape == (2, 2)
    assert ious[0, 0] == 1.0
    assert 0.0 < ious[0, 1] < 1.0


def test_evaluate_pair_rewards_exact_count_and_exact_set_match():
    gt = _target([0, 0, 10, 10], [20, 20, 40, 40])
    pred = _target([0, 0, 10, 10], [20, 20, 40, 40])

    metrics = evaluate_pair(pred, gt, valid=True)

    assert metrics["valid_json_rate"] == 1.0
    assert metrics["exact_count_accuracy"] == 1.0
    assert metrics["matched_precision@0.5"] == 1.0
    assert metrics["matched_recall@0.5"] == 1.0
    assert metrics["matched_f1@0.5"] == 1.0
    assert metrics["exact_set_match@0.5"] == 1.0
    assert metrics["mean_matched_iou"] == 1.0


def test_evaluate_pair_penalizes_overcount_even_when_gt_boxes_match():
    gt = _target([0, 0, 10, 10], [20, 20, 40, 40])
    pred = _target([0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 60])

    metrics = evaluate_pair(pred, gt, valid=True)

    assert metrics["exact_count_accuracy"] == 0.0
    assert metrics["overcount_rate"] == 1.0
    assert metrics["undercount_rate"] == 0.0
    assert math.isclose(metrics["matched_precision@0.5"], 2 / 3)
    assert metrics["matched_recall@0.5"] == 1.0
    assert metrics["exact_set_match@0.5"] == 0.0


def test_parse_prediction_invalid_json_returns_empty_prediction_without_crashing():
    valid, pred, error = parse_prediction("not json")

    assert valid is False
    assert pred.items == []
    assert error


def test_average_metrics_uses_present_denominators_for_conditional_metrics():
    metrics = average_metrics([
        {"exact_count_accuracy": 1.0, "size_small/recall@0.50": 1.0},
        {"exact_count_accuracy": 0.0},
    ])

    assert metrics["exact_count_accuracy"] == 0.5
    assert metrics["size_small/recall@0.50"] == 1.0
