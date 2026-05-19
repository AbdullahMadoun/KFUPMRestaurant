# =============================================================================
# FILE: tests/test_stage2_sam.py
# CATEGORY: TEST
# PURPOSE: Snapshot-retained source file for test_stage2_sam.py.
# DEPENDENCIES: stage2_sam.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: test_scale_normalized_boxes_projects_to_pixel_space, test_match_queries_to_prompt_boxes_prefers_high_iou_queries
# LAST MODIFIED: 2026-03-21T15:03:23.934338+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_sam import _match_queries_to_prompt_boxes, _scale_normalized_boxes


def test_scale_normalized_boxes_projects_to_pixel_space():
    boxes = torch.tensor([[0.1, 0.2, 0.9, 0.8]], dtype=torch.float32)
    scaled = _scale_normalized_boxes(boxes, height=640, width=320)
    assert torch.allclose(scaled, torch.tensor([[32.0, 128.0, 288.0, 512.0]]))


def test_match_queries_to_prompt_boxes_prefers_high_iou_queries():
    prompt_boxes = torch.tensor(
        [
            [10.0, 10.0, 50.0, 50.0],
            [60.0, 60.0, 100.0, 100.0],
        ],
        dtype=torch.float32,
    )
    query_boxes = torch.tensor(
        [
            [9.0, 9.0, 49.0, 49.0],
            [61.0, 61.0, 101.0, 101.0],
            [0.0, 0.0, 20.0, 20.0],
        ],
        dtype=torch.float32,
    )
    query_scores = torch.tensor([0.8, 0.9, 0.99], dtype=torch.float32)

    selected = _match_queries_to_prompt_boxes(prompt_boxes, query_boxes, query_scores)

    assert selected == [0, 1]
