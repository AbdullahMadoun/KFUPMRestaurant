"""End-to-end integration test on real food images.

Mocks VLM + SAM3 models but runs the full pipeline flow:
  process_image → prompter → segmenter → NMS → group → visualize

Produces actual visualization images in /tmp/grounded_test_output/.

Run:  python3 test_grounded_images.py
"""

import json
import os
import sys
import types
import shutil
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Stub heavy deps
# ---------------------------------------------------------------------------
vllm_mod = types.ModuleType("vllm")


class _FakeLLM:
    def __init__(self, **kw):
        self._responses = []
        self._call_idx = 0

    def chat(self, messages, sampling_params=None):
        resp = self._responses[self._call_idx % len(self._responses)]
        self._call_idx += 1
        out = MagicMock()
        out.outputs = [MagicMock()]
        out.outputs[0].text = resp
        return [out]


class _FakeSamplingParams:
    def __init__(self, **kw):
        pass


vllm_mod.LLM = _FakeLLM
vllm_mod.SamplingParams = _FakeSamplingParams
sys.modules["vllm"] = vllm_mod

sam3_mod = types.ModuleType("sam3")
sam3_mod.__file__ = "/fake/sam3/__init__.py"
sam3_mod.build_sam3_image_model = MagicMock()
sam3_model_mod = types.ModuleType("sam3.model")
sam3_proc_mod = types.ModuleType("sam3.model.sam3_image_processor")
sam3_proc_mod.Sam3Processor = MagicMock
sys.modules["sam3"] = sam3_mod
sys.modules["sam3.model"] = sam3_model_mod
sys.modules["sam3.model.sam3_image_processor"] = sam3_proc_mod

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import PipelineConfig
from pipeline_types import GroundedPrompt, Detection, apply_nms, group_detections
from logger import RunTracker, ImageMetrics

# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------
ASSETS = os.path.join(PROJECT_DIR, "..", "..", "assets", "v1", "context")
CHICKEN = os.path.join(ASSETS, "CHICKEN.jpg")
FISH = os.path.join(ASSETS, "FISH.jpg")
RICE = os.path.join(ASSETS, "RICE.jpg")

OUTPUT_DIR = "/tmp/grounded_test_output"


def _make_fake_mask(h, w, x1, y1, x2, y2):
    """Create a binary mask with a filled rectangle region."""
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[int(y1):int(y2), int(x1):int(x2)] = True
    return mask


class FakeSegmenter:
    """Fake SAM3 that returns rectangular masks at the bbox locations."""

    def segment(self, image_path, prompts):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        detections = []
        for prompt in prompts:
            if prompt.bbox is not None:
                x1, y1, x2, y2 = prompt.bbox
            else:
                # No bbox — generate a centered region as fallback
                x1, y1 = img_w * 0.25, img_h * 0.25
                x2, y2 = img_w * 0.75, img_h * 0.75

            mask = _make_fake_mask(img_h, img_w, x1, y1, x2, y2)
            box = torch.tensor([x1, y1, x2, y2], dtype=torch.float)
            detections.append(Detection(
                label=prompt.label,
                mask=mask,
                box=box,
                score=0.85,
            ))
        return detections, 0.1


class FakePrompter:
    """Fake VLM prompter with canned responses per image."""

    def __init__(self):
        # Map image filename -> (labels, bboxes)
        self.canned = {
            "CHICKEN.jpg": {
                "labels": ["fried chicken pieces", "white rice"],
                "bboxes": {
                    "fried chicken pieces": [50, 30, 350, 280],
                    "white rice": [360, 100, 580, 300],
                },
            },
            "FISH.jpg": {
                "labels": ["grilled fish fillet", "green salad"],
                "bboxes": {
                    "grilled fish fillet": [40, 50, 400, 300],
                    "green salad": [420, 80, 600, 280],
                },
            },
            "RICE.jpg": {
                "labels": ["steamed white rice", "curry sauce"],
                "bboxes": {
                    "steamed white rice": [60, 40, 380, 320],
                    "curry sauce": [200, 150, 500, 350],
                },
            },
        }

    def generate_prompts(self, image_path):
        name = os.path.basename(image_path)
        return self.canned.get(name, {}).get("labels", [])

    def generate_grounded_prompts(self, image_path):
        name = os.path.basename(image_path)
        entry = self.canned.get(name, {"labels": [], "bboxes": {}})
        result = []
        for label in entry["labels"]:
            bbox = entry["bboxes"].get(label)
            result.append(GroundedPrompt(label=label, bbox=bbox))
        return result


def run_pipeline_on_image(image_path, mode, output_subdir):
    """Run the full process_image flow on a single image."""
    config = PipelineConfig()
    config.prompt.prompt_mode = mode
    config.output_dir = os.path.join(OUTPUT_DIR, output_subdir)

    tracker = RunTracker(config.output_dir, asdict(config))
    prompter = FakePrompter()
    segmenter = FakeSegmenter()

    # --- replicate process_image logic ---
    metrics = ImageMetrics(image_path=str(image_path))

    if config.prompt.prompt_mode == "grounded":
        grounded = prompter.generate_grounded_prompts(image_path)
    else:
        grounded = [GroundedPrompt(label=p) for p in prompter.generate_prompts(image_path)]

    metrics.prompts = [g.label for g in grounded]

    raw_detections, threshold_used = segmenter.segment(image_path, grounded)
    metrics.num_detections_raw = len(raw_detections)
    metrics.threshold_used = threshold_used

    kept = apply_nms(raw_detections, config.nms.max_objects, config.nms.iou_threshold)
    metrics.num_detections_after_nms = len(kept)

    results = group_detections(kept)
    metrics.num_labels = len(results)

    from visualizer import visualize
    from pathlib import Path
    image_name = Path(image_path).name
    output_path = str(tracker.viz_dir / f"segmented_{image_name}")
    visualize(image_path, results, output_path, config.viz)
    metrics.visualization_path = output_path

    tracker.add_image_metrics(metrics)
    summary = tracker.finalize()

    return metrics, summary, output_path


class TestOnRealImages(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    def _check_image_exists(self, path):
        self.assertTrue(os.path.exists(path), f"Image not found: {path}")

    # ----- TEXT MODE -----
    def test_text_mode_chicken(self):
        self._check_image_exists(CHICKEN)
        metrics, summary, out = run_pipeline_on_image(CHICKEN, "text", "text_mode")
        self.assertGreater(metrics.num_detections_after_nms, 0)
        self.assertTrue(os.path.exists(out), f"Viz not saved: {out}")
        img = cv2.imread(out)
        self.assertIsNotNone(img, "Output image unreadable")
        print(f"  [text] CHICKEN: {metrics.num_detections_after_nms} detections, "
              f"labels={metrics.prompts}, saved={out}")

    def test_text_mode_fish(self):
        self._check_image_exists(FISH)
        metrics, summary, out = run_pipeline_on_image(FISH, "text", "text_mode")
        self.assertGreater(metrics.num_detections_after_nms, 0)
        self.assertTrue(os.path.exists(out))
        print(f"  [text] FISH: {metrics.num_detections_after_nms} detections, "
              f"labels={metrics.prompts}, saved={out}")

    def test_text_mode_rice(self):
        self._check_image_exists(RICE)
        metrics, summary, out = run_pipeline_on_image(RICE, "text", "text_mode")
        self.assertGreater(metrics.num_detections_after_nms, 0)
        self.assertTrue(os.path.exists(out))
        print(f"  [text] RICE: {metrics.num_detections_after_nms} detections, "
              f"labels={metrics.prompts}, saved={out}")

    # ----- GROUNDED MODE -----
    def test_grounded_mode_chicken(self):
        self._check_image_exists(CHICKEN)
        metrics, summary, out = run_pipeline_on_image(CHICKEN, "grounded", "grounded_mode")
        self.assertGreater(metrics.num_detections_after_nms, 0)
        self.assertTrue(os.path.exists(out))
        img = cv2.imread(out)
        self.assertIsNotNone(img)
        print(f"  [grounded] CHICKEN: {metrics.num_detections_after_nms} detections, "
              f"labels={metrics.prompts}, saved={out}")

    def test_grounded_mode_fish(self):
        self._check_image_exists(FISH)
        metrics, summary, out = run_pipeline_on_image(FISH, "grounded", "grounded_mode")
        self.assertGreater(metrics.num_detections_after_nms, 0)
        self.assertTrue(os.path.exists(out))
        print(f"  [grounded] FISH: {metrics.num_detections_after_nms} detections, "
              f"labels={metrics.prompts}, saved={out}")

    def test_grounded_mode_rice(self):
        self._check_image_exists(RICE)
        metrics, summary, out = run_pipeline_on_image(RICE, "grounded", "grounded_mode")
        self.assertGreater(metrics.num_detections_after_nms, 0)
        self.assertTrue(os.path.exists(out))
        print(f"  [grounded] RICE: {metrics.num_detections_after_nms} detections, "
              f"labels={metrics.prompts}, saved={out}")

    # ----- COMPARISON -----
    def test_grounded_has_tighter_masks_than_text(self):
        """In grounded mode the masks should be localized to the bbox region,
        while text-only mode uses a big centered fallback region.  So grounded
        masks should cover fewer pixels."""
        self._check_image_exists(CHICKEN)

        # text mode
        prompter = FakePrompter()
        segmenter = FakeSegmenter()
        text_prompts = [GroundedPrompt(label=p) for p in prompter.generate_prompts(CHICKEN)]
        text_dets, _ = segmenter.segment(CHICKEN, text_prompts)

        # grounded mode
        grounded_prompts = prompter.generate_grounded_prompts(CHICKEN)
        grounded_dets, _ = segmenter.segment(CHICKEN, grounded_prompts)

        text_pixels = sum(d.mask.sum().item() for d in text_dets)
        grounded_pixels = sum(d.mask.sum().item() for d in grounded_dets)

        print(f"  text mask pixels: {text_pixels:,}  vs  grounded mask pixels: {grounded_pixels:,}")
        self.assertLess(grounded_pixels, text_pixels,
                        "Grounded masks should be tighter (fewer pixels) than text-only fallback")

    # ----- SUMMARY JSON -----
    def test_run_summary_written(self):
        self._check_image_exists(CHICKEN)
        _, summary, _ = run_pipeline_on_image(CHICKEN, "grounded", "summary_test")
        self.assertGreater(summary.total_images, 0)
        self.assertEqual(summary.total_images, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
