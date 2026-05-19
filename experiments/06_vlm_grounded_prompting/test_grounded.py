"""Tests for the grounded prompting mode.

Mocks vllm and sam3 so tests run on CPU without GPU dependencies.
Run:  python3 test_grounded.py
"""

import json
import os
import sys
import types
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing project modules
# ---------------------------------------------------------------------------

# Stub vllm
vllm_mod = types.ModuleType("vllm")

class _FakeLLM:
    def __init__(self, **kw):
        self._response = ""
    def chat(self, messages, sampling_params=None):
        out = MagicMock()
        out.outputs = [MagicMock()]
        out.outputs[0].text = self._response
        return [out]

class _FakeSamplingParams:
    def __init__(self, **kw):
        pass

vllm_mod.LLM = _FakeLLM
vllm_mod.SamplingParams = _FakeSamplingParams
sys.modules["vllm"] = vllm_mod

# Stub sam3
sam3_mod = types.ModuleType("sam3")
sam3_mod.__file__ = "/fake/sam3/__init__.py"
sam3_mod.build_sam3_image_model = MagicMock()
sam3_model_mod = types.ModuleType("sam3.model")
sam3_proc_mod = types.ModuleType("sam3.model.sam3_image_processor")
sam3_proc_mod.Sam3Processor = MagicMock
sys.modules["sam3"] = sam3_mod
sys.modules["sam3.model"] = sam3_model_mod
sys.modules["sam3.model.sam3_image_processor"] = sam3_proc_mod

# Add project dir to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import PipelineConfig, PromptConfig, QwenConfig
from pipeline_types import Detection, GroundedPrompt, apply_nms, group_detections

import torch
import numpy as np


# ===================================================================
# 1. GroundedPrompt dataclass tests
# ===================================================================
class TestGroundedPrompt(unittest.TestCase):

    def test_text_only(self):
        gp = GroundedPrompt(label="fried chicken")
        self.assertEqual(gp.label, "fried chicken")
        self.assertIsNone(gp.bbox)

    def test_with_bbox(self):
        gp = GroundedPrompt(label="rice", bbox=[10.0, 20.0, 100.0, 200.0])
        self.assertEqual(gp.bbox, [10.0, 20.0, 100.0, 200.0])

    def test_default_bbox_none(self):
        gp = GroundedPrompt(label="salad")
        self.assertIsNone(gp.bbox)


# ===================================================================
# 2. Config tests
# ===================================================================
class TestPromptConfig(unittest.TestCase):

    def test_defaults(self):
        pc = PromptConfig()
        self.assertEqual(pc.prompt_mode, "text")
        self.assertIn("{label}", pc.grounding_template)

    def test_grounding_template_format(self):
        pc = PromptConfig()
        rendered = pc.grounding_template.format(label="fried chicken")
        self.assertIn("fried chicken", rendered)
        self.assertNotIn("{label}", rendered)
        # Should contain bbox_2d instruction
        self.assertIn("bbox_2d", rendered)

    def test_pipeline_config_json_roundtrip(self):
        """Ensure new fields survive to_json / from_json."""
        import tempfile
        cfg = PipelineConfig()
        cfg.prompt.prompt_mode = "grounded"

        path = os.path.join(tempfile.gettempdir(), "test_cfg.json")
        cfg.to_json(path)
        loaded = PipelineConfig.from_json(path)
        self.assertEqual(loaded.prompt.prompt_mode, "grounded")
        self.assertIn("{label}", loaded.prompt.grounding_template)
        os.unlink(path)

    def test_backward_compat_json_no_new_fields(self):
        """Old JSON configs missing new fields should still load fine."""
        import tempfile
        cfg = PipelineConfig()
        d = asdict(cfg)
        # Simulate old config: remove new fields
        del d["prompt"]["grounding_template"]
        del d["prompt"]["prompt_mode"]

        path = os.path.join(tempfile.gettempdir(), "test_old_cfg.json")
        with open(path, "w") as f:
            json.dump(d, f)

        loaded = PipelineConfig.from_json(path)
        # Should fall back to defaults
        self.assertEqual(loaded.prompt.prompt_mode, "text")
        os.unlink(path)


# ===================================================================
# 3. Prompter tests (mocked VLM)
# ===================================================================
class TestQwenFoodPrompter(unittest.TestCase):

    def _make_prompter(self):
        from qwen_food_prompter import QwenFoodPrompter
        return QwenFoodPrompter(QwenConfig(), PromptConfig())

    def test_generate_grounded_prompts_success(self):
        prompter = self._make_prompter()

        # Phase 1: generate_prompts returns food labels
        food_json = json.dumps({"food_items": ["fried chicken", "rice"]})
        # Phase 2: _locate_item returns bbox for each
        bbox1_json = json.dumps({"bbox_2d": [10, 20, 100, 200]})
        bbox2_json = json.dumps({"bbox_2d": [150, 50, 300, 250]})

        responses = [food_json, bbox1_json, bbox2_json]
        call_count = [0]

        def fake_chat(messages, sampling_params=None):
            out = MagicMock()
            out.outputs = [MagicMock()]
            out.outputs[0].text = responses[call_count[0]]
            call_count[0] += 1
            return [out]

        prompter.llm.chat = fake_chat

        # Need a real image path — create a tiny temp image
        import tempfile
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.new("RGB", (320, 240)).save(tmp.name)

        result = prompter.generate_grounded_prompts(tmp.name)
        os.unlink(tmp.name)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "fried chicken")
        self.assertEqual(result[0].bbox, [10.0, 20.0, 100.0, 200.0])
        self.assertEqual(result[1].label, "rice")
        self.assertEqual(result[1].bbox, [150.0, 50.0, 300.0, 250.0])

    def test_generate_grounded_prompts_bbox_parse_failure_fallback(self):
        """When bbox parsing fails, should fall back to text-only (bbox=None)."""
        prompter = self._make_prompter()

        food_json = json.dumps({"food_items": ["salad"]})
        bad_bbox = "not valid json at all"

        responses = [food_json, bad_bbox]
        call_count = [0]

        def fake_chat(messages, sampling_params=None):
            out = MagicMock()
            out.outputs = [MagicMock()]
            out.outputs[0].text = responses[call_count[0]]
            call_count[0] += 1
            return [out]

        prompter.llm.chat = fake_chat

        import tempfile
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.new("RGB", (320, 240)).save(tmp.name)

        result = prompter.generate_grounded_prompts(tmp.name)
        os.unlink(tmp.name)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "salad")
        self.assertIsNone(result[0].bbox)  # graceful fallback

    def test_generate_grounded_prompts_empty_labels(self):
        """When no food items detected, should return empty list."""
        prompter = self._make_prompter()

        food_json = json.dumps({"food_items": []})

        def fake_chat(messages, sampling_params=None):
            out = MagicMock()
            out.outputs = [MagicMock()]
            out.outputs[0].text = food_json
            return [out]

        prompter.llm.chat = fake_chat

        import tempfile
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.new("RGB", (320, 240)).save(tmp.name)

        result = prompter.generate_grounded_prompts(tmp.name)
        os.unlink(tmp.name)

        self.assertEqual(result, [])

    def test_generate_grounded_prompts_markdown_json(self):
        """VLM sometimes wraps JSON in ```json ... ```. Should still parse."""
        prompter = self._make_prompter()

        food_md = '```json\n{"food_items": ["bread"]}\n```'
        bbox_md = '```json\n{"bbox_2d": [5, 10, 50, 60]}\n```'

        responses = [food_md, bbox_md]
        call_count = [0]

        def fake_chat(messages, sampling_params=None):
            out = MagicMock()
            out.outputs = [MagicMock()]
            out.outputs[0].text = responses[call_count[0]]
            call_count[0] += 1
            return [out]

        prompter.llm.chat = fake_chat

        import tempfile
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.new("RGB", (320, 240)).save(tmp.name)

        result = prompter.generate_grounded_prompts(tmp.name)
        os.unlink(tmp.name)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox, [5.0, 10.0, 50.0, 60.0])

    def test_text_mode_unchanged(self):
        """generate_prompts() should still work identically."""
        prompter = self._make_prompter()

        food_json = json.dumps({"food_items": ["fried chicken", "rice"]})

        def fake_chat(messages, sampling_params=None):
            out = MagicMock()
            out.outputs = [MagicMock()]
            out.outputs[0].text = food_json
            return [out]

        prompter.llm.chat = fake_chat

        import tempfile
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.new("RGB", (320, 240)).save(tmp.name)

        result = prompter.generate_prompts(tmp.name)
        os.unlink(tmp.name)

        self.assertEqual(result, ["fried chicken", "rice"])


# ===================================================================
# 4. Segmenter bbox-coordinate conversion test
# ===================================================================
class TestSegmenterBBoxConversion(unittest.TestCase):

    def test_xyxy_to_cxcywh_normalized(self):
        """Verify the pixel→normalized conversion math used in sam3_segmenter."""
        img_w, img_h = 640, 480
        # bbox in pixel xyxy
        x1, y1, x2, y2 = 100.0, 50.0, 300.0, 250.0

        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        self.assertAlmostEqual(cx, 200.0 / 640)
        self.assertAlmostEqual(cy, 150.0 / 480)
        self.assertAlmostEqual(w, 200.0 / 640)
        self.assertAlmostEqual(h, 200.0 / 480)

        # All values should be 0-1
        for val in [cx, cy, w, h]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_full_image_bbox(self):
        """A bbox covering the whole image should map to cx=0.5, cy=0.5, w=1, h=1."""
        img_w, img_h = 320, 240
        x1, y1, x2, y2 = 0.0, 0.0, 320.0, 240.0

        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(w, 1.0)
        self.assertAlmostEqual(h, 1.0)


# ===================================================================
# 5. NMS + group_detections still work with GroundedPrompt flow
# ===================================================================
class TestNMSAndGrouping(unittest.TestCase):

    def _make_detection(self, label, score, size=10):
        mask = torch.zeros(100, 100, dtype=torch.bool)
        mask[:size, :size] = True
        box = torch.tensor([0, 0, size, size], dtype=torch.float)
        return Detection(label=label, mask=mask, box=box, score=score)

    def test_nms_keeps_best(self):
        d1 = self._make_detection("chicken", 0.9)
        d2 = self._make_detection("chicken", 0.5)  # overlapping, lower score
        kept = apply_nms([d1, d2], max_objects=5, iou_threshold=0.7)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].score, 0.9)

    def test_grouping(self):
        d1 = self._make_detection("chicken", 0.9)
        d2 = self._make_detection("rice", 0.8, size=20)
        groups = group_detections([d1, d2])
        labels = {g.label for g in groups}
        self.assertEqual(labels, {"chicken", "rice"})


# ===================================================================
# 6. main.py CLI wiring test
# ===================================================================
class TestCLIWiring(unittest.TestCase):

    def test_build_config_prompt_mode_text_default(self):
        from main import build_config_from_args
        args = MagicMock()
        args.config = None
        args.device = None
        args.output_dir = None
        args.threshold = None
        args.max_objects = None
        args.iou_threshold = None
        args.alpha = None
        args.thickness = None
        args.skip_boxes = False
        args.prompt_mode = None

        cfg = build_config_from_args(args)
        self.assertEqual(cfg.prompt.prompt_mode, "text")

    def test_build_config_prompt_mode_grounded(self):
        from main import build_config_from_args
        args = MagicMock()
        args.config = None
        args.device = None
        args.output_dir = None
        args.threshold = None
        args.max_objects = None
        args.iou_threshold = None
        args.alpha = None
        args.thickness = None
        args.skip_boxes = False
        args.prompt_mode = "grounded"

        cfg = build_config_from_args(args)
        self.assertEqual(cfg.prompt.prompt_mode, "grounded")


# ===================================================================
# 7. process_image text-mode wraps labels in GroundedPrompt
# ===================================================================
class TestProcessImageWrapping(unittest.TestCase):

    def test_text_mode_wraps_in_grounded_prompt(self):
        """In text mode, plain string labels should be wrapped as GroundedPrompt(label=..., bbox=None)."""
        labels = ["fried chicken", "rice", "salad"]
        grounded = [GroundedPrompt(label=p) for p in labels]

        self.assertEqual(len(grounded), 3)
        for gp in grounded:
            self.assertIsNone(gp.bbox)
            self.assertIn(gp.label, labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
