"""Stage 2: SAM3 Segment + Crop — text + bbox + point prompts → masks → masked crops.

Prompting strategy:
  - Text: short visual description from VLM (~10 words, SAM CLIP-friendly)
  - Box:  tight bbox from VLM (single or multi-box grid)
  - Points: 2-3 foreground points placed by VLM directly on food surface
"""

import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from config import SAMConfig
from ptypes import VisualItem, Detection, SegmentedItem

logger = logging.getLogger("pipeline")


def extract_masked_crop(image: np.ndarray, mask: np.ndarray, bbox, padding: int = 5) -> np.ndarray:
    """Extract the masked region, cropped to bbox with padding.

    Args:
        image: (H, W, 3) BGR image.
        mask: (H, W) binary mask.
        bbox: (4,) xyxy coordinates.
        padding: pixels of padding around the bbox.

    Returns:
        (crop_H, crop_W, 3) BGR crop with background zeroed out.
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Add padding (clamped to image bounds)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)

    # Apply mask (zero out background)
    masked = image.copy()
    masked[~mask.astype(bool)] = 0

    # Crop to bbox region
    crop = masked[y1:y2, x1:x2]
    return crop


class FoodSegmenter:
    """SAM3 segmenter with text + bbox geometric prompting and mask-based crop extraction.

    Extends V2's SAM3Segmenter with:
    1. Bbox geometric prompting (from grounding branch)
    2. Mask-based crop extraction (new for Stage 3 matching)
    """

    def __init__(self, config: SAMConfig, device: str = "cuda"):
        self.device = device
        self.config = config

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import sam3 as _sam3_pkg

        # Search for BPE file using config paths + runtime fallback
        search_paths = list(config.bpe_search_paths) + [
            os.path.abspath(os.path.join("sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")),
            os.path.join(os.path.dirname(_sam3_pkg.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz"),
        ]

        bpe_path = None
        for path in search_paths:
            if os.path.exists(path):
                bpe_path = path
                break

        if bpe_path is None:
            raise FileNotFoundError("Could not find bpe_simple_vocab_16e6.txt.gz in expected locations.")

        logger.info(f"Loading SAM3 model with bpe_path: {bpe_path}")
        self.model = build_sam3_image_model(bpe_path=bpe_path, device=device)
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=config.confidence_threshold)

    def segment_and_crop(self, image_path: str, items: List[VisualItem]) -> List[SegmentedItem]:
        """Segment an image using visual descriptions + bboxes, then extract masked crops.

        Args:
            image_path: Path to the input image.
            items: List of VisualItem from Stage 1 (description + bbox).

        Returns:
            List of SegmentedItem with masks, bboxes, crops, and scores.
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return []

        pil_image = Image.open(image_path).convert("RGB")
        img_w, img_h = pil_image.size

        # BGR image for crop extraction
        import cv2
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            logger.warning(f"Could not load image with cv2: {image_path}")
            return []

        thresholds = [self.config.confidence_threshold] + [
            f for f in self.config.fallback_thresholds if f < self.config.confidence_threshold
        ]

        # Encode image once (biggest perf win — avoids N re-encodes per plate)
        image_state = self.processor.set_image(pil_image)

        all_segmented: List[SegmentedItem] = []

        for item in items:
            segmented = self._segment_item(pil_image, bgr_image, item, img_w, img_h, thresholds, image_state)
            if segmented is not None:
                all_segmented.append(segmented)

        logger.info(f"Segmented {len(all_segmented)} items from {len(items)} visual descriptions")
        return all_segmented

    def _expand_bbox(self, bbox, img_w: int, img_h: int) -> List[float]:
        """Expand VLM bbox by config fraction. SAM works better with generous bboxes."""
        e = self.config.bbox_expand
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 - bw * e)
        y1 = max(0, y1 - bh * e)
        x2 = min(img_w, x2 + bw * e)
        y2 = min(img_h, y2 + bh * e)
        return [x1, y1, x2, y2]

    def _build_box_prompts(self, bbox, img_w: int, img_h: int) -> List[List[float]]:
        """Build list of [cx, cy, w, h] normalized box prompts for a single item.

        When multi_box_prompt is enabled, returns the full bbox plus an NxN grid
        of sub-boxes within it. Otherwise returns just the full bbox.
        """
        x1, y1, x2, y2 = self._expand_bbox(bbox, img_w, img_h)
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        full_box = [cx, cy, w, h]

        if not self.config.multi_box_prompt:
            return [full_box]

        n = self.config.multi_box_grid
        sub_w = w / n
        sub_h = h / n
        # Normalized top-left of the full box
        box_x0 = cx - w / 2
        box_y0 = cy - h / 2

        boxes = [full_box]
        for row in range(n):
            for col in range(n):
                sub_cx = box_x0 + sub_w * (col + 0.5)
                sub_cy = box_y0 + sub_h * (row + 0.5)
                boxes.append([sub_cx, sub_cy, sub_w, sub_h])

        return boxes

    def _normalize_vlm_points(self, points: List[List[float]], img_w: int, img_h: int) -> List[List[float]]:
        """Convert VLM pixel-coordinate points to normalized [x, y] for SAM3."""
        if not self.config.use_vlm_points or not points:
            return []
        return [[px / img_w, py / img_h] for px, py in points]

    def _segment_item(
        self, pil_image: Image.Image, bgr_image: np.ndarray,
        item: VisualItem, img_w: int, img_h: int,
        thresholds: List[float], image_state: dict,
    ) -> SegmentedItem:
        """Segment a single item with text + bbox + VLM point prompting."""

        box_prompts = self._build_box_prompts(item.bbox, img_w, img_h)
        point_prompts = self._normalize_vlm_points(item.points, img_w, img_h)

        for thresh in thresholds:
            self.processor.confidence_threshold = thresh

            # Deep-copy cached image state (prompts mutate the dict)
            state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in image_state.items()}

            # Text prompt: description is already short (~10 words, old style)
            state = self.processor.set_text_prompt(item.description, state)

            # Bbox geometric prompts
            for box in box_prompts:
                state = self.processor.add_geometric_prompt(box, 1, state)

            # VLM-provided foreground point prompts (placed directly on food)
            # SAM3 may expect [x, y] or [x, y, 0, 0] — try both formats
            for point in point_prompts:
                try:
                    state = self.processor.add_geometric_prompt(point, 1, state)
                except Exception:
                    try:
                        state = self.processor.add_geometric_prompt(point + [0, 0], 1, state)
                    except Exception as e:
                        logger.debug(f"Point prompt failed: {e}")
                        break

            if "masks" in state and state["masks"] is not None:
                masks = state["masks"]
                boxes = state.get("boxes", torch.zeros((masks.shape[0], 4)))
                scores = state.get("scores", torch.zeros(masks.shape[0]))

                if masks.shape[0] > 0:
                    best_idx = scores.argmax().item()
                    mask = masks[best_idx].squeeze().cpu().numpy()
                    box = boxes[best_idx].cpu().numpy()
                    score = float(scores[best_idx])

                    crop = extract_masked_crop(
                        bgr_image, mask, box, padding=self.config.crop_padding
                    )

                    logger.info(
                        f"Segmented '{item.description[:40]}' "
                        f"({len(point_prompts)} points) at thresh {thresh}, score={score:.3f}"
                    )
                    return SegmentedItem(
                        description=item.description,
                        mask=mask,
                        bbox=box,
                        crop=crop,
                        score=score,
                        label=item.label,
                    )

            logger.info(f"No masks for '{item.description[:40]}...' at threshold {thresh}")

        logger.warning(f"Failed to segment: '{item.description[:40]}...'")
        return None
