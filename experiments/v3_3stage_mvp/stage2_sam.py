"""Stage 2: SAM3 Segment + Crop — text + bbox prompts → masks → masked crops."""

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

        all_segmented: List[SegmentedItem] = []

        for item in items:
            segmented = self._segment_item(pil_image, bgr_image, item, img_w, img_h, thresholds)
            if segmented is not None:
                all_segmented.append(segmented)

        logger.info(f"Segmented {len(all_segmented)} items from {len(items)} visual descriptions")
        return all_segmented

    def _segment_item(
        self, pil_image: Image.Image, bgr_image: np.ndarray,
        item: VisualItem, img_w: int, img_h: int,
        thresholds: List[float],
    ) -> SegmentedItem:
        """Segment a single item with text + bbox prompting and dynamic thresholding."""

        for thresh in thresholds:
            self.processor.confidence_threshold = thresh

            state = self.processor.set_image(pil_image)

            # Text prompt: visual description
            state = self.processor.set_text_prompt(item.description, state)

            # Bbox geometric prompt (normalized center-x, center-y, width, height)
            # NOTE: add_geometric_prompt expects [cx, cy, w, h] in normalized coords — verify on GPU server
            x1, y1, x2, y2 = item.bbox
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            state = self.processor.add_geometric_prompt([cx, cy, w, h], 1, state)

            if "masks" in state and state["masks"] is not None:
                masks = state["masks"]
                boxes = state.get("boxes", torch.zeros((masks.shape[0], 4)))
                scores = state.get("scores", torch.zeros(masks.shape[0]))

                if masks.shape[0] > 0:
                    # Take best mask for this item
                    best_idx = scores.argmax().item()
                    mask = masks[best_idx].squeeze().cpu().numpy()
                    box = boxes[best_idx].cpu().numpy()
                    score = float(scores[best_idx])

                    crop = extract_masked_crop(
                        bgr_image, mask, box, padding=self.config.crop_padding
                    )

                    logger.info(f"Segmented '{item.description[:40]}...' at threshold {thresh}, score={score:.3f}")
                    return SegmentedItem(
                        description=item.description,
                        mask=mask,
                        bbox=box,
                        crop=crop,
                        score=score,
                    )

            logger.info(f"No masks for '{item.description[:40]}...' at threshold {thresh}")

        logger.warning(f"Failed to segment: '{item.description[:40]}...'")
        return None
