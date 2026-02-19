import logging
import random
from typing import List

import cv2
import numpy as np

from config import VizConfig
from pipeline_types import SegmentationResult

logger = logging.getLogger("pipeline")


def _random_color(config: VizConfig):
    """Generate a saturated, bold color using config-driven ranges."""
    lo_min, lo_max = config.color_low
    hi_min, hi_max = config.color_high
    channels = [random.randint(lo_min, lo_max), random.randint(hi_min, hi_max), random.randint(0, 255)]
    random.shuffle(channels)
    return tuple(channels)


def visualize(image_path: str, results: List[SegmentationResult], output_path: str, config: VizConfig):
    """Render segmentation masks and labels onto the original image.

    Uses a local color mapping -- does not mutate result objects.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not load image {image_path}")
        return

    output = image.copy()
    overlay = image.copy()

    # Build local color mapping (no mutation of result objects)
    color_map = {item.label: _random_color(config) for item in results}

    # First pass: draw masks and contours on overlay
    for item in results:
        color = color_map[item.label]
        for i in range(item.masks.shape[0]):
            mask = item.masks[i].squeeze()
            if mask.ndim > 2:
                mask = mask[0]
            mask_bool = mask > 0

            for c in range(3):
                overlay[:, :, c] = np.where(mask_bool, color[c], overlay[:, :, c])

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, config.thickness)

    # Blend overlay with original
    cv2.addWeighted(overlay, config.alpha, output, 1 - config.alpha, 0, output)

    # Second pass: boxes and labels (drawn opaque on blended image)
    for item in results:
        color = color_map[item.label]
        for i in range(item.masks.shape[0]):
            box = item.boxes[i] if len(item.boxes) > i else None
            num_masks = item.masks.shape[0]
            label_text = item.label
            if num_masks > 1:
                label_text += f" {i+1}/{num_masks}"

            if box is not None and config.draw_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, color, config.font_thickness)
            else:
                mask = item.masks[i]
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    M = cv2.moments(contours[0])
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(output, label_text, (cX, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, color, config.font_thickness)

    cv2.imwrite(output_path, output)
    logger.info(f"Saved visualization to {output_path}")
