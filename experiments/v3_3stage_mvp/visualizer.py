"""Visualization: render masks + match labels + confidence + prices on the original image."""

import logging
import random
from typing import List

import cv2
import numpy as np

from config import VizConfig
from ptypes import MatchResult

logger = logging.getLogger("pipeline")

# Category → color tone mapping (BGR)
CATEGORY_COLORS = {
    "protein": (60, 60, 200),     # red tones
    "carb": (50, 200, 220),       # yellow tones
    "salad": (80, 180, 80),       # green tones
    "side": (200, 150, 50),       # blue-cyan tones
    "soup": (100, 100, 200),      # warm orange
    "drink": (200, 100, 50),      # cool blue
    "dessert": (180, 100, 200),   # purple tones
    "unknown": (160, 160, 160),   # gray
}


def _get_color(category: str, config: VizConfig) -> tuple:
    """Get a color for a category, with slight random variation for visual distinction."""
    base = CATEGORY_COLORS.get(category, CATEGORY_COLORS["unknown"])
    # Add slight variation so items of same category are distinguishable
    variation = 25
    color = tuple(
        max(0, min(255, c + random.randint(-variation, variation)))
        for c in base
    )
    return color


def _build_label(match: MatchResult, config: VizConfig) -> str:
    """Build the label text for a match result."""
    parts = []

    if config.show_match_label:
        if match.menu_item == "unknown":
            parts.append(f"? Unknown")
        else:
            parts.append(match.menu_item)

    if config.show_confidence:
        parts.append(f"({match.confidence:.2f})")

    if config.show_price and match.price > 0:
        parts.append(f"— {match.price:.0f} SAR")

    return " ".join(parts)


def visualize(image_path: str, matches: List[MatchResult], output_path: str, config: VizConfig):
    """Render segmentation masks, match labels, and prices onto the original image.

    Features over V2:
    - Labels show menu item name + confidence + price instead of raw descriptions
    - Colors are category-coded (protein=red, carb=yellow, salad=green, etc.)
    - Bottom bar shows total price and item count
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not load image: {image_path}")
        return

    output = image.copy()
    overlay = image.copy()

    # Assign colors per match
    colors = [_get_color(m.category, config) for m in matches]

    # First pass: draw masks and contours on overlay
    for match, color in zip(matches, colors):
        mask = match.segmented.mask.squeeze()
        if mask.ndim > 2:
            mask = mask[0]
        mask_bool = mask > 0

        for c in range(3):
            overlay[:, :, c] = np.where(mask_bool, color[c], overlay[:, :, c])

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, config.thickness)

    # Blend overlay with original
    cv2.addWeighted(overlay, config.alpha, output, 1 - config.alpha, 0, output)

    # Second pass: boxes and labels (drawn opaque on blended image)
    for match, color in zip(matches, colors):
        label_text = _build_label(match, config)
        bbox = match.segmented.bbox

        if bbox is not None and config.draw_boxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Label background for readability
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, config.font_thickness
            )
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                output, label_text, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, config.font_scale,
                (255, 255, 255), config.font_thickness,
            )
        else:
            # Fallback: place label at mask centroid
            mask = match.segmented.mask.squeeze()
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(
                        output, label_text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, color, config.font_thickness,
                    )

    # Bottom bar: total price + item count
    known_matches = [m for m in matches if m.menu_item != "unknown"]
    total_price = sum(m.price for m in known_matches)
    item_count = len(matches)
    unknown_count = item_count - len(known_matches)

    bar_text = f"Total: {total_price:.0f} SAR | {item_count} items"
    if unknown_count > 0:
        bar_text += f" ({unknown_count} unknown)"

    h, w = output.shape[:2]
    bar_height = 40
    cv2.rectangle(output, (0, h - bar_height), (w, h), (40, 40, 40), -1)
    cv2.putText(
        output, bar_text, (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    cv2.imwrite(output_path, output)
    logger.info(f"Saved visualization to {output_path}")
