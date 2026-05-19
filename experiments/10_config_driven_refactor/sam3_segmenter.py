import os
import logging
from typing import List, Tuple

import torch
from PIL import Image

from config import SAM3Config
from pipeline_types import Detection

logger = logging.getLogger("pipeline")


class SAM3Segmenter:
    """
    Wrapper for SAM3 (Segment Anything Model 3) for food item segmentation.

    Features:
    - Text-prompted segmentation using discrete prompts.
    - Dynamic thresholding: automatically lowers confidence if no masks are found.
    - Returns raw Detection objects (NMS is handled externally).
    """

    def __init__(self, config: SAM3Config, device: str = "cuda"):
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

    def segment(self, image_path: str, prompts: List[str]) -> Tuple[List[Detection], float]:
        """Segment an image with the given text prompts.

        Returns:
            (raw_detections, threshold_used) -- caller is responsible for NMS.
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return [], self.config.confidence_threshold

        image = Image.open(image_path).convert("RGB")

        thresholds = [self.config.confidence_threshold] + [
            f for f in self.config.fallback_thresholds if f < self.config.confidence_threshold
        ]

        for thresh in thresholds:
            logger.info(f"Trying threshold: {thresh}")
            self.processor.confidence_threshold = thresh

            detections: List[Detection] = []
            for prompt in prompts:
                logger.info(f"Segmenting prompt: {prompt}")
                state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(prompt, state)

                if "masks" in state and state["masks"] is not None:
                    masks = state["masks"]
                    boxes = state.get("boxes", torch.zeros((masks.shape[0], 4)))
                    scores = state.get("scores", torch.zeros(masks.shape[0]))

                    for i in range(masks.shape[0]):
                        detections.append(Detection(
                            label=prompt,
                            mask=masks[i].squeeze(),
                            box=boxes[i],
                            score=float(scores[i]),
                        ))

            if detections:
                logger.info(f"Found {len(detections)} candidates at threshold {thresh}.")
                return detections, thresh
            else:
                logger.info(f"No candidates found at threshold {thresh}.")

        return [], thresholds[-1] if thresholds else self.config.confidence_threshold
