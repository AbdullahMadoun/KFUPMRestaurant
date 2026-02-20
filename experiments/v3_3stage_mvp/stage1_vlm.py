"""Stage 1: VLM Visual Description — Qwen3-VL generates visual descriptions + bounding boxes."""

import json
import logging
import os
import re
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams

from config import VLMConfig
from ptypes import VisualItem

logger = logging.getLogger("pipeline")


class VisualDescriber:
    """Uses Qwen3-VL via vLLM to describe food items with visual patterns and bounding boxes.

    Single VLM call per image: returns both visual descriptions and bboxes together.
    Visual descriptions (colors, textures, shapes) are better SAM3 prompts than food names.
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        try:
            self.llm = LLM(
                model=config.model_name,
                limit_mm_per_prompt={"image": 1},
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_model_len=config.max_model_len,
                enforce_eager=config.enforce_eager,
                enable_prefix_caching=config.enable_prefix_caching,
                allowed_local_media_path=config.allowed_local_media_path,
            )
            self.sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
            )
        except Exception as e:
            logger.error(f"Error initializing vLLM: {e}")
            raise

    def _build_messages(self, image_path: str) -> List[dict]:
        """Build chat messages for a single image."""
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                {"type": "text", "text": self.config.describe_template},
            ],
        })
        return messages

    def describe(self, image_path: str) -> List[VisualItem]:
        """Generate visual descriptions + bounding boxes for all food items in the image.

        Returns:
            List of VisualItem, each with a visual description and bbox in pixel coordinates.
            On parse failure for an individual item, that item is skipped (not the whole image).
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return []

        try:
            outputs = self.llm.chat(
                messages=self._build_messages(image_path),
                sampling_params=self.sampling_params,
            )
            generated_text = outputs[0].outputs[0].text.strip()
            return self._parse_response(generated_text)
        except Exception as e:
            logger.error(f"Error during VLM generation: {e}")
            return []

    def describe_batch(self, image_paths: List[str]) -> Dict[str, List[VisualItem]]:
        """Batch inference: describe multiple images in one vLLM call.

        vLLM processes all conversations concurrently with continuous batching,
        keeping the GPU saturated instead of idling between serial calls.

        Args:
            image_paths: List of absolute image paths.

        Returns:
            Dict mapping image_path → List[VisualItem]. Failed images get empty lists.
        """
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_paths:
            return {p: [] for p in image_paths}

        batch_messages = [self._build_messages(p) for p in valid_paths]

        try:
            outputs = self.llm.chat(
                messages=batch_messages,
                sampling_params=self.sampling_params,
            )
        except Exception as e:
            logger.error(f"Batch VLM error: {e}")
            return {p: [] for p in image_paths}

        results = {p: [] for p in image_paths}
        for path, output in zip(valid_paths, outputs):
            try:
                generated_text = output.outputs[0].text.strip()
                results[path] = self._parse_response(generated_text)
            except Exception as e:
                logger.error(f"Parse error for {path}: {e}")

        return results

    def _parse_response(self, text: str) -> List[VisualItem]:
        """Parse VLM JSON response into VisualItem list.

        Handles both raw JSON and markdown ```json``` wrapping.
        Skips individual items that fail to parse rather than failing the whole response.
        """
        # Strip markdown code fence if present
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        json_text = match.group(1).strip() if match else text

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse VLM JSON response: {json_text[:200]}")
            return []

        items_data = data.get("items", [])
        if not isinstance(items_data, list):
            logger.warning(f"Expected 'items' to be a list, got: {type(items_data)}")
            return []

        results = []
        for i, item in enumerate(items_data):
            try:
                description = item.get("description", "")
                bbox = item.get("bbox", [])

                if not description:
                    logger.warning(f"Item {i}: missing description, skipping")
                    continue

                if not isinstance(bbox, list) or len(bbox) != 4:
                    logger.warning(f"Item {i}: invalid bbox {bbox}, skipping")
                    continue

                bbox = [float(x) for x in bbox]
                results.append(VisualItem(description=description, bbox=bbox))
            except (ValueError, TypeError) as e:
                logger.warning(f"Item {i}: parse error ({e}), skipping")
                continue

        logger.info(f"Parsed {len(results)} visual items from VLM response")
        return results
