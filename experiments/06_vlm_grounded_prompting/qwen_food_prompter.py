import os
import json
import re
import logging

from vllm import LLM, SamplingParams

from config import QwenConfig, PromptConfig
from pipeline_types import GroundedPrompt

logger = logging.getLogger("pipeline")


class QwenFoodPrompter:
    """
    Handles prompt generation for food item detection using Qwen2.5-VL via vLLM.

    All vLLM parameters are driven by QwenConfig; the VLM prompt text comes from
    PromptConfig.template.
    """

    def __init__(self, config: QwenConfig, prompt_config: PromptConfig):
        self.prompt_template = prompt_config.template
        self.grounding_template = prompt_config.grounding_template
        try:
            self.llm = LLM(
                model=config.model_name,
                limit_mm_per_prompt={"image": 1},
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_model_len=config.max_model_len,
                enforce_eager=config.enforce_eager,
                allowed_local_media_path=config.allowed_local_media_path,
            )
            self.sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        except Exception as e:
            logger.error(f"Error initializing vLLM: {e}")
            raise

    def generate_prompts(self, image_path):
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return []

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    {"type": "text", "text": self.prompt_template},
                ],
            }
        ]

        try:
            outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()

            match = re.search(r"```json\s*(.*?)\s*```", generated_text, re.DOTALL)
            json_text = match.group(1).strip() if match else generated_text

            try:
                data = json.loads(json_text)
                return data.get("food_items", [])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {json_text}")
                return []

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return []

    def generate_grounded_prompts(self, image_path) -> list:
        """Two-phase grounding: detect food labels, then locate each with a bbox.

        Phase 1: Calls generate_prompts() to get text labels.
        Phase 2: For each label, asks the VLM to return a bounding box.

        Returns a list of GroundedPrompt objects. On parse failure for any
        individual label the bbox is set to None (graceful text-only fallback).
        """
        labels = self.generate_prompts(image_path)
        if not labels:
            return []

        grounded: list = []
        for label in labels:
            bbox = self._locate_item(image_path, label)
            grounded.append(GroundedPrompt(label=label, bbox=bbox))
        return grounded

    def _locate_item(self, image_path: str, label: str):
        """Ask the VLM for a bounding box of *label* in the image.

        Returns [x1, y1, x2, y2] pixel coords on success, None on failure.
        """
        prompt_text = self.grounding_template.format(label=label)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()

            match = re.search(r"```json\s*(.*?)\s*```", generated_text, re.DOTALL)
            json_text = match.group(1).strip() if match else generated_text

            data = json.loads(json_text)
            bbox = data.get("bbox_2d")
            if bbox and len(bbox) == 4:
                bbox = [float(v) for v in bbox]
                logger.info(f"Grounded '{label}' -> bbox {bbox}")
                return bbox

            logger.warning(f"Invalid bbox for '{label}': {bbox}")
            return None

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Grounding failed for '{label}': {e}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    prompter = QwenFoodPrompter(QwenConfig(), PromptConfig())
    items = prompter.generate_prompts(args.image)
    print(json.dumps(items, indent=2))
