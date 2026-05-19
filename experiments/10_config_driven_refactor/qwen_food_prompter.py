import os
import json
import re
import logging

from vllm import LLM, SamplingParams

from config import QwenConfig, PromptConfig

logger = logging.getLogger("pipeline")


class QwenFoodPrompter:
    """
    Handles prompt generation for food item detection using Qwen2.5-VL via vLLM.

    All vLLM parameters are driven by QwenConfig; the VLM prompt text comes from
    PromptConfig.template.
    """

    def __init__(self, config: QwenConfig, prompt_config: PromptConfig):
        self.prompt_template = prompt_config.template
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    prompter = QwenFoodPrompter(QwenConfig(), PromptConfig())
    items = prompter.generate_prompts(args.image)
    print(json.dumps(items, indent=2))
