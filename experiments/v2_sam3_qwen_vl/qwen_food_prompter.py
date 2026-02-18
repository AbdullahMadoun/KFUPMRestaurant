import os
import json
from vllm import LLM, SamplingParams
import argparse

class QwenFoodPrompter:
    """
    Handles prompt generation for food item detection using Qwen2.5-VL via vLLM.
    
    This class initializes the vLLM engine with optimized memory settings (gpu_memory_utilization=0.4)
    and provides methods to generate structured JSON lists of food items from images.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", gpu_memory_utilization=0.4):
        try:
            self.llm = LLM(
                model=model_name,
                limit_mm_per_prompt={"image": 1},
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=4096,
                enforce_eager=True,
                allowed_local_media_path="/root",
            )
            self.sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
        except Exception as e:
            print(f"Error initializing vLLM: {e}")
            raise

    def generate_prompts(self, image_path):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    {"type": "text", "text": """Analyze the image and identify all distinct items on the plate or in the container. 
For each item, provide a detailed description that includes:
1. The Count (e.g., "two pieces", "a pile", "three slices").
2. The Name or Visual Description (e.g., "fried chicken", "golden brown curved object", "white rice").

Combine these into a single descriptive string for each unique group of items.
Example: "two golden brown fried chicken pieces", "a pile of white steamed rice", "three green cucumber slices".

CRITICAL EXCLUSIONS:
- Do NOT include the dishware itself (plate, bowl, platter, cup).
- Do NOT include cutlery (fork, knife, spoon).
- Do NOT include background objects.

Return the answer strictly as a valid JSON object in the following format:

{
  "food_items": [
    "description_1",
    "description_2",
    ...
  ]
}

Ensure the JSON is syntactically valid and contains no additional text."""}
                ]
            }
        ]
        
        try:
            outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            
            try:
                import re
                match = re.search(r"```json\s*(.*?)\s*```", generated_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
                else:
                    json_text = generated_text.strip()
            except ImportError:
                # Fallback if re somehow fails or isn't imported (though it's standard)
                if generated_text.startswith("```json"):
                    generated_text = generated_text[7:]
                if generated_text.endswith("```"):
                    generated_text = generated_text[:-3]
                json_text = generated_text.strip()
            
            try:
                data = json.loads(json_text)
                return data.get("food_items", [])
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response: {json_text}")
                return []
                
        except Exception as e:
            print(f"Error during generation: {e}")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    
    prompter = QwenFoodPrompter()
    items = prompter.generate_prompts(args.image)
    print(json.dumps(items, indent=2))
