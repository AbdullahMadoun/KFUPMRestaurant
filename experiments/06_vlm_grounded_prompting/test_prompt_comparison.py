"""Quick A/B test: current prompt vs dish-level prompt on salad image."""

import base64
import json
import re
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:3b"
SALAD = "/Users/abdulrazzak/Resturant_Pipeline/menu/salad/frame_frame_009719_00.jpg"

PROMPT_CURRENT = (
    "Analyze the image and identify all distinct items on the plate or in the container.\n"
    "For each item, provide a detailed description that includes:\n"
    '1. The Count (e.g., "two pieces", "a pile", "three slices").\n'
    '2. The Name or Visual Description (e.g., "fried chicken", "golden brown curved object", "white rice").\n'
    "\n"
    "Combine these into a single descriptive string for each unique group of items.\n"
    'Example: "two golden brown fried chicken pieces", "a pile of white steamed rice", "three green cucumber slices".\n'
    "\n"
    "CRITICAL EXCLUSIONS:\n"
    "- Do NOT include the dishware itself (plate, bowl, platter, cup).\n"
    "- Do NOT include cutlery (fork, knife, spoon).\n"
    "- Do NOT include background objects.\n"
    "\n"
    "Return the answer strictly as a valid JSON object in the following format:\n"
    '{"food_items": ["description_1", "description_2", ...]}\n'
    "Ensure the JSON is syntactically valid and contains no additional text."
)

PROMPT_DISH_LEVEL = (
    "Identify the distinct DISHES or SERVINGS in this image. "
    "Treat each dish as a single item — do NOT list individual ingredients.\n"
    "\n"
    "For example:\n"
    '- A plate with rice, chicken, and salad → ["rice", "grilled chicken", "green salad"]\n'
    '- A bowl of soup → ["soup"]\n'
    '- A mixed salad → ["mixed salad"] (NOT ["lettuce", "tomato", "cucumber", ...])\n'
    "\n"
    "RULES:\n"
    "- Each item = one dish or one distinct food serving\n"
    "- Do NOT break a single dish into its ingredients\n"
    "- Do NOT include plates, bowls, cutlery, or background\n"
    "\n"
    'Return strictly as JSON: {"food_items": ["dish1", "dish2", ...]}\n'
    "Return ONLY the JSON."
)


def ask_ollama(image_path, prompt):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": MODEL, "prompt": prompt, "images": [img_b64],
        "stream": False, "options": {"temperature": 0.1},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    return resp.json()["response"].strip()


def parse(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    jt = match.group(1).strip() if match else text
    obj = re.search(r'\{[^{}]*\}', jt)
    if obj:
        jt = obj.group(0)
    return json.loads(jt)


# Also test on the other images to make sure dish-level doesn't under-detect
TEST_IMAGES = [
    ("salad", SALAD),
    ("chicken", "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/assets/v1/context/CHICKEN.jpg"),
    ("mixed_plate", "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/assets/v1/results/mixed_2.jpg"),
    ("hummus", "/Users/abdulrazzak/Resturant_Pipeline/menu/hummus/frame_frame_009060_00.jpg"),
    ("bamya", "/Users/abdulrazzak/Resturant_Pipeline/menu/Bamya/frame_frame_028414_00.jpg"),
]

for name, path in TEST_IMAGES:
    print(f"\n{'='*50}")
    print(f"IMAGE: {name}")
    print(f"{'='*50}")

    for label, prompt in [("CURRENT", PROMPT_CURRENT), ("DISH-LEVEL", PROMPT_DISH_LEVEL)]:
        t0 = time.time()
        resp = ask_ollama(path, prompt)
        t1 = time.time()
        try:
            items = parse(resp).get("food_items", [])
        except:
            items = f"PARSE FAIL: {resp[:100]}"
        print(f"  {label:12s} ({t1-t0:.1f}s): {items}")
