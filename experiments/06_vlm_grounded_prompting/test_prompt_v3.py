"""Test the balanced prompt: separate servings, not ingredients."""

import base64
import json
import re
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:3b"

PROMPT_BALANCED = (
    "Identify the distinct food SERVINGS visible in this image.\n"
    "\n"
    "RULES:\n"
    "- Each visually separate portion of food = one item\n"
    "- A mixed dish (e.g., salad, stew, soup) is ONE item — do NOT list its ingredients separately\n"
    "- If two different foods sit side by side (e.g., rice next to chicken), list them as SEPARATE items\n"
    "- Do NOT include plates, bowls, cutlery, wrapping, or background\n"
    "\n"
    "Examples:\n"
    '- Rice and chicken on same plate → ["rice", "chicken"]\n'
    '- A bowl of mixed salad → ["mixed salad"] (NOT ["lettuce", "tomato", ...])\n'
    '- Soup with bread on the side → ["soup", "bread"]\n'
    "\n"
    'Return strictly as JSON: {"food_items": ["item1", "item2", ...]}\n'
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


TEST_IMAGES = [
    ("salad", "/Users/abdulrazzak/Resturant_Pipeline/menu/salad/frame_frame_009719_00.jpg"),
    ("chicken", "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/assets/v1/context/CHICKEN.jpg"),
    ("mixed_1", "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/assets/v1/results/mixed_1_hybrid_vis.jpg"),
    ("mixed_2", "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/assets/v1/results/mixed_2.jpg"),
    ("mixed_4", "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/assets/v1/results/mixed_4_hybrid_vis.jpg"),
    ("hummus", "/Users/abdulrazzak/Resturant_Pipeline/menu/hummus/frame_frame_009060_00.jpg"),
    ("bamya", "/Users/abdulrazzak/Resturant_Pipeline/menu/Bamya/frame_frame_028414_00.jpg"),
    ("rice", "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/assets/v1/context/RICE.jpg"),
    ("macarony", "/Users/abdulrazzak/Resturant_Pipeline/menu/macarony/frame_frame_000513_00.jpg"),
    ("fish", "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/assets/v1/context/FISH.jpg"),
]

print(f"Testing BALANCED prompt on {len(TEST_IMAGES)} images\n")

for name, path in TEST_IMAGES:
    t0 = time.time()
    resp = ask_ollama(path, PROMPT_BALANCED)
    t1 = time.time()
    try:
        items = parse(resp).get("food_items", [])
    except:
        items = f"PARSE FAIL: {resp[:100]}"
    print(f"  {name:12s} ({t1-t0:.1f}s): {items}")
