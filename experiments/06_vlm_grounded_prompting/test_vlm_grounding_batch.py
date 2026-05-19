"""Batch VLM grounding test on 12+ food images via Ollama.

Usage:  python3 test_vlm_grounding_batch.py
"""

import base64
import json
import os
import re
import time

import cv2
import requests

OUTPUT_DIR = "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/experiments/v2_sam3_qwen_vl/grounding_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MENU_DIR = "/Users/abdulrazzak/Resturant_Pipeline/menu"
MIXED_DIR = "/Users/abdulrazzak/Resturant_Pipeline_Feb/KFUPMRestaurant/assets/v1/results"

MODEL = "qwen2.5vl:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

FOOD_PROMPT = (
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
    '{"food_items": ["item1", "item2", ...]}\n'
    "Return ONLY the JSON."
)

GROUNDING_TEMPLATE = (
    'Locate the "{label}" in this image. '
    'Return a JSON object with the bounding box: '
    '{{"bbox_2d": [x1, y1, x2, y2]}} '
    'where coordinates are pixel positions. Return ONLY the JSON.'
)


def ask_ollama(image_path, prompt):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def parse_json_response(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    json_text = match.group(1).strip() if match else text
    obj_match = re.search(r'\{[^{}]*\}', json_text)
    if obj_match:
        json_text = obj_match.group(0)
    return json.loads(json_text)


def parse_bbox(text):
    try:
        data = parse_json_response(text)
        bbox = data.get("bbox_2d")
        if bbox and len(bbox) == 4:
            return [float(v) for v in bbox]
    except Exception as e:
        pass
    return None


def draw_bboxes(image_path, labels_and_bboxes, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    colors = [(0,255,0), (255,100,0), (0,100,255), (255,0,255), (0,255,255), (128,255,0)]
    for i, (label, bbox) in enumerate(labels_and_bboxes):
        color = colors[i % len(colors)]
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(img, f"{label} (NO BBOX)", (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(output_path, img)


def process_image(image_path, tag):
    name = f"{tag}.jpg"
    print(f"\n  [{tag}] Phase 1: Detecting food items...")
    t0 = time.time()
    response = ask_ollama(image_path, FOOD_PROMPT)
    t1 = time.time()

    try:
        data = parse_json_response(response)
        labels = data.get("food_items", [])
    except Exception:
        print(f"  [{tag}] Parse failed: {response[:100]}")
        labels = []

    print(f"  [{tag}] ({t1-t0:.1f}s) Items: {labels}")

    labels_and_bboxes = []
    for label in labels:
        prompt = GROUNDING_TEMPLATE.format(label=label)
        t0 = time.time()
        resp = ask_ollama(image_path, prompt)
        t1 = time.time()
        bbox = parse_bbox(resp)
        status = f"bbox={bbox}" if bbox else "NO BBOX"
        print(f"  [{tag}] Grounding '{label}' ({t1-t0:.1f}s): {status}")
        labels_and_bboxes.append((label, bbox))

    output_path = os.path.join(OUTPUT_DIR, f"grounded_{name}")
    draw_bboxes(image_path, labels_and_bboxes, output_path)
    print(f"  [{tag}] Saved: {output_path}")
    return labels_and_bboxes


def main():
    print(f"Model: {MODEL} via Ollama")
    print(f"Output: {OUTPUT_DIR}/\n")

    images = []

    # One image per menu category
    for category in sorted(os.listdir(MENU_DIR)):
        cat_dir = os.path.join(MENU_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        jpgs = [f for f in os.listdir(cat_dir) if f.endswith(".jpg")]
        if jpgs:
            images.append((os.path.join(cat_dir, sorted(jpgs)[0]), category))

    # Mixed plate images
    if os.path.isdir(MIXED_DIR):
        for f in sorted(os.listdir(MIXED_DIR)):
            if f.endswith(".jpg"):
                images.append((os.path.join(MIXED_DIR, f), f.replace(".jpg", "")))

    print(f"Total images: {len(images)}\n{'='*60}")

    results = {}
    total_bboxes = 0
    total_items = 0

    for image_path, tag in images:
        lb = process_image(image_path, tag)
        total_items += len(lb)
        total_bboxes += sum(1 for _, b in lb if b is not None)
        results[tag] = lb

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Images processed: {len(images)}")
    print(f"Total food items detected: {total_items}")
    print(f"Successful bbox grounding: {total_bboxes}/{total_items} "
          f"({100*total_bboxes/max(total_items,1):.0f}%)")
    print(f"\nOutputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
