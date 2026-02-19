"""Test VLM grounding via Ollama: run Qwen2.5-VL-3B on food images, visualize bboxes.

Usage:  python3 test_vlm_grounding.py
"""

import base64
import json
import os
import re
import time

import cv2
import requests

OUTPUT_DIR = "/tmp/vlm_grounding_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "v1", "context")
IMAGES = [
    os.path.join(ASSETS, "CHICKEN.jpg"),
    os.path.join(ASSETS, "FISH.jpg"),
    os.path.join(ASSETS, "RICE.jpg"),
]

MODEL = "qwen2.5vl:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

FOOD_PROMPT = (
    "Analyze the image and identify all distinct food items on the plate or in the container.\n"
    "For each item, provide a short descriptive name.\n\n"
    "CRITICAL: Do NOT include dishware, cutlery, or background objects.\n\n"
    "Return strictly as JSON:\n"
    '{"food_items": ["item1", "item2", ...]}\n'
    "Return ONLY the JSON, no other text."
)

GROUNDING_TEMPLATE = (
    'Locate the "{label}" in this image. '
    'Return a JSON object with the bounding box: '
    '{{"bbox_2d": [x1, y1, x2, y2]}} '
    'where coordinates are pixel positions. Return ONLY the JSON.'
)


def ask_ollama(image_path, prompt):
    """Send image + prompt to Ollama and return response text."""
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
    # Also try to extract just the JSON object if there's extra text
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
    except (json.JSONDecodeError, Exception) as e:
        print(f"    Parse error: {e}")
        print(f"    Raw: {text[:200]}")
    return None


def draw_bboxes(image_path, labels_and_bboxes, output_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    colors = [
        (0, 255, 0),
        (255, 100, 0),
        (0, 100, 255),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, (label, bbox) in enumerate(labels_and_bboxes):
        color = colors[i % len(colors)]
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(img, f"{label} (NO BBOX)", (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"  Saved: {output_path}")


def main():
    print(f"Using model: {MODEL} via Ollama\n")

    for image_path in IMAGES:
        if not os.path.exists(image_path):
            print(f"Skipping {image_path} (not found)")
            continue

        name = os.path.basename(image_path)
        print(f"\n{'='*60}")
        print(f"IMAGE: {name}")
        print(f"{'='*60}")

        # Phase 1: Detect food items
        print("  Phase 1: Detecting food items...")
        t0 = time.time()
        response = ask_ollama(image_path, FOOD_PROMPT)
        t1 = time.time()
        print(f"  Response ({t1-t0:.1f}s): {response[:300]}")

        try:
            data = parse_json_response(response)
            labels = data.get("food_items", [])
        except json.JSONDecodeError:
            print(f"  FAILED to parse. Skipping.")
            labels = []

        print(f"  Found {len(labels)} items: {labels}")

        # Phase 2: Ground each label
        labels_and_bboxes = []
        for label in labels:
            print(f"\n  Phase 2: Grounding '{label}'...")
            prompt = GROUNDING_TEMPLATE.format(label=label)
            t0 = time.time()
            response = ask_ollama(image_path, prompt)
            t1 = time.time()
            print(f"    Response ({t1-t0:.1f}s): {response[:300]}")

            bbox = parse_bbox(response)
            if bbox:
                print(f"    BBOX: {bbox}")
            else:
                print(f"    NO VALID BBOX")
            labels_and_bboxes.append((label, bbox))

        # Draw results
        output_path = os.path.join(OUTPUT_DIR, f"grounded_{name}")
        draw_bboxes(image_path, labels_and_bboxes, output_path)

    print(f"\n{'='*60}")
    print(f"All outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
