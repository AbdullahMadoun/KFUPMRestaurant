import argparse
import os
import torch
import gc
import sys
import time
import json
from pathlib import Path

# Add current directory to path so we can import our modules
# Add sam3 repository to path
# Assuming sam3 is cloned in the current directory or a subdirectory
sam3_repo_path = os.path.join(os.getcwd(), "sam3")
if os.path.exists(sam3_repo_path):
    sys.path.append(sam3_repo_path)
else:
    # Fallback to the known location if current dir does not contain it
    sys.path.append("/root/.gemini/antigravity/brain/e35023a3-cf38-42a1-abe0-70c93c6f9f59/sam3")

sys.path.append(os.getcwd())

from qwen_food_prompter import QwenFoodPrompter
from sam3_segmenter import SAM3Segmenter
from visualizer import visualize

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def process_image(image_path, prompter, segmenter, output_dir, device, max_objects=5, draw_boxes=True, iou_threshold=0.7, alpha=0.7, thickness=3):
    print(f"\nProcessing: {image_path}")
    metrics = {"image": str(image_path)}
    
    # Step 1: Generate Prompts with Qwen
    start_time = time.time()
    try:
        if isinstance(prompter, QwenFoodPrompter):
            prompts = prompter.generate_prompts(image_path)
        else:
            prompts = prompter.run_inference(image_path)
            
        qwen_time = time.time() - start_time
        metrics["qwen_time_seconds"] = qwen_time
        metrics["prompts"] = prompts
        print(f"Generated prompts ({qwen_time:.2f}s): {prompts}")
        
    except Exception as e:
        print(f"Error during prompt generation: {e}")
        return None

    if not prompts:
        print("No prompts generated for this image.")
        return metrics

    # Step 2: Segment with SAM3
    start_time = time.time()
    try:
        results = segmenter.segment(image_path, prompts, max_objects=max_objects, iou_threshold=iou_threshold)
        sam3_time = time.time() - start_time
        metrics["sam3_time_seconds"] = sam3_time
        metrics["num_detected_items"] = len(results)
        print(f"Segmentation complete ({sam3_time:.2f}s). Found {len(results)} items.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during segmentation: {e}")
        return metrics

    # Step 3: Visualize
    try:
        image_name = Path(image_path).name
        output_filename = f"segmented_{image_name}"
        output_path = os.path.join(output_dir, output_filename)
        visualize(image_path, results, output_path, draw_boxes=draw_boxes, alpha=alpha, thickness=thickness)
        metrics["visualization_path"] = output_path
        print(f"Visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error during visualization: {e}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Food Segmentation on Directory or File")
    parser.add_argument("input_path", help="Path to input image or directory")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--output_dir", default="bold_results", help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold for SAM3")
    parser.add_argument("--max_objects", type=int, default=5, help="Maximum number of objects to keep (NMS)")
    parser.add_argument("--iou_threshold", type=float, default=0.7, help="IOU threshold for NMS")
    parser.add_argument("--skip_boxes", action="store_true", help="Do not draw bounding boxes in visualization")
    parser.add_argument("--alpha", type=float, default=0.7, help="Mask opacity (0-1)")
    parser.add_argument("--thickness", type=int, default=3, help="Boundary thickness")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Path {input_path} not found.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models once
    print(f"--- Initializing Models (Threshold: {args.threshold}) ---")
    try:
        prompter = QwenFoodPrompter()
        segmenter = SAM3Segmenter(device=args.device, confidence_threshold=args.threshold)
    except Exception as e:
        print(f"Failed to initialize models: {e}")
        return

    all_metrics = []

    if input_path.is_file():
        image_paths = [input_path]
    else:
        # Process all common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(input_path.glob(ext.lower())))
            image_paths.extend(list(input_path.glob(ext.upper())))
                
    image_paths = sorted(list(set(image_paths)))
    print(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        metrics = process_image(str(img_path), prompter, segmenter, str(output_dir), args.device, 
                               max_objects=args.max_objects, draw_boxes=not args.skip_boxes, 
                               iou_threshold=args.iou_threshold, alpha=args.alpha, thickness=args.thickness)
        if metrics:
            all_metrics.append(metrics)

    # Save aggregated metrics
    metrics_path = output_dir / "inference_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nProcessing complete. Metrics saved to {metrics_path}")

# Wrapper class to handle Qwen logic which might need specific instantiation
# (kept for backward compatibility if code relies on it, but main uses direct class now)
class QwenFoodPrompter_wrapper:
    def __init__(self):
        self.prompter = QwenFoodPrompter()
        
    def run_inference(self, image_path):
        return self.prompter.generate_prompts(image_path)

if __name__ == "__main__":
    main()
