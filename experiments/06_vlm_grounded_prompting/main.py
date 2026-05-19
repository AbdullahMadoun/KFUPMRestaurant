import argparse
import gc
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from config import PipelineConfig
from pipeline_types import GroundedPrompt, apply_nms, group_detections
from logger import RunTracker, ImageMetrics

logger = logging.getLogger("pipeline")


def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def build_config_from_args(args) -> PipelineConfig:
    """Build PipelineConfig from CLI args, with optional JSON config as base.

    When --config is provided, the JSON file sets the base and only
    explicitly-provided CLI flags override on top.  Without --config,
    PipelineConfig defaults are used (which match the old argparse defaults).
    """
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig()

    # CLI overrides -- None means "not provided by user"
    if args.device is not None:
        config.device = args.device
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.threshold is not None:
        config.sam3.confidence_threshold = args.threshold
    if args.max_objects is not None:
        config.nms.max_objects = args.max_objects
    if args.iou_threshold is not None:
        config.nms.iou_threshold = args.iou_threshold
    if args.alpha is not None:
        config.viz.alpha = args.alpha
    if args.thickness is not None:
        config.viz.thickness = args.thickness
    # --skip_boxes is a flag: only override when explicitly passed
    if args.skip_boxes:
        config.viz.draw_boxes = False
    if args.prompt_mode is not None:
        config.prompt.prompt_mode = args.prompt_mode

    return config


def process_image(image_path, prompter, segmenter, tracker, config):
    metrics = ImageMetrics(image_path=str(image_path))

    # Step 1: Generate prompts with Qwen
    start_time = time.time()
    try:
        if config.prompt.prompt_mode == "grounded":
            grounded = prompter.generate_grounded_prompts(image_path)
        else:
            grounded = [GroundedPrompt(label=p) for p in prompter.generate_prompts(image_path)]
        metrics.qwen_time = time.time() - start_time
        metrics.prompts = [g.label for g in grounded]
        logger.info(f"Generated prompts ({metrics.qwen_time:.2f}s, mode={config.prompt.prompt_mode}): "
                     f"{[(g.label, g.bbox) for g in grounded]}")
    except Exception as e:
        logger.error(f"Error during prompt generation: {e}")
        metrics.error = str(e)
        tracker.add_image_metrics(metrics)
        return metrics

    if not grounded:
        logger.warning("No prompts generated for this image.")
        tracker.add_image_metrics(metrics)
        return metrics

    # Step 2: Segment with SAM3
    start_time = time.time()
    try:
        raw_detections, threshold_used = segmenter.segment(image_path, grounded)
        metrics.sam3_time = time.time() - start_time
        metrics.num_detections_raw = len(raw_detections)
        metrics.threshold_used = threshold_used
        logger.info(f"Segmentation complete ({metrics.sam3_time:.2f}s). "
                     f"{len(raw_detections)} raw detections.")
    except Exception as e:
        logger.error(f"Error during segmentation: {e}\n{traceback.format_exc()}")
        metrics.error = str(e)
        tracker.add_image_metrics(metrics)
        return metrics

    # Step 3: NMS
    kept = apply_nms(raw_detections, config.nms.max_objects, config.nms.iou_threshold)
    metrics.num_detections_after_nms = len(kept)

    # Step 4: Group for visualization
    results = group_detections(kept)
    metrics.num_labels = len(results)
    logger.info(f"After NMS: {len(kept)} detections, {len(results)} labels.")

    # Step 5: Visualize
    try:
        from visualizer import visualize

        image_name = Path(image_path).name
        output_path = str(tracker.viz_dir / f"segmented_{image_name}")
        visualize(image_path, results, output_path, config.viz)
        metrics.visualization_path = output_path
    except Exception as e:
        logger.error(f"Error during visualization: {e}")

    tracker.add_image_metrics(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Food Segmentation on Directory or File")
    parser.add_argument("input_path", help="Path to input image or directory")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument("--device", default=None, help="Device to run on (default: cuda)")
    parser.add_argument("--output_dir", default=None, help="Directory to save results (default: bold_results)")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold for SAM3 (default: 0.1)")
    parser.add_argument("--max_objects", type=int, default=None, help="Maximum number of objects to keep via NMS (default: 5)")
    parser.add_argument("--iou_threshold", type=float, default=None, help="IOU threshold for NMS (default: 0.7)")
    parser.add_argument("--skip_boxes", action="store_true", help="Do not draw bounding boxes in visualization")
    parser.add_argument("--alpha", type=float, default=None, help="Mask opacity 0-1 (default: 0.7)")
    parser.add_argument("--thickness", type=int, default=None, help="Boundary thickness (default: 3)")
    parser.add_argument("--prompt_mode", default=None, choices=["text", "grounded"],
                        help="Prompt mode: 'text' (default) or 'grounded' (text+bbox)")
    args = parser.parse_args()

    config = build_config_from_args(args)

    # Add sam3 repo to sys.path
    sam3_path = os.path.abspath(config.sam3_repo_path)
    if os.path.exists(sam3_path):
        sys.path.insert(0, sam3_path)

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Path {input_path} not found.")
        return

    # Initialize run tracker (creates timestamped run dir + logger)
    tracker = RunTracker(config.output_dir, asdict(config))
    logger.info(f"Run {tracker.run_id} started. Output: {tracker.run_dir}")

    # Initialize models
    logger.info(f"Initializing models (threshold: {config.sam3.confidence_threshold})")
    try:
        from qwen_food_prompter import QwenFoodPrompter
        from sam3_segmenter import SAM3Segmenter

        prompter = QwenFoodPrompter(config.qwen, config.prompt)
        segmenter = SAM3Segmenter(config.sam3, config.device)
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return

    # Collect images
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = []
        for ext in config.image_extensions:
            image_paths.extend(list(input_path.glob(ext.lower())))
            image_paths.extend(list(input_path.glob(ext.upper())))

    image_paths = sorted(set(image_paths))
    logger.info(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        logger.info(f"Processing: {img_path}")
        process_image(str(img_path), prompter, segmenter, tracker, config)

    # Finalize run
    summary = tracker.finalize()
    logger.info(f"Run complete. {summary.total_images} images, "
                f"{summary.total_detections} detections.")
    logger.info(f"Summary saved to {tracker.run_dir / 'run_summary.json'}")

    cleanup_gpu()


if __name__ == "__main__":
    main()
