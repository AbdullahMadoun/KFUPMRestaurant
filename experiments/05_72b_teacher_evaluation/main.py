"""3-Stage MVP Pipeline: Visual Description → Segmentation → Menu Matching.

Usage:
    # Single image
    python main.py /path/to/plate.jpg --index menu.index --meta menu_meta.json

    # Directory of images
    python main.py /path/to/images/ --index menu.index --meta menu_meta.json --output_dir results/
"""

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
from ptypes import nms_segmented, PlateResult
from logger import RunTracker, ImageMetrics

logger = logging.getLogger("pipeline")


def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def build_config_from_args(args) -> PipelineConfig:
    """Build PipelineConfig from CLI args, with optional JSON config as base."""
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig()

    # CLI overrides
    if args.device is not None:
        config.device = args.device
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.threshold is not None:
        config.sam.confidence_threshold = args.threshold
    if args.similarity is not None:
        config.match.similarity_threshold = args.similarity
    if args.top_k is not None:
        config.match.top_k = args.top_k
    if args.index is not None:
        config.match.index_path = args.index
    if args.meta is not None:
        config.match.metadata_path = args.meta

    return config


def process_image(image_path, describer, segmenter, matcher, tracker, config):
    """Process a single image through the 3-stage pipeline.

    Stage 1: VLM describe → visual descriptions + bboxes
    Stage 2: SAM3 segment + crop → masks + crops
    NMS: Remove overlapping detections
    Stage 3: Match crops → menu items + prices
    Visualize: Render results
    """
    metrics = ImageMetrics(image_path=str(image_path))

    # Stage 1: VLM Visual Description
    start_time = time.time()
    try:
        visual_items = describer.describe(image_path)
        metrics.stage1_time = time.time() - start_time
        metrics.visual_items = len(visual_items)
        logger.info(f"Stage 1 ({metrics.stage1_time:.2f}s): {len(visual_items)} visual items")
        for item in visual_items:
            logger.info(f"  - '{item.description[:50]}...' bbox={item.bbox}")
    except Exception as e:
        logger.error(f"Stage 1 error: {e}\n{traceback.format_exc()}")
        metrics.error = f"Stage 1: {e}"
        tracker.add_image_metrics(metrics)
        return metrics

    if not visual_items:
        logger.warning("Stage 1: No visual items found.")
        tracker.add_image_metrics(metrics)
        return metrics

    # Stage 2: SAM3 Segment + Crop
    start_time = time.time()
    try:
        segmented_items = segmenter.segment_and_crop(image_path, visual_items)
        metrics.stage2_time = time.time() - start_time
        logger.info(f"Stage 2 ({metrics.stage2_time:.2f}s): {len(segmented_items)} segmented items")
    except Exception as e:
        logger.error(f"Stage 2 error: {e}\n{traceback.format_exc()}")
        metrics.error = f"Stage 2: {e}"
        tracker.add_image_metrics(metrics)
        return metrics

    if not segmented_items:
        logger.warning("Stage 2: No items segmented.")
        tracker.add_image_metrics(metrics)
        return metrics

    # NMS (between Stage 2 and 3)
    segmented_items = nms_segmented(segmented_items, config.nms.max_objects, config.nms.iou_threshold)
    metrics.segmented_items = len(segmented_items)
    logger.info(f"After NMS: {len(segmented_items)} items")

    # Stage 3: Match crops to menu
    start_time = time.time()
    try:
        match_results = matcher.match(segmented_items)
        metrics.stage3_time = time.time() - start_time
        logger.info(f"Stage 3 ({metrics.stage3_time:.2f}s): {len(match_results)} matches")
    except Exception as e:
        logger.error(f"Stage 3 error: {e}\n{traceback.format_exc()}")
        metrics.error = f"Stage 3: {e}"
        tracker.add_image_metrics(metrics)
        return metrics

    # Compute metrics
    known = [m for m in match_results if m.menu_item != "unknown"]
    metrics.matched_items = len(known)
    metrics.unknown_items = len(match_results) - len(known)
    metrics.total_price = sum(m.price for m in known)
    metrics.matches = [
        {
            "item": m.menu_item,
            "category": m.category,
            "price": m.price,
            "confidence": round(m.confidence, 3),
            "description": m.segmented.description[:60],
        }
        for m in match_results
    ]

    logger.info(f"Results: {metrics.matched_items} matched, {metrics.unknown_items} unknown, "
                f"total={metrics.total_price:.0f} SAR")
    for m in match_results:
        logger.info(f"  - {m.menu_item} ({m.confidence:.3f}) {m.category} {m.price:.0f} SAR")

    # Visualize
    try:
        from visualizer import visualize

        image_name = Path(image_path).name
        output_path = str(tracker.viz_dir / f"matched_{image_name}")
        visualize(image_path, match_results, output_path, config.viz)
        metrics.visualization_path = output_path
    except Exception as e:
        logger.error(f"Visualization error: {e}")

    tracker.add_image_metrics(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="3-Stage Food Pipeline: Describe → Segment → Match")
    parser.add_argument("input_path", help="Path to input image or directory")
    parser.add_argument("--index", default=None, help="Path to FAISS index file (required)")
    parser.add_argument("--meta", default=None, help="Path to metadata JSON file (required)")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument("--device", default=None, help="Device (default: cuda)")
    parser.add_argument("--output_dir", default=None, help="Results directory (default: results)")
    parser.add_argument("--threshold", type=float, default=None, help="SAM3 confidence threshold")
    parser.add_argument("--similarity", type=float, default=None, help="Min cosine similarity for match")
    parser.add_argument("--top_k", type=int, default=None, help="Number of match candidates")
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

    # Initialize run tracker
    tracker = RunTracker(config.output_dir, asdict(config))
    logger.info(f"Run {tracker.run_id} started. Output: {tracker.run_dir}")

    # Initialize all 3 stage models
    logger.info("Initializing models...")
    try:
        from stage1_vlm import VisualDescriber
        from stage2_sam import FoodSegmenter
        from stage3_match import MenuMatcher

        logger.info("Loading Stage 1: VLM (Qwen2.5-VL)...")
        describer = VisualDescriber(config.vlm)

        logger.info("Loading Stage 2: SAM3...")
        segmenter = FoodSegmenter(config.sam, config.device)

        logger.info("Loading Stage 3: MenuMatcher (SigLIP 2 + FAISS)...")
        matcher = MenuMatcher(config.match, config.device)

        logger.info("All models loaded.")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}\n{traceback.format_exc()}")
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
        logger.info(f"\n{'='*60}\nProcessing: {img_path}\n{'='*60}")
        process_image(str(img_path), describer, segmenter, matcher, tracker, config)

    # Finalize
    summary = tracker.finalize()
    logger.info(f"\nRun complete: {summary.total_images} images, "
                f"{summary.total_matched} matched, {summary.total_unknown} unknown, "
                f"total price={summary.total_price:.0f} SAR")
    logger.info(f"Avg times: Stage1={summary.avg_stage1_time:.2f}s, "
                f"Stage2={summary.avg_stage2_time:.2f}s, Stage3={summary.avg_stage3_time:.2f}s")
    logger.info(f"Summary saved to {tracker.run_dir / 'run_summary.json'}")

    cleanup_gpu()


if __name__ == "__main__":
    main()
