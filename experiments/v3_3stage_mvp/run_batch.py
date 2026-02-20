"""Batch runner for V3 pipeline: process N random images through Stage 1+2 (optionally +3).

Outputs a web-friendly directory structure where each image gets its own folder
with crops, masks, visualization, and a results.json that a website can directly consume.

Usage:
    # Basic: 50 random images, Stage 1+2 only
    python run_batch.py --source_dir /path/to/Sampled_Images_All/ --n 50

    # With Stage 3 matching
    python run_batch.py --source_dir /path/to/Sampled_Images_All/ --n 50 \
        --index menu.index --meta menu_meta.json

    # Resume after crash
    python run_batch.py --source_dir /path/to/Sampled_Images_All/ --n 50 --resume
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from config import PipelineConfig
from ptypes import MatchResult, SegmentedItem, nms_segmented
from logger import setup_logger

logger = logging.getLogger("pipeline")


def build_config(args) -> PipelineConfig:
    """Build PipelineConfig from CLI args, with optional JSON config as base."""
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig()

    config.device = args.device
    config.output_dir = args.output_dir

    # Set allowed_local_media_path to source_dir so vLLM can read the images
    config.vlm.allowed_local_media_path = str(Path(args.source_dir).resolve().parent)

    if args.index is not None:
        config.match.index_path = args.index
    if args.meta is not None:
        config.match.metadata_path = args.meta

    return config


def collect_images(source_dir: Path, config: PipelineConfig):
    """Gather all image paths from source directory."""
    image_paths = []
    for ext in config.image_extensions:
        image_paths.extend(list(source_dir.glob(ext.lower())))
        image_paths.extend(list(source_dir.glob(ext.upper())))
    return sorted(set(image_paths))


def sample_images(image_paths, n: int, seed: int):
    """Sample n images with a fixed seed, return sorted list."""
    if n >= len(image_paths):
        logger.info(f"Requested {n} images but only {len(image_paths)} available — using all.")
        return sorted(image_paths)
    rng = random.Random(seed)
    sampled = rng.sample(image_paths, n)
    return sorted(sampled)


def image_id_from_path(image_path: Path) -> str:
    """Derive a unique ID from the image filename (without extension)."""
    return image_path.stem


def save_crop(crop: np.ndarray, path: Path):
    """Save a BGR crop as JPEG."""
    cv2.imwrite(str(path), crop)


def save_mask(mask: np.ndarray, path: Path):
    """Save a binary mask as PNG."""
    mask_uint8 = (mask.astype(bool).astype(np.uint8)) * 255
    cv2.imwrite(str(path), mask_uint8)


def make_synthetic_matches(segmented_items):
    """Wrap SegmentedItems in MatchResult so the visualizer can render them."""
    return [
        MatchResult(
            segmented=item,
            menu_item="unknown",
            category="unknown",
            price=0.0,
            confidence=0.0,
            top_k=[],
        )
        for item in segmented_items
    ]


def process_single_image(
    image_path: Path,
    image_dir: Path,
    visual_items,
    stage1_time: float,
    segmenter,
    matcher,
    config: PipelineConfig,
):
    """Process one image through Stage 2 (optionally +3), save all outputs.

    Stage 1 (VLM) results are passed in pre-computed from batch inference.

    Returns a dict with status info for index.json, or None on failure.
    """
    image_id = image_id_from_path(image_path)
    crops_dir = image_dir / "crops"
    masks_dir = image_dir / "masks"
    crops_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Symlink original image
    original_link = image_dir / "original.jpg"
    if not original_link.exists():
        try:
            original_link.symlink_to(image_path.resolve())
        except OSError:
            import shutil
            shutil.copy2(str(image_path), str(original_link))

    logger.info(f"  Stage 1 ({stage1_time:.1f}s): {len(visual_items)} items described")

    if not visual_items:
        logger.warning(f"  Stage 1 returned no items, skipping {image_id}")
        return None

    # Stage 2: SAM3 Segment + Crop
    t2_start = time.time()
    segmented_items = segmenter.segment_and_crop(str(image_path), visual_items)
    stage2_time = time.time() - t2_start
    logger.info(f"  Stage 2 ({stage2_time:.1f}s): {len(segmented_items)} items segmented")

    if not segmented_items:
        logger.warning(f"  Stage 2 returned no items, skipping {image_id}")
        return None

    # NMS
    segmented_items = nms_segmented(
        segmented_items, config.nms.max_objects, config.nms.iou_threshold
    )
    logger.info(f"  After NMS: {len(segmented_items)} items")

    # Save crops and masks
    items_data = []
    for idx, seg in enumerate(segmented_items):
        crop_filename = f"item_{idx:03d}.jpg"
        mask_filename = f"item_{idx:03d}_mask.png"
        save_crop(seg.crop, crops_dir / crop_filename)
        save_mask(seg.mask, masks_dir / mask_filename)

        item_entry = {
            "index": idx,
            "description": seg.description,
            "bbox": [int(x) for x in seg.bbox],
            "crop": f"crops/{crop_filename}",
            "mask": f"masks/{mask_filename}",
            "sam_score": round(float(seg.score), 4),
        }
        items_data.append(item_entry)

    # Stage 3: Match (optional)
    match_results_data = None
    if matcher is not None:
        t3_start = time.time()
        match_results = matcher.match(segmented_items)
        stage3_time = time.time() - t3_start
        logger.info(f"  Stage 3 ({stage3_time:.1f}s): {len(match_results)} matches")

        # Enrich items_data with match info
        for idx, mr in enumerate(match_results):
            if idx < len(items_data):
                items_data[idx]["match"] = {
                    "menu_item": mr.menu_item,
                    "category": mr.category,
                    "price": mr.price,
                    "confidence": round(mr.confidence, 4),
                    "top_k": [
                        {"name": name, "score": round(score, 4)}
                        for name, score in mr.top_k
                    ],
                    "image_score": round(mr.image_score, 4),
                    "text_score": round(mr.text_score, 4),
                }

        match_results_data = {
            "stage3_time": round(stage3_time, 2),
            "total_price": sum(m.price for m in match_results if m.menu_item != "unknown"),
        }
        viz_matches = match_results
    else:
        viz_matches = make_synthetic_matches(segmented_items)

    # Visualization
    try:
        from visualizer import visualize
        viz_path = image_dir / "visualization.jpg"
        visualize(str(image_path), viz_matches, str(viz_path), config.viz)
    except Exception as e:
        logger.error(f"  Visualization error: {e}")

    # Write results.json (the done-marker)
    results = {
        "image_id": image_id,
        "original": "original.jpg",
        "visualization": "visualization.jpg",
        "num_items": len(segmented_items),
        "items": items_data,
        "stage1_time": round(stage1_time, 2),
        "stage2_time": round(stage2_time, 2),
        "match_results": match_results_data,
    }
    (image_dir / "results.json").write_text(json.dumps(results, indent=2))

    return {
        "id": image_id,
        "path": f"images/{image_id}",
        "num_items": len(segmented_items),
        "status": "success",
    }


def format_eta(elapsed: float, done: int, total: int) -> str:
    """Format a human-readable ETA string."""
    if done == 0:
        return "estimating..."
    avg = elapsed / done
    remaining = avg * (total - done)
    mins, secs = divmod(int(remaining), 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner: process N random images through V3 pipeline"
    )
    parser.add_argument("--source_dir", required=True, help="Directory of source images")
    parser.add_argument("--n", type=int, default=50, help="Number of random images (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", default="batch_results", help="Root output directory")
    parser.add_argument("--index", default=None, help="FAISS index path (enables Stage 3)")
    parser.add_argument("--meta", default=None, help="FAISS metadata JSON path")
    parser.add_argument("--config", default=None, help="JSON config override file")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed images")
    parser.add_argument("--vlm_batch_size", type=int, default=32,
                        help="Number of images per VLM batch (default: 32)")
    args = parser.parse_args()

    config = build_config(args)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (console + file)
    setup_logger("pipeline", str(output_dir))
    logger.info(f"Batch run starting — n={args.n}, seed={args.seed}, output={output_dir}")

    # Add sam3 repo to sys.path
    sam3_path = os.path.abspath(config.sam3_repo_path)
    if os.path.exists(sam3_path):
        sys.path.insert(0, sam3_path)

    # Collect and sample images
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        logger.error(f"Source directory not found: {source_dir}")
        return

    all_images = collect_images(source_dir, config)
    logger.info(f"Found {len(all_images)} total images in {source_dir}")

    if not all_images:
        logger.error("No images found. Check source_dir and image_extensions in config.")
        return

    sampled = sample_images(all_images, args.n, args.seed)
    logger.info(f"Sampled {len(sampled)} images (seed={args.seed})")

    # Save frozen config
    config.to_json(str(output_dir / "config.json"))

    # Import model classes (loaded sequentially to fit on single GPU)
    from stage1_vlm import VisualDescriber
    from stage2_sam import FoodSegmenter

    # Process images — batch VLM first, then SAM per image
    index_entries = []
    stats = {"success": 0, "failed": 0, "skipped": 0}
    batch_start = time.time()

    # Separate resumed vs to-process
    to_process = []  # (index, image_path, image_dir)
    for i, image_path in enumerate(sampled):
        image_id = image_id_from_path(image_path)
        image_dir = images_dir / image_id

        if args.resume and (image_dir / "results.json").exists():
            logger.info(f"[{i+1}/{len(sampled)}] SKIP (already done): {image_id}")
            try:
                existing = json.loads((image_dir / "results.json").read_text())
                index_entries.append({
                    "id": image_id,
                    "path": f"images/{image_id}",
                    "num_items": existing.get("num_items", 0),
                    "status": "success",
                })
                stats["skipped"] += 1
            except Exception:
                stats["failed"] += 1
            continue

        image_dir.mkdir(parents=True, exist_ok=True)
        to_process.append((i, image_path, image_dir))

    # --- Stage 1: Batch VLM inference (all images at once) ---
    # Load VLM (gets full GPU)
    logger.info("Loading Stage 1: VLM...")
    describer = VisualDescriber(config.vlm)

    vlm_batch_size = args.vlm_batch_size
    all_vlm_results = {}  # image_path_str → (List[VisualItem], stage1_time)

    if to_process:
        total_batches = (len(to_process) + vlm_batch_size - 1) // vlm_batch_size
        logger.info(f"\nStage 1: Batch VLM inference — {len(to_process)} images in {total_batches} batch(es) of {vlm_batch_size}")

        for batch_idx in range(0, len(to_process), vlm_batch_size):
            batch = to_process[batch_idx : batch_idx + vlm_batch_size]
            batch_paths = [str(p) for _, p, _ in batch]
            batch_num = batch_idx // vlm_batch_size + 1

            logger.info(f"  VLM batch {batch_num}/{total_batches} ({len(batch)} images)...")
            t1_start = time.time()
            batch_results = describer.describe_batch(batch_paths)
            batch_time = time.time() - t1_start
            per_image = batch_time / len(batch)
            logger.info(f"  VLM batch {batch_num} done in {batch_time:.1f}s ({per_image:.2f}s/image)")

            for path_str in batch_paths:
                all_vlm_results[path_str] = (batch_results.get(path_str, []), per_image)

    # Unload VLM to free GPU for SAM
    logger.info("Unloading VLM to free GPU memory...")
    del describer
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # --- Stage 2+: SAM + NMS + Match per image ---
    # Load SAM (gets full GPU now)
    logger.info("Loading Stage 2: SAM3...")
    segmenter = FoodSegmenter(config.sam, config.device)

    matcher = None
    if args.index and args.meta:
        from stage3_match import MenuMatcher
        logger.info("Loading Stage 3: MenuMatcher (SigLIP 2 + FAISS)...")
        config.match.index_path = args.index
        config.match.metadata_path = args.meta
        matcher = MenuMatcher(config.match, config.device)
    per_image_times = []  # list of dicts for timing log

    for j, (i, image_path, image_dir) in enumerate(to_process):
        image_id = image_id_from_path(image_path)
        elapsed = time.time() - batch_start
        done = stats["success"] + stats["failed"]
        total_to_do = len(to_process)
        eta = format_eta(elapsed, done, total_to_do)

        logger.info(
            f"\n[{j+1}/{total_to_do}] {image_id}  "
            f"(done={done}, ETA={eta})"
        )

        img_start = time.time()
        try:
            visual_items, stage1_time = all_vlm_results.get(str(image_path), ([], 0))
            entry = process_single_image(
                image_path, image_dir, visual_items, stage1_time,
                segmenter, matcher, config
            )
            img_total = time.time() - img_start
            if entry:
                index_entries.append(entry)
                stats["success"] += 1
                per_image_times.append({
                    "id": image_id, "s1": round(stage1_time, 2),
                    "s2": round(img_total, 2), "total": round(stage1_time + img_total, 2),
                    "items": entry["num_items"],
                })
                logger.info(
                    f"  Done: S1={stage1_time:.2f}s + S2={img_total:.2f}s = "
                    f"{stage1_time + img_total:.2f}s total, {entry['num_items']} items"
                )
            else:
                index_entries.append({
                    "id": image_id,
                    "path": f"images/{image_id}",
                    "num_items": 0,
                    "status": "failed",
                })
                stats["failed"] += 1
                logger.info(f"  Failed after {img_total:.2f}s")
        except Exception as e:
            img_total = time.time() - img_start
            logger.error(f"  FAILED ({img_total:.2f}s): {e}\n{traceback.format_exc()}")
            index_entries.append({
                "id": image_id,
                "path": f"images/{image_id}",
                "num_items": 0,
                "status": "failed",
            })
            stats["failed"] += 1

    total_time = time.time() - batch_start

    # Write index.json
    index_data = {
        "total_images": len(sampled),
        "seed": args.seed,
        "images": index_entries,
    }
    (output_dir / "index.json").write_text(json.dumps(index_data, indent=2))

    # Compute timing stats
    if per_image_times:
        s1_times = [t["s1"] for t in per_image_times]
        s2_times = [t["s2"] for t in per_image_times]
        totals = [t["total"] for t in per_image_times]
        timing_stats = {
            "stage1_vlm": {"mean": round(sum(s1_times)/len(s1_times), 2),
                           "min": min(s1_times), "max": max(s1_times), "total": round(sum(s1_times), 1)},
            "stage2_sam": {"mean": round(sum(s2_times)/len(s2_times), 2),
                           "min": min(s2_times), "max": max(s2_times), "total": round(sum(s2_times), 1)},
            "per_image":  {"mean": round(sum(totals)/len(totals), 2),
                           "min": min(totals), "max": max(totals)},
        }
    else:
        timing_stats = {}

    # Write batch_summary.json
    summary = {
        "total_images": len(sampled),
        "success": stats["success"],
        "failed": stats["failed"],
        "skipped": stats["skipped"],
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_image": round(total_time / max(stats["success"], 1), 1),
        "timing": timing_stats,
        "per_image_log": per_image_times,
        "seed": args.seed,
        "stage3_enabled": matcher is not None,
        "source_dir": str(source_dir),
        "device": config.device,
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2))

    # Print timing summary
    logger.info(f"\nBatch complete in {total_time:.0f}s")
    logger.info(f"  Success: {stats['success']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
    if timing_stats:
        logger.info(f"  Timing per image: mean={timing_stats['per_image']['mean']}s, "
                     f"min={timing_stats['per_image']['min']}s, max={timing_stats['per_image']['max']}s")
        logger.info(f"  Stage 1 (VLM):    mean={timing_stats['stage1_vlm']['mean']}s, total={timing_stats['stage1_vlm']['total']}s")
        logger.info(f"  Stage 2 (SAM):    mean={timing_stats['stage2_sam']['mean']}s, total={timing_stats['stage2_sam']['total']}s")
    logger.info(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
