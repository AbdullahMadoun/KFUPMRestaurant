"""Logging setup, run tracking, and per-stage metrics collection."""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
from datetime import datetime


def setup_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure stdlib logger with console + optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_dir:
        log_path = Path(log_dir) / "pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


@dataclass
class ImageMetrics:
    """Per-image metrics for the 3-stage pipeline."""
    image_path: str = ""
    # Stage 1: VLM
    visual_items: int = 0
    stage1_time: float = 0.0
    # Stage 2: SAM3
    segmented_items: int = 0
    stage2_time: float = 0.0
    # Stage 3: Matching
    matched_items: int = 0
    unknown_items: int = 0
    stage3_time: float = 0.0
    # Results
    total_price: float = 0.0
    matches: List[dict] = field(default_factory=list)
    visualization_path: str = ""
    error: Optional[str] = None


@dataclass
class RunSummary:
    """Aggregated metrics for a full pipeline run."""
    run_id: str = ""
    config_snapshot: dict = field(default_factory=dict)
    total_time: float = 0.0
    total_images: int = 0
    total_matched: int = 0
    total_unknown: int = 0
    total_price: float = 0.0
    avg_stage1_time: float = 0.0
    avg_stage2_time: float = 0.0
    avg_stage3_time: float = 0.0
    per_image: List[dict] = field(default_factory=list)


class RunTracker:
    """Creates a timestamped run directory and collects per-image metrics."""

    def __init__(self, base_dir: str, config_dict: dict):
        self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / self.run_id
        self.viz_dir = self.run_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        self.config_dict = config_dict
        self.image_metrics: List[ImageMetrics] = []
        self.start_time = time.time()

        # Save frozen config snapshot
        (self.run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

        # Setup logger with file output into the run directory
        self.logger = setup_logger("pipeline", str(self.run_dir))

    def add_image_metrics(self, metrics: ImageMetrics):
        self.image_metrics.append(metrics)

    def finalize(self) -> RunSummary:
        total_time = time.time() - self.start_time
        successful = [m for m in self.image_metrics if m.error is None]

        summary = RunSummary(
            run_id=self.run_id,
            config_snapshot=self.config_dict,
            total_time=total_time,
            total_images=len(self.image_metrics),
            total_matched=sum(m.matched_items for m in successful),
            total_unknown=sum(m.unknown_items for m in successful),
            total_price=sum(m.total_price for m in successful),
            avg_stage1_time=sum(m.stage1_time for m in successful) / max(len(successful), 1),
            avg_stage2_time=sum(m.stage2_time for m in successful) / max(len(successful), 1),
            avg_stage3_time=sum(m.stage3_time for m in successful) / max(len(successful), 1),
            per_image=[asdict(m) for m in self.image_metrics],
        )

        (self.run_dir / "run_summary.json").write_text(json.dumps(asdict(summary), indent=2))
        return summary
