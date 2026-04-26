from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STAGE1_PROMPT = (
    'List every food/beverage region visible in this image as JSON.\n'
    'Format: {"items":[{"name":"<short name>","bbox":[x1,y1,x2,y2],"descriptor":"<5-10 visual words>"}]}\n'
    "Use absolute pixel coordinates. One item per food/beverage region. Mixed dishes "
    "(stew, salad) are one item. Toppings stay with their base dish. Do not output "
    "the dish, plate, tray, table, utensils, or empty container areas. Descriptors "
    "must describe visible appearance only: color, texture, shape, surface cues. "
    "No prose or markdown."
)


@dataclass
class Stage1Config:
    export_root: Path
    output_dir: Path = Path("outputs/stage1_kcfd")
    run_name: str | None = None
    seed: int = 1337
    split: str = "train"
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    split_seed: int = 1337
    expected_version: str | None = "v3"
    expected_hash: str | None = None
    splits_path: Path | None = None
    reference_policy: str = "pause"  # pause | exclude | train | include
    allow_incomplete_export: bool = False
    train_max_images: int = 0
    eval_max_samples: int = 0
    prompt: str = STAGE1_PROMPT

    def __post_init__(self) -> None:
        self.export_root = Path(self.export_root)
        self.output_dir = Path(self.output_dir)
        if self.splits_path is not None:
            self.splits_path = Path(self.splits_path)
        if self.reference_policy not in {"pause", "exclude", "train", "include"}:
            raise ValueError("reference_policy must be one of pause, exclude, train, include")
        if self.split not in {"train", "val", "dev", "test"}:
            raise ValueError("split must be train, val/dev, or test")
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if ratio_sum <= 0:
            raise ValueError("split ratios must sum to > 0")


# Compatibility name used by contract tests.
Stage1KCFDConfig = Stage1Config
