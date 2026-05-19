from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image


def get_bbox_scale_factor(item: dict, export_root: Path) -> Tuple[float, float]:
    img_rel = item.get("image_path") or ""
    mask_rel = item.get("mask_path") or ""
    if not img_rel or not mask_rel:
        return (1.0, 1.0)
    img_path = Path(export_root) / img_rel
    mask_path = Path(export_root) / mask_rel
    if not img_path.exists() or not mask_path.exists():
        return (1.0, 1.0)
    with Image.open(img_path) as img:
        img_w, img_h = img.size
    with Image.open(mask_path) as mask:
        mask_w, mask_h = mask.size
    if mask_w == 0 or mask_h == 0:
        return (1.0, 1.0)
    return (img_w / mask_w, img_h / mask_h)


def scale_bbox(bbox: Sequence[float], sx: float, sy: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def load_image_at_mask_resolution(item: dict, export_root: Path) -> Image.Image:
    image = Image.open(Path(export_root) / item["image_path"]).convert("RGB")
    mask_path = Path(export_root) / item.get("mask_path", "")
    if mask_path.exists():
        with Image.open(mask_path) as mask:
            mask_size = mask.size
        if image.size != mask_size:
            image = image.resize(mask_size, Image.Resampling.LANCZOS)
    return image


def load_image_at_v3_resolution(item: dict, export_root: Path) -> Image.Image:
    return Image.open(Path(export_root) / item["image_path"]).convert("RGB")


def load_mask_at_native_resolution(item: dict, export_root: Path) -> np.ndarray:
    return np.array(Image.open(Path(export_root) / item["mask_path"]).convert("L"))
