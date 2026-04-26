from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont

from .bbox_canonical import get_bbox_scale_factor, load_image_at_mask_resolution, load_image_at_v3_resolution, scale_bbox
from .dataset import (
    BBOX_MINOR_OUT_OF_FRAME_TOLERANCE_PX,
    read_jsonl,
    validate_stage1_export,
)


BOX_COLOR = (37, 157, 77)
MASK_TIGHT_COLOR = (20, 135, 210)
WRONG_SPACE_COLOR = (220, 68, 55)
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)


def _font(size: int = 15) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _is_reference(row: dict) -> bool:
    return bool(row.get("is_reference"))


def _label(row: dict) -> str:
    return str(row.get("class_slug") or row.get("class_display_name") or row.get("name") or "food")


def _image_id(row: dict) -> str:
    return str(row.get("src_image_id") or row.get("image_id"))


def _group_rows(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(_image_id(row), []).append(row)
    return dict(sorted(grouped.items()))


def _filter_reference_policy(rows: Sequence[dict], reference_policy: str) -> List[dict]:
    if reference_policy == "include":
        return list(rows)
    reference_images = {_image_id(row) for row in rows if _is_reference(row)}
    if reference_policy == "exclude":
        return [row for row in rows if _image_id(row) not in reference_images]
    if reference_policy == "train":
        return list(rows)
    raise ValueError("reference_policy must be exclude, train, or include")


def _mask_bbox(row: dict, export_root: Path) -> List[int] | None:
    mask_path = export_root / str(row.get("mask_path", ""))
    if not mask_path.exists():
        return None
    with Image.open(mask_path) as mask:
        return list(mask.convert("L").getbbox() or []) or None


def _clamp_box(box: Sequence[float], size: tuple[int, int]) -> List[float]:
    width, height = size
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def _frame_excess(box: Sequence[float], size: tuple[int, int]) -> float:
    width, height = size
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(0.0, -x1, -y1, x2 - width, y2 - height)


def _max_abs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(float(a[idx]) - float(b[idx])) for idx in range(4))


def _summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "count": float(len(ordered)),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "max": float(max(ordered)),
    }


def compute_bbox_audit(rows: Sequence[dict], export_root: str | Path, *, reference_policy: str = "exclude") -> Dict[str, Any]:
    root = Path(export_root)
    selected_rows = _filter_reference_policy(rows, reference_policy)
    raw_to_mask_tight: List[float] = []
    scaled_to_full_tight: List[float] = []
    wrong_raw_on_full: List[float] = []
    image_mask_mismatch = 0
    out_of_frame_minor = 0
    out_of_frame_major = 0

    for row in selected_rows:
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        mask_box = _mask_bbox(row, root)
        if mask_box is None:
            continue
        with Image.open(root / str(row["mask_path"])) as mask:
            mask_size = mask.size
        with Image.open(root / str(row["image_path"])) as image:
            image_size = image.size
        if image_size != mask_size:
            image_mask_mismatch += 1
        excess = _frame_excess(bbox, mask_size)
        if excess > 0:
            if excess <= BBOX_MINOR_OUT_OF_FRAME_TOLERANCE_PX:
                out_of_frame_minor += 1
            else:
                out_of_frame_major += 1
        raw_to_mask_tight.append(_max_abs_delta(_clamp_box(bbox, mask_size), mask_box))
        sx, sy = get_bbox_scale_factor(row, root)
        scaled_bbox = _clamp_box(scale_bbox(bbox, sx, sy), image_size)
        scaled_mask_box = scale_bbox(mask_box, sx, sy)
        scaled_to_full_tight.append(_max_abs_delta(scaled_bbox, scaled_mask_box))
        wrong_raw_on_full.append(_max_abs_delta(_clamp_box(bbox, image_size), scaled_mask_box))

    reference_images = {_image_id(row) for row in rows if _is_reference(row)}
    return {
        "reference_policy": reference_policy,
        "total_rows": len(rows),
        "audited_rows": len(selected_rows),
        "reference_images_excluded": len(reference_images) if reference_policy == "exclude" else 0,
        "image_mask_size_mismatch_count": image_mask_mismatch,
        "bbox_out_of_frame_minor_count": out_of_frame_minor,
        "bbox_out_of_frame_major_count": out_of_frame_major,
        "raw_bbox_vs_mask_tight_px": _summary(raw_to_mask_tight),
        "scaled_bbox_vs_fullres_mask_tight_px": _summary(scaled_to_full_tight),
        "wrong_raw_bbox_drawn_on_fullres_px": _summary(wrong_raw_on_full),
        "coordinate_contract": (
            "Training/eval: load image at mask resolution and use raw bbox. "
            "Full-res visualization: scale bbox by image_size/mask_size before drawing."
        ),
    }


def _select_image_ids(rows: Sequence[dict], export_root: Path, *, max_samples: int, seed: int) -> List[str]:
    grouped = _group_rows(rows)
    rng = random.Random(seed)
    image_ids = list(grouped)
    rng.shuffle(image_ids)

    boundary_ids: List[str] = []
    for image_id, image_rows in grouped.items():
        for row in image_rows:
            if "bbox" not in row:
                continue
            with Image.open(export_root / str(row["mask_path"])) as mask:
                if _frame_excess(row["bbox"], mask.size) > 0:
                    boundary_ids.append(image_id)
                    break
    boundary_ids = sorted(set(boundary_ids))

    selected: List[str] = []
    seen_labels: set[str] = set()
    for image_id in boundary_ids:
        if len(selected) >= max_samples:
            return selected
        selected.append(image_id)
        seen_labels.update(_label(row) for row in grouped[image_id])

    remaining = [image_id for image_id in image_ids if image_id not in set(selected)]
    while remaining and len(selected) < max_samples:
        best_pos = 0
        best_score = None
        for pos, image_id in enumerate(remaining):
            labels = {_label(row) for row in grouped[image_id]}
            score = (len(labels - seen_labels), len(labels), len(grouped[image_id]), -pos)
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
        image_id = remaining.pop(best_pos)
        selected.append(image_id)
        seen_labels.update(_label(row) for row in grouped[image_id])
    return selected


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill=TEXT_BG)
    draw.text((x, y), text, fill=TEXT_FG, font=font)


def _draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], color: tuple[int, int, int], label: str, *, width: int = 4) -> None:
    x1, y1, x2, y2 = [float(value) for value in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    _draw_label(draw, (x1 + 2, max(0, y1 + 2)), label, _font(13))


def _overlay_mask(base: Image.Image, mask_path: Path, color: tuple[int, int, int], *, target_size: tuple[int, int]) -> Image.Image:
    with Image.open(mask_path) as mask:
        alpha = mask.convert("L")
        if alpha.size != target_size:
            alpha = alpha.resize(target_size, Image.Resampling.NEAREST)
    alpha = alpha.point(lambda value: 72 if value > 0 else 0)
    overlay = Image.new("RGBA", target_size, (*color, 0))
    overlay.putalpha(alpha)
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def _resize_panel(image: Image.Image, width: int) -> Image.Image:
    if image.width <= width:
        return image
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def render_audit_panel(image_id: str, rows: Sequence[dict], export_root: str | Path, output_path: str | Path, *, panel_width: int = 900) -> Path:
    root = Path(export_root)
    first = rows[0]
    native = load_image_at_mask_resolution(first, root)
    full = load_image_at_v3_resolution(first, root)
    native_draw = native.copy()
    full_draw = full.copy()

    for row in rows:
        mask_path = root / str(row["mask_path"])
        native_draw = _overlay_mask(native_draw, mask_path, MASK_TIGHT_COLOR, target_size=native_draw.size)
        full_draw = _overlay_mask(full_draw, mask_path, MASK_TIGHT_COLOR, target_size=full_draw.size)

    native_canvas = ImageDraw.Draw(native_draw)
    full_canvas = ImageDraw.Draw(full_draw)
    for index, row in enumerate(rows, start=1):
        raw_box = row["bbox"]
        mask_box = _mask_bbox(row, root)
        sx, sy = get_bbox_scale_factor(row, root)
        scaled_box = scale_bbox(raw_box, sx, sy)
        label = str(row.get("name") or _label(row))[:30]
        _draw_box(native_canvas, _clamp_box(raw_box, native_draw.size), BOX_COLOR, f"{index} raw {label}")
        if mask_box:
            _draw_box(native_canvas, mask_box, MASK_TIGHT_COLOR, f"{index} mask-tight", width=2)
        _draw_box(full_canvas, _clamp_box(scaled_box, full_draw.size), BOX_COLOR, f"{index} scaled {label}")
        if mask_box:
            _draw_box(full_canvas, scale_bbox(mask_box, sx, sy), MASK_TIGHT_COLOR, f"{index} mask-tight", width=2)
        _draw_box(full_canvas, _clamp_box(raw_box, full_draw.size), WRONG_SPACE_COLOR, f"{index} raw-on-full wrong", width=2)

    native_panel = _resize_panel(native_draw, panel_width)
    full_panel = _resize_panel(full_draw, panel_width)
    title_font = _font(18)
    gap = 24
    title_h = 74
    out_w = native_panel.width + full_panel.width + gap
    out_h = title_h + max(native_panel.height, full_panel.height)
    out = Image.new("RGB", (out_w, out_h), "white")
    draw = ImageDraw.Draw(out)
    labels = sorted({_label(row) for row in rows})
    sx, sy = get_bbox_scale_factor(first, root)
    draw.text(
        (8, 6),
        f"{image_id} | items={len(rows)} | mask-native={native.size} | full-res={full.size} | scale=({sx:.3f},{sy:.3f})",
        fill=(0, 0, 0),
        font=title_font,
    )
    draw.text((8, 34), "green=training bbox, blue=mask tight bbox/overlay, red=wrong raw box on full-res image", fill=(0, 0, 0), font=_font(14))
    draw.text((8, 54), "classes: " + ", ".join(labels)[:180], fill=(0, 0, 0), font=_font(14))
    out.paste(native_panel, (0, title_h))
    out.paste(full_panel, (native_panel.width + gap, title_h))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


def write_bbox_audit(
    export_root: str | Path,
    output_dir: str | Path,
    *,
    reference_policy: str = "exclude",
    max_samples: int = 24,
    seed: int = 20260426,
    expected_version: str | None = "v3",
    expected_hash: str | None = None,
) -> Dict[str, Any]:
    root = Path(export_root)
    validate_stage1_export(root, expected_version=expected_version, expected_hash=expected_hash)
    rows = read_jsonl(root / "items.jsonl")
    rows = _filter_reference_policy(rows, reference_policy)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = compute_bbox_audit(read_jsonl(root / "items.jsonl"), root, reference_policy=reference_policy)
    grouped = _group_rows(rows)
    selected_ids = _select_image_ids(rows, root, max_samples=max_samples, seed=seed)
    files: List[str] = []
    for idx, image_id in enumerate(selected_ids):
        path = output / f"{idx:02d}_{image_id}_bbox_audit.png"
        render_audit_panel(image_id, grouped[image_id], root, path)
        files.append(str(path))
    manifest = {
        "export_root": str(root),
        "reference_policy": reference_policy,
        "seed": seed,
        "selected_image_ids": selected_ids,
        "files": files,
        "report": report,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a visual audit of v3 bbox coordinate alignment.")
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-policy", choices=["exclude", "train", "include"], default="exclude")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--expected-version", default="v3")
    parser.add_argument("--expected-hash", default=None)
    parser.add_argument("--allow-major-out-of-frame", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_bbox_audit(
        args.export_root,
        args.output_dir,
        reference_policy=args.reference_policy,
        max_samples=args.max_samples,
        seed=args.seed,
        expected_version=args.expected_version,
        expected_hash=args.expected_hash,
    )
    print(json.dumps(manifest["report"], indent=2, sort_keys=True))
    major = int(manifest["report"].get("bbox_out_of_frame_major_count", 0) or 0)
    if major and not args.allow_major_out_of_frame:
        raise SystemExit(f"bbox audit failed: {major} major out-of-frame boxes")


if __name__ == "__main__":
    main()
