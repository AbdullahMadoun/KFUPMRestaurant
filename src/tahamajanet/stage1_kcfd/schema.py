from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Stage1Item:
    name: str
    bbox: List[int | float]
    descriptor: str


@dataclass(frozen=True)
class Stage1Target:
    items: List[Stage1Item]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _json_number(value: float) -> int | float:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-6:
        return int(rounded)
    return round(float(value), 3)


def validate_bbox(bbox: Any) -> List[int | float]:
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in bbox)
    ):
        raise ValueError("bbox must be a list of four finite numbers")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox must be xyxy with non-negative width and height")
    return [_json_number(v) for v in (x1, y1, x2, y2)]


def parse_stage1_target(text_or_payload: str | Dict[str, Any]) -> Stage1Target:
    if isinstance(text_or_payload, str):
        payload = json.loads(text_or_payload.strip())
    else:
        payload = text_or_payload
    if not isinstance(payload, dict) or set(payload.keys()) != {"items"}:
        raise ValueError("target must be exactly one object with key: items")
    if not isinstance(payload["items"], list):
        raise ValueError("items must be a list")
    items: List[Stage1Item] = []
    for row in payload["items"]:
        if not isinstance(row, dict) or set(row.keys()) != {"name", "bbox", "descriptor"}:
            raise ValueError("each item must have exactly name, bbox, descriptor")
        name = normalize_text(row["name"])
        descriptor = normalize_text(row["descriptor"])
        if not name:
            raise ValueError("name must be non-empty")
        if not descriptor:
            raise ValueError("descriptor must be non-empty")
        items.append(Stage1Item(name=name, bbox=validate_bbox(row["bbox"]), descriptor=descriptor))
    return Stage1Target(items=items)


def target_to_payload(target: Stage1Target) -> Dict[str, Any]:
    return {
        "items": [
            {
                "name": item.name,
                "bbox": list(item.bbox),
                "descriptor": item.descriptor,
            }
            for item in target.items
        ]
    }


def target_to_json(target: Stage1Target) -> str:
    return json.dumps(target_to_payload(target), ensure_ascii=False, separators=(",", ":"))


def parse_prediction(text: str) -> Tuple[bool, Stage1Target, str]:
    try:
        parsed = parse_stage1_target(text)
        return True, parsed, ""
    except Exception as exc:
        return False, Stage1Target(items=[]), exc.__class__.__name__


def descriptor_word_count(descriptor: str) -> int:
    return len([part for part in re.split(r"\s+", descriptor.strip()) if part])
