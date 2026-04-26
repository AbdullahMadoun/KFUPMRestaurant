from __future__ import annotations

import importlib
from dataclasses import asdict, is_dataclass
from typing import Any

import pytest


REQUIRED_SCHEMA_API = ("parse_stage1_target",)


def _load_schema_api():
    try:
        api = importlib.import_module("stage1_kcfd.schema")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Expected Stage 1 schema layer at stage1_kcfd.schema exposing "
            f"{', '.join(REQUIRED_SCHEMA_API)}"
        )
        raise AssertionError from exc

    missing = [name for name in REQUIRED_SCHEMA_API if not hasattr(api, name)]
    if missing:
        pytest.fail(f"stage1_kcfd.schema is missing required API(s): {', '.join(missing)}")
    return api


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain(child) for child in value]
    return value


def test_parse_stage1_target_accepts_name_bbox_descriptor_schema():
    api = _load_schema_api()

    parsed = api.parse_stage1_target(
        """
        {
          "items": [
            {
              "name": "rice",
              "bbox": [10, 20, 40, 60],
              "descriptor": "loose white grains"
            }
          ]
        }
        """
    )

    assert _plain(parsed) == {
        "items": [
            {
                "name": "rice",
                "bbox": [10, 20, 40, 60],
                "descriptor": "loose white grains",
            }
        ]
    }


@pytest.mark.parametrize(
    "payload",
    [
        '[{"name": "rice", "bbox": [0, 0, 1, 1], "descriptor": "loose"}]',
        '{"items": [{"label": "rice", "bbox": [0, 0, 1, 1], "descriptor": "loose"}]}',
        '{"items": [{"name": "rice", "bbox": [0, 0, 1, 1]}]}',
        '{"items": [{"name": "rice", "bbox": [0, 0, 1, 1], "descriptor": "loose", "id": 1}]}',
        '{"items": [{"name": "rice", "bbox": [0, 0, 1], "descriptor": "loose"}]}',
        '{"items": [{"name": "rice", "bbox": ["0", 0, 1, 1], "descriptor": "loose"}]}',
        '{"items": [{"name": "rice", "bbox": [10, 0, 1, 1], "descriptor": "loose"}]}',
        '{"items": [{"name": "", "bbox": [0, 0, 1, 1], "descriptor": "loose"}]}',
        '{"items": [{"name": "rice", "bbox": [0, 0, 1, 1], "descriptor": ""}]}',
    ],
)
def test_parse_stage1_target_rejects_non_strict_payloads(payload: str):
    api = _load_schema_api()

    with pytest.raises((TypeError, ValueError)):
        api.parse_stage1_target(payload)
