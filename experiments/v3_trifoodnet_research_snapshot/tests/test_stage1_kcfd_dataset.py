from __future__ import annotations

import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


REQUIRED_DATASET_API = (
    "Stage1KCFDConfig",
    "Stage1KCFDDataset",
    "preflight_stage1_kcfd_export",
)


def _load_dataset_api():
    try:
        api = importlib.import_module("stage1_kcfd.dataset")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Expected Stage 1 dataset layer at stage1_kcfd.dataset exposing "
            f"{', '.join(REQUIRED_DATASET_API)}"
        )
        raise AssertionError from exc

    missing = [name for name in REQUIRED_DATASET_API if not hasattr(api, name)]
    if missing:
        pytest.fail(f"stage1_kcfd.dataset is missing required API(s): {', '.join(missing)}")
    return api


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_item(handle, **row: Any) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")


def _paint_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (80, 80), color)
    image.save(path)


def _paint_mask(path: Path, bbox: list[int]) -> None:
    mask = Image.new("L", (80, 80), 0)
    x1, y1, x2, y2 = bbox
    for y in range(y1, y2):
        for x in range(x1, x2):
            mask.putpixel((x, y), 255)
    mask.save(path)


@pytest.fixture
def v3_export_root(tmp_path: Path) -> Path:
    root = tmp_path / "v3_export"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)

    classes = [
        {
            "id": 10,
            "compact_id": 0,
            "slug": "rice",
            "display_name": "rice",
            "size": 4,
            "name_distribution": {"rice": 4},
        },
        {
            "id": 20,
            "compact_id": 1,
            "slug": "chicken",
            "display_name": "chicken pieces",
            "size": 3,
            "name_distribution": {"chicken": 3},
        },
    ]
    _write_json(root / "classes.json", classes)
    _write_json(
        root / "manifest.json",
        {
            "version": "v3",
            "content_hash_sha8": "stage1fx",
            "counts": {"classes": 2, "items": 7, "images": 4},
            "schema": {
                "fields": [
                    "sample_id",
                    "class_id",
                    "compact_id",
                    "class_slug",
                    "class_display_name",
                    "src_image_id",
                    "src_item_index",
                    "name",
                    "description",
                    "bbox",
                    "is_reference",
                    "image_path",
                    "mask_path",
                ]
            },
        },
    )

    image_specs = {
        "img_001": (200, 20, 20),
        "img_002": (20, 200, 20),
        "img_003": (20, 20, 200),
        "img_004": (160, 160, 20),
    }
    for image_id, color in image_specs.items():
        _paint_image(root / "images" / f"{image_id}.jpg", color)
        (root / "masks" / image_id).mkdir()

    item_rows = [
        {
            "sample_id": "img_001__right_top",
            "class_id": 20,
            "compact_id": 1,
            "class_slug": "chicken",
            "class_display_name": "chicken pieces",
            "src_image_id": "img_001",
            "src_item_index": 2,
            "name": "right top chicken",
            "description": "glossy browned chicken on the right",
            "bbox": [50, 10, 70, 30],
            "is_reference": False,
        },
        {
            "sample_id": "img_001__left_bottom",
            "class_id": 10,
            "compact_id": 0,
            "class_slug": "rice",
            "class_display_name": "rice",
            "src_image_id": "img_001",
            "src_item_index": 1,
            "name": "left bottom rice",
            "description": "rice below the top row",
            "bbox": [10, 50, 30, 70],
            "is_reference": False,
        },
        {
            "sample_id": "img_001__left_top",
            "class_id": 10,
            "compact_id": 0,
            "class_slug": "rice",
            "class_display_name": "rice",
            "src_image_id": "img_001",
            "src_item_index": 0,
            "name": "left top rice",
            "description": "fluffy white rice at upper left",
            "bbox": [10, 10, 30, 30],
            "is_reference": True,
        },
        {
            "sample_id": "img_002__missing_descriptor",
            "class_id": 20,
            "compact_id": 1,
            "class_slug": "chicken",
            "class_display_name": "chicken pieces",
            "src_image_id": "img_002",
            "src_item_index": 0,
            "name": "plain chicken",
            "description": "",
            "bbox": [8, 8, 28, 28],
            "is_reference": True,
        },
        {
            "sample_id": "img_003__missing_name",
            "class_id": 10,
            "compact_id": 0,
            "class_slug": "rice",
            "class_display_name": "rice",
            "src_image_id": "img_003",
            "src_item_index": 0,
            "name": "",
            "description": "small mound with visible grains",
            "bbox": [12, 12, 32, 32],
            "is_reference": False,
        },
        {
            "sample_id": "img_004__rice",
            "class_id": 10,
            "compact_id": 0,
            "class_slug": "rice",
            "class_display_name": "rice",
            "src_image_id": "img_004",
            "src_item_index": 0,
            "name": "center rice",
            "description": "centered mound of rice",
            "bbox": [20, 20, 42, 42],
            "is_reference": False,
        },
        {
            "sample_id": "img_004__chicken",
            "class_id": 20,
            "compact_id": 1,
            "class_slug": "chicken",
            "class_display_name": "chicken pieces",
            "src_image_id": "img_004",
            "src_item_index": 1,
            "name": "lower chicken",
            "description": "small chicken piece below rice",
            "bbox": [44, 48, 68, 70],
            "is_reference": False,
        },
    ]

    with (root / "items.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(item_rows):
            image_id = row["src_image_id"]
            mask_path = f"masks/{image_id}/item_{index:03d}_mask.png"
            row = {
                **row,
                "image_width": 80,
                "image_height": 80,
                "image_path": f"images/{image_id}.jpg",
                "mask_path": mask_path,
            }
            _paint_mask(root / mask_path, list(row["bbox"]))
            _write_item(handle, **row)

    return root


def _make_config(api, export_root: Path, **overrides: Any):
    kwargs = {
        "export_root": export_root,
        "split": "train",
        "train_ratio": 1.0,
        "val_ratio": 0.0,
        "test_ratio": 0.0,
        "split_seed": 123,
        "reference_policy": "include",
        "allow_incomplete_export": True,
    }
    kwargs.update(overrides)
    return api.Stage1KCFDConfig(**kwargs)


def _make_dataset(api, export_root: Path, **overrides: Any):
    return api.Stage1KCFDDataset(_make_config(api, export_root, **overrides))


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


def _example_image_id(example: Any) -> str:
    if isinstance(example, dict):
        for key in ("image_id", "src_image_id"):
            if key in example:
                return str(example[key])
    if hasattr(example, "image_id"):
        return str(example.image_id)
    pytest.fail("Dataset example must expose image_id or src_image_id")


def _target_payload(example: Any) -> dict:
    if isinstance(example, tuple) and len(example) >= 2:
        target = example[1]
    elif isinstance(example, dict):
        for key in ("target", "target_json", "stage1_target", "labels"):
            if key in example:
                target = example[key]
                break
        else:
            pytest.fail("Dataset example must expose a Stage 1 target payload")
    elif hasattr(example, "target"):
        target = example.target
    else:
        pytest.fail("Dataset example must expose a Stage 1 target payload")

    if isinstance(target, str):
        return json.loads(target)
    return _plain(target)


def _image_payload(example: Any) -> Image.Image:
    if isinstance(example, tuple):
        image = example[0]
    elif isinstance(example, dict):
        for key in ("image", "pil_image", "input_image"):
            if key in example:
                image = example[key]
                break
        else:
            pytest.fail("Dataset example must expose the loaded PIL image")
    elif hasattr(example, "image"):
        image = example.image
    else:
        pytest.fail("Dataset example must expose the loaded PIL image")

    assert isinstance(image, Image.Image)
    return image


def _examples(dataset: Any) -> list[Any]:
    return [dataset[index] for index in range(len(dataset))]


def _image_ids(dataset: Any) -> set[str]:
    if hasattr(dataset, "image_ids"):
        return {str(image_id) for image_id in dataset.image_ids}
    if hasattr(dataset, "samples"):
        ids = {
            str(sample["image_id"] if isinstance(sample, dict) else sample.image_id)
            for sample in dataset.samples
        }
        if ids:
            return ids
    return {_example_image_id(example) for example in _examples(dataset)}


def test_dataset_loads_v3_export_images_and_masks_without_crops(v3_export_root: Path):
    api = _load_dataset_api()

    dataset = _make_dataset(api, v3_export_root)
    example = next(example for example in _examples(dataset) if _example_image_id(example) == "img_001")

    image = _image_payload(example)
    target = _target_payload(example)

    assert image.size == (80, 80)
    assert len(target["items"]) == 3
    assert all(set(item) == {"name", "bbox", "descriptor"} for item in target["items"])
    assert target["items"][0] == {
        "name": "left top rice",
        "bbox": [10, 10, 30, 30],
        "descriptor": "fluffy white rice at upper left",
    }


def test_reference_exclude_policy_removes_whole_source_images(v3_export_root: Path):
    api = _load_dataset_api()

    dataset = _make_dataset(api, v3_export_root, reference_policy="exclude")

    assert "img_001" not in _image_ids(dataset)
    assert "img_002" not in _image_ids(dataset)
    assert _image_ids(dataset) == {"img_003", "img_004"}


def test_default_pause_policy_rejects_reference_exports(v3_export_root: Path):
    api = _load_dataset_api()

    with pytest.raises(ValueError, match="reference_policy='pause'"):
        api.Stage1KCFDDataset(
            api.Stage1KCFDConfig(
                export_root=v3_export_root,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
                split_seed=123,
            )
        )


def test_targets_are_sorted_by_bbox_center_top_to_bottom_then_left_to_right(v3_export_root: Path):
    api = _load_dataset_api()

    dataset = _make_dataset(api, v3_export_root)
    example = next(example for example in _examples(dataset) if _example_image_id(example) == "img_001")
    target = _target_payload(example)

    assert [item["name"] for item in target["items"]] == [
        "left top rice",
        "right top chicken",
        "left bottom rice",
    ]


def test_image_level_split_is_deterministic_and_has_no_cross_split_leakage(v3_export_root: Path):
    api = _load_dataset_api()
    split_kwargs = {
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "split_seed": 7,
    }

    first = {
        split: _image_ids(_make_dataset(api, v3_export_root, split=split, **split_kwargs))
        for split in ("train", "val", "test")
    }
    second = {
        split: _image_ids(_make_dataset(api, v3_export_root, split=split, **split_kwargs))
        for split in ("train", "val", "test")
    }

    assert first == second
    assert first["train"].isdisjoint(first["val"])
    assert first["train"].isdisjoint(first["test"])
    assert first["val"].isdisjoint(first["test"])
    assert first["train"] | first["val"] | first["test"] == {
        "img_001",
        "img_002",
        "img_003",
        "img_004",
    }


def _stat(stats: Any, *names: str) -> int:
    stats = _plain(stats)
    for name in names:
        if isinstance(stats, dict) and name in stats:
            return int(stats[name])
    pytest.fail(f"Preflight stats missing any of: {', '.join(names)}")


def test_preflight_reports_reference_and_missing_metadata_counts(v3_export_root: Path):
    api = _load_dataset_api()

    stats = api.preflight_stage1_kcfd_export(v3_export_root)

    assert _stat(stats, "reference_count", "reference_items", "references") == 2
    assert _stat(stats, "missing_descriptor_count", "missing_descriptors") == 1
    assert _stat(stats, "missing_name_count", "missing_names") == 1
    assert _stat(stats, "bbox_checked_count") == 7
    assert _stat(stats, "bbox_out_of_frame_count") == 0
    assert _stat(stats, "bbox_out_of_frame_minor_count") == 0
    assert _stat(stats, "bbox_out_of_frame_major_count") == 0


def test_minor_boundary_box_overruns_are_reported_clipped_and_not_training_blockers(v3_export_root: Path):
    api = _load_dataset_api()
    rows = [
        json.loads(line)
        for line in (v3_export_root / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["sample_id"] == "img_004__rice":
            row["bbox"] = [-1, -2, 82, 81]
    with (v3_export_root / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            _write_item(handle, **row)

    stats = api.preflight_stage1_kcfd_export(v3_export_root)
    bad = api.incomplete_export_counts(stats)
    dataset = _make_dataset(api, v3_export_root)
    example = next(example for example in _examples(dataset) if _example_image_id(example) == "img_004")
    target = _target_payload(example)

    assert stats["bbox_out_of_frame_count"] == 1
    assert stats["bbox_out_of_frame_minor_count"] == 1
    assert stats["bbox_out_of_frame_major_count"] == 0
    assert "bbox_out_of_frame_count" not in bad
    assert "bbox_out_of_frame_major_count" not in bad
    assert target["items"][0]["bbox"] == [0, 0, 80, 80]


def test_dataset_rejects_incomplete_exports_by_default(v3_export_root: Path):
    api = _load_dataset_api()

    with pytest.raises(ValueError, match="incomplete Stage 1 training data"):
        api.Stage1KCFDDataset(
            api.Stage1KCFDConfig(
                export_root=v3_export_root,
                reference_policy="include",
                allow_incomplete_export=False,
            )
        )


def test_manifest_hash_mismatch_is_rejected(v3_export_root: Path):
    api = _load_dataset_api()

    with pytest.raises(ValueError, match="dataset hash mismatch"):
        _make_dataset(api, v3_export_root, expected_hash="wronghash")


def test_split_mapping_is_persisted_with_dataset_metadata(v3_export_root: Path, tmp_path: Path):
    api = _load_dataset_api()
    splits_path = tmp_path / "splits.json"

    first = _make_dataset(api, v3_export_root, splits_path=splits_path)
    second = _make_dataset(api, v3_export_root, splits_path=splits_path)
    payload = json.loads(splits_path.read_text(encoding="utf-8"))

    assert splits_path.exists()
    assert payload["dataset_hash"] == "stage1fx"
    assert payload["method"] == "stage1_image_level_stratified_class_item_balance_v2"
    assert first.split_mapping == second.split_mapping


def test_training_previews_render_mask_native_and_full_resolution_boxes(v3_export_root: Path, tmp_path: Path):
    api = _load_dataset_api()
    visualize = importlib.import_module("stage1_kcfd.visualize")

    dataset = _make_dataset(api, v3_export_root)
    paths = visualize.save_training_previews(dataset, tmp_path / "previews", max_samples=2)

    assert len(paths) == 4
    assert all(path.exists() for path in paths)
    assert (tmp_path / "previews" / "manifest.json").exists()


def test_preflight_only_prints_reference_counts_without_training_gate(v3_export_root: Path, tmp_path: Path, capsys, monkeypatch):
    stage1_train = importlib.import_module("stage1_kcfd.train")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--export-root",
            str(v3_export_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--preflight-only",
        ],
    )

    stage1_train.main()
    printed = json.loads(capsys.readouterr().out)

    assert printed["reference_items"] == 2
    assert printed["reference_images"] == 2


@pytest.mark.parametrize(
    ("extra_args", "expected_split_seed"),
    [
        ([], 420),
        (["--split-seed", "314"], 314),
    ],
)
def test_train_cli_uses_canonical_split_seed_and_allows_override(
    v3_export_root: Path,
    tmp_path: Path,
    monkeypatch,
    extra_args: list[str],
    expected_split_seed: int,
):
    stage1_train = importlib.import_module("stage1_kcfd.train")
    captured: dict[str, Any] = {}

    def capture_config(config):
        captured["split_seed"] = config.split_seed
        raise RuntimeError("captured Stage 1 config")

    monkeypatch.setattr(stage1_train, "build_datasets_from_config", capture_config)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--export-root",
            str(v3_export_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "77",
            "--reference-policy",
            "include",
            "--allow-incomplete-export",
            *extra_args,
        ],
    )

    with pytest.raises(RuntimeError, match="captured Stage 1 config"):
        stage1_train.main()

    assert captured["split_seed"] == expected_split_seed


def test_visualize_cli_keeps_split_seed_separate_from_preview_selection_seed(
    v3_export_root: Path,
    tmp_path: Path,
    monkeypatch,
):
    visualize = importlib.import_module("stage1_kcfd.visualize")
    captured: dict[str, Any] = {}

    class FakeDataset:
        def __init__(self, config):
            captured["split_seed"] = config.split_seed
            self.config = config

    def fake_save_training_previews(dataset, output_dir, *, max_samples, seed, selection, **kwargs):
        captured["selection_seed"] = seed
        captured["selection"] = selection
        return []

    monkeypatch.setattr(visualize, "Stage1KCFDDataset", FakeDataset)
    monkeypatch.setattr(visualize, "save_training_previews", fake_save_training_previews)
    monkeypatch.setattr(
        "sys.argv",
        [
            "visualize.py",
            "--export-root",
            str(v3_export_root),
            "--output-dir",
            str(tmp_path / "previews"),
            "--split-seed",
            "420",
            "--seed",
            "20260426",
            "--selection",
            "class-diverse",
        ],
    )

    visualize.main()

    assert captured == {
        "split_seed": 420,
        "selection_seed": 20260426,
        "selection": "class-diverse",
    }


def test_train_cli_blocks_incomplete_exports_before_dataset_build(v3_export_root: Path, tmp_path: Path, monkeypatch):
    stage1_train = importlib.import_module("stage1_kcfd.train")

    def fail_if_called(*args, **kwargs):
        pytest.fail("dataset build should not run after incomplete export preflight failure")

    monkeypatch.setattr(stage1_train, "build_datasets_from_config", fail_if_called)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--export-root",
            str(v3_export_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--reference-policy",
            "include",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        stage1_train.main()

    message = str(exc_info.value)
    assert "incomplete Stage 1 training data" in message
    assert "missing_name_count" in message
    assert "missing_descriptor_count" in message
