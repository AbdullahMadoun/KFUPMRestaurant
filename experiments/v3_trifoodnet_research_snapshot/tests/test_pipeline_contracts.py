# =============================================================================
# FILE: tests/test_pipeline_contracts.py
# CATEGORY: TEST
# PURPOSE: Snapshot-retained source file for test_pipeline_contracts.py.
# DEPENDENCIES: config_loader.py, dataset_integration.py, validate_pipeline_contracts.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: _cfg, test_class_mapping_contract, test_split_contract, test_stage3_episode_contract, test_annotation_contract, test_stage3_support_capacity
# LAST MODIFIED: 2026-03-21T14:50:31.848923+00:00
# SNAPSHOT NOTES: contains hardcoded absolute paths that must be updated for a new environment
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_loader import load_config
from dataset_integration import build_export_paths
from validate_pipeline_contracts import (
    validate_annotation_contract,
    validate_class_mapping,
    validate_episode_contract,
    validate_split_contract,
    validate_stage3_support_capacity,
)


def _cfg():
    snapshot_root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(snapshot_root / "master_config.yaml"), overrides=["run.name=trial-20260321-cleandata1"])
    for candidate in [snapshot_root, Path("/root/rest_model")]:
        if (candidate / "Sampled_Images_All").exists():
            cfg.data.integration.repo_root = str(candidate)
            break
    return cfg


def _require_pointer_assets(cfg):
    repo_root = Path(str(cfg.data.integration.repo_root or "")).resolve()
    if not (repo_root / "Sampled_Images_All").exists():
        pytest.skip("Sampled_Images_All is not available; pointer-backed data contract checks are skipped.")


def test_class_mapping_contract():
    cfg = _cfg()
    export_root = build_export_paths(
        cfg.data.integration.batch_root,
        export_root=(cfg.data.integration.export_root or None),
        repo_root=(cfg.data.integration.repo_root or None),
    ).export_root
    report = validate_class_mapping(export_root)
    assert report["num_stage3_classes"] >= 18


def test_split_contract():
    cfg = _cfg()
    report = validate_split_contract(cfg)
    assert len(report["supported_classes"]) >= 18


def test_stage3_episode_contract():
    cfg = _cfg()
    _require_pointer_assets(cfg)
    report = validate_episode_contract(cfg)
    assert report["queries_checked"] > 0


def test_annotation_contract():
    cfg = _cfg()
    _require_pointer_assets(cfg)
    report = validate_annotation_contract(cfg)
    assert report["train"]["images"] >= 90
    assert report["dev"]["images"] >= 10
    assert report["test"]["images"] >= 10


def test_stage3_support_capacity():
    cfg = _cfg()
    report = validate_stage3_support_capacity(cfg)
    assert {"Fasoolia", "Koshari", "Muttabal"}.issubset(set(report["one_item_per_split_classes"]))
