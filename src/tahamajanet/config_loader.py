# =============================================================================
# FILE: config_loader.py
# CATEGORY: UTIL
# PURPOSE: YAML config loader with dot-override support and DotDict accessors.
# DEPENDENCIES: PyYAML
# USED BY: benchmark_runtime.py, check_trainable.py, run_dev_inference.py, run_isolated_inference.py, run_single_inference.py, test_pictsure_lora.py, tests/test_pipeline_contracts.py, train_joint.py, train_stage3_hf.py, validate_pipeline_contracts.py, verify_split.py, visualize_val_predictions.py
# KEY CLASSES/FUNCTIONS: DotDict, _deep_merge, _default_config_path, load_config, _apply_override
# LAST MODIFIED: 2026-03-21T07:09:56+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
trifoodnet/utils/config_loader.py
───────────────────────────────────
Single entry-point for all configs.

    from trifoodnet.utils.config_loader import load_config, cfg

    # Read master_config.yaml (default)
    cfg = load_config()

    # Override one field at a time
    cfg.stage1.training.learning_rate = 5e-4

    # Override from CLI  (key=value pairs)
    cfg = load_config(overrides=["stage1.lora.r=32", "joint.loss_weights.lambda1=2.0"])

    # Save current config next to a checkpoint
    cfg.save("checkpoints/v3-run1/config_snapshot.yaml")

Design
──────
Uses a thin DotDict wrapper so attributes are accessible with
  cfg.stage1.lora.r          instead of  cfg["stage1"]["lora"]["r"]
and still serialisable back to plain dicts / YAML.
"""

from __future__ import annotations
import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# DotDict — recursive attribute-access wrapper around dicts
# ──────────────────────────────────────────────────────────────────────────────

class DotDict:
    """
    A dict whose keys are also accessible as attributes.
    Nested dicts are automatically wrapped.
    """

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, DotDict(v) if isinstance(v, dict) else v)

    # ── attribute access ──────────────────────────────────────────────────────

    def __getattr__(self, name: str):
        # Only called when normal lookup fails
        raise AttributeError(f"Config has no key '{name}'")

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, dict):
            value = DotDict(value)
        super().__setattr__(name, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    # ── dict-like serialisation ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, DotDict) else v
        return out

    def __repr__(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    # ── deep merge ────────────────────────────────────────────────────────────

    def merge(self, other: Union[dict, "DotDict"]):
        """Recursively merge another dict/DotDict into self (other wins)."""
        other_dict = other.to_dict() if isinstance(other, DotDict) else other
        _deep_merge(self.__dict__, other_dict)

    # ── save ─────────────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"[Config] Snapshot saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Deep-merge helper
# ──────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], DotDict) and isinstance(v, dict):
            _deep_merge(base[k].__dict__, v)
        elif k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = DotDict(v) if isinstance(v, dict) else v


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def _default_config_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.with_name("master_config.yaml"),
        here.parent / "configs" / "master_config.yaml",
        here.parent.parent / "configs" / "master_config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_config(
    config_path: Optional[str | Path] = None,
    overrides:   Optional[List[str]]  = None,
) -> DotDict:
    """
    Load config from YAML, then apply any key=value overrides.

    Parameters
    ----------
    config_path : path to YAML file (defaults to configs/master_config.yaml)
    overrides   : list of "section.key=value" strings, e.g.
                  ["stage1.lora.r=32", "joint.loss_weights.lambda2=0.5"]

    Returns
    -------
    DotDict  — fully resolved config object
    """
    path = Path(config_path) if config_path else _default_config_path()

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = DotDict(raw)

    if overrides:
        for override in overrides:
            _apply_override(cfg, override)

    # Static (dataset-independent) validation. Dataset-dependent checks run
    # in train_joint.py after the adapter has reported class counts.
    try:
        from config_validation import validate_config, ConfigValidationError
    except ImportError:
        # Validation module is optional; fall through if unavailable.
        return cfg
    warnings = validate_config(cfg)
    for msg in warnings:
        print(f"[Config] WARN: {msg}")
    return cfg


def _apply_override(cfg: DotDict, override: str):
    """
    Apply a single "dotted.key=value" override to cfg in-place.
    Values are parsed with yaml.safe_load so "true", "1e-4", "[1,3,5]" all work.
    """
    if "=" not in override:
        raise ValueError(f"Override must be 'key=value', got: '{override}'")

    key_path, raw_value = override.split("=", 1)
    value = yaml.safe_load(raw_value)

    parts = key_path.strip().split(".")
    node  = cfg
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)
    print(f"[Config] Override applied: {key_path} = {value!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Convenience singleton  (import and use directly in notebooks / scripts)
# ──────────────────────────────────────────────────────────────────────────────

# Not pre-loaded — call load_config() explicitly so paths resolve correctly.
# Example:
#   from trifoodnet.utils.config_loader import load_config
#   cfg = load_config()
#   print(cfg.stage1.lora.r)           # → 16
#   print(cfg.joint.loss_weights.lambda1)  # → 1.0
