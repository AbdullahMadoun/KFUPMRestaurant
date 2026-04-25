#!/usr/bin/env python3
# =============================================================================
# FILE: scripts/registry_append.py
# CATEGORY: UTIL
# PURPOSE: Append a single line to experiments/registry.jsonl summarizing one
#          finished run. Reads events.jsonl + run_metadata.json from the run
#          dir; produces a flat dict suitable for grep / jq queries later.
# DEPENDENCIES: stdlib only
# USAGE:
#   python scripts/registry_append.py <log_dir>
#   # e.g. logs/trial-20260321-cleandata1/joint
# =============================================================================
"""Append a finished run to ``experiments/registry.jsonl``.

The registry is the single answer to "what was the best result for ablation X".
Each line is a complete record — no joins, no nested objects beyond a flat
metric dict, so a one-line ``jq`` query can answer "best dev/combined where
adapter.kind = oracle".
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "experiments" / "registry.jsonl"


def read_events(p: Path) -> List[dict]:
    if not p.exists():
        return []
    rows: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def best_dev_event(events: List[dict]) -> Optional[dict]:
    candidates = [e for e in events if e.get("event_type") == "eval_epoch"]
    if not candidates:
        return None
    return max(candidates, key=lambda e: float(e.get("dev/combined", float("-inf"))))


def final_train_event(events: List[dict]) -> Optional[dict]:
    candidates = [e for e in events if e.get("event_type") == "train_eval_final"]
    return candidates[-1] if candidates else None


def run_status(log_dir: Path) -> str:
    p = log_dir / "run_status.json"
    if not p.exists():
        return "unknown"
    try:
        return json.loads(p.read_text()).get("status", "unknown")
    except json.JSONDecodeError:
        return "unknown"


def build_record(log_dir: Path) -> dict:
    md_path = log_dir / "run_metadata.json"
    md = json.loads(md_path.read_text()) if md_path.exists() else {}

    events = read_events(log_dir / "events.jsonl")
    best = best_dev_event(events) or {}
    final_train = final_train_event(events) or {}

    return {
        "ts_appended_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": md.get("run_name") or log_dir.parent.name,
        "log_dir": str(log_dir),
        "status": run_status(log_dir),
        "elapsed_sec": md.get("elapsed_sec") or final_train.get("elapsed_sec"),
        # provenance
        "git_sha": (md.get("git") or {}).get("sha"),
        "git_dirty": (md.get("git") or {}).get("dirty"),
        "git_branch": (md.get("git") or {}).get("branch"),
        "seed": md.get("seed"),
        "determinism_mode": md.get("determinism_mode"),
        "torch_version": md.get("torch"),
        "cuda_devices": (md.get("cuda") or {}).get("devices"),
        # dataset
        "dataset_version": (md.get("dataset") or {}).get("version"),
        "dataset_hash": (md.get("dataset") or {}).get("hash"),
        "dataset_export_root": (md.get("dataset") or {}).get("export_root"),
        # headline metrics
        "best_epoch": best.get("epoch"),
        "best_dev_combined": best.get("dev/combined"),
        "best_dev_combined_formula_version": best.get("dev/combined_formula_version"),
        "best_dev_stage1_recall@0.5": best.get("dev/stage1_recall@0.5"),
        "best_dev_stage1_precision@0.5": best.get("dev/stage1_precision@0.5"),
        "best_dev_stage2_mIoU": best.get("dev/stage2_mIoU"),
        "best_dev_stage3_acc": best.get("dev/stage3_acc"),
        "best_dev_stage3_episode_acc": best.get("dev/stage3_episode_acc"),
        "best_dev_pred_items_per_image": best.get("dev/pred_items_per_image"),
        "best_dev_loss_total": best.get("dev/loss_total"),
        # final train (after restoring best)
        "final_train_stage3_episode_acc": final_train.get("train/stage3_episode_acc"),
        "final_train_stage3_acc": final_train.get("train/stage3_acc"),
        "final_train_loss_total": final_train.get("train/loss_total"),
        # health signals
        "train_nan_total": final_train.get("train/nan_total") or 0,
        "train_episode_leak_fallback_total": final_train.get("train/episode_leak_fallback_total") or 0,
    }


def append(record: dict, registry_path: Path = REGISTRY_PATH) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a finished run to experiments/registry.jsonl")
    parser.add_argument("log_dir", help="path to the joint log dir (contains events.jsonl + run_metadata.json)")
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="path to registry.jsonl")
    parser.add_argument("--print", action="store_true", help="print the record without appending")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"[FATAL] not a directory: {log_dir}", file=sys.stderr)
        return 2
    if not (log_dir / "events.jsonl").exists():
        print(f"[FATAL] missing events.jsonl in: {log_dir}", file=sys.stderr)
        return 2

    record = build_record(log_dir)
    if args.print:
        print(json.dumps(record, indent=2, default=str))
        return 0
    registry_path = Path(args.registry)
    append(record, registry_path)
    print(f"[registry] appended to {registry_path}: run={record['run_name']} dev/combined={record['best_dev_combined']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
