#!/usr/bin/env python3
# =============================================================================
# FILE: scripts/compare_runs.py
# CATEGORY: TEST
# PURPOSE: Side-by-side comparison of two training runs by reading their
#          events.jsonl. Used in Phase 3 to quantify how the audit-bug fixes
#          + per-stage LRs + dataset migration moved the headline metrics.
# DEPENDENCIES: stdlib only
# USAGE:
#   python scripts/compare_runs.py <old_logdir> <new_logdir>
#   python scripts/compare_runs.py logs/trial-20260321-cleandata1/joint logs/trial-20260425-rerun/joint
# =============================================================================
"""Compare two runs against the Phase-3 expectations matrix.

For each side we extract:
    best_dev   : the eval_epoch event with the highest dev/combined
    final_train: the last train_eval_final event (run-end teacher-forced + e2e eval)

We then print:
    1. Per-metric side-by-side table
    2. The Phase-3 expectations matrix (predictions written before the rerun)
    3. A pass/fail verdict on each predicted delta

Exit code: 0 if all expectations passed, 1 if any failed, 2 on bad input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 expectations matrix
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Expectation:
    metric: str                  # key in the events.jsonl, with the prefix
    where: str                   # "best_dev" | "final_train"
    direction: str               # "down" | "up" | "near_zero" | "stable"
    threshold: float             # how much movement counts as a pass
    rationale: str               # why we expect this

EXPECTATIONS: List[Expectation] = [
    Expectation(
        metric="train/stage3_episode_acc",
        where="final_train",
        direction="down",
        threshold=0.20,
        rationale="Leak fix: query no longer in its own support set, "
                   "so train episode acc should drop 20+ points toward dev's level.",
    ),
    Expectation(
        metric="dev/stage3_episode_acc",
        where="best_dev",
        direction="stable",
        threshold=0.05,
        rationale="Dev episode sampling already used train support, no leak fix impact expected.",
    ),
    Expectation(
        metric="dev/pred_items_per_image",
        where="best_dev",
        direction="down",
        threshold=0.4,
        rationale="NMS is now active (iou_threshold=0.5 vs 0.0 before), "
                   "so overlapping detections get suppressed.",
    ),
    Expectation(
        metric="dev/stage1_precision@0.5",
        where="best_dev",
        direction="up",
        threshold=0.03,
        rationale="With overlap suppressed, fewer redundant predictions → precision rises.",
    ),
    Expectation(
        metric="dev/stage1_recall@0.5",
        where="best_dev",
        direction="stable",
        threshold=0.05,
        rationale="NMS only removes near-duplicates, recall should be unchanged.",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# events.jsonl parsing
# ──────────────────────────────────────────────────────────────────────────────


def read_events(events_path: Path) -> List[dict]:
    rows: List[dict] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] skipping malformed line in {events_path.name}: {exc}", file=sys.stderr)
    return rows


def find_best_dev_event(events: List[dict]) -> Optional[dict]:
    candidates = [e for e in events if e.get("event_type") == "eval_epoch"]
    if not candidates:
        return None
    return max(candidates, key=lambda e: float(e.get("dev/combined", float("-inf"))))


def find_final_train_eval_event(events: List[dict]) -> Optional[dict]:
    candidates = [e for e in events if e.get("event_type") == "train_eval_final"]
    if not candidates:
        return None
    return candidates[-1]


def find_run_summary(logdir: Path) -> dict:
    summary_path = logdir / "best_metrics.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def find_run_metadata(logdir: Path) -> dict:
    md_path = logdir / "run_metadata.json"
    if md_path.exists():
        with md_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────────────


def fmt_value(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def fmt_delta(old: Optional[float], new: Optional[float]) -> str:
    if old is None or new is None:
        return "—"
    delta = new - old
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
    return f"{arrow}{abs(delta):.4f}"


def run_comparison(old_dir: Path, new_dir: Path) -> int:
    old_events = read_events(old_dir / "events.jsonl")
    new_events = read_events(new_dir / "events.jsonl")
    if not old_events:
        print(f"[FATAL] no events read from {old_dir}", file=sys.stderr)
        return 2
    if not new_events:
        print(f"[FATAL] no events read from {new_dir}", file=sys.stderr)
        return 2

    old_best = find_best_dev_event(old_events) or {}
    new_best = find_best_dev_event(new_events) or {}
    old_final = find_final_train_eval_event(old_events) or {}
    new_final = find_final_train_eval_event(new_events) or {}

    old_md = find_run_metadata(old_dir)
    new_md = find_run_metadata(new_dir)

    # ── header ────────────────────────────────────────────────────────────
    print("=" * 78)
    print("Run comparison")
    print("=" * 78)
    print(f"  OLD: {old_dir}")
    print(f"       run_name={old_md.get('run_name', '?')}")
    print(f"       dataset={(old_md.get('dataset') or {}).get('version', '?')}/{(old_md.get('dataset') or {}).get('hash', '?')}")
    print(f"       seed={old_md.get('seed', '?')}, determinism={old_md.get('determinism_mode', '?')}")
    print(f"       git={(old_md.get('git') or {}).get('sha', '?')[:8]} dirty={(old_md.get('git') or {}).get('dirty', '?')}")
    print()
    print(f"  NEW: {new_dir}")
    print(f"       run_name={new_md.get('run_name', '?')}")
    print(f"       dataset={(new_md.get('dataset') or {}).get('version', '?')}/{(new_md.get('dataset') or {}).get('hash', '?')}")
    print(f"       seed={new_md.get('seed', '?')}, determinism={new_md.get('determinism_mode', '?')}")
    print(f"       git={(new_md.get('git') or {}).get('sha', '?')[:8]} dirty={(new_md.get('git') or {}).get('dirty', '?')}")
    print()

    # ── headline metrics table ────────────────────────────────────────────
    print("Headline metrics")
    print("-" * 78)
    rows = [
        ("best_dev epoch",        old_best.get("epoch"),                   new_best.get("epoch")),
        ("best_dev combined",     old_best.get("dev/combined"),            new_best.get("dev/combined")),
        ("dev stage1 recall@0.5", old_best.get("dev/stage1_recall@0.5"),   new_best.get("dev/stage1_recall@0.5")),
        ("dev stage1 prec@0.5",   old_best.get("dev/stage1_precision@0.5"),new_best.get("dev/stage1_precision@0.5")),
        ("dev stage2 mIoU",       old_best.get("dev/stage2_mIoU"),         new_best.get("dev/stage2_mIoU")),
        ("dev stage3 acc",        old_best.get("dev/stage3_acc"),          new_best.get("dev/stage3_acc")),
        ("dev stage3 ep acc",     old_best.get("dev/stage3_episode_acc"),  new_best.get("dev/stage3_episode_acc")),
        ("dev pred/img",          old_best.get("dev/pred_items_per_image"),new_best.get("dev/pred_items_per_image")),
        ("dev loss_total",        old_best.get("dev/loss_total"),          new_best.get("dev/loss_total")),
        ("train stage3 ep acc",   old_final.get("train/stage3_episode_acc"),new_final.get("train/stage3_episode_acc")),
        ("train stage3 acc",      old_final.get("train/stage3_acc"),       new_final.get("train/stage3_acc")),
    ]
    name_w = max(len(r[0]) for r in rows)
    print(f"  {'metric':<{name_w}}  {'old':>10}  {'new':>10}  {'delta':>10}")
    for name, old_v, new_v in rows:
        # Convert to float when possible
        ov = float(old_v) if isinstance(old_v, (int, float)) else None
        nv = float(new_v) if isinstance(new_v, (int, float)) else None
        print(f"  {name:<{name_w}}  {fmt_value(old_v):>10}  {fmt_value(new_v):>10}  {fmt_delta(ov, nv):>10}")

    # ── new-only fields (per-stage LRs, NaN counters, leak fallback) ──────
    print()
    print("New-only signals (only present in fixed-code runs)")
    print("-" * 78)
    for k in ("optimizer/stage1_lr", "optimizer/stage2_lr", "optimizer/stage3_lr",
              "train/nan_total", "train/nan_stage1", "train/nan_stage2", "train/nan_stage3",
              "train/nan_stage2_internal", "train/episode_leak_fallback_total",
              "dev/n_nan_batches", "dev/combined_formula_version"):
        v = new_best.get(k) if k.startswith("dev/") else None
        if v is None:
            # try the run_start event
            for e in new_events:
                if k in e:
                    v = e[k]
                    break
        print(f"  {k:<40}  {fmt_value(v):>15}")

    # ── expectations check ────────────────────────────────────────────────
    print()
    print("Phase 3 expectations matrix")
    print("-" * 78)
    n_pass = 0
    n_total = 0
    for exp in EXPECTATIONS:
        source_old = old_best if exp.where == "best_dev" else old_final
        source_new = new_best if exp.where == "best_dev" else new_final
        old_v = source_old.get(exp.metric)
        new_v = source_new.get(exp.metric)
        if old_v is None or new_v is None:
            verdict = "?"
            note = "metric missing"
        else:
            delta = float(new_v) - float(old_v)
            if exp.direction == "down":
                ok = delta <= -exp.threshold
            elif exp.direction == "up":
                ok = delta >= exp.threshold
            elif exp.direction == "stable":
                ok = abs(delta) <= exp.threshold
            elif exp.direction == "near_zero":
                ok = abs(float(new_v)) <= exp.threshold
            else:
                ok = False
            verdict = "PASS" if ok else "FAIL"
            note = f"old={float(old_v):.4f} new={float(new_v):.4f} delta={delta:+.4f} (need {exp.direction} ≥{exp.threshold:.2f})"
            n_total += 1
            if ok:
                n_pass += 1
        print(f"  [{verdict:<4}] {exp.metric}")
        print(f"           {note}")
        print(f"           why: {exp.rationale}")

    print()
    print(f"Expectations: {n_pass}/{n_total} passed")
    print()

    # ── gate ──────────────────────────────────────────────────────────────
    if n_total == 0:
        print("[GATE] inconclusive — no shared metrics found.")
        return 1
    if n_pass == n_total:
        print("[GATE] PASS — proceed to Phase 4.")
        return 0
    if n_pass >= n_total - 1:
        print("[GATE] SOFT — most predictions held; investigate the failed row before ablating.")
        return 1
    print("[GATE] FAIL — multiple predictions broke; the fixes did not produce expected behavior.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two TriFoodNet runs.")
    parser.add_argument("old_dir", help="path to the old run's joint log dir (contains events.jsonl)")
    parser.add_argument("new_dir", help="path to the new run's joint log dir")
    args = parser.parse_args()

    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)
    for d in (old_dir, new_dir):
        if not d.is_dir():
            print(f"[FATAL] not a directory: {d}", file=sys.stderr)
            return 2
        if not (d / "events.jsonl").exists():
            print(f"[FATAL] missing events.jsonl in: {d}", file=sys.stderr)
            return 2

    return run_comparison(old_dir, new_dir)


if __name__ == "__main__":
    sys.exit(main())
