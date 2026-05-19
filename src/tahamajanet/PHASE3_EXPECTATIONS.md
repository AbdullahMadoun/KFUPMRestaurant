# Phase 3 — Baseline Rerun Expectations

**Pre-committed:** 2026-04-25, before the rerun.
**Purpose:** lock the goalposts so we cannot post-rationalize after the run.

## What we are running

Same `master_config.yaml` as `trial-20260321-cleandata1`, with:

- All four Tier 0 audit-bug fixes landed (query-in-support leak, best-checkpoint mismatch, silent NaN, NMS-off).
- Phase 1 dataset bridge active — pipeline reads from `v3_2026-04-24_61ac038c` instead of the legacy `_review/dataset/`.
- Phase 2 trustworthy base active — deterministic seed, single eval harness, per-stage learning rates, config validation, full provenance metadata.
- 40 epochs joint training on RTX 5090, ~7 hours.

## Reference point (the registry-frozen baseline)

| metric | value | source |
|---|---|---|
| best epoch | 38 | `logs/trial-20260321-cleandata1/joint/run_summary.md` |
| `dev/combined` | 1.9376 | same |
| `dev/stage1_recall@0.5` | 0.8636 | same |
| `dev/stage1_precision@0.5` | 0.7600 | same |
| `dev/stage2_mIoU` | 0.5740 | same |
| `dev/stage3_acc` | 0.5000 | same |
| `dev/stage3_episode_acc` | 0.6364 | same |
| `dev/pred_items_per_image` | 2.0833 | same |
| `dev/loss_total` | 3.2000 | same |
| `train/stage3_episode_acc` | 0.9329 | same |
| `train/stage3_acc` (e2e) | 0.3537 | same |

This row is in `experiments/registry.jsonl` — `compare_runs.py` reads from there.

## Predicted deltas — must hold for Phase 3 to PASS

| metric | direction | threshold | rationale |
|---|---|---|---|
| `train/stage3_episode_acc` | ↓ | ≥ 0.20 | **The leak fix.** Query no longer appears in its own support set. Train episode accuracy should drop substantially toward dev's level (~0.65–0.70). |
| `dev/stage3_episode_acc` | stable | ±0.05 | Dev episodes already pulled support from train — leak fix shouldn't move dev. |
| `dev/pred_items_per_image` | ↓ | ≥ 0.40 | NMS active (iou_threshold 0.0 → 0.5). Overlapping mask predictions get suppressed. |
| `dev/stage1_precision@0.5` | ↑ | ≥ 0.03 | With redundant detections gone, precision rises. |
| `dev/stage1_recall@0.5` | stable | ±0.05 | NMS only removes near-duplicates; recall should be unchanged. |

## Predicted but not gated on (still informative)

- **`dev/combined`**: any direction. Could be slightly better (per-stage LRs let Stage 3 converge further) or slightly worse (retraining noise). Either is acceptable — we are buying *trustworthiness*, not a higher number.
- **`dev/stage3_acc`**: likely small improvement if Stage 3 trains further at LR 1e-4 vs. 5e-6.
- **`dev/loss_total`**: directional uncertain.
- **Run wall-clock**: ~10-20% slower because deterministic mode disables cuDNN benchmark.

## Health signals — must be at zero

| metric | gate value |
|---|---|
| `train/nan_total` | exactly 0 |
| `train/nan_stage1` | exactly 0 |
| `train/nan_stage2` | exactly 0 |
| `train/nan_stage3` | exactly 0 |
| `train/nan_stage2_internal` | < 5% of train batches |
| `train/episode_leak_fallback_total` | 0 or small (only fires for the 6 tail classes) |
| dev eval did not raise `RuntimeError: Non-finite ... in strict mode` | true |

## Provenance signals — must be present

| metric | gate value |
|---|---|
| `optimizer/stage1_lr` in run_start log | 2e-5 |
| `optimizer/stage2_lr` in run_start log | 5e-5 |
| `optimizer/stage3_lr` in run_start log | 1e-4 |
| `dataset_version` in `run_metadata.json` | `"v3"` |
| `dataset_hash` in `run_metadata.json` | `"61ac038c"` |
| `seed` in `run_metadata.json` | 1337 |
| `determinism_mode` in `run_metadata.json` | `"deterministic"` |
| `git_sha` in `run_metadata.json` | non-null |
| `requirements_resolved.txt` | exists in log dir |

## Best-checkpoint coherence

| check | gate value |
|---|---|
| `best/` directory exists | true |
| `best_by_monitor/` directory exists | true |
| Loading `best/` reproduces `dev/combined` reported in `events.jsonl` | identical to 4 decimals |
| If monitor differs from combined, `best/` ≠ `best_by_monitor/` | true |

## Gate decision

Counting only the five **must-hold predicted deltas**:

- **PASS (5/5)**: proceed to Phase 4.
- **SOFT (4/5)**: investigate the failed row before ablating. Most common failure mode is the precision delta being smaller than 0.03 — acceptable to weaken to 0.02, document the call.
- **HARD FAIL (≤3/5)**: a fix did not produce the predicted effect. Stop, root-cause it, do not ablate on a base whose behavior we do not understand.

The first signal in a hard fail is almost always `train/stage3_episode_acc`. If that does not drop, the leak was not the dominant cause of the inflated train-vs-dev gap and we have a different bug to find.

## What we deliberately do NOT predict

- **The "right" `dev/combined`**: it is what it is. The point of Phase 3 is to make the number trustworthy, not to beat the previous one.
- **K-shot sensitivity**: separate ablation in Phase 5.
- **Whether Stage 3 should be a closed-set head**: separate ablation.
- **Whether the dataset is large enough**: separate question.

If Phase 3 produces a worse `dev/combined` than 1.9376 *and* all five must-hold deltas pass, we still proceed. The fixes worked; we just have a more honest number.

## After the run

1. `python scripts/compare_runs.py logs/trial-20260321-cleandata1/joint logs/<new-run>/joint`
2. `python scripts/registry_append.py logs/<new-run>/joint`
3. Write the result against this doc — green/yellow/red on each row.
4. If green → start Phase 4 (stage registry).
5. If yellow → fix the soft-fail row and re-evaluate.
6. If red → debug session.

## Authors and history

- Pre-committed: aj.salkini@aajil.ai, 2026-04-25
- This document is read-only. After the run, results live in `outputs/PHASE3_BASELINE.md` and the registry — not here.
