# Experiment Reporting Guide

This repository now treats experiment reporting as a first-class output, not as
an afterthought.

The training side writes structured logs under:

- `logs/<run_name>/joint/events.jsonl`
- `logs/<run_name>/joint/latest_metrics.json`
- `logs/<run_name>/joint/best_metrics.json`
- `logs/<run_name>/joint/run_metadata.json`
- `logs/<run_name>/joint/run_status.json`
- `logs/<run_name>/joint/config_snapshot.json`
- `logs/<run_name>/joint/run_summary.md`

The report side turns those files into:

- markdown summaries
- CSV and JSON summaries for later analysis
- SVG graphs for quick visual review
- run cards for per-run context

## Quick Start

Generate a report for every run under `logs/`:

```bash
python experiment_report.py --logs-root logs --output reports/all_runs
```

Generate a report for a specific subset:

```bash
python experiment_report.py ^
  --logs-root logs ^
  --run-name v3-run1 ^
  --run-name v3-run2 ^
  --baseline v3-run1 ^
  --output reports/v3_compare
```

You can also point directly at run folders:

```bash
python experiment_report.py ^
  --run-dir logs/v3-run1/joint ^
  --run-dir logs/v3-run2/joint ^
  --baseline v3-run1 ^
  --output reports/v3_compare
```

If the package is installed, the same workflow is available as:

```bash
trifoodnet-report --logs-root logs --output reports/all_runs
```

## Generated Files

Each generated report directory contains:

- `index.md`
- `summary.csv`
- `summary.json`
- `plots/*.svg`
- `runs/*.md`

`index.md` is the main document. Open it first.

## What The Report Shows

### Core metrics table

This is the scoreboard.

It reports:

- best joint combined score
- best Stage 1 validation recall
- best Stage 2 validation mIoU
- best Stage 3 validation accuracy
- minimum validation total loss

Use this table to decide whether a run is worth keeping.

### Efficiency table

This is the cost side of the experiment.

It reports:

- device
- average training throughput
- peak GPU memory
- Stage 3 loss type
- joint learning rate
- effective batch size

Use this table when a run is better but much more expensive.

### Improvement table

This table compares every run to a chosen baseline.

Positive numbers mean improvement. Negative numbers mean regression.

Use this when you are running ablations and want one clean answer to:
"did the change help?"

### Trend charts

These are line plots over training progress.

They show:

- train total and per-stage losses
- validation total loss
- Stage 1 validation recall
- Stage 2 validation mIoU
- Stage 3 validation accuracy
- learning rate
- throughput
- peak GPU memory

Use the plots to catch:

- unstable training
- late overfitting
- metric saturation
- data pipeline bottlenecks
- memory regressions

### Run cards

Each run also gets a small per-run markdown card with:

- run status
- device
- notes
- config highlights
- best metrics

These are meant to make old runs understandable without reopening code or
config files.

## Recommended Workflow

Use a simple loop:

1. Change one thing.
2. Update `run.name` and `run.notes`.
3. Train or benchmark.
4. Generate a report.
5. Compare against a fixed baseline.
6. Keep the change only if the report shows a real gain.

## Suggested Naming

Keep run names short and hypothesis-driven.

Good examples:

- `bsm-baseline`
- `bsm-nway12`
- `bsm-lr5e5`
- `logit-adjusted-tau1`
- `compile-stage3-on`

Avoid names like:

- `run3`
- `newtest`
- `latest-final-really-final`

## Minimal Research Hygiene

For publishable comparisons, each run should have:

- a stable split
- a clear baseline
- one hypothesis
- a run note describing the change
- a generated report saved with the run outputs

## Interpreting Changes

Prefer the following order when reading reports:

1. best validation metrics
2. deltas against baseline
3. loss curves
4. throughput and memory

If a change improves only training loss but not held-out metrics, treat it as
non-evidence.

If a change improves metrics slightly but causes large throughput or memory
regressions, keep it only if the gain matters.

## Practical Notes

- `run_status.json` exists so failed runs do not get mistaken for good runs.
- SVG plots are used so reports remain lightweight and portable.
- `summary.csv` is intended for spreadsheet review or paper-table assembly.
- `summary.json` is intended for scripted post-processing.
