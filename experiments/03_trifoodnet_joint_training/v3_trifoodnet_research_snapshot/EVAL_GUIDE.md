# Evaluation Guide

## Metrics Defined In Code

- `joint/combined`: checkpoint-selection score used by the trainer and report generator.
- `dev/stage1_recall@0.5`: Stage 1 grounding recall at IoU 0.5.
- `dev/stage2_mIoU`: mean mask IoU for end-to-end segmentation.
- `dev/stage3_acc`: end-to-end item classification accuracy.
- `dev/stage3_matched_acc`: classification accuracy restricted to matched predicted boxes.
- `dev/stage3_episode_acc`: teacher-forced episodic Stage 3 accuracy.
- `dev/loss_total`: teacher-forced dev loss used for overfitting tracking.
- `dev/latency_total_ms`: per-image end-to-end latency.

Metric implementations live in `metrics.py`; the experiment summarization logic lives in `experiment_report.py`.

## Run Evaluation From The Snapshot

Restore the packaged best checkpoint first:

```bash
./restore_best_checkpoint.sh
```

This script restores the winning `epoch_038` payload for provenance and mirrors it into `checkpoints/trial-20260321-cleandata1/joint/best/`, which is the path expected by `validate_pipeline_contracts.py` and the inference helpers inside the snapshot.

Then run the rigorous validation/evaluation entry point:

```bash
python validate_pipeline_contracts.py   --config ./master_config.yaml   --run-name trial-20260321-cleandata1   --split dev   --max-images 5   --output ./validation_report.json
```

You can also regenerate experiment summaries from copied logs:

```bash
python experiment_report.py --logs-root ./logs --output ./reports/all_runs
```

## Expected Output

- `validate_pipeline_contracts.py` writes a JSON report summarizing dataset, split, episode, and inference-contract checks.
- `experiment_report.py` writes `index.md`, `summary.csv`, `summary.json`, and SVG plots.

## Best Experiment Metrics

| Run ID | Best `joint/combined` | Min `dev/loss_total` | Status |
| --- | ---: | ---: | --- |
| trial-20260321-cleandata1 ⭐ | 1.9375961198969618 | 3.1939461330572763 | completed |
| trial-20260321-converge4 | n/a | n/a | failed |
| trial-20260321-converge5 | n/a | n/a | unknown |
| trial-20260321-converge6 | n/a | n/a | unknown |
| trial-20260321-converge7 | 0.8181818181818181 | 13.38074533144633 | unknown |
| trial-20260321-full3 | 1.9064984766601074 | 4.562163972854615 | completed |
| trial-20260321-full40-crossent1 | 1.3521015969531631 | 4.000943183898926 | completed |
| trial-20260321-full40-puretf1 | 1.3101694644245667 | 3.7452661395072937 | unknown |
| trial-20260321-full40-tf08-1 | 1.3970592483157867 | 4.115770796934764 | completed |
| trial-20260321-full40-tfmajority1 | n/a | n/a | completed |
| trial-20260321-stability1 | n/a | n/a | unknown |
| trial-20260321-stability2 | 0.7727272727272727 | 13.037922859191895 | unknown |
| trial-20260321-stability3 | 0.8181818181818181 | 12.732777198155722 | completed |
| trial-20260321-stability4 | 1.4442723853831299 | 11.06284475326538 | unknown |
| trial-20260321-stability5 | 1.4450006238541302 | 4.059867084026337 | completed |
