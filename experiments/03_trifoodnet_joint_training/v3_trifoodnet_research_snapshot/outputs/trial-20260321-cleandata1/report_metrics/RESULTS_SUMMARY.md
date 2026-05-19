# Trial Report: trial-20260321-cleandata1

## Status

- Log source: `/root/rest_model/logs/trial-20260321-cleandata1/joint`
- Run status file: `completed` with `elapsed_sec=7354.59` and `finished_utc=2026-03-21T23:00:56.888899+00:00`
- Completed dev epochs captured: `1-40`
- Latest event stored in `events.jsonl`: `train_eval_final` at epoch `40` step `480`
- Note: `run_status.json` exists and marks the run complete, but the event log itself still does not contain a separate `run_end` record.

## Final Dev Metrics

- epoch `40` dev/loss_total: `3.20090651512146`
- epoch `40` dev/stage1_precision@0.5: `0.76`
- epoch `40` dev/stage1_recall@0.5: `0.8636363636363636`
- epoch `40` dev/stage2_mIoU: `0.5733921838932934`
- epoch `40` dev/stage3_acc: `0.5`
- epoch `40` dev/stage3_matched_acc: `0.5789473684210527`
- epoch `40` dev/stage3_episode_acc: `0.6363636363636364`
- epoch `40` dev/pred_items_per_image: `2.0833333333333335`
- epoch `40` dev/latency_total_ms: `6144.5216666666665`
- epoch `40` dev/combined: `1.937028547529657`

## Final Train Eval Metrics

- epoch `40` train/loss_total: `2.0714402698701426`
- epoch `40` train/stage1_precision@0.5: `0.8451612903225807`
- epoch `40` train/stage1_recall@0.5: `0.7987804878048781`
- epoch `40` train/stage2_mIoU: `0.4919433713018482`
- epoch `40` train/stage3_acc: `0.35365853658536583`
- epoch `40` train/stage3_matched_acc: `0.44274809160305345`
- epoch `40` train/stage3_episode_acc: `0.9329268292682927`
- epoch `40` train/pred_items_per_image: `1.6666666666666667`
- epoch `40` train/latency_total_ms: `5976.930645161291`

## Best Observed Epochs

- Best `dev/combined`: epoch `38` with `1.9375961198969618`
- Lowest `dev/loss_total`: epoch `26` with `3.1939461330572763`

## Loss Visuals

- Total Loss: [plots/train_vs_dev_loss_total_by_epoch.svg](plots/train_vs_dev_loss_total_by_epoch.svg)
- Stage 1 Loss: [plots/train_vs_dev_loss_stage1_by_epoch.svg](plots/train_vs_dev_loss_stage1_by_epoch.svg)
- Stage 2 Loss: [plots/train_vs_dev_loss_stage2_by_epoch.svg](plots/train_vs_dev_loss_stage2_by_epoch.svg)
- Stage 3 Loss: [plots/train_vs_dev_loss_stage3_by_epoch.svg](plots/train_vs_dev_loss_stage3_by_epoch.svg)

## Existing Auto Report

- Main index: [index.md](index.md)
- Summary JSON: [summary.json](summary.json)
- Summary CSV: [summary.csv](summary.csv)
- Run card: [runs/trial-20260321-cleandata1.md](runs/trial-20260321-cleandata1.md)

## Notes

- The preserved `best` checkpoint directory is governed by the configured loss monitor, not by the best end-to-end dev combined score.
- The strongest late-run end-to-end dev behavior is represented by epoch `38` in the logs, while epoch `40` is the latest checkpoint.
- Dev latency remained around 6 seconds per image in the late plateau.