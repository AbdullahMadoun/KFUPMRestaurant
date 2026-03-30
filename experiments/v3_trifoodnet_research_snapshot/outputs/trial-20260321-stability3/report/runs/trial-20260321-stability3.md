# trial-20260321-stability3

- root_dir: `logs/trial-20260321-stability3/joint`
- event_count: 55

## Best Metrics

| Metric | Value |
| --- | --- |
| Best Joint Combined | 0.818182 |
| Best Dev Stage 1 Recall@0.5 | 0.772727 |
| Best Dev Stage 2 mIoU | 0.000000 |
| Best Dev Stage 3 Accuracy | 0.045455 |
| Min Dev Total Loss | 12.7328 |

## Config Highlights

| Field | Value |
| --- | --- |
| run | trial-20260321-stability3 |
| status | completed |
| device | cuda |
| elapsed_sec | 962.94 |
| notes | 3-epoch stability rerun after Stage3 full-checkpoint fix and no-NMS defaults |
| batch_root | /root/dataset |
| stage3_loss | balanced_softmax |
| joint_lr | 0.000005 |
| joint_batch_size | 1.0000 |
| joint_grad_accum | 8.0000 |
| compile | true |
| gradient_checkpointing | true |
