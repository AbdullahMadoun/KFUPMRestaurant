# trial-20260321-stability5

- root_dir: `logs/trial-20260321-stability5/joint`
- event_count: 57

## Best Metrics

| Metric | Value |
| --- | --- |
| Best Joint Combined | 1.4450 |
| Best Dev Stage 1 Recall@0.5 | 0.772727 |
| Best Dev Stage 2 mIoU | 0.491058 |
| Best Dev Stage 3 Accuracy | 0.181818 |
| Min Dev Total Loss | 4.0599 |

## Config Highlights

| Field | Value |
| --- | --- |
| run | trial-20260321-stability5 |
| status | completed |
| device | cuda |
| elapsed_sec | 1,012.9 |
| notes | 3-epoch stability rerun after Stage3 loss alignment fix |
| batch_root | /root/dataset |
| stage3_loss | cross_entropy |
| joint_lr | 0.000005 |
| joint_batch_size | 1.0000 |
| joint_grad_accum | 8.0000 |
| compile | true |
| gradient_checkpointing | true |
