# trial-20260321-cleandata1

- root_dir: `/root/rest_model/logs/trial-20260321-cleandata1/joint`
- event_count: 716

## Best Metrics

| Metric | Value |
| --- | --- |
| Best Joint Combined | 1.9376 |
| Best Dev Stage 1 Recall@0.5 | 0.863636 |
| Best Dev Stage 2 mIoU | 0.588312 |
| Best Dev Stage 3 Accuracy | 0.500000 |
| Min Dev Total Loss | 3.1939 |

## Config Highlights

| Field | Value |
| --- | --- |
| run | trial-20260321-cleandata1 |
| status | completed |
| device | cuda |
| elapsed_sec | 7,354.6 |
| notes | 40-epoch retrain after raw-image recovery and corrupted sample removal |
| batch_root | /root/dataset |
| stage3_loss | cross_entropy |
| joint_lr | 0.000005 |
| joint_batch_size | 1.0000 |
| joint_grad_accum | 8.0000 |
| compile | true |
| gradient_checkpointing | true |
