# trial-20260321-full40-puretf1

- root_dir: `/root/rest_model/logs/trial-20260321-full40-puretf1/joint`
- event_count: 137

## Best Metrics

| Metric | Value |
| --- | --- |
| Best Joint Combined | 1.3102 |
| Best Dev Stage 1 Recall@0.5 | 0.772727 |
| Best Dev Stage 2 mIoU | 0.491988 |
| Best Dev Stage 3 Accuracy | 0.181818 |
| Min Dev Total Loss | 3.7453 |

## Config Highlights

| Field | Value |
| --- | --- |
| run | trial-20260321-full40-puretf1 |
| status | unknown |
| device | cuda |
| elapsed_sec | - |
| notes | 40-epoch full retrain with pure teacher forcing in training and full inference on dev |
| batch_root | /root/dataset |
| stage3_loss | cross_entropy |
| joint_lr | 0.000005 |
| joint_batch_size | 1.0000 |
| joint_grad_accum | 8.0000 |
| compile | true |
| gradient_checkpointing | true |
