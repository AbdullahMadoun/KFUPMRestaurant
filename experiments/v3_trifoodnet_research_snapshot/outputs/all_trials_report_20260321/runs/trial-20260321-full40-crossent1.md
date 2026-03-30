# trial-20260321-full40-crossent1

- root_dir: `/root/rest_model/logs/trial-20260321-full40-crossent1/joint`
- event_count: 61

## Best Metrics

| Metric | Value |
| --- | --- |
| Best Joint Combined | 1.3521 |
| Best Dev Stage 1 Recall@0.5 | 0.772727 |
| Best Dev Stage 2 mIoU | 0.488981 |
| Best Dev Stage 3 Accuracy | 0.090909 |
| Min Dev Total Loss | 4.0009 |

## Config Highlights

| Field | Value |
| --- | --- |
| run | trial-20260321-full40-crossent1 |
| status | completed |
| device | cuda |
| elapsed_sec | 1,849.7 |
| notes | 40-epoch full run after SAM3 query decoding fix and Stage3 CE alignment |
| batch_root | /root/dataset |
| stage3_loss | cross_entropy |
| joint_lr | 0.000005 |
| joint_batch_size | 1.0000 |
| joint_grad_accum | 8.0000 |
| compile | true |
| gradient_checkpointing | true |
