# trial-20260321-stability4

- root_dir: `/root/rest_model/logs/trial-20260321-stability4/joint`
- event_count: 54

## Best Metrics

| Metric | Value |
| --- | --- |
| Best Joint Combined | 1.4443 |
| Best Dev Stage 1 Recall@0.5 | 0.772727 |
| Best Dev Stage 2 mIoU | 0.490860 |
| Best Dev Stage 3 Accuracy | 0.181818 |
| Min Dev Total Loss | 11.0628 |

## Config Highlights

| Field | Value |
| --- | --- |
| run | trial-20260321-stability4 |
| status | unknown |
| device | cuda |
| elapsed_sec | - |
| notes | 3-epoch stability rerun after Stage2 SAM3 query decoding fix |
| batch_root | /root/dataset |
| stage3_loss | balanced_softmax |
| joint_lr | 0.000005 |
| joint_batch_size | 1.0000 |
| joint_grad_accum | 8.0000 |
| compile | true |
| gradient_checkpointing | true |
