# trial-20260321-converge7

- root_dir: `/root/rest_model/logs/trial-20260321-converge7/joint`
- event_count: 62

## Best Metrics

| Metric | Value |
| --- | --- |
| Best Joint Combined | 0.818182 |
| Best Dev Stage 1 Recall@0.5 | 0.772727 |
| Best Dev Stage 2 mIoU | 0.000000 |
| Best Dev Stage 3 Accuracy | 0.045455 |
| Min Dev Total Loss | 13.3807 |

## Config Highlights

| Field | Value |
| --- | --- |
| run | trial-20260321-converge7 |
| status | unknown |
| device | cuda |
| elapsed_sec | - |
| notes | Full convergence run after split coverage, PEFT-safe Stage3 compile handling, gradient remainder fix, and scheduler step-count fix |
| batch_root | /root/dataset |
| stage3_loss | balanced_softmax |
| joint_lr | 0.000005 |
| joint_batch_size | 1.0000 |
| joint_grad_accum | 8.0000 |
| compile | true |
| gradient_checkpointing | true |
