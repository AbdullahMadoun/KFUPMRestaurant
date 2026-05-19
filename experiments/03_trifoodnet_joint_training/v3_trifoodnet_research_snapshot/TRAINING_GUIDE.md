# Training Guide

## Active Training Loop

`train_joint.py` is the main end-to-end trainer. The script:

1. Loads `master_config.yaml` plus dot-style overrides through `config_loader.py`.
2. Resolves device, AMP dtype, optional 4-bit loading, and runtime flags.
3. Builds `QwenGrounder`, `SAM3Segmenter`, `FoodClassifier`, and wraps them inside `TriFoodNet`.
4. Creates `JointFoodDataset` instances for train, train-eval, and dev splits.
5. Builds a Stage 3 reference library from the training split.
6. Trains with gradient accumulation, AdamW, and a Transformers scheduler.
7. Evaluates after each epoch with both teacher-forced objective metrics and end-to-end inference metrics.
8. Saves the live `joint/best` alias whenever the configured early-stopping monitor improves (`dev/loss_total` in the best run), while separately tracking `joint/combined = dev/stage1_recall@0.5 + dev/stage2_mIoU + dev/stage3_acc` to identify the strongest end-to-end epoch.

## Best Run Hyperparameters

- Run name: `trial-20260321-cleandata1`
- Stage 1 model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Stage 2 model: `facebook/sam3`
- Stage 3 model: `pictsure/pictsure-vit`
- Image size: `640`
- Joint epochs: `40`
- Joint batch size: `1`
- Joint grad accumulation: `8`
- Joint learning rate: `5e-06`
- Joint weight decay: `0.0001`
- Joint warmup ratio: `0.05`
- Joint scheduler: `cosine`
- Joint max grad norm: `0.3`
- Teacher forcing: `{'enabled': True, 'sustain_epochs': 40, 'transition_epochs': 0, 'start_prob': 1.0, 'end_prob': 1.0}`
- Loss weights: `{'lambda1': 1.0, 'lambda2': 0.5, 'lambda3': 1.5}`
- AMP fp16: `True`
- AMP bf16: `True`
- `torch.compile`: `True`
- Gradient checkpointing: `True`

## Loss Functions

- Stage 1: language-model loss from Qwen.
- Stage 2: BCE + Dice with weights `{'bce_weight': 0.5, 'dice_weight': 2.0}`.
- Stage 3: `cross_entropy` with settings `{'name': 'cross_entropy', 'logit_adjust_tau': 0.5}`.
- Joint total: `lambda1 * loss_stage1 + lambda2 * loss_stage2 + lambda3 * loss_stage3`.

## Optimizer And Scheduler

- Optimizer: AdamW over trainable parameters only.
- Scheduler: Transformers scheduler `cosine`.
- Warmup strategy: ratio-based warmup with `0.05` of total optimizer steps.

## Resume Training From The Snapshot Checkpoint

The original project saves directory checkpoints rather than one monolithic weight file. In this snapshot the winning `joint/epoch_038` directory is packed into `best_checkpoint.tar` so the snapshot still carries a single best-checkpoint artifact.

Important selection note: the packaged `epoch_038` checkpoint is the strongest end-to-end dev checkpoint by `joint/combined`. It is not the same thing as the live `joint/best` alias, which followed `dev/loss_total`.

```bash
# Restore the winning joint checkpoint payload
mkdir -p ./checkpoints/trial-20260321-cleandata1/joint/epoch_038
tar -xf ./weights/best_checkpoint.tar -C ./checkpoints/trial-20260321-cleandata1/joint/epoch_038 --strip-components=1
```

Important limitation: `train_joint.py` does not implement true joint-training resume from `joint/epoch_*` or `joint/best`. It only warm-starts from per-stage checkpoints (`stage1/best`, `stage2/best.pt`, `stage3/best_icl.pt`), and the saved joint checkpoints are weight-only bundles with no optimizer, scheduler, or scaler state. Treat the packaged checkpoint as the correct continuation point for evaluation, inference, and manual research continuation rather than a drop-in optimizer-state resume.
