# CONFIG_GUIDE — every field in `master_config.yaml`

This is the single source of truth for what each config knob does, what it defaults to, and which knobs are safe to override per experiment.

---

## How config loading works

1. `python -m train_joint` reads `master_config.yaml` via `config_loader.load_config()`.
2. CLI overrides (`key=value` positional args) are applied on top, in order.
3. `config_validation.validate_config(cfg)` runs — fails fast on bad ratios, missing adapter, etc.
4. After the dataset adapter loads, validation runs again with `dataset_class_count` + `dataset_min_train_class_size` for the dataset-aware checks.

Override syntax:
```bash
python -m train_joint joint.training.epochs=10 stage3.eval.k_shot=10 logging.wandb=false
```

Values are parsed with `yaml.safe_load`, so booleans, scientific notation, and lists all work.

---

## Section 0 — `run`

| field | default | meaning | safe to override? |
|---|---|---|---|
| `name` | `trial-20260321-cleandata1` | run identifier; used for log/checkpoint dir names | **always override** for new experiments |
| `notes` | snapshot string | freeform text saved into run_metadata.json | yes |
| `tags` | `[joint, snapshot, best_run]` | freeform tags | yes |
| `seed` | `1337` | seeds torch/numpy/python/CUDA — captured in run_metadata | yes; use a different seed per fold/replicate |
| `determinism_mode` | `deterministic` | `loose` / `deterministic` / `strict`. Strict slows ~30%; loose is fastest but non-reproducible. | rarely; default is fine |

---

## Section 1 — `paths`

Almost all paths are relative to the cwd when training launches. The vast scripts cd to the training dir before running, so these resolve to `experiments/v3_trifoodnet_research_snapshot/{logs,checkpoints,outputs}/`.

| field | default | meaning |
|---|---|---|
| `checkpoints` | `checkpoints` | where `pipeline.save()` writes |
| `outputs` | `outputs` | reports + visualizations |
| `logs` | `logs` | events.jsonl + tensorboard/ + run_metadata.json |
| `images`, `annotations`, `splits`, `reference_library`, `price_table` | various | mostly legacy; the v3 adapter doesn't use them |

**Don't override** these casually — they're used by helper scripts that assume the layout.

---

## Section 2 — `data`

| field | default | meaning | safe to override? |
|---|---|---|---|
| `image_size` | `640` | longest-edge resize for images fed to Stage 1+2 | rarely |
| `num_classes` | `32` | size of Stage 3 classifier head; must be ≥ max compact_id+1 in dataset | only if dataset changes |
| `num_workers` | `8` | DataLoader workers | drop to 4 on rented boxes with limited CPUs |
| `pin_memory` | `true` | speeds up host→GPU copies | yes if low RAM |

### `data.integration` — dataset wiring

| field | default | meaning |
|---|---|---|
| `adapter.kind` | `v3_export` | which adapter to use; only `v3_export` is supported |
| `adapter.export_root` | local Mac path | where the v3 export lives — **MUST update for each environment** |
| `adapter.expected_version` | `v3` | manifest version pin; mismatch raises |
| `adapter.expected_hash` | `61ac038c` | manifest content_hash_sha8 pin; mismatch raises |
| `adapter.splits_path` | `""` (default = `<export_root>/splits.json`) | where to persist computed splits |
| `train_ratio`, `dev_ratio`, `test_ratio` | `0.8`, `0.1`, `0.1` | split fractions; must sum to 1.0 |
| `val_ratio` | `0.1` | legacy alias for `dev_ratio`; ignored if `dev_ratio` set |
| `split_seed` | `420` | RNG seed for stratified split |
| `stage1_prompt` | (long string) | prompt sent to Qwen for box detection |

---

## Section 3 — `hardware`

| field | default | meaning |
|---|---|---|
| `device` | `auto` | `auto` / `cuda` / `cpu` |
| `fp16` | `true` | use AMP fp16 (off if `bf16=true`) |
| `bf16` | `true` | use AMP bf16 — preferred on H100/A100/5090 |
| `compile` | `true` | `torch.compile` Stage 3 transformer; off for short/debug runs |
| `gradient_checkpointing` | `true` | saves ~40% VRAM, slows ~10% |
| `load_in_4bit` | `false` | 4-bit quantization for Qwen; needed if VRAM tight |

---

## Section 4 — `logging`

| field | default | meaning |
|---|---|---|
| `tensorboard` | `true` | write scalars to `<run_dir>/tensorboard/` — **the canonical tracker** |
| `wandb` | `true` | also write to W&B if `wandb` package + login present |
| `wandb_project` | `trifoodnet` | W&B project name |
| `jsonl` | `true` | always-on; events.jsonl is the source of truth |
| `log_system_metrics` | `true` | CPU%, RAM, GPU mem per logged event |
| `log_gpu_metrics` | `true` | GPU memory + utilization |
| `log_every_n_steps` | `10` | how often to print metrics to stdout |
| `save_every_n_epochs` | `1` | epoch checkpoint cadence (the `best/` checkpoint is independent of this) |
| `keep_last_n_ckpts` | `3` | rotates epoch-numbered checkpoints |
| `visualizations.every_n_epochs` | `3` | dev-image visualizations |
| `profiler.enabled` | `false` | torch profiler |

---

## Section 5 — `stage1` (Qwen 2.5-VL)

| field | default | safe to override? |
|---|---|---|
| `model_name` | `Qwen/Qwen2.5-VL-3B-Instruct` | swap to 7B if budget allows |
| `lora.r`, `lora.alpha` | `16`, `32` | LoRA rank/alpha; standard PEFT knobs |
| `lora.dropout` | `0.05` | dropout on LoRA layers |
| `lora.use_rslora` | `true` | rank-stabilized LoRA |
| `lora.target_modules` | `[q,k,v,o]_proj` | which linear layers get LoRA |
| `training.epochs` | `60` | standalone Stage 1 training (not used in joint flow) |
| `training.learning_rate` | `2e-5` | Stage 1's LR — **fed into joint via per_stage_lr** |
| `eval.iou_threshold` | `0.5` | for `stage1_recall@0.5` / `precision@0.5` |

---

## Section 6 — `stage2` (SAM 3)

| field | default | safe to override? |
|---|---|---|
| `model_name` | `facebook/sam3` | gated:manual on HF; needs Meta approval |
| `freeze.image_encoder` | `true` | rarely unfreeze |
| `freeze.prompt_encoder` | `true` | rarely unfreeze |
| `freeze.mask_decoder` | `false` | the trainable part |
| `loss.bce_weight`, `loss.dice_weight` | `0.5`, `2.0` | mask-loss balance |
| `nms.iou_threshold` | `0.5` | mask-IoU NMS at inference (was `0.0` pre-fix; that disabled NMS) |
| `nms.score_threshold` | `0.0` | SAM scores aren't well calibrated; fallback handles empty case |

---

## Section 7 — `stage3` (PictSure-style ICL)

| field | default | safe to override? |
|---|---|---|
| `clip_model` | `pictsure/pictsure-vit` | the embedding backbone |
| `embed_dim` | `512` | |
| `train_embedding` | `true` | whether to backprop into the ViT |
| `lora.*` | `r=16, alpha=32` | LoRA on the transformer FFN |
| `transformer.{num_layers, num_heads, ff_dim, dropout}` | `4, 8, 1024, 0.1` | ICL transformer architecture |
| `episode.n_way` | `5` | classes per training episode (random sampling) |
| `episode.k_shot` | `5` | reference shots per class per episode |
| `episode.query_per_class` | `1` | queries per class per episode |
| `episode.episodes_per_epoch` | `1000` | for the standalone Stage 3 trainer (unused in joint) |
| `eval.n_way` | `5` | classes per eval episode |
| `eval.k_shot` | `5` | shots per class at eval |
| `reference_library.{min,max}_images_per_class` | `1`, `5` | reference set construction caps |
| `loss.name` | `cross_entropy` | also `balanced_softmax`, `logit_adjusted` |

---

## Section 8 — `joint` (the joint trainer, the main code path)

### `joint.training`

| field | default | safe to override? |
|---|---|---|
| `epochs` | `40` | yes |
| `batch_size` | `1` | rarely change; bumps VRAM hard |
| `grad_accum_steps` | `8` | effective batch = batch_size × grad_accum |
| `max_batches_per_epoch` | `0` (= full pass) | set to e.g. `50` for fast smoke runs |
| `learning_rate` | `5e-6` | **global fallback only** — per-stage LR overrides this |
| `weight_decay` | `1e-4` | |
| `per_stage_lr.{stage1,stage2,stage3}` | `2e-5, 5e-5, 1e-4` | the actual LRs each stage runs at |
| `per_stage_weight_decay.{stage1,stage2,stage3}` | `1e-4 each` | per-stage WD overrides |
| `warmup_ratio` | `0.05` | fraction of total steps for LR warmup |
| `lr_scheduler` | `cosine` | passed to HF `get_scheduler` |
| `max_grad_norm` | `0.3` | global grad clipping |
| `early_stopping.enabled` | `false` | turn on for long runs |
| `early_stopping.monitor` | `dev/loss_total` | what to monitor; `dev/combined` is the headline |
| `early_stopping.{mode, patience, min_delta, min_epochs}` | `min, 8, 1e-4, 28` | standard ES |

### `joint.loss_weights`

| field | default | safe to override? |
|---|---|---|
| `lambda1` | `1.0` | Stage 1 LM loss weight |
| `lambda2` | `0.5` | Stage 2 mask loss weight |
| `lambda3` | `1.5` | Stage 3 ICL loss weight |

These are heuristic. No ablation has been run on them.

### `joint.curriculum.teacher_forcing`

| field | default | safe to override? |
|---|---|---|
| `enabled` | `true` | |
| `sustain_epochs` | `40` | epochs at `start_prob` before decay begins |
| `transition_epochs` | `0` | epochs to ramp from `start_prob` to `end_prob` |
| `start_prob` | `1.0` | initial teacher-forcing probability |
| `end_prob` | `1.0` | final teacher-forcing probability |

**Today's config = teacher forcing at p=1.0 forever.** This means Stage 2/3 always see GT boxes during training. Decaying it is a real ablation (Phase 5 candidate).

### `joint.eval`

| field | default | safe to override? |
|---|---|---|
| `interval` | `1` | how often to run dev eval (every N epochs) |

---

## Section 9 — `inference`

Used by `pipeline.run()` (end-to-end inference path). Most fields mirror Stage 2's NMS settings.

| field | default | meaning |
|---|---|---|
| `nms_iou_threshold` | `0.5` | mask NMS at inference |
| `score_threshold` | `0.5` | SAM score filter |
| `top_k_classes` | `1` | how many classes Stage 3 predicts per item |
| `target_latency_ms` | `2000` | informational |
| `device` | `cuda` | |

---

## Section 10 — legacy fields (ignore)

`pictsure.*`, `official_pictsure.*`, `benchmark.*` — kept for backward compat with snapshot-era scripts. Not consumed by the training path.

---

## Common override recipes

### Quick smoke (5 epochs, 50 batches per epoch, no eval mid-run)
```bash
python -m train_joint \
  joint.training.epochs=5 \
  joint.training.max_batches_per_epoch=50 \
  joint.eval.interval=5 \
  hardware.compile=false \
  logging.wandb=false \
  run.name=smoke-$(date +%Y%m%d-%H%M)
```

### Try Stage 3 LR sweep
```bash
for lr in 5e-5 1e-4 2e-4; do
  python -m train_joint \
    joint.training.per_stage_lr.stage3=$lr \
    run.name=lr-stage3-$lr
done
```

### Disable teacher forcing decay (curriculum experiment)
```bash
python -m train_joint \
  joint.curriculum.teacher_forcing.start_prob=1.0 \
  joint.curriculum.teacher_forcing.end_prob=0.0 \
  joint.curriculum.teacher_forcing.transition_epochs=20 \
  joint.curriculum.teacher_forcing.sustain_epochs=20 \
  run.name=curriculum-decay
```

### Test on a different dataset version
```bash
python -m train_joint \
  data.integration.adapter.export_root=/data/exports/v4_2026-05-01_abc123 \
  data.integration.adapter.expected_version=v4 \
  data.integration.adapter.expected_hash=abc12345 \
  run.name=on-v4-baseline
```

---

## What validation will refuse

`config_validation.py` raises `ConfigValidationError` if:
- `train_ratio + dev_ratio + test_ratio != 1.0` (within 1e-3)
- `data.num_classes < max(compact_id)+1` from the dataset adapter
- `stage3.episode.k_shot > min(class size)` and `allow_replacement` is false
- `joint.training.max_batches_per_epoch < grad_accum_steps` (would never step the optimizer)
- `run.determinism_mode` is not in `{loose, deterministic, strict}`
- Both `data.integration.adapter.kind` empty AND `data.integration.batch_root` empty

These are deliberate fail-fast guards. Don't try to silence them — fix the config.
