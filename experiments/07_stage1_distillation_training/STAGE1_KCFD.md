# Stage 1 Qwen-VL Training

This implementation trains only Stage 1 of the cafeteria tray pipeline. It fine-tunes `Qwen/Qwen2.5-VL-7B-Instruct` to emit strict JSON food/beverage regions for one cropped dish image. It does not train or simulate Stage 0, SAM, or the Stage 3 vector DB.

## Target Schema

The assistant target is exactly:

```json
{"items":[{"name":"short name","bbox":[x1,y1,x2,y2],"descriptor":"short visual descriptor"}]}
```

`name` comes from `items.jsonl["name"]`. `descriptor` comes from `description`, falling back to `vlm_description`. Boxes are the v3 export boxes in mask-native `xyxy` coordinates. Training images are resized to the corresponding mask resolution so the model target coordinates and pixels share the same frame. Full-resolution visual previews scale the boxes before drawing.

The prompt tells Qwen to ignore dish, plate, tray, table, utensils, and empty container areas. It does not mention metrics.

## Local Responsibilities

Local machine:

- Edit code, run unit tests, and inspect generated preview PNGs.
- Keep secrets out of the repo. Put `HF_TOKEN` in a local `.env` or export it in the shell; do not commit it.
- Use SSH port forwarding for TensorBoard at `http://127.0.0.1:6006`.
- Pull run artifacts after training.

Remote Vast instance:

- Standard Vast container, not a VM.
- Install `requirements-stage1.txt`; it pins `numpy<2.3` for the current SciPy/Transformers stack.
- Store the v3 dataset under `/root/work/dataset_v3_export`.
- Run preflight, render training previews, run TensorBoard, and train in tmux.
- Write runs under `/root/work/stage1_runs`.

Shared branch scripts such as `00_state.sh` may still contain legacy comments from the larger project, but the Stage 1 workflow uses only the `stage1_*` scripts plus dataset download, tmux attach, and destroy helpers. The remote command launched for training is `python train.py`; no Stage 3 process is started.

## Preflight

Always start with preflight:

```bash
python train.py \
  --export-root /path/to/v3_export \
  --output-dir outputs/stage1_kcfd \
  --preflight-only
```

If the export contains reference items, default training exits before model load. Choose the policy explicitly:

- `--reference-policy exclude`: safest no-leakage policy; excludes every source image containing a reference item. This is the default for remote previews and training.
- `--reference-policy train`: keeps reference images in train only.
- `--reference-policy include`: diagnostic only; keeps references eligible for any split.

Full training also exits before dataset/model construction if preflight reports missing images, masks, names, or descriptors. Fix the export first, or use `--allow-incomplete-export` only for a deliberate diagnostic run where skipped targets are acceptable.

Use `--expected-hash <content_hash_sha8>` to pin the dataset version.

## Training

Single command:

```bash
python train.py \
  --export-root /root/work/dataset_v3_export \
  --output-dir /root/work/stage1_runs \
  --run-name stage1-qwen7b-v1 \
  --reference-policy exclude \
  --seed 1337 \
  --split-seed 420 \
  --expected-hash 61ac038c
```

Defaults match the requested setup: Qwen2.5-VL-7B-Instruct, LoRA `r=32`, alpha `32`, dropout `0.1`, targets `q/k/v/o/gate/up/down`, unfrozen vision encoder, AdamW, LR `1e-4`, cosine schedule, warmup `100`, batch size `1`, grad accumulation `16`, bf16 when CUDA is available, clip `1.0`, and 10 epochs.

`--seed` controls training RNGs. `--split-seed` controls the image-level split cache; when omitted it defaults to `--seed`, so set it explicitly when you want to vary training RNGs without changing train/dev/test membership.

Training backpropagates teacher-forced CE over the strict JSON target. Parsed generated GIoU is logged as an eval-only loss component because generated text boxes are discrete and not differentiable to Qwen tokens. Checkpoint selection uses exact count first, then `exact_set_match@0.5`, then `matched_f1@0.5`.

## Metrics

The evaluator is class-agnostic. It does not score food labels.

Core metrics include `valid_json_rate`, `json_schema_accuracy`, `exact_count_accuracy`, `count_mae`, `count_rmse`, `count_bias`, `overcount_rate`, `undercount_rate`, `zero_pred_rate`, matched precision/recall/F1 at IoU `0.50`, `0.75`, and `0.90`, `mean_matched_iou`, `mean_matched_giou`, `exact_set_match@0.5`, AR@1/3/5/10, AR@[.50:.95], count buckets, size buckets, and descriptor/schema diagnostics.

## Visualization

Render five training examples before long training:

```bash
python -m stage1_kcfd.visualize \
  --export-root /root/work/dataset_v3_export \
  --output-dir /root/work/stage1_runs/previews \
  --split train \
  --max-samples 12 \
  --reference-policy exclude \
  --seed 20260426 \
  --selection class-diverse
```

Each sample writes a mask-native preview and a full-resolution preview. The preview manifest records selected image ids and distinct class slugs so the sample can be audited for diversity. Inspect these locally before starting the full run.

Every 5 training epochs, validation predictions are drawn against GT boxes and logged to TensorBoard.

## Checkpoints

Every epoch writes:

```text
checkpoints/epoch_XXX/
  model/                  # PEFT LoRA adapter
  processor/
  vision_encoder.pt       # unfrozen Qwen vision encoder state
  checkpoint_manifest.json
  metrics.json
```

Use `stage1_kcfd.model.load_stage1_checkpoint()` so both the LoRA adapter and `vision_encoder.pt` are loaded. Loading only the PEFT adapter silently drops the unfrozen vision updates.
