# Stage 1 Remote Run Notes

This note tracks the active Stage 1-only Qwen2.5-VL run. It intentionally does
not cover Stage 0, Stage 2, or Stage 3 training.

## Active Remote

- Vast instance: `35646445`
- SSH: `root@ssh3.vast.ai -p 16444`
- GPU: NVIDIA RTX PRO 5000 Blackwell, about 48 GB VRAM
- Remote code: `/root/work/code`
- Remote dataset: `/root/work/dataset_v3_export`
- Remote run dir: `/root/work/stage1_runs/stage1-qwen7b-20260426-2009`
- Local pulled artifacts: `outputs/stage1-qwen7b-20260426-2009/`
- TensorBoard: `http://127.0.0.1:6006`

## Training Contract

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Adaptation: LoRA plus unfrozen vision encoder
- Reference policy: `exclude`
- Training seed: `1337`
- Split seed: `420`
- Split method: `stage1_image_level_stratified_class_item_balance_v2`
- Output schema: strict top-level `{"items":[...]}` JSON with `name`, `bbox`, and `descriptor`

## BBox Audit

The v3 export stores boxes in mask-native coordinates, while `images/*.jpg`
are original-resolution images. Stage 1 training therefore loads each image at
the corresponding mask resolution and trains on raw `items.jsonl["bbox"]`.

Remote bbox audit, reference images excluded:

- Audited rows: `4526`
- Image/mask size mismatches: `3276`
- Major out-of-frame boxes: `0`
- Minor boundary overruns: `17` in the trainable subset, clipped to image frame
- Raw bbox vs mask-tight median error: `4.0 px`
- Scaled bbox vs full-res mask-tight median error: `5.28125 px`
- Wrong raw-on-full-res median error: `214.7265625 px`

Audit panels are in:

- Remote: `/root/work/stage1_runs/stage1-qwen7b-20260426-2009/bbox_audit/`
- Local: `outputs/stage1-qwen7b-20260426-2009/bbox_audit/`

Training previews are in:

- Remote: `/root/work/stage1_runs/stage1-qwen7b-20260426-2009/previews/`
- Local: `outputs/stage1-qwen7b-20260426-2009/previews/`

## Start Command

```bash
bash scripts/vast/stage1_05_train.sh
```

The startup batch probe selected:

- `per_device_batch_size=4`
- `gradient_accumulation_steps=4`
- Effective batch size: `16`
- Probe peak allocated VRAM: about `25.5 GiB`
- Probe reserved VRAM: about `28.3 GiB`

## Monitoring

```bash
TMUX_SESSION=stage1-train bash scripts/vast/99_attach_tmux.sh
bash scripts/vast/stage1_04_tensorboard.sh
```

TensorBoard should show at minimum:

- `train/loss`
- `train/ce_loss`
- `train/lr`
- validation metrics after each epoch

Best checkpoint is selected by validation `exact_count_accuracy`, then
`exact_set_match@0.5`, then `matched_f1@0.5`.
