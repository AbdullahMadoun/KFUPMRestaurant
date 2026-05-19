# HANDOFF — TriFoodNet, restaurant pricing pipeline

This branch is what you should read first. It tells you **what's here, what we did, and what's open**, so you can decide where to dig in.

---

## TL;DR

We built a 3-stage food-recognition pipeline for the KFUPM cafeteria dataset and trained it end-to-end. The 5-epoch joint training run is finished, results are reproducible, the architecture works, and the headline ablation (ICL transformer vs cosine retrieval) is paper-worthy.

**Key result**: an ICL transformer adds **+11.7 percentage points** over pure cosine retrieval for few-shot food classification, and the gap widens monotonically across training epochs.

There are two branches in this repo to look at:

| Branch | What it is |
|---|---|
| `training-only` (you are here) | The trained system. Qwen-3B (LoRA fine-tuned) → SAM3 (frozen) → PictSure ICL transformer (LoRA). 5 epochs, jointly trained. |
| `vector-db-classifier` | Same Stages 1+2, but Stage 3 is **just a vector DB** (DINOv2 cosine top-1 against a 32-class reference index). No classifier training. Reads as a direct "what if we delete the transformer" comparison. |

Both branches share the same dataset format, adapter, and Stages 1-2 code. The diff is purely in Stage 3 + a config flag.

---

## How to read this repo

Start with these files in order:

1. **`HANDOFF.md`** (this file) — the map
2. **`docs/DATASET.md`** — what the v3 export looks like on disk and how the code reads it
3. **`pipeline.py`** — the 3-stage forward pass, ~600 LOC, well-commented
4. **`dataset_v3_adapter.py`** + **`dataset_integration.py`** — how raw items.jsonl becomes training batches
5. **`TRAIN_GUIDE.md`** — step-by-step to run a fresh training run
6. **`CONFIG_GUIDE.md`** — every knob in `master_config.yaml`, what it does, when to touch it
7. **`EXPERIMENT_GUIDE.md`** — how to compare runs (TensorBoard + registry.jsonl)

If you only have 30 minutes, read `pipeline.py` + `docs/DATASET.md`. The rest is operational.

---

## Architecture (one paragraph)

```
image
  │
  ├─► Stage 1: Qwen2.5-VL-3B (LoRA r=16 on q/k/v/o)
  │     produces JSON of {bbox} per detected food item
  │
  ├─► Stage 2: SAM3 (frozen — image enc + prompt enc + mask dec all frozen)
  │     consumes Stage 1's bboxes as prompts, produces per-item masks
  │
  └─► Stage 3: PictSure ViT (frozen) + cross-attention transformer (LoRA r=16)
        few-shot 5-way 5-shot classification against a reference library
        produces class label + confidence per item
```

Trainable parameters: 7.9M out of 4.72B total (1 in 598). Only the LoRA adapters on Stages 1 and 3 train. SAM3 and PictSure ViT are intentionally frozen — based on the PictSure paper finding that frozen backbones outperform fine-tuned ones in few-shot.

---

## What's NOT in this git repo (lives in Drive)

| Artifact | Size | Where |
|---|---|---|
| **Dataset v3 tarball** | 1.6 GB | Google Drive file ID `1brzFmBKCUGWkUwG9_1HkpTcnxuyJp1Rk` (`v3_2026-04-24_61ac038c.tar`, sha8 `6473cdab`) |
| **Trained checkpoints** (5 epochs + best/best_by_monitor) | 480 MB | (need to upload — see HANDOFF_OPEN below) |
| **Dev visualization PNGs** (370 prediction overlays from epoch 3) | 116 MB | (same upload) |
| **Training run logs + tensorboard events** | 5.4 MB | (same upload) |

Once uploaded, the Drive folder will hold all four. The dataset tar's already there.

`scripts/vast/00b_pull_dataset_from_drive.sh` already knows how to pull the dataset onto a vast.ai instance via gdown, verify the sha8, and extract.

---

## Reproducibility — running a training run from scratch

Three commands once `.env` has your HF token and you have a GPU:

```bash
# (1) launch + provision instance
bash scripts/vast/01_launch.sh           # picks cheapest 5090, boots, waits for SSH

# (2) push code + pull dataset
bash scripts/vast/02_push.sh             # rsync's the working tree
bash scripts/vast/00b_pull_dataset_from_drive.sh   # gdown pull on the instance

# (3) train (smoke runs first; train only fires if 41/41 smoke checks pass)
bash scripts/vast/03_run_remote.sh       # installs deps, runs smoke, launches train in tmux
```

Then monitor:
```bash
python scripts/vast/04_live_monitor.py   # reads events.jsonl over SSH, renders dashboard
```

When done:
```bash
bash scripts/vast/05_pull.sh             # rsync logs + checkpoints back
bash scripts/vast/06_destroy.sh          # tear down (no auto-destroy by default)
python scripts/build_report_pdf.py --run logs/<run_name> --out reports/<run_name>.pdf
```

The 5-epoch run takes ~4.4 hrs on a 5090, ~$1.70 at $0.39/hr.

---

## Headline results (5-epoch run `trial-20260425-2313-real-1hr`)

Full PDF report at `reports/trial-20260425-2313-real-1hr.pdf`.

| Metric | Epoch 1 | Epoch 5 | Δ |
|---|---|---|---|
| **Combined score** (recall + mIoU + s3_acc) | 1.221 | **1.253** | +0.032 |
| Stage 1 recall@0.5 | 0.535 | 0.549 | +0.014 |
| Stage 1 precision@0.5 | 0.677 | 0.760 | +0.083 |
| Stage 2 mIoU (SAM3 frozen) | 0.378 | 0.397 | +0.019 |
| Stage 3 acc overall | 0.307 | 0.307 | 0.000 |
| Stage 3 acc on detected items | 0.575 | 0.560 | -0.015 |
| **Cosine top-1 baseline** (no transformer) | 0.498 | 0.443 | **-0.055** |
| **Transformer LIFT over cosine** | +0.076 | **+0.117** | **+0.041** |
| Dev loss total | 2.259 | 2.005 | -0.254 |

**Read**: Stage 1 got more conservative (precision +8.3 pt, recall flat). Stage 2 free-rode on Stage 1 (mIoU +1.8 pt despite SAM3 being frozen). Stage 3 is the interesting one — joint training **degraded** the embedding's cosine geometry (cosine top-1 −5.5 pt) but the **transformer learned to compensate**, so net stage3_acc stayed flat. The transformer's lift over a retrieval-only baseline grew from +7.6 → +11.7 pt monotonically — that's the publishable finding.

---

## Open architectural questions

Two we discussed but didn't run yet:

1. **Stage 1 recall is the binding bottleneck at 54.9%.** 45% of GT items are never detected, so Stage 3 can't classify them. Lifting recall is a bigger lever than tuning Stage 3. Options: longer Stage 1 LoRA training; detection-specific loss (not just LM loss); larger backbone (e.g., distilled 7B Qwen — see point 2); or augment with additional data.

2. **Distill Qwen 72B → 7B for Stage 1.** Pipeline_v2 (in the parent repo) used Qwen 72B AWQ to label the dataset. We have 5000 `vlm_outputs/<image_id>.json` files containing 72B's `{name, description, bbox}` outputs. Standard sequence-level SFT on `(image, prompt) → JSON` would distill 72B → 7B. The 7B then drops into Stage 1 as a more capable detector with the same fine-tune recipe we already use for 3B.

3. **(Side side experiment, ruled out)**: We tried YOLO-World x as a Stage 1 replacement (zero-shot, no fine-tuning). Best F1 was 0.326 with 32-class prompts at score_threshold=0.25 — vs Qwen-3B-LoRA's 0.638. Conclusion: open-vocab YOLO without fine-tuning isn't competitive on this dataset's specific class distribution. Useful to know, not worth pursuing.

---

## What the `vector-db-classifier` branch shows

Same TriFoodNet, but Stage 3 is **just `cosine_top1(DINOv2(crop), reference_index)`** — no transformer, no LoRA, no episodic training. The branch exists to:

- Make the architectural alternative inspectable in code, not just in slides
- Be a starting point for the "fork pipeline_v2 + vector DB + distilled 7B" direction (open question 2 above)
- Document that we considered the simpler architecture deliberately, not by oversight

See `docs/VECTOR_DB.md` on that branch for the design + expected trade-off analysis.

---

## Hardware + cost notes

- **5090 (Blackwell sm_120)** requires `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`. Earlier cu124 wheels lack Blackwell kernels and crash with "no kernel image."
- **4090 (sm_89)** works with stock `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.
- **SAM3 is gated:manual** on HuggingFace. Need Meta approval before training. Use the approved Madoun token in `.env` until your own is approved.
- A 5-epoch run on 5090 = ~$1.70. Smoke + setup overhead = ~$0.30. Budget ~$2-3 per experiment.
- **Auto-destroy is off by default** (`AUTO_DESTROY_MINUTES=0`). The watchdog will silently kill mid-epoch on long runs. Manually `bash scripts/vast/06_destroy.sh` when done.

---

## Files you should NOT touch without understanding why

- `dataset_v3_adapter.py` — the stratified split logic is load-bearing. Splits are cached deterministically; if the seed or hash changes, every previous comparison becomes apples-to-oranges.
- `pipeline.py::_safe_loss` — NaN protection. Without it, one bad batch kills the run.
- `master_config.yaml::stage2.freeze` — all three SAM3 components must stay `true`. Mask decoder unfrozen showed visual mask quality regression.

---

## Stuff that's still TODO before this is fully clean for handoff

- [ ] Upload the 480 MB checkpoints + 116 MB visualizations + 5.4 MB logs to the shared Drive folder
- [ ] (Optional) Distillation experiment: fine-tune Qwen 7B on `pipeline_v2/vlm_outputs/`
- [ ] (Optional) DINOv2-large vs PictSure cosine top-1 comparison (would tell us if the vector-db branch's choice of embedding matters)

---

## Who knows what

- **Razak**: full context on this codebase, the 5-epoch run, and the YOLO experiment
- **Madoun**: full context on `pipeline_v2/` (the dataset creation pipeline that produced the v3 export)
- **You (the reader)**: probably have questions — `HANDOFF.md` is the place to extend with them
