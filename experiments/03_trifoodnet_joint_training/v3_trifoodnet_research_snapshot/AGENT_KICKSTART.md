# TriFoodNet — Agent Kickstart Guide

> Read this first. It contains everything a coding agent (or new developer) needs
> to understand the project, navigate the codebase, run training, and make changes
> without any prior context.

> Snapshot note:
> The architecture and target packaged layout below describe the intended
> research repository. This specific workspace is a flat working snapshot with
> the core model files, an official PictSure integration path, and some missing
> training/data utilities relative to the full paper repo.

---

## 1. What This Project Is

**TriFoodNet** is a three-stage computer-vision pipeline that identifies and prices
individual food items on a university cafeteria dish plate:

```
Dish image
  │
  ▼ Stage 1  QwenGrounder (Qwen2.5-VL-3B + LoRA)
  │          → bounding boxes + coarse labels
  │
  ▼ Stage 2  SAMSegmenter (SAM2, decoder-only trainable)
  │          → pixel-level mask per box
  │
  ▼ Stage 3  FoodClassifier (frozen CLIP + ICL Transformer)
             → fine-grained class label + price
```

Each stage has its own loss and can be trained independently.
All three are also jointly fine-tuned end-to-end.

---

## 2. Repository Layout

```
trifoodnet/
│
├── configs/
│   ├── master_config.yaml        ← THE FILE YOU EDIT for all hyperparameters
│   ├── base_config.yaml          (inherited defaults)
│   ├── stage1_config.yaml        (stage-specific defaults)
│   ├── stage2_config.yaml
│   ├── stage3_config.yaml
│   └── joint_config.yaml
│
├── data/
│   ├── food_dataset.py           FoodDataset — loads images + annotations
│   ├── episode_sampler.py        EpisodeDataset — N-way K-shot episodic sampler
│   ├── transforms.py             Image augmentation pipelines
│   └── __init__.py
│
├── models/
│   ├── stage1_qwen.py            QwenGrounder — LoRA-wrapped Qwen2.5-VL-3B
│   ├── stage2_sam.py             SAMSegmenter — SAM2 with trainable mask decoder
│   ├── stage3_icl.py             FoodClassifier — FrozenCLIP + ICLTransformer
│   ├── pictsure_baseline.py      PictSurePipeline — local V1 baseline (cosine retrieval)
│   ├── pictsure_official.py      Official pretrained PictSure HF wrapper
│   ├── pipeline.py               TriFoodNet — full end-to-end pipeline
│   └── __init__.py
│
├── losses/
│   ├── losses.py                 Stage1Loss, Stage2Loss, Stage3Loss, JointLoss
│   └── __init__.py
│
├── training/
│   ├── train_stage1.py           Stage 1 training loop
│   ├── train_stage2.py           Stage 2 training loop
│   ├── train_stage3.py           Stage 3 episodic training loop
│   └── train_joint.py            Joint end-to-end fine-tuning
│
├── utils/
│   ├── config_loader.py          load_config() — reads master_config.yaml
│   ├── metrics.py                Recall@IoU, mIoU, EpisodicAccumulator
│   └── visualization.py         draw_pipeline_output(), save_comparison_grid()
│
└── setup.py
```

---

## 3. The One Config File to Rule Them All

**`configs/master_config.yaml`** is the single place to change anything.

```yaml
# Most-changed parameters — quick reference:

stage1:
  lora:
    r: 16                    # LoRA rank  (try 8, 16, 32)
  training:
    learning_rate: 2.0e-4

stage2:
  loss:
    bce_weight:  1.0
    dice_weight: 1.0

stage3:
  transformer:
    num_layers: 4            # 2 (fast) / 4 (default) / 6 (heavy)
  episode:
    n_way: 10
    k_shot: 5

joint:
  loss_weights:
    lambda1: 1.0             # raise if Stage 1 grounding degrades
    lambda2: 1.0
    lambda3: 1.0
```

You can also pass overrides directly on the CLI without editing the file:

```bash
python -m trifoodnet.training.train_stage1 \
    stage1.lora.r=32 \
    stage1.training.learning_rate=1e-4
```

### Loading config in code

```python
from trifoodnet.utils.config_loader import load_config

cfg = load_config()                          # reads master_config.yaml
cfg = load_config("configs/my_exp.yaml")     # custom file
cfg = load_config(overrides=["stage1.lora.r=32"])

# Attribute access — no dict brackets needed
print(cfg.stage1.lora.r)               # 32
print(cfg.joint.loss_weights.lambda1)  # 1.0

# Mutate at runtime
cfg.stage1.training.learning_rate = 5e-4

# Save snapshot alongside a checkpoint
cfg.save("checkpoints/v3-run1/config_snapshot.yaml")
```

---

## 4. Installation

```bash
# 1. Clone
git clone https://github.com/AbdullahMadoun/KFUPMRestaurant
cd KFUPMRestaurant

# 2. Create env
conda create -n trifoodnet python=3.11 -y
conda activate trifoodnet

# 3. Install PyTorch (adjust CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install project + dependencies
pip install -e .

# 5. (Optional) SAM2 from source for latest weights
pip install git+https://github.com/facebookresearch/sam2.git

# 6. Verify
python -c "from trifoodnet import TriFoodNet; print('OK')"
```

---

## 5. Data Format

### Annotation JSON (one file, list of image dicts)

```json
[
  {
    "image_id": "img_0042",
    "file_name": "img_0042.jpg",
    "split": "train",
    "items": [
      {
        "box":      [x1, y1, x2, y2],
        "label":    "grilled_chicken",
        "label_id": 3,
        "mask_rle": { "counts": "...", "size": [H, W] }
      }
    ]
  }
]
```

- `box` — tightest bounding box around the SAM3 mask
- `mask_rle` — COCO RLE format (requires `pycocotools`)
- `label_id` — integer class id (0-indexed, contiguous)

### Splits JSON

```json
{
  "train": ["img_0001", "img_0042", ...],
  "val":   ["img_0100", ...],
  "test":  ["img_0200", ...],
  "held_out_class_ids": [5, 12, 17, 23, 31]
}
```

`held_out_class_ids` — class ids withheld entirely from Stage 3 training,
used only for zero-shot generalization evaluation (Table 3 in the paper).

### Reference Library (Stage 3 inference)

```
data/reference_library/
  grilled_chicken/
    ref1.jpg
    ref2.jpg
    ref3.jpg
  white_rice/
    ref1.jpg
    ...
```

Minimum 1 image per class. 3–5 is recommended.

### Price Table

```json
{
  "grilled_chicken": 15.00,
  "white_rice":       5.00,
  "garden_salad":     8.00
}
```

---

## 6. Training — Step by Step

### Step 1 — Train Stage 1 (Qwen grounding)

```bash
python -m trifoodnet.training.train_stage1
# With overrides:
python -m trifoodnet.training.train_stage1 stage1.lora.r=32 stage1.training.epochs=15
```

Saves LoRA adapter weights to `checkpoints/<run.name>/stage1/best/`.

### Step 2 — Train Stage 2 (SAM mask decoder)

```bash
python -m trifoodnet.training.train_stage2
```

Saves full model state dict to `checkpoints/<run.name>/stage2/best.pt`.

### Step 3 — Train Stage 3 (ICL Transformer)

```bash
python -m trifoodnet.training.train_stage3
```

Saves ICL transformer weights to `checkpoints/<run.name>/stage3/best_icl.pt`.
CLIP weights are frozen and never saved (reloaded from HF hub each time).

### Step 4 — Joint fine-tuning

```bash
python -m trifoodnet.training.train_joint
```

Loads the three checkpoints above, then fine-tunes them together.
Uses GT box prompts for SAM for the first `joint.curriculum.gt_boxes_epochs` epochs,
then switches to Qwen predicted boxes.

---

## 7. Inference

```python
from PIL import Image
from trifoodnet.models import TriFoodNet, QwenGrounder, SAMSegmenter, FoodClassifier, PriceLookup
from trifoodnet.utils.config_loader import load_config

cfg = load_config()

# Build pipeline
stage1 = QwenGrounder(cfg.stage1.model_name, lora_r=cfg.stage1.lora.r)
stage2 = SAMSegmenter(cfg.stage2.model_name)
stage3 = FoodClassifier(cfg.stage3.clip_model)

prices = PriceLookup()
prices.load_json(cfg.paths.price_table)

pipeline = TriFoodNet(stage1, stage2, stage3, prices)
pipeline.load("checkpoints/v3-run1/joint/best")

# Build reference library (Stage 3)
import os
from pathlib import Path
ref_library = {}
for cls_dir in Path(cfg.paths.reference_library).iterdir():
    ref_library[cls_dir.name] = [Image.open(p) for p in cls_dir.glob("*.jpg")]
stage3.build_reference_library(ref_library)

# Run inference
image  = Image.open("test_dish.jpg")
result = pipeline.run(image)

print(f"Total price: SAR {result.total_price:.2f}")
for item in result.items:
    print(f"  {item.label:25s} {item.confidence:.0%}  SAR {item.price:.2f}")
print(f"Latency: {result.latency_ms}")
```

### Adding a new food class (zero retraining)

```python
from PIL import Image

new_refs = [Image.open(f"refs/pasta_{i}.jpg") for i in range(3)]
stage3.add_class("pasta", new_refs, device="cuda")
# Pipeline now recognises pasta without any retraining.
```

---

## 8. PictSure Baselines (Table 2)

There are now two PictSure paths in this workspace:

- `pictsure_official.py` uses the upstream `PictSure` package and public
  Hugging Face checkpoints such as `pictsure/pictsure-vit`.
- `pictsure_baseline.py` keeps the older local CLIP retrieval baseline for
  ablations and architecture comparisons.

```python
from trifoodnet.models.pictsure_baseline import CLIPEncoder, PictSureClassifier

encoder    = CLIPEncoder(device="cuda")
classifier = PictSureClassifier(encoder)

# Add references
from PIL import Image
import os
for cls_name in os.listdir("data/reference_library"):
    imgs = [Image.open(p) for p in Path(f"data/reference_library/{cls_name}").glob("*.jpg")]
    classifier.add_reference_images(cls_name, imgs)
classifier.build()

# Classify a crop
results = classifier.classify([Image.open("crop.jpg")], top_k=3)
print(results)   # [("grilled_chicken", 0.82), ("roast_beef", 0.10), ...]

# Save / reload index
classifier.save("data/pictsure_index")
classifier2 = PictSureClassifier.load("data/pictsure_index")
```

---

## 9. How to Modify Things

### Change a loss weight

Edit `configs/master_config.yaml`:
```yaml
stage2:
  loss:
    bce_weight:  0.5    # was 1.0
    dice_weight: 2.0    # was 1.0
```
Or pass on CLI: `python -m trifoodnet.training.train_stage2 stage2.loss.dice_weight=2.0`

### Add a new loss term to Stage 2

Open `trifoodnet/losses/losses.py`, find `Stage2Loss.forward()`:

```python
def forward(self, pred_logits, gt_masks):
    bce  = F.binary_cross_entropy_with_logits(pred_logits, gt_masks)
    dice = dice_loss(torch.sigmoid(pred_logits), gt_masks)
    # ── ADD your new term here ──
    focal = focal_loss(pred_logits, gt_masks)   # example
    loss  = self.bce_weight * bce + self.dice_weight * dice + 0.5 * focal
    return loss, {"bce": bce.item(), "dice": dice.item(), "focal": focal.item(), "total": loss.item()}
```

Then add a `focal_weight` field to `Stage2Loss.__init__` and to `master_config.yaml`.

### Change LoRA targets

```yaml
# master_config.yaml
stage1:
  lora:
    target_modules: ["q_proj", "v_proj", "k_proj"]   # add k_proj
```

### Swap CLIP backbone (Stage 3)

```yaml
stage3:
  clip_model: "openai/clip-vit-base-patch32"   # smaller / faster
  embed_dim:  512                               # stays 512 for both ViT-L and ViT-B
```

### Run a single experiment tag

```yaml
run:
  name: "v3-run2-lora32"
  notes: "Testing larger LoRA rank"
```

All checkpoints, logs, and config snapshots will be saved under that tag automatically.

---

## 10. Key Design Decisions

| Decision | Rationale |
|---|---|
| LoRA on Qwen (not full fine-tune) | ~1% of params updated; keeps VRAM < 24 GB |
| Freeze SAM image encoder | ViT-H is 307M params; freezing saves ~3× VRAM |
| Frozen CLIP in Stage 3 | CLIP already generalises well; training it would overfit on 1K images |
| Episodic training for Stage 3 | Forces the model to learn comparison rather than memorisation |
| Global NMS on masks | Inherited from V2; eliminates ghost detections on dish edges |
| GT boxes for SAM in epoch 1 | Curriculum isolates Stage 2 from Stage 1 errors early in training |

---

## 11. Evaluation Tables (from paper)

### Table 1 — Per-stage, independent vs joint

| Stage | Metric | Independent | Joint |
|---|---|---|---|
| Stage 1 (Qwen) | Recall@IoU0.5 | `TBD` | `TBD` |
| Stage 2 (SAM)  | mIoU | `TBD` | `TBD` |
| Stage 3 (ICL)  | Top-1 Acc | `TBD` | `TBD` |

### Table 2 — Stage 3 classifier comparison

| Classifier | Top-1 Acc | New class onboarding |
|---|---|---|
| Cosine sim (PictSure) | `TBD` | Instant |
| **ICL Transformer (ours)** | `TBD` | Instant |
| CLIP linear probe | `TBD` | Requires retraining |

### Table 3 — Zero-shot on held-out classes

| References | Top-1 Acc |
|---|---|
| 1 | `TBD` |
| 3 | `TBD` |
| 5 | `TBD` |

### Table 4 — Latency breakdown

| Stage | Target |
|---|---|
| Stage 1 (Qwen) | TBD ms |
| Stage 2 (SAM)  | TBD ms |
| Stage 3 (ICL)  | TBD ms |
| **Total** | **< 2000 ms** |

---

## 12. Common Issues

**OOM on Stage 1**
→ Lower `stage1.training.batch_size` to 2 and raise `grad_accum_steps` to 16.

**SAM not found**
→ `pip install git+https://github.com/facebookresearch/sam2.git`
→ Or use `transformers>=4.44` and the model id `facebook/sam2-hiera-large`.

**pycocotools not found**
→ `pip install pycocotools`  (Linux) or `pip install pycocotools-windows` (Windows)

**Wandb disabled**
→ Set `logging.wandb: false` in `master_config.yaml`.

**Zero-shot eval shows 0 classes**
→ Check `splits.json` has `held_out_class_ids` populated.

---

## 13. Citation

```bibtex
@article{madoun2026trifoodnet,
  title   = {TriFoodNet: A Jointly Trainable Grounding-Segmentation-Recognition
             Pipeline for Open-World Food Item Analysis},
  author  = {Madoun, Abdullah},
  journal = {arXiv preprint},
  year    = {2026}
}
```
