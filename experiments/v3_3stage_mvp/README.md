# V3 3-Stage MVP Pipeline

**Describe → Segment → Match**

A 3-stage food detection pipeline that identifies and prices food items from a known restaurant menu. Unlike V2 (zero-shot), this pipeline closes the loop: it detects food on a plate, segments it precisely, then identifies each item against a reference menu database with pricing.

---

## How It Works

```
Input Image (plate photo)
        │
   ┌────▼─────────────────────┐
   │  Stage 1: VLM Describe   │  Qwen2.5-VL-3B
   │  → visual descriptions   │  "golden crispy surface with ridged texture"
   │  → bounding boxes        │  [120, 45, 380, 290]
   └────┬─────────────────────┘
        │
   ┌────▼─────────────────────┐
   │  Stage 2: SAM3 Segment   │  text + bbox → pixel masks
   │  → precise masks         │  crop each item from original
   │  → masked crops          │  (background zeroed out)
   └────┬─────────────────────┘
        │
   ┌────▼─────────────────────┐
   │  Stage 3: Vector Match   │  SigLIP 2 encode crop
   │  → FAISS nearest search  │  → top-k menu matches
   │  → item ID + price       │  → price lookup
   └────┬─────────────────────┘
        │
        ▼
   Final Output: identified items + prices + masks + visualization
```

### Stage 1 — VLM Visual Description (`stage1_vlm.py`)

Qwen2.5-VL-3B analyzes the plate image in a **single call** and returns:
- A **visual description** for each food item (colors, textures, shapes — NOT food names)
- A **bounding box** in pixel coordinates `[x1, y1, x2, y2]`

**Why visual descriptions instead of food names?**
- SAM3's text encoder handles visual features ("golden brown crispy") better than semantic names ("fried chicken")
- Avoids VLM misidentification bias — if the VLM calls fish "chicken", SAM3 would search for the wrong thing
- Identification is Stage 3's job, where embedding similarity is more reliable

### Stage 2 — SAM3 Segment + Crop (`stage2_sam.py`)

For each item from Stage 1, SAM3 receives **two prompts**:
1. **Text prompt** — the visual description (e.g., "golden brown crispy surface")
2. **Geometric prompt** — the bounding box converted to normalized `[cx, cy, w, h]`

This dual prompting (from the grounding branch) improves segmentation by ~40% compared to text-only.

After segmentation:
- **NMS** (Non-Maximum Suppression) removes overlapping detections
- Each item's mask is applied to the original image and **cropped** to its bounding box region

### Stage 3 — Vector Matching (`stage3_match.py`)

Each masked crop is:
1. Encoded with **SigLIP 2** into a 768-dim embedding vector
2. Searched against a pre-built **FAISS index** of reference menu images
3. Matched to the nearest menu item (or labeled "unknown" if similarity is below threshold)

The match result includes: item name, category, price, confidence score, and top-k candidates.

---

## Setup

### Dependencies

Install on top of the existing V2 requirements:

```bash
pip install faiss-cpu>=1.7.4 transformers>=4.40.0
```

Full list (from V2 + new):
```
torch>=2.4.0
torchvision>=0.19.0
vllm>=0.6.0
transformers>=4.40.0
accelerate
opencv-python
numpy
Pillow
faiss-cpu>=1.7.4
```

> Use `faiss-gpu` instead of `faiss-cpu` if you want GPU-accelerated index search (not needed for <1000 vectors).

### GPU Memory Budget (~9.5 GB)

| Model | VRAM | Notes |
|-------|------|-------|
| Qwen2.5-VL-3B (vLLM, 0.4 util) | ~6 GB | Stage 1 |
| SAM3 | ~3 GB | Stage 2 |
| SigLIP 2 base | ~0.4 GB | Stage 3 |
| FAISS flat index | <1 MB | Negligible |

All three models stay loaded simultaneously. Fits on a 16GB T4/V100 with headroom.

---

## Usage

### Step 1: Prepare Reference Images

Organize reference food images into subdirectories, one per menu item:

```
reference_images/
├── chicken/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── rice/
│   ├── img1.jpg
│   └── img2.jpg
├── salad/
│   └── img1.jpg
└── ...
```

Each subdirectory name must match a key in `menu_schema.json`. Use multiple images per item (different angles, portions, lighting) for more robust matching.

### Step 2: Edit the Menu Schema

Edit `menu_schema.json` to reflect your actual menu:

```json
{
  "chicken": {
    "category": "protein",
    "price": 15.0,
    "description": "Grilled or fried chicken pieces"
  },
  "rice": {
    "category": "carb",
    "price": 5.0,
    "description": "White or yellow steamed rice"
  }
}
```

Categories: `protein`, `carb`, `salad`, `side`, `soup`, `drink`, `dessert`

### Step 3: Build the Vector Index

Run once (or whenever the menu changes):

```bash
python build_index.py \
    --menu_dir /path/to/reference_images/ \
    --schema menu_schema.json \
    --output_index menu.index \
    --output_meta menu_meta.json \
    --device cuda
```

This encodes all reference images with SigLIP 2 and builds a FAISS flat index. Outputs:
- `menu.index` — FAISS index file
- `menu_meta.json` — metadata mapping (name, category, price per vector)

### Step 4: Run the Pipeline

**Single image:**
```bash
python main.py /path/to/plate.jpg \
    --index menu.index \
    --meta menu_meta.json
```

**Directory of images:**
```bash
python main.py /path/to/images/ \
    --index menu.index \
    --meta menu_meta.json \
    --output_dir results/
```

**With config file:**
```bash
python main.py /path/to/plate.jpg \
    --index menu.index \
    --meta menu_meta.json \
    --config my_config.json
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input_path` | (required) | Path to image or directory of images |
| `--index` | `menu.index` | Path to FAISS index file |
| `--meta` | `menu_meta.json` | Path to metadata JSON file |
| `--config` | None | Path to JSON config file (overrides defaults) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--output_dir` | `results` | Directory for run outputs |
| `--threshold` | `0.1` | SAM3 confidence threshold |
| `--similarity` | `0.5` | Min cosine similarity for a menu match (below = "unknown") |
| `--top_k` | `3` | Number of match candidates to return |

CLI arguments override config file values, which override defaults.

---

## Configuration Reference

All config is defined as nested dataclasses in `config.py`. You can export defaults to JSON, edit, and reload:

```python
from config import PipelineConfig
config = PipelineConfig()
config.to_json("my_config.json")  # export defaults
```

Then pass `--config my_config.json` to override.

### VLMConfig (Stage 1)

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace model ID for the VLM |
| `gpu_memory_utilization` | `0.4` | vLLM GPU memory fraction (0.0–1.0) |
| `max_model_len` | `4096` | Max context length for vLLM |
| `enforce_eager` | `true` | Disable CUDA graphs (saves memory, slightly slower) |
| `allowed_local_media_path` | `/root` | Base path vLLM is allowed to read images from |
| `temperature` | `0.1` | Sampling temperature (low = deterministic) |
| `max_tokens` | `512` | Max output tokens (needs room for descriptions + bboxes) |
| `describe_template` | *(see below)* | The VLM prompt template |

**Prompt template** instructs the VLM to:
- Describe visual appearance only (color, texture, shape) — not name the food
- Provide bounding boxes as `[x1, y1, x2, y2]` pixel coordinates
- Treat mixed dishes as one item, side-by-side items as separate
- Ignore plates, bowls, cutlery, background
- Return strictly as JSON: `{"items": [{"description": "...", "bbox": [...]}, ...]}`

### SAMConfig (Stage 2)

| Field | Default | Description |
|-------|---------|-------------|
| `model_path` | `facebook/sam3` | SAM3 model identifier |
| `confidence_threshold` | `0.1` | Initial mask confidence threshold |
| `fallback_thresholds` | `[0.05, 0.02, 0.01]` | If no masks at initial threshold, retry with these (descending) |
| `crop_padding` | `5` | Pixels of padding around mask crop (prevents edge clipping) |
| `bpe_search_paths` | *(see config.py)* | Paths to search for SAM3's BPE vocabulary file |

**Dynamic thresholding**: If SAM3 finds no masks at 0.1, it automatically retries at 0.05, then 0.02, then 0.01. This handles low-contrast or unusual items.

### MatchConfig (Stage 3)

| Field | Default | Description |
|-------|---------|-------------|
| `embedding_model` | `google/siglip2-base-patch16-224` | SigLIP 2 model for crop encoding |
| `embedding_dim` | `768` | Embedding vector dimension |
| `index_path` | `menu.index` | Path to FAISS index file |
| `metadata_path` | `menu_meta.json` | Path to metadata JSON |
| `top_k` | `3` | Number of nearest neighbors to retrieve |
| `similarity_threshold` | `0.5` | Cosine similarity below this = "unknown" |

**Tuning `similarity_threshold`**:
- **Lower (e.g., 0.3)**: More items matched, but more false positives
- **Higher (e.g., 0.7)**: Fewer matches, but higher confidence in each
- Start at 0.5, adjust based on your reference image quality and variety

### NMSConfig

| Field | Default | Description |
|-------|---------|-------------|
| `max_objects` | `8` | Maximum items to keep after NMS (restaurant plates can be busy) |
| `iou_threshold` | `0.7` | Mask overlap above this = duplicate → lower-scoring one is removed |

### VizConfig

| Field | Default | Description |
|-------|---------|-------------|
| `draw_boxes` | `true` | Draw bounding boxes on visualization |
| `alpha` | `0.7` | Mask overlay opacity (0.0 = transparent, 1.0 = opaque) |
| `thickness` | `3` | Contour line thickness in pixels |
| `font_scale` | `0.9` | Label text size |
| `font_thickness` | `2` | Label text weight |
| `show_match_label` | `true` | Show menu item name (e.g., "Chicken") instead of visual description |
| `show_price` | `true` | Show price on label (e.g., "— 15 SAR") |
| `show_confidence` | `true` | Show match confidence (e.g., "(0.92)") |

**Visualization features**:
- Category-coded colors: proteins=red, carbs=yellow, salads=green, sides=blue, etc.
- Labels: `Chicken (0.92) — 15 SAR` or `? Unknown (0.38)` if below threshold
- Bottom bar: `Total: 28 SAR | 3 items (1 unknown)`

---

## Output Structure

Each run creates a timestamped directory:

```
results/
└── run_20260220_143052/
    ├── config.json              # Frozen config snapshot for reproducibility
    ├── pipeline.log             # Full pipeline log with per-stage timing
    ├── run_summary.json         # Aggregated metrics (see below)
    └── visualizations/
        ├── matched_plate1.jpg   # Annotated image with masks + labels + prices
        ├── matched_plate2.jpg
        └── ...
```

### Run Summary (`run_summary.json`)

```json
{
  "run_id": "run_20260220_143052",
  "total_time": 45.2,
  "total_images": 5,
  "total_matched": 12,
  "total_unknown": 2,
  "total_price": 156.0,
  "avg_stage1_time": 1.8,
  "avg_stage2_time": 3.2,
  "avg_stage3_time": 0.4,
  "per_image": [...]
}
```

---

## File Structure

```
experiments/v3_3stage_mvp/
├── config.py              # All configuration dataclasses + JSON I/O
├── ptypes.py              # Data types: VisualItem, SegmentedItem, MatchResult, etc.
├── stage1_vlm.py          # Qwen VLM: image → visual descriptions + bboxes
├── stage2_sam.py           # SAM3: descriptions + bboxes → masks → crops
├── stage3_match.py         # SigLIP2 + FAISS: crops → embeddings → menu matches
├── vector_store.py         # FAISS index: build / save / load / query
├── visualizer.py           # Render masks + match labels + confidence + prices
├── logger.py               # Run tracking + per-stage metrics
├── main.py                 # CLI orchestrator
├── build_index.py          # CLI: build vector DB from reference images
├── menu_schema.json        # Menu metadata: name, category, price per item
└── README.md               # This file
```

> **Note**: `ptypes.py` is named to avoid shadowing Python's stdlib `types` module. It corresponds to `types.py` in the design spec.

---

## What Changed from V2

| Aspect | V2 (zero-shot) | V3 (3-stage MVP) |
|--------|----------------|-------------------|
| VLM output | Food names | Visual descriptions + bboxes |
| VLM calls | 1 call (names only) | 1 call (descriptions + bboxes) |
| SAM3 input | Food name text only | Visual description + bbox geometric |
| After segmentation | Visualize with label | Crop → embed → match against DB |
| Identification | VLM names it (can be wrong) | Vector similarity (more reliable) |
| Menu awareness | None | Full menu + pricing |
| Output | Segmented image | Segmented image + item IDs + total price |
