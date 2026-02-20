# V3 Pipeline Changes — 2026-02-21

## Summary

Upgrade VLM model, redesign prompt to eliminate description copying, add multi-box
SAM prompting, and tune sampling/threshold parameters.

---

## 1. VLM Model Upgrade (`config.py`)

| Parameter | Before | After |
|---|---|---|
| `model_name` | `Qwen/Qwen2.5-VL-3B-Instruct` | `Qwen/Qwen3-VL-8B-Instruct` |
| `gpu_memory_utilization` | 0.4 | 0.6 |

**Why:** The 3B model copied prompt examples verbatim — 475/1130 descriptions were
identical to the rice example, 461/1130 identical to the meat example (only 172 unique
descriptions out of 1130). The 8B model has much stronger instruction-following and
will generate actual per-image descriptions instead of parroting the template.

**Requires:** `vllm >= 0.11.0`, `transformers >= 4.57.0`

---

## 2. Prompt Redesign (`config.py`)

### New system prompt (Qwen3-VL has no default)
```
You are a food visual analyzer. You describe exactly what you see
in images — colors, textures, shapes, positions. You never guess
food names. You output only valid JSON.
```

### New describe_template — key changes:

- **Abstract placeholders** instead of concrete food examples:
  ```
  "<main_colors> <surface_texture> <shape_and_form>, <position_on_plate>"
  ```
  Old prompt had `"yellowish rice grains..."` and `"dark brown glazed meat pieces..."`
  which the 3B model copied verbatim 936/1130 times.

- **Explicit "this is a TEMPLATE" callout** so the model knows not to output it literally.

- **Grouping rules** for edge cases:
  - Kebab (scattered pieces of same food) → ONE item, ONE bbox around all pieces
  - Salad/stew (mixed ingredients) → ONE item
  - Different foods touching → SEPARATE items
  - Same food with visible gap → SEPARATE items

- **"Do NOT default to any specific number"** — the old prompt's 2-item example caused
  72% of images to get exactly 2 items.

- **Structured description format** (colors → texture → shape → position) for consistency
  without giving copyable content.

---

## 3. Sampling Parameters (`config.py`, `stage1_vlm.py`)

| Parameter | Before | After |
|---|---|---|
| `temperature` | 0.1 | 0.3 |
| `max_tokens` | 512 | 768 |
| `top_p` | 1.0 (default) | 0.8 |
| `top_k` | -1 (default) | 20 |

**Why:**
- `temperature 0.1` was too deterministic — model always picked highest-probability
  tokens (the example text). 0.3 adds diversity while staying coherent.
- `max_tokens 768` gives room for 4-5 detailed item descriptions.
- `top_p=0.8, top_k=20` are Qwen3's recommended values for vision tasks.

---

## 4. Multi-Box SAM Prompting (`config.py`, `stage2_sam.py`)

### What
Instead of sending SAM a single bbox per item, we now send a 2x2 grid of sub-boxes
plus the full bbox (5 prompts total). The `add_geometric_prompt` calls accumulate,
giving SAM spatial guidance across the full region rather than just the center.

```
┌─────────────┐
│ [1]  │  [2] │
│──────┼──────│    + [full bbox]  = 5 box prompts
│ [3]  │  [4] │
└─────────────┘
```

### Config flags (can disable to revert to single-bbox behavior)
```python
multi_box_prompt: bool = True   # set False to use single bbox (old behavior)
multi_box_grid: int = 2         # NxN grid (2 → 4 sub-boxes + 1 full = 5)
```

### Code changes
- New `_build_box_prompts()` method computes the sub-box grid in normalized coords
- `_segment_item()` loops over all box prompts instead of sending one

---

## 5. SAM Threshold & Padding (`config.py`)

| Parameter | Before | After |
|---|---|---|
| `confidence_threshold` | 0.1 | 0.15 |
| `crop_padding` | 5px | 10px |

**Why:**
- 0.15 filters out the 4 garbage masks (scores < 0.17) earlier in the cascade.
  The median SAM score is 0.975 so this doesn't affect good detections.
- 10px padding gives crops slightly more context for Stage 3 matching.

---

## 6. System Prompt Support (`stage1_vlm.py`)

- Added conditional system message injection (Qwen3-VL ships with no default system prompt)
- The system prompt is configurable via `VLMConfig.system_prompt` — set to empty string to disable

---

## Files Changed

| File | What |
|---|---|
| `config.py` | Model name, GPU mem, prompt, sampling params, SAM thresholds, multi-box config |
| `stage1_vlm.py` | System prompt in messages, `top_p`/`top_k` in SamplingParams |
| `stage2_sam.py` | Multi-box prompting via `_build_box_prompts()`, module docstring |

## How to Revert Multi-Box Prompting

In `config.json` or code, set:
```json
{"sam": {"multi_box_prompt": false}}
```
This restores the original single-bbox behavior with zero code changes.
