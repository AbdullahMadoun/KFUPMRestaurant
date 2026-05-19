# VECTOR_DB.md — Stage 3 vector DB classifier

This branch (`vector-db-classifier`) replaces TriFoodNet's Stage 3 ICL transformer with a **DINOv2 cosine top-1 retrieval** against per-class reference prototypes. Stages 1 and 2 are unchanged. Read this with `stage3_vector_db.py` open.

---

## TL;DR

- **What**: classify each food crop by cosine similarity to a per-class prototype, computed once from the reference library at startup.
- **Why**: removes the entire Stage 3 training surface (no LoRA, no episodic batches, no joint training of the embedding) — which we observed actually *degrades* the embedding's cosine geometry over training.
- **Trade-off**: gives up the **+11.7 pt transformer lift** we measured on the `training-only` branch, but gains architectural simplicity and zero retraining when adding a class.
- **Switch**: `master_config.yaml::stage3.backend = "vector_db"` (default on this branch).

---

## Why this branch exists

On `training-only` we measured (5-epoch run, dev split):

| | Value |
|---|---|
| Stage 3 acc with ICL transformer | 56.0 % (matched_acc) |
| Stage 3 acc with cosine top-1 only | 44.3 % (cosine_top1_acc) |
| **Transformer lift** | **+11.7 pt** |

Interesting wrinkle: the cosine baseline **dropped 5.5 pt over training** (49.8 → 44.3). The PictSure embedding's cosine geometry got *worse* during joint training, but the transformer learned to extract more signal from it, so net accuracy stayed flat.

**This branch tests two hypotheses simultaneously:**
1. If we never touch the embedding, does cosine top-1 stay at ~50 % (the un-degraded baseline)?
2. Does swapping PictSure → DINOv2-large (a stronger general-purpose encoder) close more of the gap to the transformer?

Realistic expectation:
- Pure cosine on PictSure (no joint training): ~50 % matched_acc (same as epoch 1 of the trained run)
- Cosine on DINOv2-large: ~50-55 % (DINOv2 is generally a stronger image encoder)
- Still 5-10 pt below the trained ICL transformer

So we're trading roughly 5-10 pt of accuracy for:
- **Zero Stage 3 training** (no LoRA, no episodic sampling, no leak protection)
- **Add new classes at inference** without retraining (just drop 5 reference images in the right folder, restart)
- **Embedding stays clean** — no joint-training contamination
- **One-line classifier** — `cosine_top1(DINOv2(crop), prototypes)`

---

## Architecture diff vs `training-only`

```
                training-only                              vector-db-classifier
                ─────────────                              ────────────────────
Stage 1         Qwen 3B + LoRA (trained)                   Qwen 3B + LoRA (trained)         ← unchanged
Stage 2         SAM3 (frozen)                              SAM3 (frozen)                     ← unchanged
Stage 3         PictSure ViT + cross-attn transformer      DINOv2-large + cosine top-1       ← swapped
                + LoRA + episodic 5-way 5-shot             + per-class mean prototype
                + joint training                           + no training
                ~524K trainable params                     ~0 trainable params
                ~7-12 pt lift over cosine                  no lift (it IS cosine)
```

That's it. Same dataset, same pipeline.py, same eval_harness — the swap happens in `train_joint.py`'s Stage-3 construction:

```python
if cfg.stage3.backend == "vector_db":
    from stage3_vector_db import FoodVectorDB
    stage3 = FoodVectorDB(reference_root=..., class_names=..., encoder=..., device=...)
else:
    stage3 = FoodClassifier(...)  # the existing ICL transformer
pipeline = TriFoodNet(stage1, stage2, stage3)
```

---

## How `FoodVectorDB` works

### At init (one-time)

1. Load DINOv2-large (frozen, fp16 on GPU)
2. For each of the 32 classes:
   - List `reference/<class_slug>/*.jpg`
   - Embed each reference image: `cls_embedding = DINOv2(image)[CLS]`
   - L2-normalize
   - Mean-pool across the class's references → per-class **prototype**
   - L2-normalize again
3. Stack into a `(32, 1024)` matrix `prototypes`

### At inference (per crop)

```python
query_emb = F.normalize(DINOv2(crop)[CLS], dim=-1)
scores    = prototypes @ query_emb            # (32,) cosine similarities
top_k     = scores.topk(k=1)
return [(class_names[idx], float(score)) for ...]
```

That's the entire classify path. ~10 lines, no learnable parameters.

### Eval-harness diagnostic compatibility

`stage3_vector_db.classify()` populates `self.last_candidate_class_names` exactly like `stage3_icl.classify()` does, so `eval_harness.evaluate_inference_loop` keeps emitting:

- `dev/stage3_acc` (overall classification accuracy)
- `dev/stage3_matched_acc` (accuracy on items Stage 1 detected)
- `dev/stage3_cosine_top1_acc` (the retrieval baseline — same as stage3_acc here, by construction)
- `dev/stage3_retrieval_recall@K` (whether GT class is in top-K)
- `dev/stage3_acc_given_retrieved` (if retrieval surfaced GT, did we pick it?)
- `dev/stage3_transformer_lift_over_top1` (will be **≈0** here — the "transformer" IS retrieval)

So you can directly diff the JSONL events between branches and see exactly where the transformer was contributing vs where retrieval alone is enough.

---

## How to run on this branch

Same flow as `training-only`, but Stage 3 is inference-only so the joint-training loop only optimizes Stages 1+2:

```bash
# 1. Confirm config
grep -A 2 "^stage3:" master_config.yaml
#   should show backend: "vector_db"

# 2. Smoke (should pass — same as training-only minus Stage 3 trainable checks)
python scripts/smoke_phase3.py

# 3. (Optional) Just eval, no training, on the dev split
python scripts/run_eval_only.py    # if it exists; otherwise:
python -m train_joint joint.training.epochs=0 joint.eval.interval=1
#   → loads pipeline, evaluates dev once, writes events.jsonl + dev_visualizations
```

To launch a full training run on this branch, the per-epoch behavior is:
- Stages 1+2 train as usual (Qwen LoRA + SAM3 frozen with mask supervision)
- Stage 3 evaluates only — no gradients flow through the DINOv2 encoder
- Dev eval emits Stage 3 metrics that are entirely cosine top-K based

---

## What to compare across branches

After running both:

| Metric | training-only ep5 | vector-db-classifier (when run) | Delta |
|---|---|---|---|
| stage1_recall@0.5 | 0.549 | (should be similar — Stage 1 unchanged) | ~0 |
| stage1_precision@0.5 | 0.760 | (should be similar) | ~0 |
| stage2_mIoU | 0.397 | (should be similar — SAM3 frozen + same boxes) | ~0 |
| stage3_matched_acc | 0.560 | ? (predicted ~50%) | ~ -5 to -10 |
| stage3_cosine_top1_acc | 0.443 | ? (should be HIGHER, embedding never degraded) | ~ +5 to +10 |
| transformer_lift | +0.117 | ~0 by construction | -11.7 |
| combined score | 1.253 | ? | -0.05 to -0.10 |

If the cosine_top1 number on this branch is meaningfully higher than 0.443 (say, ≥0.55), it confirms the joint-training-degrades-embedding hypothesis. That's a useful finding even if the absolute classifier accuracy is lower.

---

## Reference library expectations

The vector DB needs **at least one reference image per class** in `data/reference_library/<class_slug>/`. Today the v3 export's `reference/` folder has ~5 images per class.

If a class folder is empty, that class gets a zero-vector prototype and will never be selected. There's a one-line warning at startup if any class has zero references.

To add a new class:
1. Drop 3-5 representative images in `data/reference_library/<new_class_slug>/`
2. Add the class name to `classes.json`
3. Restart inference — the prototype index rebuilds at startup (~30 sec for 32 × 5 images on GPU)
4. No training. No checkpoints. No retraining the existing classes.

---

## Open work

- [ ] Run a fresh dev eval on this branch and fill in the comparison table above
- [ ] Try `text_weight > 0` (use the VLM's `name + description` text via SigLIP, fuse with image cosine) — could meaningfully close the gap to the trained ICL transformer
- [ ] If DINOv2-large doesn't beat PictSure cosine top-1 by a clear margin, fall back to PictSure encoder (set `vector_db.encoder` accordingly)

---

## Related

- `stage3_vector_db.py` — the implementation (~200 LOC, well-commented)
- `stage3_icl.py` — the trained ICL transformer this swaps out
- `HANDOFF.md` — top-level overview, references this branch
- `docs/DATASET.md` — explains the reference library + items.jsonl `is_reference` flag
