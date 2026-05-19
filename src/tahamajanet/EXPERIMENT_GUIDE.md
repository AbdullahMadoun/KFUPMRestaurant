# EXPERIMENT_GUIDE — how to track, document, and compare runs

The training-only branch ships with one place to track results (TensorBoard) and one place to record what was tried (`experiments/registry.jsonl`). This guide is the discipline for using both.

---

## The single source of truth

```
experiments/registry.jsonl
```

Every completed run gets one line. This is non-negotiable:
- the file is grep-able forever
- it survives team changes, branch deletions, and dead instances
- if a run isn't in the registry, it didn't happen

Append after each run:
```bash
python scripts/registry_append.py logs/<run-name>/joint
```

This pulls run_metadata.json + the best/final eval events and writes one JSON object per line. Schema includes: `run_name`, `dataset_version`, `dataset_hash`, `git_sha`, `git_dirty`, `seed`, `best_dev_combined`, all `dev/stage*` numbers, `train/nan_*`, `train/episode_leak_fallback_total`, `elapsed_sec`. Full schema in `scripts/registry_append.py`.

---

## Naming convention for runs

Use this template:

```
trial-<YYYYMMDD>-<short-tag>
```

Examples:
- `trial-20260425-cleandata1-rerun` — re-running the cleandata1 baseline with fixes
- `trial-20260501-stage3-lr-sweep-1e-4` — single point in an LR sweep
- `trial-20260503-no-teacher-forcing` — single ablation
- `trial-20260510-fold2-of-3` — fold of a CV experiment

Date first so registries sort chronologically. Tag last and descriptive — anyone reading the registry should be able to figure out what was tried without opening the config.

Set the name explicitly per run:
```bash
python -m train_joint run.name=trial-20260501-no-teacher-forcing ...
```

If `run.name` is left at the default (`trial-20260321-cleandata1`), `registry_append.py` will warn but still record. Don't rely on the default.

---

## TensorBoard — the live result tracker

```bash
tensorboard --logdir logs/
```

Default port `6006`. If running on a remote box, tunnel via SSH:
```bash
ssh -L 6006:localhost:6006 root@$HOST
```

### What gets logged

The `ExperimentLogger` mirrors every numeric scalar from `events.jsonl` to TB. You will see:

**Training (every train_step)**:
- `train_step/loss_stage1`, `_stage2`, `_stage3`, `_total`
- `train_step/stage1_lm_loss`, `stage2_bce_loss`, `stage2_dice_loss`, `stage3_ce_loss`, `stage3_acc`
- `train_step/lr`, `grad_norm`, `samples_per_sec`, `step_time_sec`
- `train_step/use_gt_boxes`, `teacher_forcing_prob`

**Per epoch**:
- `train/epoch_time_sec`
- `train/nan_stage1`, `nan_stage2`, `nan_stage2_internal`, `nan_stage3`, `nan_total`
- `train/episode_leak_fallback_total`

**Per dev eval**:
- `dev/stage1_recall@0.5`, `stage1_precision@0.5`
- `dev/stage2_mIoU`
- `dev/stage3_acc`, `stage3_matched_acc`, `stage3_episode_acc`
- `dev/combined`, `dev/loss_total`
- `dev/pred_items_per_image`, `dev/latency_total_ms`
- The diagnostic four: `dev/stage3_retrieval_recall@K`, `stage3_acc_given_retrieved`, `stage3_cosine_top1_acc`, `stage3_transformer_lift_over_top1`

**Run start (one-shot)**:
- `optimizer/stage1_lr`, `stage2_lr`, `stage3_lr` (per-stage LRs as configured)
- `model/stage1_trainable_params`, `stage2_trainable_params`, `stage3_trainable_params`
- A text card under "config_snapshot" with the full resolved config

### How to compare runs in TB

Multiple runs in the same `logs/` directory show up as separate lines automatically. Toggle them on/off in the left sidebar. Use the regex filter to focus on one metric family at a time.

Save TB views as URLs (`?run=...&tag=...`) and paste them into experiment notes.

---

## Documenting a single experiment

For each run, write a one-page markdown after it completes. Template:

```markdown
# Run: trial-YYYYMMDD-<tag>

## What changed (vs which baseline)

One paragraph, link to the registry baseline row.

## Hypothesis

What we expected to see, and why.

## Result

| metric | baseline | this run | delta |
|---|---|---|---|
| dev/combined | 1.94 | 1.97 | +0.03 |
| dev/stage3_acc | 0.50 | 0.53 | +0.03 |
| ... | | | |

## Interpretation

What the deltas tell us. Was the hypothesis confirmed? Reject? Inconclusive?

## Side observations

Anything weird in the training curves. NaN counts. Leak fallbacks. Step time.

## Next step

What this enables / what to try next.
```

Save these in `outputs/<run-name>/RUN_NOTES.md`. They'll get pulled along with logs by `scripts/vast/05_pull.sh`.

---

## Documenting a series of experiments (ablation campaign)

Create a campaign-level markdown:

```
outputs/campaign_<topic>_<YYYYMMDD>/
  ├── README.md                  # the campaign overview + final table
  ├── trial-XXX-fold1/RUN_NOTES.md
  ├── trial-XXX-fold2/RUN_NOTES.md
  └── trial-XXX-fold3/RUN_NOTES.md
```

The campaign README starts with: hypothesis, design (which configs differ between runs), expected outcome. It ends with: actual outcome, conclusion, what to do next.

Use TB's run-comparison view + the per-run RUN_NOTES.md as your evidence.

---

## When to update the master config vs. add a config override

| change | what to do |
|---|---|
| Test a new value of an existing field | CLI override or per-experiment yaml — never edit master |
| Add a NEW field that an experiment needs | edit master with the safe default, document in CONFIG_GUIDE.md |
| Found a fix the whole project should use | edit master, commit on `training-only`, push, tell the team |
| Trying something exploratory | feature branch off `training-only`, do NOT merge until validated |

The master config should reflect "the current best understanding of what's right." If you change it, that's a statement about the project, not about your experiment.

---

## Writing commit messages for the training branch

Two categories:

### Code change commits

```
<area>: <short imperative>

<longer paragraph explaining why and any tradeoffs>

<bullet list of metrics affected, if any>
```

Examples:
- `eval_harness: add stage3_cosine_top1_acc baseline metric`
- `pipeline: surface candidate_classes on FoodItem for retrieval diagnostic`
- `adapter: invalidate splits cache when held_out_classes changes` (would be — we reverted that)

### Run-result commits (registry updates)

```
registry: <run_name> result

<one-line summary of the change tested + dev/combined>
<one-paragraph interpretation>
```

Example:
```
registry: trial-20260425-cleandata1-rerun result

First Phase 3 baseline rerun on the cleaned codebase. dev/combined=1.94, all 4
audit-fix predicted deltas hit. Train stage3_episode_acc dropped from 0.93 to
0.62 confirming the leak fix worked.
```

---

## What goes in the registry, what goes in TB, what goes in markdown

| info | registry.jsonl | TensorBoard | markdown notes |
|---|---|---|---|
| Headline metrics (best dev/*) | ✅ source of truth | ✅ | reference only |
| Loss curves | ❌ | ✅ source of truth | reference only |
| Hyperparameters (config snapshot) | ✅ | ✅ (text card) | summary only |
| Run identity (name, git sha, dataset hash) | ✅ source of truth | ✅ (text card) | reference only |
| Why we did it | ❌ | ❌ | ✅ source of truth |
| Interpretation, next steps | ❌ | ❌ | ✅ source of truth |
| Confusion matrices, per-class breakdowns | ❌ | ✅ (added later) | summary only |

If any of those sources contradicts another, **registry.jsonl wins** for numbers, **markdown wins** for narrative. TB is for live viewing only — never the durable record.

---

## How to read the diagnostic metrics

The four metrics added in the last refactor let you decompose any classification result:

```
dev/stage3_acc                       = 0.50    ← end-to-end (existing)
dev/stage3_retrieval_recall@K        = 0.78    ← was GT in cosine top-K?
dev/stage3_acc_given_retrieved       = 0.64    ← of items where it was, did transformer pick it?
dev/stage3_cosine_top1_acc           = 0.55    ← would skipping the transformer have worked?
dev/stage3_transformer_lift_over_top1 = +0.18  ← transformer adds vs cosine top-1
```

Three diagnostic patterns:

### Healthy
```
recall@K = 0.85
cosine_top1_acc = 0.55
acc_given_retrieved = 0.78
lift = +0.20
```
Transformer adds real value. Architecture justified. Continue training.

### Transformer is dead weight
```
recall@K = 0.85
cosine_top1_acc = 0.62
acc_given_retrieved = 0.65
lift = +0.03
```
Transformer barely helps. Strong case for replacing Stage 3 with cosine k-NN. Try DINOv2 retriever.

### Retriever is the bottleneck
```
recall@K = 0.45
cosine_top1_acc = 0.30
acc_given_retrieved = 0.85
lift = N/A (transformer can't beat what retrieval doesn't surface)
```
Embedding can't find GT in top-K. Swap retriever to DINOv2 / CLIP-large. No amount of Stage 3 training will help here.

---

## What to NEVER do

- **Don't run experiments without setting `run.name`**. If two runs share a name, one's logs overwrite the other.
- **Don't tune metrics on the test split.** Test is reserved for the final report.
- **Don't manually edit `events.jsonl` or `run_metadata.json`**. They're append-only / immutable per run.
- **Don't delete log directories** unless you've appended to the registry first. Once they're gone, the registry's row is the only record.
- **Don't compare across different `dataset_hash` values** without flagging it. The data changed; numbers aren't apples-to-apples.

---

## When in doubt

Run the smoke first (`python scripts/smoke_phase3.py`). If it doesn't pass, your eval numbers are not trustworthy and shouldn't be added to the registry.
