# TRAIN_GUIDE — deterministic agent guide

**This guide is read by both humans and LLM agents. It must be followed in order. Do not skip steps.** If any step fails, stop and report — do not improvise around it.

---

## What you are doing

You are launching a TriFoodNet training run. The codebase is a 3-stage pipeline:

```
Qwen 2.5-VL (Stage 1)  →  SAM 3 (Stage 2)  →  PictSure ICL transformer (Stage 3)
   detection + labels         segmentation             few-shot classification
```

Trained jointly with per-stage losses (LM, BCE+Dice, episodic CE), summed with weights `(λ1, λ2, λ3)`.

---

## Hard prerequisites

You **cannot** start training without all of:

- A GPU with **≥ 24 GB VRAM** (RTX 4090, 5090, A6000, A100, H100). Stage 2 mask decoder + Qwen LoRA + ICL transformer at `bs=1 grad_accum=8` peaks ~22 GB.
- **CUDA driver compatible with the target GPU.** RTX 5090 (sm_120, Blackwell) requires torch nightly cu128 — see `requirements.txt`. RTX 4090 (sm_89) works with stock torch ≥ 2.5.
- A **HuggingFace account approved for `facebook/sam3`** — SAM 3 is gated:manual, requires Meta approval (hours to days). Without approval, training crashes at Stage 2 model load with HTTP 403. See `SECRETS.md`.
- The **v3 export dataset** accessible at the path in `master_config.yaml` (`data.integration.adapter.export_root`), or available via the Google Drive workflow in `scripts/vast/`.

---

## Step 0 — environment

```bash
cd experiments/v3_trifoodnet_research_snapshot
pip install -r requirements.txt
```

If you are on **RTX 5090**, additionally:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --upgrade --force-reinstall "bitsandbytes>=0.45"
```

Set the HuggingFace token (one-time):
```bash
cat > .env <<'EOF'
HF_TOKEN=hf_REPLACE_WITH_YOUR_TOKEN
EOF
chmod 600 .env
```

The `.env` file is gitignored. Vast scripts read it and push it to the remote automatically.

---

## Step 1 — pre-flight smoke (5 seconds, CPU only)

**This must pass before every training run.** It catches integration drift, broken adapters, missing dataset files, config validation failures, and so on. It does NOT load models.

```bash
python scripts/smoke_phase3.py
```

Expected output ends with:
```
Smoke test: NN/NN passed
All checks green. Safe to launch full training.
```

If anything fails, **STOP**. Investigate the failure. Do not proceed.

If the smoke passes but you suspect dataset path issues, override:
```bash
python scripts/smoke_phase3.py --export-root /actual/path/to/v3_export
```

---

## Step 2 — kick off training

### Option A — local GPU

```bash
python -m train_joint
```

This reads `master_config.yaml` end-to-end. Override individual fields with positional `key=value` args:
```bash
python -m train_joint joint.training.epochs=10 joint.training.max_batches_per_epoch=50 logging.wandb=false
```

### Option B — rented vast.ai instance

The full workflow is in `scripts/vast/README.md`. Short version:

```bash
bash scripts/vast/01_launch.sh                       # provisions cheapest 4090
bash scripts/vast/02_push.sh                          # pushes code only (~5 MB)
bash scripts/vast/00b_pull_dataset_from_drive.sh    # instance pulls dataset from Drive
bash scripts/vast/03_run_remote.sh                   # installs deps, runs smoke, starts training in tmux
```

Then in another terminal, monitor:
```bash
python scripts/vast/04_live_monitor.py    # local terminal dashboard refreshing every 2s
```

When done:
```bash
bash scripts/vast/05_pull.sh              # pulls logs + checkpoints back
bash scripts/vast/06_destroy.sh           # destroys instance (asks for 'yes')
```

---

## Step 3 — monitor while training runs

**Three options, use whichever you prefer.**

### A. TensorBoard (canonical — use this)

In a second terminal on the training machine:
```bash
tensorboard --logdir logs/
```

Open `http://localhost:6006/` (or via SSH tunnel if remote: `ssh -L 6006:localhost:6006 root@<host>`).

You'll see scalars updating in real time:
- `train_step/loss_*`, `train_step/stage3_acc`, `train_step/lr`, `train_step/grad_norm`
- `train/nan_*` and `train/episode_leak_fallback_total` per epoch
- `dev/stage1_recall@0.5`, `dev/stage2_mIoU`, `dev/stage3_acc`, `dev/combined`
- The diagnostic trio: `dev/stage3_retrieval_recall@K`, `dev/stage3_acc_given_retrieved`, `dev/stage3_cosine_top1_acc`, `dev/stage3_transformer_lift_over_top1`

### B. Live monitor script (vast.ai only)

```bash
python scripts/vast/04_live_monitor.py
```

ANSI table refreshing every 2 seconds. Shows latest train_step losses, NaN counters, leak-fallback total, dev metrics when they land.

### C. Raw stdout (debugging only)

```bash
ssh -t -p $PORT root@$HOST 'tmux attach -t train'
```

Detach with Ctrl-b then d. Training keeps running.

---

## Step 4 — interpret the headline numbers

After training finishes, the comparison script gives you a one-shot read:

```bash
python scripts/compare_runs.py logs/<old-run>/joint logs/<new-run>/joint
```

It prints a side-by-side table for the predicted-deltas matrix from `PHASE3_EXPECTATIONS.md` and exits with `[GATE] PASS|SOFT|FAIL`.

The metrics that matter (in priority order):

| metric | what it measures | target |
|---|---|---|
| `dev/combined` | headline (stage1_recall + stage2_mIoU + stage3_acc) | as high as possible |
| `dev/stage3_acc` | end-to-end classification accuracy | ≥ 0.50 baseline |
| `dev/stage3_retrieval_recall@K` | did cosine sim find GT in top-K? | ≥ 0.85 |
| `dev/stage3_cosine_top1_acc` | "transformer-removed" baseline | compare against `stage3_acc` |
| `dev/stage3_transformer_lift_over_top1` | how much the transformer adds | should be > 0 |
| `train/nan_*` | health check | exactly 0 |
| `train/episode_leak_fallback_total` | health check | small (≤ tail-class count) |

See `EXPERIMENT_GUIDE.md` for how to read these in combination.

---

## Step 5 — record the run

```bash
python scripts/registry_append.py logs/<run-name>/joint
```

Appends one line to `experiments/registry.jsonl` with: run_name, dataset_version, git SHA, all headline metrics, NaN counts, leak fallback. This is the **only** record of "what got tried, when, with what numbers." Maintain it religiously.

---

## What you must NOT change without coordination

- **`master_config.yaml`** — change ONLY by adding override files in `configs/<experiment>.yaml` (Phase 4) or via CLI overrides. Direct edits to the master file affect everyone.
- **`experiment_logging.py`** — the metric schema is contractual. Adding fields is fine; renaming or removing breaks the registry.
- **`eval_harness.py` `combined_score`** — the formula is locked. If you change it, bump `COMBINED_FORMULA_VERSION` and document why in the commit message.
- **The four Tier-0 fixes** — query-in-support leak filter, best-by-combined checkpoint, NaN counters, NMS thresholds. Reverting any of these silently restores the old leakage and metric inflation.

---

## What you SHOULD change for new experiments

- Per-stage learning rates: `joint.training.per_stage_lr.{stage1,stage2,stage3}`
- Loss weights: `joint.loss_weights.{lambda1,lambda2,lambda3}`
- Curriculum (teacher forcing): `joint.curriculum.teacher_forcing.*`
- Number of epochs, batch size, gradient accumulation
- Run name + tags

For the full list, see `CONFIG_GUIDE.md`.

---

## Common failure modes (and the exact fix)

| symptom | cause | fix |
|---|---|---|
| `ImportError: cannot import name 'Sam3Model'` | transformers < 4.56 | `pip install --upgrade 'transformers>=4.56'` |
| `OSError: gated repo facebook/sam3` HTTP 401/403 | account not approved for SAM 3 | accept license at huggingface.co/facebook/sam3, wait for Meta to approve manually (hours-days) |
| `RuntimeError: no kernel image is available for execution` | torch wheel doesn't have GPU's compute capability | install nightly cu128 (Blackwell) or switch to 4090 |
| `ModuleNotFoundError: triton.ops` | bitsandbytes < 0.45 mismatched with new triton | `pip install --upgrade --force-reinstall 'bitsandbytes>=0.45'` |
| Smoke fails on `adapter.export_root not found` | dataset path wrong | edit master_config.yaml `data.integration.adapter.export_root` or pass `--export-root` to smoke |
| Training crashes at Stage 1 with `cannot find HF_TOKEN` | env var didn't reach tmux | use `tmux new-session -e HF_TOKEN=hf_...` (see `scripts/vast/03_run_remote.sh`) |

When in doubt, run the smoke first. It catches the integration-level problems faster than waiting for the model load to fail.

---

## When the run finishes

1. `python scripts/registry_append.py logs/<run>/joint`
2. `python scripts/compare_runs.py logs/<reference-run>/joint logs/<run>/joint`
3. Pull artifacts back to local if you ran on a rented instance: `bash scripts/vast/05_pull.sh`
4. Destroy the instance: `bash scripts/vast/06_destroy.sh`
5. Write a one-paragraph summary in `experiments/registry.jsonl` notes field (or in a separate run-card markdown).
6. If the result was notable (good or bad), update `EXPERIMENT_GUIDE.md`'s "results so far" table.

---

## Asking for help

If a step fails and the table above doesn't cover it:

1. Look at the last 30 lines of `train.stdout.log` (vast.ai) or `events.jsonl` (always).
2. Grep `git log` for keywords from the error.
3. Check `SECRETS.md` if it's auth-related.
4. Only escalate after the above three turn up nothing.
