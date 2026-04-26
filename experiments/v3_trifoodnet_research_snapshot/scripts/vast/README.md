# Vast.ai mini Phase-3 smoke

A 30-minute, ~$0.15 run on a rented RTX 5090 to validate the full integration
end-to-end on real GPU hardware. Catches model-load issues, VRAM problems, and
real-data NaN events that the local CPU smoke can't see.

## What gets checked

- Qwen2.5-VL-3B + SAM3 actually load with `peft` + `bitsandbytes`
- Forward+backward produces finite losses on real batches
- VRAM doesn't OOM at the configured batch shape
- `train/nan_*` counters and `episode_leak_fallback_total` populate from real data
- `pipeline.save()/load()` round-trips real model state
- The new `evaluate_split` produces a complete `EvalReport` end-to-end
- Per-stage LRs make it from config → `param_groups` → AdamW → log

## Workflow

```bash
# 1. Pick + launch (creates instance, ~2-3 min)
bash scripts/vast/01_launch.sh

# 2. Push code + dataset to remote (~1-5 min depending on network)
bash scripts/vast/02_push.sh

# 3. SSH in and install deps (~2 min); kick off training in tmux (~15-20 min)
bash scripts/vast/03_run_remote.sh

# 4. Live monitor (in a second terminal — refreshes every 2 sec)
python scripts/vast/04_live_monitor.py

# 5. Pull artifacts back
bash scripts/vast/05_pull.sh

# 6. Compare locally
python scripts/compare_runs.py logs/trial-20260321-cleandata1/joint logs/trial-20260425-mini-smoke/joint

# 7. Destroy instance (will prompt for confirmation)
bash scripts/vast/06_destroy.sh
```

## Files

| Script | Role |
|---|---|
| `00_state.sh` | Source this — exports `INSTANCE_ID`, `REMOTE` paths used by every other script |
| `01_launch.sh` | `vastai create instance` with the chosen offer + auto-destroy timer |
| `02_push.sh` | rsync code + v3 dataset to remote |
| `03_run_remote.sh` | SSH in, install pip deps, start training in detached tmux |
| `04_live_monitor.py` | Local Python — SSH-tails events.jsonl, renders a refreshing metrics table |
| `05_pull.sh` | rsync logs + checkpoints back |
| `06_destroy.sh` | `vastai destroy instance` with explicit confirmation |
| `99_attach_tmux.sh` | Convenience: `ssh -t <host> 'tmux attach'` for full stdout view |

## Monitoring options

You have three ways to watch the run:

1. **`04_live_monitor.py`** — local terminal, structured table refreshing every 2 sec.
   Best for "is it healthy and progressing".
2. **`99_attach_tmux.sh`** — interactive terminal on the instance.
   Best for "show me the raw stdout" or debugging.
3. **`ssh <host> 'tail -f /root/work/logs/.../events.jsonl | jq .'`** — raw event stream.
   Best for grep-style queries.

## Cost guard rails

`01_launch.sh` sets a 1-hour auto-destroy schedule on the instance. If anything
goes wrong and you walk away, the instance dies automatically. Hard cap on cost:
~$0.30. Confirm balance with `vastai show user`.

## Stage 1 Qwen standard-container flow

Use the `stage1_*` scripts for Stage 1-only Qwen training. They keep their own
`.stage1_state`, launch a standard Vast container only, and never start training
until the explicit tmux start command is run.

```bash
# 1. Pick fastest expected offer under $2.10/hr and launch a standard container.
bash scripts/vast/stage1_launch.sh

# 2. Push Stage 1 code, including requirements-stage1.txt.
bash scripts/vast/stage1_push.sh

# 3. Pull the Drive dataset on the remote container.
bash scripts/vast/stage1_pull_dataset_from_drive.sh

# 4. Install Stage 1 deps and run the dataset preflight.
bash scripts/vast/stage1_preflight.sh

# 5. Render five training previews remotely and pull them local.
bash scripts/vast/stage1_render_previews.sh

# 6. Start TensorBoard remotely and an SSH tunnel at http://127.0.0.1:6006.
bash scripts/vast/stage1_tensorboard.sh start

# 7. Start training only after inspecting previews.
bash scripts/vast/stage1_train_tmux.sh start
```

Useful overrides:

```bash
STAGE1_MAX_DPH=2.10 STAGE1_MIN_GPU_RAM_GB=48 bash scripts/vast/stage1_launch.sh
STAGE1_RUN_NAME=stage1-qwen7b-v1 STAGE1_REFERENCE_POLICY=exclude bash scripts/vast/stage1_train_tmux.sh start
STAGE1_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct bash scripts/vast/stage1_train_tmux.sh start
```

The offer picker ranks by Vast `dlperf` first, falling back to a GPU-name speed
estimate only when `dlperf` is missing. Price is used as a tie breaker, not as
the primary sort. HF tokens are not stored in these scripts; if a gated model is
used, create `/root/.stage1_hf_env` on the instance yourself or source it in the
remote shell before starting training.
