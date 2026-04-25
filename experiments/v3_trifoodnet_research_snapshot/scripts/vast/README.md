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
