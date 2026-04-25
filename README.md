# TriFoodNet — `training-only` branch

This branch contains **only what you need to train**. No legacy V1/V2 code, no inference scripts, no demo assets. Pull this branch on any GPU server and you can launch a training run with one command.

## Where to look first

All training code, configs, and docs live in:

```
experiments/v3_trifoodnet_research_snapshot/
```

Open these in order:

1. **`experiments/v3_trifoodnet_research_snapshot/TRAIN_GUIDE.md`** — the deterministic, step-by-step guide for getting a run going. **An LLM agent should read this first and follow it literally.**
2. **`experiments/v3_trifoodnet_research_snapshot/CONFIG_GUIDE.md`** — what every config field does and which ones to override per experiment.
3. **`experiments/v3_trifoodnet_research_snapshot/EXPERIMENT_GUIDE.md`** — how to name runs, document changes, and read TensorBoard.
4. **`experiments/v3_trifoodnet_research_snapshot/SECRETS.md`** — HuggingFace token handling, gitignored `.env` workflow.

## One-command quick start

```bash
# --single-branch keeps clone size small (skips main's heavy history)
git clone --single-branch --branch training-only https://github.com/AbdullahMadoun/KFUPMRestaurant.git
cd KFUPMRestaurant/experiments/v3_trifoodnet_research_snapshot
pip install -r requirements.txt
# Set HF token (one-time per machine; see SECRETS.md)
echo 'HF_TOKEN=hf_your_token_here' > .env && chmod 600 .env
# Point at your dataset (or set TRIFOODNET_DATASET_DIR env var)
export TRIFOODNET_DATASET_DIR=/path/to/v3_2026-04-24_61ac038c
python scripts/smoke_phase3.py    # ~5s, must show "All checks green"
python -m train_joint              # launches training with master_config.yaml
```

Plus, in another terminal:

```bash
tensorboard --logdir logs/
```

## What this branch is

- The complete TriFoodNet pipeline: Qwen 2.5-VL (Stage 1) → SAM 3 (Stage 2) → PictSure-style ICL transformer (Stage 3)
- All four Tier-0 audit fixes applied (query-in-support leak, best-checkpoint mismatch, silent NaN, NMS off)
- Phase-1 dataset adapter for the v3 export format
- Phase-2 trustworthy base: deterministic seed, single eval harness, per-stage learning rates, config validation, full provenance metadata
- Retrieval-vs-transformer diagnostic metrics built into eval
- TensorBoard wired in as the canonical result tracker
- Vast.ai launcher scripts that pull data from Google Drive and start training in tmux

## What this branch is not

- It is **not** a place to add new exploratory features, demos, or one-off scripts. Those go on a feature branch off `main`.
- It is **not** the inference repo. Production inference belongs in a separate deploy branch.
- It does **not** contain the dataset. Dataset is hosted on Google Drive and pulled by `scripts/vast/00b_pull_dataset_from_drive.sh` directly to the training machine.

## Where things live

| What | Where |
|---|---|
| Training entry point | `experiments/v3_trifoodnet_research_snapshot/train_joint.py` |
| Master config | `experiments/v3_trifoodnet_research_snapshot/master_config.yaml` |
| Local pre-flight smoke | `experiments/v3_trifoodnet_research_snapshot/scripts/smoke_phase3.py` |
| Vast.ai launcher | `experiments/v3_trifoodnet_research_snapshot/scripts/vast/` |
| Run comparison | `experiments/v3_trifoodnet_research_snapshot/scripts/compare_runs.py` |
| Experiment registry | `experiments/v3_trifoodnet_research_snapshot/experiments/registry.jsonl` |
| Contract tests | `experiments/v3_trifoodnet_research_snapshot/tests/` |

## Branch policy

- `training-only` is a long-lived branch. Bug fixes and improvements to the training loop land here.
- New features go on `feature/<name>` branches off `training-only`, then PR back.
- The branch must always pass `python scripts/smoke_phase3.py`. If it doesn't, fix that first.
- Update `EXPERIMENT_GUIDE.md` whenever the run / docs / metric naming convention changes.

## Asking for help

If something is unclear, the hierarchy is:
1. The four guide MDs above
2. The committed history (`git log --oneline experiments/v3_trifoodnet_research_snapshot/`)
3. `experiments/v3_trifoodnet_research_snapshot/PHASE3_EXPECTATIONS.md` — locked predictions for the baseline rerun
4. `experiments/v3_trifoodnet_research_snapshot/experiments/registry.jsonl` — every previous run's headline metrics

Ping the team only after all four come up dry.
