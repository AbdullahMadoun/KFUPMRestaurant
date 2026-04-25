# Secrets handling

**This file is gitignored. Do not commit. Do not paste real values into Slack / PRs / docs.**

## What lives where

| Secret | Local path | Remote path (vast.ai) | Used by |
|---|---|---|---|
| HuggingFace token | `.env` (mode 600) | `/root/.hf_env` (mode 600) | SAM3 + gated HF model downloads |

## Where to rotate / regenerate

- **HuggingFace token** → https://huggingface.co/settings/tokens
  - Type: **Read**
  - Account must be **approved for `facebook/sam3`** (gated: manual; Meta reviews requests in hours-to-days)
  - Without SAM3 approval, training crashes at `Sam3Model.from_pretrained` with HTTP 403

## Workflow

### Set or rotate the local token

```bash
cd /Users/abdulrazzak/MADOUN_PIPELINE/KFUPMRestaurant/experiments/v3_trifoodnet_research_snapshot
cat > .env <<'EOF'
HF_TOKEN=hf_REDACTED_PASTE_NEW_VALUE_HERE
EOF
chmod 600 .env
```

### Push to a fresh vast.ai instance

`scripts/vast/03_run_remote.sh` reads `.env` and pushes `HF_TOKEN` to the
instance at `/root/.hf_env`. The training tmux session is started with
`tmux -e HF_TOKEN=...` so the env var reaches the python process directly
(sourcing inside tmux doesn't always propagate; the `-e` flag is reliable).

### Verify token is live on a running instance

```bash
ssh -p $SSH_PORT root@$SSH_HOST 'source /root/.hf_env && python3 -c "
from huggingface_hub import HfApi
api = HfApi()
print(\"whoami:\", api.whoami()[\"name\"])
print(\"sam3 metadata reachable:\", api.model_info(\"facebook/sam3\").id)
"'
```

`whoami` returning a username confirms the token is valid. **But** `model_info`
succeeding does NOT prove SAM3 download access — check actual file access:

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
    -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/facebook/sam3/resolve/main/config.json"
```

200 → approved, training will work.
403 → license accepted but Meta hasn't approved yet (or token lacks read scope).
401 → token invalid.

## Why this pattern

- `.env` is a single source of truth on the laptop
- `03_run_remote.sh` propagates from `.env` → remote `/root/.hf_env` → tmux env
- `.bashrc` sources `/root/.hf_env` so any interactive ssh session also has it
- `tmux -e HF_TOKEN=...` is the belt-and-suspenders for the training process
- Nothing about secrets ends up in `events.jsonl`, `run_metadata.json`, or commits

## What to do if a token leaks

1. Revoke immediately at https://huggingface.co/settings/tokens
2. Generate a new one
3. Re-run the "Set or rotate the local token" steps above
4. Re-run training; new instances pick up the new token automatically

## Token rotation history (no values, just context)

| Date | Account | Why rotated |
|---|---|---|
| 2026-04-25 | Razak111 | initial token; not yet SAM3-approved → cleandata1 trainer's account substituted |
