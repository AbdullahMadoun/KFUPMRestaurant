# GitHub Publishing

This snapshot is ready for GitHub publication, but one artifact requires special
handling:

- `weights/best_checkpoint.tar` is about 555 MB and must use Git LFS or a
  release asset workflow.

## Recommended Publishing Flow

### 1. Initialize Git

```bash
git init
git add .gitattributes .gitignore
git add .
git status
```

### 2. Enable Git LFS Before the First Commit

```bash
git lfs install
git lfs track "weights/*.tar"
```

The repository already includes `.gitattributes` entries for the large retained
checkpoint formats.

### 3. Review What Will Be Published

Check the following before the first push:

- no local dataset copies were added
- no `__pycache__/`, virtualenv, or editor folders are staged
- no restored checkpoint payloads under `checkpoints/.../epoch_*` are staged
- the only heavyweight binary intended for the repo is `weights/best_checkpoint.tar`

## What This Repo Intentionally Keeps

- source code and config
- tests and validation helpers
- curated logs and generated reports
- one packaged best checkpoint
- research and resume documentation

## What Should Stay Out Of Git

- reviewed export dataset
- `Sampled_Images_All/`
- local cache directories
- ad hoc restored checkpoint payloads
- machine-specific notebooks or scratch outputs unless intentionally curated

## If You Do Not Want Git LFS

If you prefer a lighter GitHub repo:

1. remove `weights/best_checkpoint.tar` from the initial commit
2. keep `weights/CHECKPOINT_PROVENANCE.md`
3. upload the checkpoint as a GitHub Release asset or external download
4. update `README.md` to point to that release URL

## Suggested First Commit Layout

Use one initial commit for the repo contents and one follow-up commit for any
post-publication cleanup. That keeps the archival snapshot easy to audit.

Example:

```bash
git commit -m "Initial GitHub-ready TriFoodNet research snapshot"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Final Checklist

- `README.md` reads cleanly on GitHub
- `docs/README.md` and `docs/REPOSITORY_MAP.md` make navigation obvious
- Git LFS is enabled before pushing
- no secrets or local paths were added during cleanup
- dataset requirements are stated clearly so users do not mistake this snapshot
  for a fully self-contained benchmark release
