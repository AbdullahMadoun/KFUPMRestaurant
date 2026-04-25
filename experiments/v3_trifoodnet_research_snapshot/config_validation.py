# =============================================================================
# FILE: config_validation.py
# CATEGORY: UTIL
# PURPOSE: Fail-fast assertions on a loaded config so a misconfigured run errors
#          out at load time instead of after hours of training.
# DEPENDENCIES: config_loader.py (DotDict shape only — duck-typed)
# USED BY: config_loader.py (called automatically from load_config), train_joint.py
# KEY CLASSES/FUNCTIONS: ConfigValidationError, validate_config
# =============================================================================
"""
Config-time validation for TriFoodNet runs.

The goal is "fail in seconds, not in hours". Each check below targets a
realistic past-or-likely-future failure mode:

- ratios that don't sum to 1.0 → silent split mis-allocation
- num_classes < dataset's compact_id range → missized classifier head
- k_shot > smallest train class → episodic sampling falls into replacement mode
                                  unless explicitly allowed
- batches_per_epoch < grad_accum → optimizer never steps
- determinism mode typo → silently runs in 'loose' mode

The function does NOT mutate the config; callers wrap it after load_config
and either log or raise on the returned issue list.
"""

from __future__ import annotations

from typing import List, Optional


class ConfigValidationError(Exception):
    """Raised when one or more config-validation checks fail."""

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__("Config validation failed:\n  - " + "\n  - ".join(errors))


def validate_config(cfg, *, dataset_class_count: Optional[int] = None,
                     dataset_min_train_class_size: Optional[int] = None) -> List[str]:
    """Run a battery of static checks on the config.

    Parameters
    ----------
    cfg : DotDict-like config object
    dataset_class_count : if known, the maximum compact_id+1 from the dataset adapter.
                          When provided, asserts cfg.data.num_classes >= this value.
    dataset_min_train_class_size : minimum count of any class in the train split.
                                   Used to gate k_shot.

    Returns the list of warning messages collected. Raises ConfigValidationError
    on any *fatal* check failure.
    """
    fatal: List[str] = []
    warn: List[str] = []

    # ── ratios sum to 1 ───────────────────────────────────────────────────
    integration = _safe(cfg, "data", "integration")
    if integration is not None:
        train_r = float(_safe(integration, "train_ratio", default=0.0) or 0.0)
        dev_r = float(_safe(integration, "dev_ratio", default=_safe(integration, "val_ratio", default=0.0)) or 0.0)
        test_r = float(_safe(integration, "test_ratio", default=0.0) or 0.0)
        total = train_r + dev_r + test_r
        if not (0.999 <= total <= 1.001):
            fatal.append(
                f"data.integration.{{train,dev,test}}_ratio sums to {total:.4f} — must be 1.0 "
                f"(train={train_r}, dev={dev_r}, test={test_r}). val_ratio is a legacy alias "
                f"for dev_ratio; if both are set they must agree or one must be 0."
            )
        # If both val_ratio and dev_ratio are non-zero and disagree, flag.
        explicit_val = float(_safe(integration, "val_ratio", default=0.0) or 0.0)
        explicit_dev = float(_safe(integration, "dev_ratio", default=0.0) or 0.0)
        if explicit_val and explicit_dev and abs(explicit_val - explicit_dev) > 1e-6:
            warn.append(
                f"data.integration.val_ratio ({explicit_val}) and dev_ratio ({explicit_dev}) "
                "both set with different values; dev_ratio takes precedence."
            )

    # ── num_classes covers compact_id range ──────────────────────────────
    cfg_num_classes = int(_safe(cfg, "data", "num_classes", default=0) or 0)
    if dataset_class_count is not None and cfg_num_classes < dataset_class_count:
        fatal.append(
            f"data.num_classes={cfg_num_classes} is smaller than the dataset's class count "
            f"({dataset_class_count}). The classifier head would be too small. "
            "Update master_config.yaml: data.num_classes."
        )

    # ── k_shot vs min train class size ───────────────────────────────────
    k_shot_train = int(_safe(cfg, "stage3", "episode", "k_shot", default=0) or 0)
    k_shot_eval = int(_safe(cfg, "stage3", "eval", "k_shot", default=k_shot_train) or k_shot_train)
    allow_replacement = bool(_safe(cfg, "stage3", "episode", "allow_replacement", default=True))
    if dataset_min_train_class_size is not None and not allow_replacement:
        if k_shot_train > dataset_min_train_class_size:
            fatal.append(
                f"stage3.episode.k_shot={k_shot_train} > smallest train class "
                f"({dataset_min_train_class_size}). Set stage3.episode.allow_replacement: true "
                "or reduce k_shot, or drop tail classes from the dataset."
            )
        if k_shot_eval > dataset_min_train_class_size:
            warn.append(
                f"stage3.eval.k_shot={k_shot_eval} > smallest train class "
                f"({dataset_min_train_class_size}); episodic eval will sample with replacement."
            )

    # ── effective optimizer step count is at least 1 ─────────────────────
    bs = int(_safe(cfg, "joint", "training", "batch_size", default=1) or 1)
    accum = int(_safe(cfg, "joint", "training", "grad_accum_steps", default=1) or 1)
    max_batches = int(_safe(cfg, "joint", "training", "max_batches_per_epoch", default=0) or 0)
    if accum < 1:
        fatal.append("joint.training.grad_accum_steps must be >= 1")
    if 0 < max_batches < accum:
        fatal.append(
            f"joint.training.max_batches_per_epoch={max_batches} < grad_accum_steps={accum}; "
            "the optimizer would never step. Increase max_batches or reduce accum."
        )

    # ── determinism mode is recognized ───────────────────────────────────
    mode = str(_safe(cfg, "run", "determinism_mode", default="deterministic") or "deterministic")
    if mode not in ("loose", "deterministic", "strict"):
        fatal.append(
            f"run.determinism_mode={mode!r} is invalid. Use 'loose', 'deterministic', or 'strict'."
        )

    # ── adapter or batch_root must exist ─────────────────────────────────
    if integration is not None:
        adapter_cfg = _safe(integration, "adapter")
        adapter_kind = str(_safe(adapter_cfg, "kind", default="") or "")
        batch_root = str(_safe(integration, "batch_root", default="") or "")
        if not adapter_kind and not batch_root:
            fatal.append(
                "data.integration must set either `adapter.kind` (preferred) or `batch_root`."
            )

    # ── per-stage learning rates exist when joint cfg references them ────
    joint_per_stage = _safe(cfg, "joint", "training", "per_stage_lr")
    if joint_per_stage is not None:
        for stage_name in ("stage1", "stage2", "stage3"):
            lr = _safe(joint_per_stage, stage_name)
            if lr is None:
                warn.append(
                    f"joint.training.per_stage_lr is set but {stage_name} entry is missing; "
                    "that stage will fall back to the global joint lr."
                )

    if fatal:
        raise ConfigValidationError(fatal)
    return warn


def _safe(obj, *path, default=None):
    """Safe nested attribute/key access: returns ``default`` if any segment is missing."""
    cur = obj
    for key in path:
        if cur is None:
            return default
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
            continue
        return default
    return cur if cur is not None else default
