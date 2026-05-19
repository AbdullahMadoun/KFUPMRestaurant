# =============================================================================
# FILE: determinism.py
# CATEGORY: TRAIN
# PURPOSE: Single entry point for seeding torch/numpy/python/CUDA and producing
#          deterministic-mode DataLoader generators + worker init functions.
# DEPENDENCIES: None (torch + numpy at runtime)
# USED BY: train_joint.py
# KEY CLASSES/FUNCTIONS: SeedConfig, set_seed, dataloader_generator, worker_init_fn
# =============================================================================
"""
Determinism helpers.

Three modes:

- ``loose``         : default PyTorch behavior. cuDNN benchmark on. Fast, non-reproducible.
- ``deterministic`` : cuDNN deterministic + benchmark off. ~10-20% slower.
                      Reasonable default for research runs.
- ``strict``        : everything in deterministic + ``torch.use_deterministic_algorithms(True)``.
                      Some ops will fail if no deterministic implementation exists.
                      Use for paper-grade reproducibility, expect throughput hit.

Usage in train scripts:

    from determinism import set_seed, dataloader_generator, worker_init_fn

    seed_info = set_seed(cfg.run.seed, mode=cfg.run.determinism_mode)
    logger.log("seed", seed_info, ...)

    DataLoader(
        ...,
        generator=dataloader_generator(seed_info["seed"]),
        worker_init_fn=worker_init_fn(seed_info["seed"]),
    )
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional


VALID_MODES = ("loose", "deterministic", "strict")


@dataclass
class SeedConfig:
    seed: int
    mode: str
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    use_deterministic_algorithms: bool

    def to_dict(self) -> Dict[str, object]:
        return {f"system/seed_{k}": v for k, v in asdict(self).items()}


def set_seed(seed: Optional[int] = None, mode: str = "deterministic") -> Dict[str, object]:
    """Seed all RNG sources and configure cuDNN/torch determinism.

    Returns a flat dict of seed-related metadata suitable for logging.
    The actual effective seed (after substituting a default if seed is None) is
    captured in the returned dict so downstream callers and ``run_metadata.json``
    record what really ran, not what was requested.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown determinism mode {mode!r}; expected one of {VALID_MODES}")

    if seed is None:
        # If env var PYTHONHASHSEED is set, prefer that for cross-process consistency.
        env_seed = os.environ.get("TRIFOODNET_SEED") or os.environ.get("PYTHONHASHSEED")
        seed = int(env_seed) if env_seed and env_seed.isdigit() else 1337
    seed = int(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    cudnn_det = mode in ("deterministic", "strict")
    cudnn_bench = mode == "loose"
    use_det_algos = mode == "strict"

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = cudnn_det
            torch.backends.cudnn.benchmark = cudnn_bench
        if use_det_algos and hasattr(torch, "use_deterministic_algorithms"):
            # CUBLAS workspace config required for fully deterministic matmul on CUDA.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=False)
    except ImportError:
        pass

    cfg = SeedConfig(
        seed=seed,
        mode=mode,
        cudnn_deterministic=cudnn_det,
        cudnn_benchmark=cudnn_bench,
        use_deterministic_algorithms=use_det_algos,
    )
    return cfg.to_dict()


def dataloader_generator(seed: int):
    """Return a torch.Generator seeded with the given seed.

    Pass to ``DataLoader(..., generator=dataloader_generator(seed))`` so shuffle
    order is reproducible. Returns ``None`` if torch is unavailable so this
    module remains importable in tooling-only contexts.
    """
    try:
        import torch
    except ImportError:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Build a worker_init_fn that seeds each DataLoader worker deterministically.

    Without this, ``num_workers > 0`` makes runs non-reproducible even with a
    pinned base seed because each worker has its own Python and numpy RNG.
    """
    base_seed = int(base_seed)

    def _init(worker_id: int) -> None:
        worker_seed = (base_seed + int(worker_id)) % (2**31 - 1)
        random.seed(worker_seed)
        try:
            import numpy as np
            np.random.seed(worker_seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(worker_seed)
        except ImportError:
            pass

    return _init
