# =============================================================================
# FILE: experiment_logging.py
# CATEGORY: UTIL
# PURPOSE: Structured JSONL experiment logging, best-metric tracking, and run summaries.
# DEPENDENCIES: torch, psutil (optional), wandb (optional)
# USED BY: train_joint.py
# KEY CLASSES/FUNCTIONS: utc_now_iso, flatten_metrics, collect_system_metrics, build_run_metadata, BestMetric, ExperimentLogger
# LAST MODIFIED: 2026-03-21T12:00:37.932122+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
Structured experiment logging for research runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Dict, Optional

import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    import wandb
except ImportError:
    wandb = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float | int | str]:
    flat: Dict[str, float | int | str] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=full_key))
        elif isinstance(value, (int, float, str, bool)):
            flat[full_key] = value
    return flat


def collect_system_metrics(device: torch.device) -> Dict[str, float | int | str]:
    metrics: Dict[str, float | int | str] = {}
    if psutil is not None:
        vm = psutil.virtual_memory()
        metrics["system/cpu_percent"] = float(psutil.cpu_percent(interval=None))
        metrics["system/ram_used_gb"] = round((vm.total - vm.available) / (1024 ** 3), 3)
        metrics["system/ram_total_gb"] = round(vm.total / (1024 ** 3), 3)
    if device.type == "cuda" and torch.cuda.is_available():
        current = torch.cuda.current_device()
        metrics["gpu/index"] = int(current)
        metrics["gpu/name"] = torch.cuda.get_device_name(current)
        metrics["gpu/mem_allocated_gb"] = round(torch.cuda.memory_allocated(current) / (1024 ** 3), 3)
        metrics["gpu/mem_reserved_gb"] = round(torch.cuda.memory_reserved(current) / (1024 ** 3), 3)
        metrics["gpu/mem_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(current) / (1024 ** 3), 3)
    return metrics


def build_run_metadata(cfg, device: torch.device) -> Dict[str, Any]:
    meta = {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "cwd": os.getcwd(),
        "run_name": getattr(cfg.run, "name", "unknown"),
        "notes": getattr(cfg.run, "notes", ""),
        # ── reproducibility / determinism ─────────────────────────────────
        "seed": getattr(cfg.run, "seed", None),
        "determinism_mode": getattr(cfg.run, "determinism_mode", "deterministic"),
        # ── environment knobs that affect numeric outcomes ────────────────
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "TRIFOODNET_SEED": os.environ.get("TRIFOODNET_SEED"),
        },
        # ── git / source provenance ───────────────────────────────────────
        "git": _collect_git_state(),
        # ── package versions (full pip freeze written to a sibling file) ──
        "packages": _collect_pip_freeze_summary(),
        # ── dataset provenance (filled in by train_joint after adapter loads) ──
        "dataset": {
            "version": None,
            "hash": None,
            "export_root": None,
        },
    }
    if device.type == "cuda" and torch.cuda.is_available():
        meta["cuda"] = {
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    return meta


def _collect_git_state() -> Dict[str, Any]:
    """Capture the current commit + dirty flag without raising on non-git checkouts."""
    import subprocess
    state: Dict[str, Any] = {"sha": None, "dirty": None, "branch": None}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, cwd=os.getcwd(),
        ).strip()
        state["sha"] = sha
        diff = subprocess.run(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL, cwd=os.getcwd(),
        )
        state["dirty"] = diff.returncode != 0
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL, text=True, cwd=os.getcwd(),
            ).strip()
            state["branch"] = branch
        except subprocess.CalledProcessError:
            pass
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Not a git repo, or git not installed — leave fields as None.
        pass
    return state


def _collect_pip_freeze_summary() -> Dict[str, Any]:
    """Collect installed package versions for the most relevant deps.

    Full pip freeze is also written to ``<run_dir>/requirements_resolved.txt``
    by ExperimentLogger so that exhaustive reproduction is possible without
    bloating run_metadata.json.
    """
    summary: Dict[str, Any] = {}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "numpy", "PIL", "Pillow"):
        try:
            import importlib.metadata as md
            summary[pkg] = md.version(pkg)
        except Exception:
            summary[pkg] = None
    return summary


@dataclass
class BestMetric:
    name: str
    value: float
    step: int
    epoch: int


# --- Snapshot note: Structured metrics logger that writes JSONL, best metrics, run status, and summaries. ---
class ExperimentLogger:
    def __init__(self, root_dir: str | Path, cfg, device: torch.device):
        self.root_dir = Path(root_dir)
        self.cfg = cfg
        self.device = device
        self.start_time = time.perf_counter()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root_dir / "events.jsonl"
        self.latest_path = self.root_dir / "latest_metrics.json"
        self.best_path = self.root_dir / "best_metrics.json"
        self.status_path = self.root_dir / "run_status.json"
        self.summary_path = self.root_dir / "run_summary.md"
        self.profiles_dir = self.root_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._best_metrics: Dict[str, BestMetric] = {}
        self._latest_payload: Dict[str, Any] = {}
        self._wandb_enabled = bool(getattr(cfg.logging, "wandb", False) and wandb is not None)
        if self._wandb_enabled:
            wandb.init(
                project=cfg.logging.wandb_project,
                name=f"{cfg.run.name}-joint",
                config=cfg.to_dict(),
                tags=getattr(cfg.run, "tags", None),
            )

        self._write_json(self.root_dir / "run_metadata.json", build_run_metadata(cfg, device))
        self._write_json(self.root_dir / "config_snapshot.json", cfg.to_dict())
        self._write_pip_freeze(self.root_dir / "requirements_resolved.txt")

    def update_run_metadata(self, **fields: Any) -> None:
        """Merge fields into run_metadata.json so callers can fill in late-arriving
        provenance such as the dataset version/hash reported by the adapter.
        """
        path = self.root_dir / "run_metadata.json"
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
        for key, value in fields.items():
            if isinstance(value, dict) and isinstance(payload.get(key), dict):
                payload[key].update(value)
            else:
                payload[key] = value
        self._write_json(path, payload)

    @staticmethod
    def _write_pip_freeze(path: Path) -> None:
        """Snapshot installed packages for reproducibility. Best-effort only."""
        import subprocess
        try:
            output = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.DEVNULL, text=True,
            )
            path.write_text(output, encoding="utf-8")
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            # Pip not available or sandboxed — skip silently.
            pass

    def log(
        self,
        event_type: str,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        split: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        commit_wandb: bool = True,
        update_latest: bool = True,
    ):
        payload: Dict[str, Any] = {
            "timestamp_utc": utc_now_iso(),
            "event_type": event_type,
            "elapsed_sec": round(time.perf_counter() - self.start_time, 3),
        }
        if step is not None:
            payload["step"] = int(step)
        if epoch is not None:
            payload["epoch"] = int(epoch)
        if split is not None:
            payload["split"] = split

        flat_metrics = flatten_metrics(metrics)
        payload.update(flat_metrics)
        payload.update(collect_system_metrics(self.device))
        if extra:
            payload.update(flatten_metrics(extra, prefix="extra"))

        with open(self.events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

        if update_latest:
            self._latest_payload = payload
            self._write_json(self.latest_path, payload)

        if self._wandb_enabled:
            wandb.log({k: v for k, v in payload.items() if isinstance(v, (int, float))}, step=step, commit=commit_wandb)

    def update_best(self, name: str, value: float, step: int, epoch: int):
        current = self._best_metrics.get(name)
        if current is None or value > current.value:
            self._best_metrics[name] = BestMetric(name=name, value=float(value), step=step, epoch=epoch)
            self._write_json(
                self.best_path,
                {metric_name: vars(metric) for metric_name, metric in self._best_metrics.items()},
            )
            return True
        return False

    def record_checkpoint(self, checkpoint_path: str | Path, metrics: Dict[str, Any], *, step: int, epoch: int, is_best: bool = False):
        payload = {
            "checkpoint_path": str(checkpoint_path),
            "is_best": bool(is_best),
            "metrics": flatten_metrics(metrics),
        }
        self.log("checkpoint", payload, step=step, epoch=epoch, split="checkpoint", update_latest=False)

    def record_profile(self, trace_path: str | Path, description: str):
        self.log(
            "profile",
            {"trace_path": str(trace_path), "description": description},
            split="profile",
            update_latest=False,
        )

    def close(self, status: str = "completed"):
        finished_utc = utc_now_iso()
        elapsed_sec = round(time.perf_counter() - self.start_time, 3)
        self._write_json(
            self.status_path,
            {
                "status": status,
                "run_name": getattr(self.cfg.run, "name", "unknown"),
                "finished_utc": finished_utc,
                "elapsed_sec": elapsed_sec,
            },
        )
        summary_lines = [
            "# Run Summary",
            "",
            f"- status: {status}",
            f"- run_name: {getattr(self.cfg.run, 'name', 'unknown')}",
            f"- finished_utc: {finished_utc}",
            f"- elapsed_sec: {elapsed_sec}",
            "",
            "## Best Metrics",
        ]
        if self._best_metrics:
            for name, metric in sorted(self._best_metrics.items()):
                summary_lines.append(
                    f"- {name}: {metric.value:.6f} at epoch {metric.epoch}, step {metric.step}"
                )
        else:
            summary_lines.append("- none")
        self.summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        if self._wandb_enabled:
            wandb.finish()

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
