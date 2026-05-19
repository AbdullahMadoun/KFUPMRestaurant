#!/usr/bin/env python3
# =============================================================================
# FILE: scripts/vast/04_live_monitor.py
# CATEGORY: TEST
# PURPOSE: Local-side live dashboard for the remote training run. SSH-tails
#          events.jsonl, parses each new event, and renders a refreshing
#          terminal dashboard with key metrics + health signals.
# DEPENDENCIES: stdlib only (uses ssh subprocess, no fancy libs)
# USAGE:
#   python scripts/vast/04_live_monitor.py
#   # Reads .state for SSH host/port + RUN_NAME automatically.
# EXIT: Ctrl-C cleanly tears down the SSH child.
# =============================================================================
"""Live dashboard.

What you see while training runs:

    ┌─ trial-...-mini-smoke ────────────── elapsed 02m 14s ─┐
    │ status: training (tmux 'train' alive)                 │
    │ epoch: 3/5    step: 18    lr: 2.0e-5/5.0e-5/1.0e-4    │
    │ train/loss_total ▼ 5.42 → 4.10 (-1.32)                │
    │ train/stage1_lm  ▼ 1.83 → 1.19                        │
    │ train/stage2     ▼ 1.21 → 0.95                        │
    │ train/stage3     ▼ 2.38 → 1.96                        │
    │ ────────────── health ────────────────────             │
    │ NaN counters     ▶ 0 / 0 / 0 / 0 / 0  (stage1/2/2int/3/total) │
    │ leak fallback    ▶ 0                                   │
    │ ────────────── eval (when present) ───────             │
    │ best dev/combined: — (no dev eval yet)                │
    └──────────────────────────────────────────────────────┘

Press Ctrl-C to exit; training continues unaffected.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_FILE = SCRIPT_DIR / ".state"
REFRESH_SECONDS = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# State + SSH wiring
# ──────────────────────────────────────────────────────────────────────────────


def load_state() -> Dict[str, str]:
    if not STATE_FILE.exists():
        sys.exit(f"[monitor] FAIL: {STATE_FILE} not found. Run 01_launch.sh first.")
    out: Dict[str, str] = {}
    for line in STATE_FILE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    if not out.get("SSH_HOST"):
        sys.exit("[monitor] FAIL: SSH_HOST missing from .state")
    return out


def remote_run_name() -> str:
    """Read the run_name we exported in 00_state.sh."""
    # Bash 00_state.sh writes RUN_NAME to env when sourced. Recompute the same way.
    state_sh = SCRIPT_DIR / "00_state.sh"
    out = subprocess.check_output(
        ["bash", "-c", f"source {state_sh}; echo $RUN_NAME"],
        text=True,
    ).strip()
    if not out:
        sys.exit("[monitor] FAIL: could not derive RUN_NAME from 00_state.sh")
    return out


def remote_log_path(run_name: str) -> str:
    state_sh = SCRIPT_DIR / "00_state.sh"
    out = subprocess.check_output(
        ["bash", "-c", f"source {state_sh}; echo $REMOTE_LOGS/{run_name}/joint/events.jsonl"],
        text=True,
    ).strip()
    return out


def open_ssh_tail(state: Dict[str, str], remote_path: str) -> subprocess.Popen:
    """Stream events.jsonl from the instance line by line. We use `tail -F`
    (capital F) so the tail survives the file being created late."""
    cmd = [
        "ssh", "-p", state.get("SSH_PORT", "22"),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ServerAliveInterval=15",
        f"root@{state['SSH_HOST']}",
        # `tail -F` retries when file is created or renamed
        f"tail -F -n +1 {remote_path} 2>/dev/null",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)


def ssh_check_tmux_alive(state: Dict[str, str]) -> bool:
    cmd = [
        "ssh", "-p", state.get("SSH_PORT", "22"),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=5",
        f"root@{state['SSH_HOST']}",
        "tmux has-session -t train 2>/dev/null && echo alive || echo dead",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10).strip()
        return out == "alive"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Event aggregation
# ──────────────────────────────────────────────────────────────────────────────


class RunningState:
    """Aggregates the latest values seen for the metrics we display."""

    def __init__(self):
        self.start_ts: Optional[float] = None
        self.latest_event_type: str = "—"
        self.latest_step: Optional[int] = None
        self.latest_epoch: Optional[int] = None
        self.lr_per_stage: Dict[str, Optional[float]] = {"stage1": None, "stage2": None, "stage3": None}
        self.loss_history: Deque[Dict[str, float]] = deque(maxlen=2)
        self.last_train_metrics: Dict[str, float] = {}
        self.last_dev_metrics: Dict[str, float] = {}
        self.best_combined: Optional[float] = None
        self.best_combined_epoch: Optional[int] = None
        self.nan_counts: Dict[str, int] = {"stage1": 0, "stage2": 0, "stage2_internal": 0, "stage3": 0, "total": 0}
        self.leak_fallback_total: int = 0
        self.events_seen: int = 0
        self.last_event_ts: Optional[float] = None

    def consume(self, event: Dict[str, Any]) -> None:
        self.events_seen += 1
        self.last_event_ts = time.time()
        et = event.get("event_type")
        if et:
            self.latest_event_type = et
        if "step" in event and event["step"] is not None:
            self.latest_step = int(event["step"])
        if "epoch" in event and event["epoch"] is not None:
            self.latest_epoch = int(event["epoch"])
        if self.start_ts is None and "elapsed_sec" in event:
            self.start_ts = time.time() - float(event["elapsed_sec"])

        # Per-stage LR (logged at run_start)
        for k in ("optimizer/stage1_lr", "optimizer/stage2_lr", "optimizer/stage3_lr"):
            if k in event:
                stage = k.split("/")[1].split("_")[0]
                self.lr_per_stage[stage] = float(event[k])

        # Train-step losses
        train_loss_keys = (
            "train_step/loss_stage1", "train_step/loss_stage2",
            "train_step/loss_stage3", "train_step/loss_total",
            "train_step/stage1_lm_loss", "train_step/stage2_bce_loss",
            "train_step/stage2_dice_loss", "train_step/stage3_ce_loss",
            "train_step/stage3_acc",
        )
        if any(k in event for k in train_loss_keys):
            this = {k: float(event[k]) for k in train_loss_keys if k in event}
            self.last_train_metrics.update(this)
            # Track delta from prior step
            if self.loss_history:
                self.loss_history.append(this)
            else:
                self.loss_history.append(this)

        # NaN counters (logged on train_epoch_end)
        if et == "train_epoch_end":
            for k in ("train/nan_stage1", "train/nan_stage2", "train/nan_stage2_internal",
                      "train/nan_stage3", "train/nan_total"):
                if k in event:
                    short = k.split("nan_")[1]
                    self.nan_counts[short] = int(float(event[k]))
            if "train/episode_leak_fallback_total" in event:
                self.leak_fallback_total = int(float(event["train/episode_leak_fallback_total"]))

        # Dev eval
        if et == "eval_epoch":
            for k, v in event.items():
                if k.startswith("dev/") and isinstance(v, (int, float)):
                    self.last_dev_metrics[k] = float(v)
            combined = event.get("dev/combined")
            if combined is not None and (self.best_combined is None or combined > self.best_combined):
                self.best_combined = float(combined)
                self.best_combined_epoch = self.latest_epoch


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────


def fmt_loss_line(label: str, history: Deque[Dict[str, float]], key: str) -> str:
    if not history:
        return f"{label:<22} —"
    cur = history[-1].get(key)
    if cur is None:
        return f"{label:<22} —"
    if len(history) >= 2:
        prev = history[-2].get(key)
        if prev is not None:
            delta = cur - prev
            arrow = "▼" if delta < 0 else ("▲" if delta > 0 else "▶")
            return f"{label:<22} {arrow} {cur:.4f} ({delta:+.4f})"
    return f"{label:<22} ▶ {cur:.4f}"


def fmt_lr_triple(lr: Dict[str, Optional[float]]) -> str:
    def f(x):
        return f"{x:.1e}" if x is not None else "—"
    return f"{f(lr['stage1'])}/{f(lr['stage2'])}/{f(lr['stage3'])}"


def fmt_elapsed(start: Optional[float]) -> str:
    if start is None:
        return "—"
    sec = int(time.time() - start)
    return f"{sec // 60:02d}m {sec % 60:02d}s"


def render(state: RunningState, run_name: str, tmux_alive: bool, age: float) -> str:
    cols = shutil.get_terminal_size((80, 30)).columns
    width = max(60, min(cols - 2, 90))

    border = "─" * (width - 2)
    out = []
    title = f" {run_name} "
    pad = (width - 2 - len(title) - len(f" elapsed {fmt_elapsed(state.start_ts)} "))
    out.append("┌" + title + "─" * max(1, pad) + f" elapsed {fmt_elapsed(state.start_ts)} " + "┐")

    status = "alive" if tmux_alive else "DEAD (training stopped)"
    age_label = f"{age:.0f}s" if age < 999 else "stale"
    out.append(f"│ tmux: {status:<12}  events seen: {state.events_seen:<5}  last event: {age_label:>7}".ljust(width - 1) + "│")
    out.append(f"│ event_type: {state.latest_event_type:<25} epoch: {str(state.latest_epoch or '—'):<4} step: {str(state.latest_step or '—'):<6}".ljust(width - 1) + "│")
    out.append(f"│ LR (stage1/2/3): {fmt_lr_triple(state.lr_per_stage)}".ljust(width - 1) + "│")
    out.append("├" + border + "┤")
    out.append("│ losses (most recent train_step)".ljust(width - 1) + "│")
    for label, key in [
        ("train/loss_total",  "train_step/loss_total"),
        ("train/loss_stage1", "train_step/loss_stage1"),
        ("train/loss_stage2", "train_step/loss_stage2"),
        ("train/loss_stage3", "train_step/loss_stage3"),
        ("stage3 step acc",   "train_step/stage3_acc"),
    ]:
        out.append(f"│   {fmt_loss_line(label, state.loss_history, key)}".ljust(width - 1) + "│")

    out.append("├" + border + "┤")
    out.append("│ health".ljust(width - 1) + "│")
    nan_str = "/".join(str(state.nan_counts[k]) for k in ("stage1", "stage2", "stage2_internal", "stage3", "total"))
    nan_warn = " ⚠" if any(v > 0 for v in state.nan_counts.values()) else ""
    out.append(f"│   NaN s1/s2/s2int/s3/total: {nan_str}{nan_warn}".ljust(width - 1) + "│")
    fb_warn = " ⚠ tail class triggered" if state.leak_fallback_total > 0 else ""
    out.append(f"│   leak_fallback total:      {state.leak_fallback_total}{fb_warn}".ljust(width - 1) + "│")

    out.append("├" + border + "┤")
    out.append("│ eval (best dev so far)".ljust(width - 1) + "│")
    if state.best_combined is None:
        out.append("│   no dev eval yet (interval=5 → only at epoch 5)".ljust(width - 1) + "│")
    else:
        out.append(f"│   best dev/combined: {state.best_combined:.4f} @ epoch {state.best_combined_epoch}".ljust(width - 1) + "│")
        for k in ("dev/stage1_recall@0.5", "dev/stage1_precision@0.5",
                  "dev/stage2_mIoU", "dev/stage3_acc", "dev/stage3_episode_acc",
                  "dev/pred_items_per_image"):
            v = state.last_dev_metrics.get(k)
            if v is not None:
                out.append(f"│     {k:<28} {v:.4f}".ljust(width - 1) + "│")

    out.append("└" + border + "┘")
    out.append("(Ctrl-C to exit; training keeps running)")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    state_kv = load_state()
    run_name = remote_run_name()
    log_path = remote_log_path(run_name)

    print(f"[monitor] watching {state_kv['SSH_HOST']}:{log_path}")
    print(f"[monitor] run name: {run_name}")
    print(f"[monitor] starting tail in 2 sec...")
    time.sleep(2)

    state = RunningState()
    proc = open_ssh_tail(state_kv, log_path)

    last_render = 0.0
    last_tmux_check = 0.0
    tmux_alive = True

    def cleanup(_sig=None, _frame=None):
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    try:
        while True:
            line = proc.stdout.readline() if proc.stdout else None
            if line:
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        state.consume(event)
                    except json.JSONDecodeError:
                        pass

            now = time.time()
            if now - last_tmux_check > 10:
                tmux_alive = ssh_check_tmux_alive(state_kv)
                last_tmux_check = now

            if now - last_render >= REFRESH_SECONDS:
                age = (now - state.last_event_ts) if state.last_event_ts else 999.0
                # Clear screen + render
                print("\033[H\033[J", end="")  # cursor home + clear
                print(render(state, run_name, tmux_alive, age))
                sys.stdout.flush()
                last_render = now

            # If the process is gone, exit
            if proc.poll() is not None:
                err = proc.stderr.read() if proc.stderr else ""
                print()
                print("[monitor] ssh tail exited.")
                if err:
                    print(f"[monitor] stderr: {err.strip()}")
                return 1

            if not line:
                time.sleep(0.2)

    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
