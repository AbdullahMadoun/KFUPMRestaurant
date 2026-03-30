# =============================================================================
# FILE: benchmark_runtime.py
# CATEGORY: UTIL
# PURPOSE: CLI benchmarks for Stage 3 synthetic training and joint-training throughput.
# DEPENDENCIES: config_loader.py, dataset_integration.py, losses.py, metrics.py, pipeline.py, stage1_qwen.py, stage2_sam.py, stage3_icl.py
# USED BY: None
# KEY CLASSES/FUNCTIONS: resolve_device, parse_args, benchmark_stage3_train, benchmark_joint_train, _move_to_device, main
# LAST MODIFIED: 2026-03-21T09:37:48+00:00
# SNAPSHOT NOTES: appears stale against the current JointBatchCollator signature
# =============================================================================
"""
Benchmark and profiling entrypoints for training and inference experiments.

Examples:
    python benchmark_runtime.py stage3-train --steps 50 --warmup 10
    python benchmark_runtime.py joint-train --config master_config.yaml --steps 20
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config_loader import load_config
from dataset_integration import JointBatchCollator, JointFoodDataset
from losses import Stage3Loss
from metrics import format_metrics_table
from stage1_qwen import DEFAULT_PROMPT, QwenGrounder
from stage2_sam import SAM3Segmenter
from stage3_icl import FoodClassifier
from pipeline import TriFoodNet


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage3 = subparsers.add_parser("stage3-train", help="Synthetic Stage 3 training benchmark.")
    stage3.add_argument("--steps", type=int, default=50)
    stage3.add_argument("--warmup", type=int, default=10)
    stage3.add_argument("--batch-size", type=int, default=4)
    stage3.add_argument("--n-way", type=int, default=10)
    stage3.add_argument("--k-shot", type=int, default=5)
    stage3.add_argument("--query-per-class", type=int, default=1)
    stage3.add_argument("--device", default="auto")

    joint = subparsers.add_parser("joint-train", help="Real-data joint benchmark using the integration dataset.")
    joint.add_argument("--config", default="master_config.yaml")
    joint.add_argument("--steps", type=int, default=20)
    joint.add_argument("--warmup", type=int, default=5)
    joint.add_argument("--device", default="auto")
    joint.add_argument("overrides", nargs="*")

    return parser.parse_args()


def benchmark_stage3_train(args: argparse.Namespace):
    device = resolve_device(args.device)
    model = FoodClassifier().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.icl.parameters(), lr=1e-4, foreach=("foreach" in torch.optim.AdamW.__init__.__code__.co_varnames))

    support = torch.randn(args.batch_size, args.n_way * args.k_shot, 3, 224, 224, device=device)
    query = torch.randn(args.batch_size, args.n_way * args.query_per_class, 3, 224, 224, device=device)
    labels = torch.tensor(
        [class_idx for _ in range(args.batch_size) for class_idx in range(args.n_way) for _ in range(args.query_per_class)],
        device=device,
    )

    step_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        t0 = time.perf_counter()
        logits = model(support, query, n_way=args.n_way, k_shot=args.k_shot)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if step >= args.warmup:
            step_times.append(dt)

    stats = {
        "benchmark/command": "stage3-train",
        "benchmark/device": str(device),
        "benchmark/steps_measured": len(step_times),
        "benchmark/mean_step_ms": 1000.0 * sum(step_times) / max(len(step_times), 1),
        "benchmark/samples_per_sec": (args.batch_size * max(len(step_times), 1)) / max(sum(step_times), 1e-9),
    }
    if device.type == "cuda":
        stats["benchmark/max_memory_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(format_metrics_table(stats))


def benchmark_joint_train(args: argparse.Namespace):
    cfg = load_config(args.config, args.overrides or None)
    device = resolve_device(args.device if args.device != "auto" else cfg.hardware.device)

    integration_cfg = cfg.data.integration
    if not getattr(integration_cfg, "batch_root", ""):
        raise ValueError("Set data.integration.batch_root in the config or via overrides before running joint-train.")

    stage1 = QwenGrounder(
        model_name=cfg.stage1.model_name,
        lora_r=cfg.stage1.lora.r,
        lora_alpha=cfg.stage1.lora.alpha,
        lora_dropout=cfg.stage1.lora.dropout,
        lora_target_modules=cfg.stage1.lora.target_modules,
        gradient_checkpointing=cfg.hardware.gradient_checkpointing,
    )
    stage2 = SAM3Segmenter(
        model_name=cfg.stage2.model_name,
        freeze_image_encoder=cfg.stage2.freeze.image_encoder,
        freeze_prompt_encoder=cfg.stage2.freeze.prompt_encoder,
        gradient_checkpointing=cfg.hardware.gradient_checkpointing,
    ).to(device)
    stage3 = FoodClassifier(
        clip_model=cfg.stage3.clip_model,
        num_layers=cfg.stage3.transformer.num_layers,
        num_heads=cfg.stage3.transformer.num_heads,
        ff_dim=cfg.stage3.transformer.ff_dim,
        dropout=cfg.stage3.transformer.dropout,
    ).to(device)
    pipeline = TriFoodNet(stage1, stage2, stage3)

    dataset = JointFoodDataset(
        batch_root=integration_cfg.batch_root,
        export_root=getattr(integration_cfg, "export_root", None),
        repo_root=getattr(integration_cfg, "repo_root", None),
        split="train",
        image_size=cfg.data.image_size,
        train_ratio=integration_cfg.train_ratio,
        val_ratio=integration_cfg.val_ratio,
        test_ratio=integration_cfg.test_ratio,
        split_seed=integration_cfg.split_seed,
        n_way=cfg.stage3.episode.n_way,
        k_shot=cfg.stage3.episode.k_shot,
        query_per_class=cfg.stage3.episode.query_per_class,
    )
    collator = JointBatchCollator(
        stage1_processor=stage1.processor,
        stage3_processor=stage3.clip.processor,
        stage1_prompt=getattr(cfg.data.integration, "stage1_prompt", DEFAULT_PROMPT),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.joint.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=bool(cfg.data.pin_memory and device.type == "cuda"),
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        [p for p in pipeline.parameters() if p.requires_grad],
        lr=cfg.joint.training.learning_rate,
        weight_decay=cfg.joint.training.weight_decay,
    )
    stage3_loss_fn = Stage3Loss(
        label_smoothing=cfg.stage3.training.label_smoothing,
        kind=getattr(cfg.stage3.loss, "name", "cross_entropy"),
        logit_adjust_tau=getattr(cfg.stage3.loss, "logit_adjust_tau", 1.0),
    )
    amp_dtype = torch.float16 if device.type == "cuda" and cfg.hardware.fp16 else None
    iterator = iter(loader)
    measured = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.steps):
        batch = next(iterator)
        batch = _move_to_device(batch, device)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        t0 = time.perf_counter()
        with autocast_ctx:
            losses = pipeline.forward(
                batch,
                use_gt_boxes=True,
                loss_weights=(
                    cfg.joint.loss_weights.lambda1,
                    cfg.joint.loss_weights.lambda2,
                    cfg.joint.loss_weights.lambda3,
                ),
                stage3_loss_fn=stage3_loss_fn,
            )
        optimizer.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if step >= args.warmup:
            measured.append(dt)

    stats = {
        "benchmark/command": "joint-train",
        "benchmark/device": str(device),
        "benchmark/steps_measured": len(measured),
        "benchmark/mean_step_ms": 1000.0 * sum(measured) / max(len(measured), 1),
        "benchmark/samples_per_sec": (cfg.joint.training.batch_size * max(len(measured), 1)) / max(sum(measured), 1e-9),
    }
    if device.type == "cuda":
        stats["benchmark/max_memory_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(format_metrics_table(stats))


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=(device.type == "cuda"))
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def main():
    args = parse_args()
    if args.command == "stage3-train":
        benchmark_stage3_train(args)
    elif args.command == "joint-train":
        benchmark_joint_train(args)


if __name__ == "__main__":
    main()
