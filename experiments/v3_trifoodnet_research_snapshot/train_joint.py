# =============================================================================
# FILE: train_joint.py
# CATEGORY: TRAIN
# PURPOSE: Main joint trainer that builds the full pipeline, datasets, optimizer, scheduler, evaluation, and checkpointing.
# DEPENDENCIES: config_loader.py, dataset_integration.py, experiment_logging.py, losses.py, metrics.py, pipeline.py, post_training_artifacts.py, stage1_qwen.py, stage2_sam.py, stage3_icl.py
# USED BY: validate_pipeline_contracts.py
# KEY CLASSES/FUNCTIONS: train_joint, _load_existing_checkpoints, _build_adamw_kwargs, _build_param_groups, _prune_old_checkpoints, _maybe_generate_epoch_visualizations, _accumulate_metrics, _to_device, _move_to_device, _resolve_amp_dtype, _resolve_dev_ratio
# LAST MODIFIED: 2026-03-21T21:22:00.859567+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
Joint end-to-end fine-tuning for the TriFoodNet research snapshot.

This version is wired to the exported review dataset contract described in
`INTEGRATION_DATASET_GUIDE.md` and provides:

- manifest-based data loading
- detailed per-stage loss reporting
- structured experiment logging
- optional export snapshotting for reproducibility
- optional profiler traces for performance analysis
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import math
import os
from pathlib import Path
import shutil
import time
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_scheduler

from config_loader import load_config
from dataset_integration import (
    JointBatchCollator,
    JointFoodDataset,
    build_class_name_index,
    build_export_paths,
    read_json,
    snapshot_export_contract,
)
from dataset_v3_adapter import V3ExportAdapter, adapter_from_config
from determinism import set_seed, dataloader_generator, worker_init_fn
from eval_harness import EvalMode, evaluate_split, combined_score, COMBINED_FORMULA_VERSION
from experiment_logging import ExperimentLogger
from losses import Stage3Loss
from metrics import EpisodicAccumulator, format_metrics_table, greedy_box_matches, mask_iou
from pipeline import TriFoodNet
from post_training_artifacts import (
    build_stage3_reference_library,
    generate_split_summary,
    generate_split_visualizations,
    generate_training_report,
)
from stage1_qwen import DEFAULT_PROMPT, QwenGrounder
from stage2_sam import SAM3Segmenter
from stage3_icl import FoodClassifier


# --- Snapshot note: Primary training entry point used for the best reported experiments. ---
def _load_dotenv():
    """Lightweight .env loader. Reads KEY=VALUE pairs from ./.env if present
    and exports them into os.environ unless the key is already set. Used to
    propagate HF_TOKEN to local training without requiring users to remember
    `export HF_TOKEN=...` before every run. Vast launcher already pushes
    .env contents to the remote separately."""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def train_joint(config_path: Optional[str] = None, overrides=None):
    _load_dotenv()
    cfg = load_config(config_path, overrides)
    jc = cfg.joint
    h = cfg.hardware
    log_cfg = cfg.logging
    integration = cfg.data.integration

    # Either an adapter must be configured, or the legacy batch_root path.
    adapter_cfg_pre = getattr(integration, "adapter", None)
    has_adapter_cfg = bool(adapter_cfg_pre is not None and getattr(adapter_cfg_pre, "kind", ""))
    if not has_adapter_cfg and not getattr(integration, "batch_root", ""):
        raise ValueError(
            "Configure either `data.integration.adapter` (preferred) or "
            "`data.integration.batch_root` before running joint training."
        )

    # Pin all RNGs early so dataset construction and model init are reproducible.
    seed_meta = set_seed(
        seed=getattr(cfg.run, "seed", None),
        mode=str(getattr(cfg.run, "determinism_mode", "deterministic")),
    )

    device = _resolve_device(getattr(h, "device", "auto"))
    _configure_training_runtime(
        device,
        deterministic=str(getattr(cfg.run, "determinism_mode", "deterministic")) != "loose",
    )
    amp_dtype = _resolve_amp_dtype(h, device)

    ckpt_dir = Path(cfg.paths.checkpoints) / cfg.run.name / "joint"
    log_dir = Path(cfg.paths.logs) / cfg.run.name / "joint"
    # Refuse to silently overwrite a previous run with the same name. Either
    # bump run.name or pass run.allow_overwrite=true (off by default). Without
    # this guard, two runs with the same name would interleave events into
    # the same jsonl, corrupting the source of truth.
    allow_overwrite = bool(getattr(cfg.run, "allow_overwrite", False))
    if log_dir.exists() and any(log_dir.iterdir()) and not allow_overwrite:
        raise ValueError(
            f"Log directory already exists and is non-empty: {log_dir}\n"
            f"Either change run.name (preferred) or set run.allow_overwrite=true."
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(ckpt_dir / "config_snapshot.yaml")
    logger = ExperimentLogger(log_dir, cfg, device)

    if getattr(integration, "snapshot_export", False):
        snapshot_root = Path(integration.snapshot_dir) / cfg.run.name / "joint_export_snapshot"
        snapshot_export_contract(
            batch_root=integration.batch_root,
            export_root=(integration.export_root or None),
            snapshot_root=snapshot_root,
            include_assets=True,
        )
        logger.log(
            "dataset_snapshot",
            {"dataset/snapshot_path": str(snapshot_root)},
            split="system",
        )

    c1, c2, c3 = cfg.stage1, cfg.stage2, cfg.stage3

    bnb_config = _build_bnb_config(h, device)
    dev_ratio = _resolve_dev_ratio(integration)

    # Build the v3-export adapter once and share it across train/dev/train_eval
    # datasets so manifest reads, split computation, and asset roots are all
    # cached. When `data.integration.adapter` is unset, we fall back to the
    # legacy on-disk export contract (batch_root/_review/dataset/...).
    adapter_cfg = getattr(integration, "adapter", None)
    adapter_kind = str(getattr(adapter_cfg, "kind", "")) if adapter_cfg is not None else ""
    use_adapter = bool(adapter_kind)

    if use_adapter:
        adapter = adapter_from_config(integration)
        adapter_data = adapter.load()
        export_paths = build_export_paths(
            adapter_data.export_root,
            export_root=adapter_data.export_root,
            repo_root=(integration.repo_root or None),
        )
        class_records = list(adapter_data.classes)
        class_names, class_name_to_id = build_class_name_index(
            adapter_data.export_root,
            classes=class_records,
            stage3_rows=adapter_data.stage3_rows,
        )
        # Stamp dataset provenance into run_metadata.json.
        logger.update_run_metadata(
            dataset={
                "version": adapter_data.dataset_version,
                "hash": adapter_data.dataset_hash,
                "export_root": str(adapter_data.export_root),
                "n_classes": len(adapter_data.classes),
                "n_images_total": len(adapter_data.retained_image_ids),
                "n_stage3_items_total": len(adapter_data.stage3_rows),
            },
            seed_meta=seed_meta,
        )
        # Run dataset-dependent validation now that adapter has reported counts.
        try:
            from config_validation import validate_config
            train_class_counts = (
                adapter_data.stage3_split_summary.get("train", {}).get("class_item_counts", {})
            )
            min_train_class_size = min(train_class_counts.values()) if train_class_counts else 0
            num_classes_in_dataset = max(adapter_data.classes_compact_ids, default=-1) + 1
            extra_warnings = validate_config(
                cfg,
                dataset_class_count=num_classes_in_dataset,
                dataset_min_train_class_size=min_train_class_size,
            )
            for msg in extra_warnings:
                print(f"[Config] WARN: {msg}")
        except ImportError:
            pass
    else:
        adapter = None
        export_paths = build_export_paths(
            integration.batch_root,
            export_root=(integration.export_root or None),
            repo_root=(integration.repo_root or None),
        )
        class_records = read_json(export_paths.export_root / "classes.json")
        class_names, class_name_to_id = build_class_name_index(
            export_paths.export_root,
            classes=class_records if isinstance(class_records, list) else None,
        )
        logger.update_run_metadata(
            dataset={
                "version": "legacy",
                "hash": "",
                "export_root": str(export_paths.export_root),
            },
            seed_meta=seed_meta,
        )

    stage1 = QwenGrounder(
        model_name=c1.model_name,
        lora_r=c1.lora.r,
        lora_alpha=c1.lora.alpha,
        lora_dropout=c1.lora.dropout,
        lora_target_modules=c1.lora.target_modules,
        use_rslora=c1.lora.get("use_rslora", False),
        device=device,
        gradient_checkpointing=h.gradient_checkpointing,
        quantization_config=bnb_config,
    )
    stage2 = SAM3Segmenter(
        model_name=c2.model_name,
        freeze_image_encoder=c2.freeze.image_encoder,
        freeze_prompt_encoder=c2.freeze.prompt_encoder,
        device=device,
        gradient_checkpointing=h.gradient_checkpointing,
        quantization_config=bnb_config,
        torch_dtype=amp_dtype,
        bce_weight=float(getattr(c2.loss, "bce_weight", 1.0)),
        dice_weight=float(getattr(c2.loss, "dice_weight", 1.0)),
    )
    stage3 = FoodClassifier(
        clip_model=c3.clip_model,
        num_layers=c3.transformer.num_layers,
        num_heads=c3.transformer.num_heads,
        ff_dim=c3.transformer.ff_dim,
        dropout=c3.transformer.dropout,
        lora_cfg=getattr(c3, "lora", None),
        num_classes=max(int(cfg.data.num_classes), len(class_names), (max(class_name_to_id.values()) + 1) if class_name_to_id else 0),
        class_names=class_names,
        train_embedding=bool(getattr(c3, "train_embedding", True)),
        inference_n_way=int(c3.eval.n_way),
        inference_k_shot=int(c3.eval.k_shot),
    ).to(device)
    pipeline = TriFoodNet(stage1, stage2, stage3)

    _load_existing_checkpoints(cfg, pipeline, logger)

    collator = JointBatchCollator(
        stage1_processor=stage1.processor,
        stage1_prompt=getattr(integration, "stage1_prompt", DEFAULT_PROMPT),
    )
    max_batches_per_epoch = int(getattr(jc.training, "max_batches_per_epoch", 0) or 0)
    if 0 < max_batches_per_epoch < int(jc.training.grad_accum_steps):
        raise ValueError("joint.training.max_batches_per_epoch must be >= grad_accum_steps when enabled.")
    common_dataset_kwargs = dict(
        batch_root=(None if use_adapter else integration.batch_root),
        export_root=(None if use_adapter else (integration.export_root or None)),
        repo_root=(integration.repo_root or None),
        image_size=cfg.data.image_size,
        train_ratio=integration.train_ratio,
        val_ratio=dev_ratio,
        test_ratio=integration.test_ratio,
        split_seed=integration.split_seed,
        adapter=adapter,
    )
    train_ds = JointFoodDataset(
        split="train",
        episode_support_split="train",
        n_way=c3.episode.n_way,
        k_shot=c3.episode.k_shot,
        query_per_class=c3.episode.query_per_class,
        **common_dataset_kwargs,
    )
    train_eval_ds = JointFoodDataset(
        split="train",
        episode_support_split="train",
        n_way=c3.eval.n_way,
        k_shot=c3.eval.k_shot,
        query_per_class=1,
        **common_dataset_kwargs,
    )
    dev_ds = JointFoodDataset(
        split="dev",
        episode_support_split="train",
        n_way=c3.eval.n_way,
        k_shot=c3.eval.k_shot,
        query_per_class=1,
        **common_dataset_kwargs,
    )
    reference_support_images, reference_support_labels, reference_support_stats = build_stage3_reference_library(cfg)
    pipeline.stage3.set_support_set(reference_support_images, reference_support_labels)

    effective_seed = int(seed_meta.get("system/seed_seed", 1337))
    loader_kwargs = {
        "batch_size": jc.training.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": bool(cfg.data.pin_memory and device.type == "cuda"),
        "collate_fn": collator,
        "worker_init_fn": worker_init_fn(effective_seed) if cfg.data.num_workers > 0 else None,
    }
    if cfg.data.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    # Distinct generators per loader so train shuffle order is independent
    # from eval order — but each is reproducible from the effective seed.
    train_loader = DataLoader(
        train_ds, shuffle=True,
        generator=dataloader_generator(effective_seed),
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_eval_ds, shuffle=False,
        generator=dataloader_generator(effective_seed + 1),
        **loader_kwargs,
    )
    dev_loader = DataLoader(
        dev_ds, shuffle=False,
        generator=dataloader_generator(effective_seed + 2),
        **loader_kwargs,
    )
    batches_per_epoch = len(train_loader)
    if max_batches_per_epoch > 0:
        batches_per_epoch = min(batches_per_epoch, max_batches_per_epoch)

    # Per-stage param groups so heterogeneous stages can run at their natural LRs.
    # Falls back to the joint global lr/weight_decay when a stage has no override.
    param_groups, param_group_summary = _build_param_groups(pipeline, jc.training, c1, c2, c3)
    if not param_groups:
        raise RuntimeError("Pipeline has no trainable parameters across all stages.")
    adamw_kwargs = _build_adamw_kwargs(device, jc.training)
    # The per-group lr/weight_decay overrides the global kwargs, so AdamW
    # honors them per-stage. We keep `lr`/`weight_decay` in adamw_kwargs as
    # the fallback for any group that does not specify them.
    optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
    trainable = [p for g in param_groups for p in g["params"]]
    optimizer_steps_per_epoch = max(1, math.ceil(batches_per_epoch / max(int(jc.training.grad_accum_steps), 1)))
    total_steps = max(1, optimizer_steps_per_epoch * int(jc.training.epochs))
    warmup_steps = int(total_steps * jc.training.warmup_ratio)
    scheduler = get_scheduler(
        jc.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    stage3_loss_fn = Stage3Loss(
        label_smoothing=c3.training.label_smoothing,
        kind=getattr(c3.loss, "name", "cross_entropy"),
        logit_adjust_tau=getattr(c3.loss, "logit_adjust_tau", 1.0),
    )
    early_cfg = getattr(jc.training, "early_stopping", None)
    early_stopping_enabled = bool(getattr(early_cfg, "enabled", False))
    early_monitor = str(getattr(early_cfg, "monitor", "joint/combined"))
    early_mode = str(getattr(early_cfg, "mode", "max")).strip().lower()
    if early_mode not in {"max", "min"}:
        raise ValueError(f"Unsupported early stopping mode: {early_mode}")
    early_patience = int(getattr(early_cfg, "patience", 3))
    early_min_delta = float(getattr(early_cfg, "min_delta", 0.0))
    early_min_epochs = int(getattr(early_cfg, "min_epochs", 1))

    stage3_uses_peft = hasattr(pipeline.stage3.icl, "peft_config")
    if device.type == "cuda" and getattr(h, "compile", False) and hasattr(torch, "compile"):
        if stage3_uses_peft:
            logger.log(
                "compile",
                {
                    "system/compiled_stage3": 0,
                    "system/compile_skipped": 1,
                    "system/compile_skip_reason": "stage3_peft_transformer_cudagraph_incompatible",
                },
                split="system",
            )
        else:
            try:
                pipeline.stage3.compile_transformer(mode="reduce-overhead")
                logger.log("compile", {"system/compiled_stage3": 1}, split="system")
            except Exception as exc:
                logger.log("compile", {"system/compiled_stage3": 0, "system/compile_error": str(exc)}, split="system")

    profiler = _start_profiler_if_enabled(log_cfg, logger.profiles_dir, device)
    report_root = Path(cfg.paths.outputs) / cfg.run.name / "report"
    per_stage_lr_log = {
        f"optimizer/{entry['name']}_lr": float(entry["lr"]) for entry in param_group_summary
    }
    per_stage_wd_log = {
        f"optimizer/{entry['name']}_weight_decay": float(entry["weight_decay"])
        for entry in param_group_summary
    }
    per_stage_params_log = {
        f"model/{entry['name']}_trainable_params": float(entry["param_count"])
        for entry in param_group_summary
    }

    logger.log(
        "run_start",
        {
            "dataset/train_images": len(train_ds),
            "dataset/dev_images": len(dev_ds),
            "dataset/test_images": int(train_ds.split_summary.get("test", {}).get("images", 0)),
            "optimizer/batches_per_epoch": batches_per_epoch,
            "optimizer/optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "optimizer/total_steps": total_steps,
            "optimizer/warmup_steps": warmup_steps,
            "model/trainable_params": float(sum(p.numel() for p in trainable)),
            **per_stage_lr_log,
            **per_stage_wd_log,
            **per_stage_params_log,
            "stage3/loss_kind": getattr(c3.loss, "name", "cross_entropy"),
            "curriculum/gt_boxes_epochs_legacy": int(getattr(jc.curriculum, "gt_boxes_epochs", 0)),
            "curriculum/teacher_forcing_sustain_epochs": int(
                getattr(getattr(jc.curriculum, "teacher_forcing", None), "sustain_epochs", getattr(jc.curriculum, "gt_boxes_epochs", 0))
            ),
            "curriculum/teacher_forcing_transition_epochs": int(
                getattr(getattr(jc.curriculum, "teacher_forcing", None), "transition_epochs", 0)
            ),
            "curriculum/teacher_forcing_start_prob": float(
                getattr(getattr(jc.curriculum, "teacher_forcing", None), "start_prob", 1.0)
            ),
            "curriculum/teacher_forcing_end_prob": float(
                getattr(getattr(jc.curriculum, "teacher_forcing", None), "end_prob", 0.0)
            ),
        },
        split="system",
    )
    logger.log(
        "dataset_split_summary",
        {"dataset/split_summary": train_ds.split_summary},
        split="system",
        update_latest=False,
    )
    logger.log(
        "dataset_class_contract",
        {
            "dataset/supported_classes": list(getattr(train_ds, "supported_classes", [])),
            "dataset/removed_classes": list(getattr(train_ds, "removed_classes", [])),
            "dataset/supported_class_ids": list(getattr(train_ds, "supported_class_ids", [])),
        },
        split="system",
        update_latest=False,
    )
    logger.log(
        "stage3_reference_library",
        {
            "stage3/reference_support_images": reference_support_stats["num_support_images"],
            "stage3/reference_available_classes": len(reference_support_stats["available_class_ids"]),
            "stage3/reference_missing_configured_classes": len(reference_support_stats["missing_configured_classes"]),
        },
        split="system",
        update_latest=False,
    )

    # `best/` (alias for best-by-combined) is the headline checkpoint matching the
    # `dev/combined` metric reported in summaries. `best_by_monitor/` tracks whatever
    # `early_stopping.monitor` is configured to (e.g., dev/loss_total). Saving both
    # avoids the historical mismatch where the on-disk `best/` did not reproduce the
    # reported best combined score.
    best_checkpoint_path = ckpt_dir / "best"               # = best by combined
    best_by_monitor_path = ckpt_dir / "best_by_monitor"
    best_monitor_value = float("-inf") if early_mode == "max" else float("inf")
    best_stage2_state_by_combined = None
    best_stage2_state_by_monitor = None
    best_combined = float("-inf")
    epochs_without_improvement = 0
    early_stopped = False
    global_step = 0
    interval_metrics: Dict[str, float] = {}
    interval_count = 0
    run_status = "completed"

    try:
        for epoch in range(1, jc.training.epochs + 1):
            pipeline.train()
            pipeline.reset_nan_counts()
            optimizer.zero_grad(set_to_none=True)
            teacher_forcing_prob = _resolve_teacher_forcing_probability(
                jc.curriculum,
                epoch=epoch,
                total_epochs=int(jc.training.epochs),
            )
            logger.log(
                "curriculum_epoch",
                {"train/teacher_forcing_prob": float(teacher_forcing_prob)},
                step=global_step,
                epoch=epoch,
                split="train",
                update_latest=False,
            )
            epoch_start = time.perf_counter()
            data_fetch_start = time.perf_counter()
            accum_counter = 0
            current_accum_target = min(int(jc.training.grad_accum_steps), max(batches_per_epoch, 1))
            epoch_leak_fallback_total = 0

            for step, batch in enumerate(train_loader):
                if max_batches_per_epoch > 0 and step >= max_batches_per_epoch:
                    break
                if accum_counter == 0:
                    remaining_batches = max(batches_per_epoch - step, 1)
                    current_accum_target = min(int(jc.training.grad_accum_steps), remaining_batches)
                data_time = time.perf_counter() - data_fetch_start
                step_start = time.perf_counter()
                batch = _to_device(batch, device)
                epoch_leak_fallback_total += int(batch.get("leak_fallback_count", 0) or 0)
                use_gt = _sample_teacher_forcing(teacher_forcing_prob)

                autocast_ctx = (
                    torch.autocast(device_type=device.type, dtype=amp_dtype)
                    if amp_dtype is not None
                    else nullcontext()
                )
                with autocast_ctx:
                    losses = pipeline.forward(
                        batch,
                        use_gt_boxes=use_gt,
                        loss_weights=(
                            jc.loss_weights.lambda1,
                            jc.loss_weights.lambda2,
                            jc.loss_weights.lambda3,
                        ),
                        stage3_loss_fn=stage3_loss_fn,
                    )
                    total_loss = losses["loss_total"] / max(current_accum_target, 1)

                scaler.scale(total_loss).backward()
                accum_counter += 1

                if accum_counter >= current_accum_target:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, jc.training.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accum_counter = 0
                    if profiler is not None:
                        profiler.step()

                    step_time = time.perf_counter() - step_start
                    effective_batch = batch["images"].shape[0] * current_accum_target
                    train_metrics = {
                        "train_step/loss_stage1": float(losses["loss_stage1"].item()),
                        "train_step/loss_stage2": float(losses["loss_stage2"].item()),
                        "train_step/loss_stage3": float(losses["loss_stage3"].item()),
                        "train_step/loss_total": float(losses["loss_total"].item()),
                        "train_step/stage1_lm_loss": float(losses["metrics"]["stage1/lm_loss"]),
                        "train_step/stage2_bce_loss": float(losses["metrics"]["stage2/bce_loss"]),
                        "train_step/stage2_dice_loss": float(losses["metrics"]["stage2/dice_loss"]),
                        "train_step/stage3_ce_loss": float(losses["metrics"]["stage3/ce_loss"]),
                        "train_step/stage3_acc": float(losses["metrics"]["stage3/acc"]),
                        "train_step/lr": float(scheduler.get_last_lr()[0]),
                        "train_step/grad_norm": float(grad_norm),
                        "train_step/data_time_sec": data_time,
                        "train_step/step_time_sec": step_time,
                        "train_step/samples_per_sec": effective_batch / max(step_time, 1e-9),
                        "train_step/use_gt_boxes": int(use_gt),
                        "train_step/teacher_forcing_prob": float(teacher_forcing_prob),
                        "train_step/accum_target": int(current_accum_target),
                    }
                    logger.log(
                        "train_step",
                        train_metrics,
                        step=global_step,
                        epoch=epoch,
                        split="train",
                    )
                    interval_metrics = _accumulate_metrics(interval_metrics, train_metrics)
                    interval_count += 1

                    if global_step % log_cfg.log_every_n_steps == 0:
                        averaged = {k: v / max(interval_count, 1) for k, v in interval_metrics.items()}
                        print(format_metrics_table({"epoch": epoch, "step": global_step, **averaged}))
                        logger.log(
                            "train_interval",
                            averaged,
                            step=global_step,
                            epoch=epoch,
                            split="train",
                        )
                        interval_metrics = {}
                        interval_count = 0

                data_fetch_start = time.perf_counter()

            train_nan_counts = pipeline.get_nan_counts()
            epoch_metrics = {
                "train/epoch_time_sec": time.perf_counter() - epoch_start,
                "train/episode_leak_fallback_total": float(epoch_leak_fallback_total),
                **{f"train/nan_{k}": float(v) for k, v in train_nan_counts.items()},
            }
            logger.log("train_epoch_end", epoch_metrics, step=global_step, epoch=epoch, split="train")
            # Stage 3 caches its support-set embeddings; weights changed this epoch,
            # so the cache is now stale. Invalidate so the cosine retriever uses
            # the current encoder when dev eval runs below.
            if hasattr(pipeline.stage3, "invalidate_support_cache"):
                pipeline.stage3.invalidate_support_cache()

            if epoch % jc.eval.interval == 0:
                dev_report = evaluate_split(
                    pipeline,
                    dataset=dev_ds,
                    loader=dev_loader,
                    mode=EvalMode.END_TO_END,
                    cfg=cfg,
                    device=device,
                    amp_dtype=amp_dtype,
                    stage3_loss_fn=stage3_loss_fn,
                    split_name="dev",
                )
                dev_metrics = dev_report.flat_metrics(prefix="dev")
                print(format_metrics_table({"epoch": epoch, "split": "dev", **dev_metrics}))

                combined = dev_report.combined
                monitor_value = _resolve_early_stop_metric(early_monitor, {}, dev_metrics, combined)
                dev_metrics["dev/early_stop_monitor"] = monitor_value
                logger.log("eval_epoch", dev_metrics, step=global_step, epoch=epoch, split="dev")

                logger.update_best("joint/combined", combined, step=global_step, epoch=epoch)
                improved_monitor = _is_improvement(
                    current=monitor_value,
                    best=best_monitor_value,
                    mode=early_mode,
                    min_delta=early_min_delta,
                )
                improved_combined = combined > best_combined

                # Save by monitor (early-stopping driver, possibly dev/loss_total).
                if improved_monitor:
                    best_monitor_value = monitor_value
                    epochs_without_improvement = 0
                    pipeline.save(str(best_by_monitor_path))
                    logger.record_checkpoint(
                        best_by_monitor_path, dev_metrics,
                        step=global_step, epoch=epoch, is_best=False,
                    )
                    best_stage2_state_by_monitor = _capture_module_state_cpu(pipeline.stage2)
                elif epoch >= early_min_epochs:
                    epochs_without_improvement += 1

                # Save by combined (the headline metric). The `best/` directory is the
                # canonical "best" so downstream loaders reproduce the reported number.
                if improved_combined:
                    best_combined = combined
                    pipeline.save(str(best_checkpoint_path))
                    logger.record_checkpoint(
                        best_checkpoint_path, dev_metrics,
                        step=global_step, epoch=epoch, is_best=True,
                    )
                    best_stage2_state_by_combined = _capture_module_state_cpu(pipeline.stage2)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                _maybe_generate_epoch_visualizations(
                    pipeline,
                    cfg,
                    logger,
                    epoch=epoch,
                    step=global_step,
                    report_root=report_root,
                    device=device,
                )

                if (
                    early_stopping_enabled
                    and epoch >= early_min_epochs
                    and epochs_without_improvement >= max(early_patience, 1)
                ):
                    early_stopped = True
                    logger.log(
                        "early_stop",
                        {
                            "system/early_stopped": 1,
                            "system/early_stop_monitor": early_monitor,
                            "system/early_stop_mode": early_mode,
                            "system/early_stop_patience": early_patience,
                            "system/early_stop_wait_count": epochs_without_improvement,
                            "system/best_monitor_value": float(best_monitor_value),
                            "system/current_monitor_value": float(monitor_value),
                        },
                        step=global_step,
                        epoch=epoch,
                        split="system",
                        update_latest=False,
                    )

            if epoch % log_cfg.save_every_n_epochs == 0:
                epoch_path = ckpt_dir / f"epoch_{epoch:03d}"
                pipeline.save(str(epoch_path))
                logger.record_checkpoint(epoch_path, {"epoch": epoch}, step=global_step, epoch=epoch, is_best=False)
                _prune_old_checkpoints(ckpt_dir, keep_last_n=int(log_cfg.keep_last_n_ckpts))
            if early_stopped:
                break
        if best_checkpoint_path.exists():
            pipeline.load(str(best_checkpoint_path))
            if best_stage2_state_by_combined is not None:
                pipeline.stage2.load_state_dict(best_stage2_state_by_combined, strict=False)
            # Stage 3 weights just changed → drop the support-embedding cache.
            if hasattr(pipeline.stage3, "invalidate_support_cache"):
                pipeline.stage3.invalidate_support_cache()
            pipeline.eval()
            train_eval_report = evaluate_split(
                pipeline,
                dataset=train_eval_ds,
                loader=train_eval_loader,
                mode=EvalMode.END_TO_END,
                cfg=cfg,
                device=device,
                amp_dtype=amp_dtype,
                stage3_loss_fn=stage3_loss_fn,
                split_name="train_final",
            )
            final_train_metrics = train_eval_report.flat_metrics(prefix="train")
            print(format_metrics_table({"epoch": epoch, "split": "train_final", **final_train_metrics}))
            logger.log(
                "train_eval_final",
                final_train_metrics,
                step=global_step,
                epoch=epoch,
                split="train",
                update_latest=False,
            )
        # Post-training artifacts. Inside the main try-block so they share the
        # try/finally lifecycle. We catch their failures separately so a bad
        # artifact step downgrades status to `completed_with_artifact_errors`
        # instead of `completed` (the run trained fine, only the report failed).
        if run_status == "completed":
            try:
                if best_checkpoint_path.exists():
                    pipeline.load(str(best_checkpoint_path))
                    if best_stage2_state_by_combined is not None:
                        pipeline.stage2.load_state_dict(best_stage2_state_by_combined, strict=False)
                    if hasattr(pipeline.stage3, "invalidate_support_cache"):
                        pipeline.stage3.invalidate_support_cache()
                pipeline.eval()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                generate_split_visualizations(
                    pipeline,
                    cfg,
                    report_root / "dev_visualizations",
                    split="dev",
                    include_ground_truth=False,
                )
                generate_split_summary(cfg, report_root)
                generate_training_report(cfg, log_dir, report_root)
            except Exception as artifact_exc:
                run_status = "completed_with_artifact_errors"
                logger.log(
                    "post_training_artifacts_error",
                    {
                        "system/error_type": artifact_exc.__class__.__name__,
                        "system/error_message": str(artifact_exc),
                    },
                    step=global_step or None,
                    split="system",
                )
                # Don't re-raise — training succeeded; only artifacts failed.
    except Exception as exc:
        run_status = "failed"
        logger.log(
            "run_error",
            {
                "system/error_type": exc.__class__.__name__,
                "system/error_message": str(exc),
            },
            step=global_step or None,
            split="system",
        )
        raise
    finally:
        if profiler is not None:
            profiler.stop()
        logger.close(status=run_status)

    return pipeline


# NOTE: ``evaluate_objective`` and ``evaluate_inference_dataset`` were removed
# in Phase 2 of the base refactor. Their behavior is now in eval_harness.py:
# call ``evaluate_split(pipeline, dataset=..., loader=..., mode=EvalMode.END_TO_END)``
# for the same effect, and pull a flat metric dict via ``report.flat_metrics(prefix=...)``.


def _load_existing_checkpoints(cfg, pipeline: TriFoodNet, logger: ExperimentLogger):
    stage1_ckpt = Path(cfg.paths.checkpoints) / cfg.run.name / "stage1" / "best"
    if stage1_ckpt.exists():
        pipeline.stage1.load_lora(str(stage1_ckpt))
        logger.log("checkpoint_load", {"system/stage1_checkpoint": str(stage1_ckpt)}, split="system")

    stage2_ckpt = Path(cfg.paths.checkpoints) / cfg.run.name / "stage2" / "best.pt"
    if stage2_ckpt.exists():
        pipeline.stage2.load_state_dict(torch.load(stage2_ckpt, map_location="cpu"), strict=False)
        logger.log("checkpoint_load", {"system/stage2_checkpoint": str(stage2_ckpt)}, split="system")

    stage3_ckpt = Path(cfg.paths.checkpoints) / cfg.run.name / "stage3" / "best_icl.pt"
    if stage3_ckpt.exists():
        target_module = getattr(pipeline.stage3.icl, "_orig_mod", pipeline.stage3.icl)
        target_module.load_state_dict(torch.load(stage3_ckpt, map_location="cpu"), strict=False)
        logger.log("checkpoint_load", {"system/stage3_checkpoint": str(stage3_ckpt)}, split="system")


def _build_adamw_kwargs(device: torch.device, training_cfg) -> Dict[str, float | bool]:
    """Backwards-compatible AdamW kwargs for single-LR optimizer construction.

    For per-stage param groups, callers should instead use
    ``_build_param_groups`` and pass the resulting list as the first argument
    to ``torch.optim.AdamW``.
    """
    kwargs: Dict[str, float | bool] = {
        "lr": training_cfg.learning_rate,
        "weight_decay": training_cfg.weight_decay,
    }
    adamw_params = inspect.signature(torch.optim.AdamW).parameters
    if device.type == "cuda" and "fused" in adamw_params:
        kwargs["fused"] = True
    elif "foreach" in adamw_params:
        kwargs["foreach"] = True
    return kwargs


def _build_param_groups(pipeline, joint_training_cfg, c1, c2, c3) -> tuple[list, list]:
    """Construct per-stage AdamW param groups + a parallel summary list.

    Each stage gets its own LR + weight_decay. Resolution order per stage:
        joint.training.per_stage_lr.<stage>          (preferred)
        stage{N}.training.learning_rate              (fallback to standalone cfg)
        joint.training.learning_rate                 (final fallback)
    Same precedence applies to weight_decay.

    Returns
    -------
    groups : list of dicts ready for AdamW(...)
    summary : list of {"name", "lr", "weight_decay", "param_count"} dicts for logging
    """
    per_stage = getattr(joint_training_cfg, "per_stage_lr", None)
    per_stage_wd = getattr(joint_training_cfg, "per_stage_weight_decay", None)
    fallback_lr = float(joint_training_cfg.learning_rate)
    fallback_wd = float(joint_training_cfg.weight_decay)

    stage_specs = [
        ("stage1", pipeline.stage1, getattr(c1, "training", None)),
        ("stage2", pipeline.stage2, getattr(c2, "training", None)),
        ("stage3", pipeline.stage3, getattr(c3, "training", None)),
    ]

    def _resolve(name: str, attr: str, fallback: float, stage_cfg) -> float:
        # Tier 1: explicit joint per-stage override.
        # Critical: per_stage is the LR map; only use it for learning_rate lookups.
        # Previously, weight_decay would also pick up the LR value from this map,
        # silently writing the LR into the optimizer's weight_decay arg (a real
        # correctness bug surfaced by code review).
        if attr == "learning_rate":
            if per_stage is not None and getattr(per_stage, name, None) is not None:
                return float(getattr(per_stage, name))
        elif attr == "weight_decay":
            if per_stage_wd is not None and getattr(per_stage_wd, name, None) is not None:
                return float(getattr(per_stage_wd, name))
        # Tier 2: stageN.training.<attr> if present
        if stage_cfg is not None and getattr(stage_cfg, attr, None) is not None:
            return float(getattr(stage_cfg, attr))
        # Tier 3: shared joint default
        return fallback

    groups = []
    summary = []
    for name, module, stage_cfg in stage_specs:
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            summary.append({"name": name, "lr": 0.0, "weight_decay": 0.0, "param_count": 0,
                            "trainable": False})
            continue
        lr = _resolve(name, "learning_rate", fallback_lr, stage_cfg)
        wd = _resolve(name, "weight_decay", fallback_wd, stage_cfg)
        groups.append({"params": params, "lr": lr, "weight_decay": wd, "name": name})
        summary.append({"name": name, "lr": lr, "weight_decay": wd,
                        "param_count": sum(p.numel() for p in params), "trainable": True})
    return groups, summary


def _prune_old_checkpoints(ckpt_dir: Path, keep_last_n: int):
    if keep_last_n <= 0:
        return
    epoch_dirs = sorted(
        [path for path in ckpt_dir.glob("epoch_*") if path.is_dir()],
        key=lambda path: path.name,
    )
    if len(epoch_dirs) <= keep_last_n:
        return
    for stale in epoch_dirs[:-keep_last_n]:
        shutil.rmtree(stale, ignore_errors=True)


def _maybe_generate_epoch_visualizations(
    pipeline,
    cfg,
    logger: ExperimentLogger,
    *,
    epoch: int,
    step: int,
    report_root: Path,
    device: torch.device,
):
    viz_cfg = getattr(cfg.logging, "visualizations", None)
    every_n_epochs = int(getattr(viz_cfg, "every_n_epochs", 0) or 0)
    if every_n_epochs <= 0 or epoch % every_n_epochs != 0:
        return

    max_images = getattr(viz_cfg, "max_images", None)
    max_images = None if max_images in (None, 0) else int(max_images)
    output_dir = report_root / "dev_visualizations" / f"epoch_{epoch:03d}"
    was_training = bool(pipeline.training)
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        summary = generate_split_visualizations(
            pipeline,
            cfg,
            output_dir,
            split="dev",
            max_images=max_images,
            include_ground_truth=False,
        )
        logger.log(
            "artifact_visualizations",
            {
                "artifacts/dev_visualizations_epoch": int(epoch),
                "artifacts/dev_visualizations_path": str(output_dir),
                "artifacts/dev_visualizations_num_images": int(summary.get("num_images", 0)),
                "artifacts/dev_visualizations_elapsed_sec": float(time.perf_counter() - started),
            },
            step=step,
            epoch=epoch,
            split="system",
            update_latest=False,
        )
    except Exception as exc:
        logger.log(
            "artifact_visualizations_error",
            {
                "artifacts/dev_visualizations_epoch": int(epoch),
                "artifacts/dev_visualizations_path": str(output_dir),
                "artifacts/dev_visualizations_error_type": exc.__class__.__name__,
                "artifacts/dev_visualizations_error_message": str(exc),
            },
            step=step,
            epoch=epoch,
            split="system",
            update_latest=False,
        )
    finally:
        pipeline.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _accumulate_metrics(store: Dict[str, float], values: Dict[str, float]) -> Dict[str, float]:
    for key, value in values.items():
        store[key] = store.get(key, 0.0) + float(value)
    return store


def _to_device(batch, device):
    return _move_to_device(batch, device, non_blocking=(device.type == "cuda"))


def _move_to_device(value, device, non_blocking: bool = False):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device, non_blocking=non_blocking) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device, non_blocking=non_blocking) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device, non_blocking=non_blocking) for v in value)
    return value


def _resolve_amp_dtype(hardware_cfg, device: torch.device):
    if device.type != "cuda":
        return None
    if getattr(hardware_cfg, "bf16", False):
        return torch.bfloat16
    if getattr(hardware_cfg, "fp16", False):
        return torch.float16
    return None


def _resolve_dev_ratio(integration_cfg) -> float:
    return float(getattr(integration_cfg, "dev_ratio", getattr(integration_cfg, "val_ratio", 0.1)))


def _capture_module_state_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for key, value in module.state_dict().items():
        if isinstance(value, torch.Tensor):
            state[key] = value.detach().cpu().clone()
    return state


def _resolve_teacher_forcing_probability(curriculum_cfg, *, epoch: int, total_epochs: int) -> float:
    tf_cfg = getattr(curriculum_cfg, "teacher_forcing", None)
    if tf_cfg is None or not bool(getattr(tf_cfg, "enabled", True)):
        legacy_epochs = int(getattr(curriculum_cfg, "gt_boxes_epochs", 0))
        return 1.0 if epoch <= legacy_epochs else 0.0

    start_prob = float(getattr(tf_cfg, "start_prob", 1.0))
    end_prob = float(getattr(tf_cfg, "end_prob", 0.0))
    sustain_epochs = int(getattr(tf_cfg, "sustain_epochs", getattr(curriculum_cfg, "gt_boxes_epochs", 0)))
    transition_epochs = int(getattr(tf_cfg, "transition_epochs", max(total_epochs - sustain_epochs, 0)))

    start_prob = min(max(start_prob, 0.0), 1.0)
    end_prob = min(max(end_prob, 0.0), 1.0)

    if epoch <= sustain_epochs:
        return start_prob
    if transition_epochs <= 0:
        return end_prob

    progress = min(max(epoch - sustain_epochs, 0), transition_epochs)
    fraction = progress / max(transition_epochs, 1)
    return float(start_prob + (end_prob - start_prob) * fraction)


def _sample_teacher_forcing(probability: float) -> bool:
    probability = min(max(float(probability), 0.0), 1.0)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool(torch.rand(1).item() < probability)


def _resolve_early_stop_metric(
    monitor_name: str,
    train_metrics: Dict[str, float],
    dev_metrics: Dict[str, float],
    combined: float,
) -> float:
    if monitor_name == "joint/combined":
        return float(combined)
    if monitor_name in dev_metrics:
        return float(dev_metrics[monitor_name])
    if monitor_name in train_metrics:
        return float(train_metrics[monitor_name])
    raise KeyError(f"Early stopping monitor not found in metrics: {monitor_name}")


def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


def _resolve_device(requested: str | torch.device | None) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    name = str(requested or "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _build_bnb_config(hardware_cfg, device: torch.device):
    if not getattr(hardware_cfg, "load_in_4bit", False):
        return None
    if device.type != "cuda":
        return None
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if getattr(hardware_cfg, "bf16", False) else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _configure_training_runtime(device: torch.device, deterministic: bool = False):
    """Set torch matmul/cudnn knobs.

    When ``deterministic`` is True, do NOT enable cudnn.benchmark — that auto-tuner
    picks different kernels run-to-run, undoing whatever ``set_seed`` arranged.
    Caller passes the determinism flag from the seed setup so the two stay in sync.
    """
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not deterministic:
            torch.backends.cudnn.benchmark = True
        # else: leave whatever set_seed configured (benchmark=False, deterministic=True)


def _start_profiler_if_enabled(log_cfg, profile_dir: Path, device: torch.device):
    profiler_cfg = getattr(log_cfg, "profiler", None)
    if not profiler_cfg or not getattr(profiler_cfg, "enabled", False):
        return None

    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    profiler = profile(
        activities=activities,
        schedule=schedule(
            wait=int(profiler_cfg.wait_steps),
            warmup=int(profiler_cfg.warmup_steps),
            active=int(profiler_cfg.active_steps),
            repeat=int(profiler_cfg.repeat),
        ),
        on_trace_ready=tensorboard_trace_handler(str(profile_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    profiler.start()
    return profiler


if __name__ == "__main__":
    import sys

    train_joint(overrides=sys.argv[1:] or None)
