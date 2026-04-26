from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .config import Stage1Config
from .dataset import build_datasets_from_config, incomplete_export_counts, preflight_stage1_kcfd_export
from .eval import Stage1Evaluator
from .model import ModelConfig, build_model_and_processor, trainable_parameter_summary
from .trainer import Stage1Trainer, TrainingConfig, save_json
from .visualize import Stage1Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1-only Qwen-VL trainer for v3 export boxes")
    parser.add_argument("--export-root", required=True, help="Path to v3 export root containing manifest.json/items.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage1_kcfd")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--reference-policy", choices=["pause", "exclude", "train", "include"], default="pause")
    parser.add_argument("--splits-path", default=None)
    parser.add_argument("--expected-version", default="v3")
    parser.add_argument("--expected-hash", default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--freeze-vision", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--visualize-every-epochs", type=int, default=5)
    parser.add_argument("--visualize-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--train-max-images", type=int, default=0)
    parser.add_argument("--skip-preflight-gate", action="store_true")
    parser.add_argument("--allow-incomplete-export", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_pause_for_reference_policy(config: Stage1Config, stats: Dict[str, Any], *, skip_gate: bool) -> Stage1Config:
    if config.reference_policy != "pause":
        return config
    ref_items = int(stats.get("reference_items", 0) or 0)
    ref_images = int(stats.get("reference_images", 0) or 0)
    if ref_items == 0:
        return config
    if skip_gate:
        raise SystemExit(
            "Refusing to train with --reference-policy pause even though --skip-preflight-gate was set. "
            f"The dataset contains {ref_items} reference items across {ref_images} images. "
            "Choose --reference-policy exclude, train, or include explicitly."
        )
    raise SystemExit(
        "Preflight gate: dataset contains "
        f"{ref_items} reference items across {ref_images} images. "
        "Rerun with --reference-policy exclude|train|include after choosing the no-leakage policy."
    )


def fail_on_incomplete_export(stats: Dict[str, Any], *, allow: bool) -> None:
    bad = incomplete_export_counts(stats)
    if bad and not allow:
        raise SystemExit(
            "Preflight gate: export has incomplete Stage 1 training data "
            f"{bad}. Fix the export or rerun with --allow-incomplete-export."
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = Stage1Config(
        export_root=Path(args.export_root),
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        seed=args.seed,
        split_seed=args.split_seed if args.split_seed is not None else args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        reference_policy=args.reference_policy,
        expected_version=args.expected_version,
        expected_hash=args.expected_hash,
        splits_path=Path(args.splits_path) if args.splits_path else None,
        allow_incomplete_export=args.allow_incomplete_export,
        train_max_images=args.train_max_images,
        eval_max_samples=args.eval_max_samples,
    )

    preflight_stats = preflight_stage1_kcfd_export(
        config.export_root,
        expected_version=config.expected_version,
        expected_hash=config.expected_hash,
    )
    run_name = args.run_name or "preflight"
    preflight_dir = Path(args.output_dir) / run_name / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    save_json(preflight_dir / "dataset_report.json", preflight_stats)
    if args.preflight_only:
        print(json.dumps(preflight_stats, indent=2, sort_keys=True))
        return
    config = maybe_pause_for_reference_policy(config, preflight_stats, skip_gate=args.skip_preflight_gate)
    fail_on_incomplete_export(preflight_stats, allow=args.allow_incomplete_export)
    datasets = build_datasets_from_config(config)
    save_json(preflight_dir / "dataset_report.json", datasets.report)

    model_config = ModelConfig(
        model_id=args.model_id,
        device_map=args.device_map,
        unfreeze_vision=not args.freeze_vision,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        visualize_every_epochs=args.visualize_every_epochs,
        visualize_samples=args.visualize_samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    model, processor, vision_params = build_model_and_processor(model_config)
    model_summary = trainable_parameter_summary(model)
    model_summary["vision_params_unfrozen"] = float(vision_params)

    collator = datasets.make_collator(processor)
    evaluator = Stage1Evaluator(prompt=config.prompt, max_samples=args.eval_max_samples or None)
    visualizer = Stage1Visualizer(prompt=config.prompt)
    trainer = Stage1Trainer(
        model=model,
        processor=processor,
        train_dataset=datasets.train,
        val_dataset=datasets.val,
        collator=collator,
        config=training_config,
        evaluator=evaluator,
        visualizer=visualizer,
        metadata={
            "stage1_config": datasets.serializable_config,
            "dataset_report": datasets.report,
            "model_config": asdict(model_config),
            "model_summary": model_summary,
        },
    )
    run_dir = trainer.train()
    save_json(run_dir / "model_summary.json", model_summary)
    print(f"Stage 1 run complete: {run_dir}")


if __name__ == "__main__":
    main()
