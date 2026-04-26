from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from .config import CANONICAL_STAGE1_SPLIT_SEED, Stage1Config
from .dataset import Stage1Collator, Stage1KCFDDataset
from .model import ModelConfig, build_model_and_processor, first_parameter_device
from .trainer import forward_ce_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe safe Stage 1 Qwen-VL microbatch size on this GPU.")
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--reference-policy", choices=["exclude", "train", "include"], default="exclude")
    parser.add_argument("--split-seed", type=int, default=CANONICAL_STAGE1_SPLIT_SEED)
    parser.add_argument("--expected-hash", default=None)
    parser.add_argument("--candidate-batches", default="4,2,1")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--freeze-vision", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _try_batch(model, dataset, collator, batch_size: int) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    batch = next(iter(loader))
    model.train()
    model.zero_grad(set_to_none=True)
    device = first_parameter_device(model)
    loss = forward_ce_loss(model, batch, device)
    loss.backward()
    model.zero_grad(set_to_none=True)
    peak_gb = 0.0
    reserved_gb = 0.0
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return {
        "ok": True,
        "batch_size": batch_size,
        "loss": float(loss.detach().cpu()),
        "peak_allocated_gb": peak_gb,
        "max_memory_reserved_gb": reserved_gb,
    }


def main() -> None:
    args = parse_args()
    candidates = [int(part) for part in args.candidate_batches.split(",") if part.strip()]
    candidates = sorted(set(candidates), reverse=True)
    data_config = Stage1Config(
        export_root=Path(args.export_root),
        split="train",
        reference_policy=args.reference_policy,
        split_seed=args.split_seed,
        expected_hash=args.expected_hash,
        allow_incomplete_export=False,
        train_max_images=max(args.num_samples, max(candidates)),
    )
    dataset = Stage1KCFDDataset(data_config)
    model, processor, _ = build_model_and_processor(
        ModelConfig(
            model_id=args.model_id,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            unfreeze_vision=not args.freeze_vision,
        )
    )
    collator = Stage1Collator(processor, prompt=data_config.prompt)

    results: List[Dict[str, Any]] = []
    selected = 1
    for batch_size in candidates:
        if len(dataset) < batch_size:
            continue
        try:
            result = _try_batch(model, dataset, collator, batch_size)
            results.append(result)
            selected = batch_size
            break
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append({"ok": False, "batch_size": batch_size, "error": "CUDA OOM", "detail": str(exc)[:300]})
        except RuntimeError as exc:
            text = str(exc)
            if "out of memory" in text.lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results.append({"ok": False, "batch_size": batch_size, "error": "OOM", "detail": text[:300]})
            else:
                raise

    payload = {
        "selected_per_device_batch_size": selected,
        "recommended_gradient_accumulation_steps": max(1, (16 + selected - 1) // selected),
        "effective_batch_size": selected * max(1, (16 + selected - 1) // selected),
        "results": results,
        "config": asdict(data_config),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
