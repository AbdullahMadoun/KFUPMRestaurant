from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from .loss import eval_loss_summary
from .model import first_parameter_device, model_input_tensors, tensor_batch_to_device
from .model import save_vision_encoder_state


@dataclass
class TrainingConfig:
    epochs: int = 10
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    num_workers: int = 4
    eval_every_epochs: int = 1
    visualize_every_epochs: int = 5
    visualize_samples: int = 8
    max_new_tokens: int = 512
    output_dir: str = "outputs/stage1_kcfd"
    run_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.eval_every_epochs <= 0:
            raise ValueError("eval_every_epochs must be > 0")
        if self.visualize_every_epochs < 0:
            raise ValueError("visualize_every_epochs must be >= 0; use 0 to disable visualization")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def log_metrics(writer: SummaryWriter, prefix: str, metrics: Dict[str, float], step: int) -> None:
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", float(value), step)


def log_image_file(writer: SummaryWriter, tag: str, path: Path, step: int) -> None:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(rgb.tobytes()))
        tensor = data.view(height, width, 3).permute(2, 0, 1)
        writer.add_image(tag, tensor, step)


def forward_ce_loss(model, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    batch = tensor_batch_to_device(batch, device)
    outputs = model(**model_input_tensors(batch))
    return outputs.loss


@torch.no_grad()
def evaluate_ce_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        losses.append(float(forward_ce_loss(model, batch, device).detach().cpu()))
    return sum(losses) / max(len(losses), 1)


class Stage1Trainer:
    def __init__(
        self,
        *,
        model,
        processor,
        train_dataset,
        val_dataset,
        collator,
        config: TrainingConfig,
        evaluator,
        visualizer=None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collator = collator
        self.config = config
        self.evaluator = evaluator
        self.visualizer = visualizer
        self.run_name = config.run_name or time.strftime("stage1-kcfd-%Y%m%d-%H%M%S")
        self.run_dir = Path(config.output_dir) / self.run_name
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self.metadata = metadata or {}

    def _loader(self, dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            pin_memory=torch.cuda.is_available(),
        )

    def _save_checkpoint(self, name: str, metrics: Dict[str, Any]) -> None:
        path = self.run_dir / "checkpoints" / name
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "model")
        vision_params_saved = save_vision_encoder_state(self.model, path / "vision_encoder.pt")
        self.processor.save_pretrained(path / "processor")
        manifest = {
            "base_model": (self.metadata.get("model_config") or {}).get("model_id", ""),
            "adapter_dir": "model",
            "processor_dir": "processor",
            "vision_encoder": "vision_encoder.pt",
            "vision_params_saved": vision_params_saved,
            "checkpoint_loader": "stage1_kcfd.model.load_stage1_checkpoint",
            "note": "Load both the PEFT adapter and vision_encoder.pt; the vision encoder was unfrozen during training.",
        }
        save_json(path / "checkpoint_manifest.json", manifest)
        save_json(path / "metrics.json", {**metrics, "vision_params_saved": vision_params_saved})

    def train(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.run_dir / "trainer_config.json", asdict(self.config))
        save_json(self.run_dir / "run_metadata.json", self.metadata)

        train_loader = self._loader(self.train_dataset, shuffle=True)
        val_loader = self._loader(self.val_dataset, shuffle=False)
        optimizer = AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        updates_per_epoch = math.ceil(len(train_loader) / max(self.config.gradient_accumulation_steps, 1))
        total_steps = max(1, updates_per_epoch * self.config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        device = first_parameter_device(self.model)
        best_score = (-1.0, -1.0, -1.0)
        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                loss = forward_ce_loss(self.model, batch, device)
                (loss / max(self.config.gradient_accumulation_steps, 1)).backward()
                running_loss += float(loss.detach().cpu())

                if step % self.config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                            self.config.max_grad_norm,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    self.writer.add_scalar("train/loss", running_loss / max(step, 1), global_step)
                    self.writer.add_scalar("train/ce_loss", running_loss / max(step, 1), global_step)
                    self.writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            train_loss = running_loss / max(len(train_loader), 1)
            self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            self._save_checkpoint(f"epoch_{epoch:03d}", {"epoch": epoch, "train_loss": train_loss})

            if epoch % self.config.eval_every_epochs == 0:
                val_ce = evaluate_ce_loss(self.model, val_loader, device)
                eval_report = self.evaluator.evaluate(
                    self.model,
                    self.processor,
                    self.val_dataset,
                    output_path=self.run_dir / "eval" / f"epoch_{epoch:03d}.json",
                    max_new_tokens=self.config.max_new_tokens,
                )
                metrics = dict(eval_report)
                metrics.update(eval_loss_summary(val_ce, metrics.get("mean_matched_giou", 0.0)))
                save_json(self.run_dir / "latest_metrics.json", {"epoch": epoch, **metrics})
                log_metrics(self.writer, "val", metrics, epoch)

                score = (
                    float(metrics.get("exact_count_accuracy", 0.0)),
                    float(metrics.get("exact_set_match@0.5", 0.0)),
                    float(metrics.get("matched_f1@0.5", 0.0)),
                )
                if score > best_score:
                    best_score = score
                    self._save_checkpoint("best", {"epoch": epoch, **metrics})

            if (
                self.visualizer is not None
                and self.config.visualize_every_epochs > 0
                and epoch % self.config.visualize_every_epochs == 0
            ):
                image_paths = self.visualizer.generate_validation_panel(
                    self.model,
                    self.processor,
                    self.val_dataset,
                    self.run_dir / "visualizations" / f"epoch_{epoch:03d}",
                    max_samples=self.config.visualize_samples,
                    max_new_tokens=self.config.max_new_tokens,
                )
                for idx, image_path in enumerate(image_paths):
                    self.writer.add_text(f"visualizations/epoch_{epoch}_{idx}", str(image_path), epoch)
                    log_image_file(self.writer, f"visualizations/epoch_{epoch}_{idx}", Path(image_path), epoch)

        self.writer.close()
        return self.run_dir
