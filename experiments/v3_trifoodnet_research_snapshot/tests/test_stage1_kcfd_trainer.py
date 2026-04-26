from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stage1_kcfd.trainer import Stage1Trainer, TrainingConfig


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"value": float(index + 1)}


class TinyCollator:
    def __call__(self, examples):
        values = torch.tensor([[example["value"]] for example in examples], dtype=torch.float32)
        return {"input_ids": values, "labels": values.clone()}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, 1)

    def forward(self, input_ids, labels=None):
        pred = self.head(self.visual(input_ids))
        loss = ((pred - labels) ** 2).mean()
        return SimpleNamespace(loss=loss)

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "tiny_model.pt")


class TinyProcessor:
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "processor.txt").write_text("ok", encoding="utf-8")


def test_trainer_smoke_updates_and_saves_checkpoint(tmp_path: Path):
    model = TinyModel()
    config = TrainingConfig(
        epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=0,
        num_workers=0,
        eval_every_epochs=99,
        output_dir=str(tmp_path),
        run_name="tiny",
    )
    trainer = Stage1Trainer(
        model=model,
        processor=TinyProcessor(),
        train_dataset=TinyDataset(),
        val_dataset=TinyDataset(),
        collator=TinyCollator(),
        config=config,
        evaluator=None,
        metadata={"model_config": {"model_id": "tiny/base"}},
    )

    run_dir = trainer.train()

    checkpoint = run_dir / "checkpoints" / "epoch_001"
    assert (checkpoint / "model" / "tiny_model.pt").exists()
    assert (checkpoint / "processor" / "processor.txt").exists()
    assert (checkpoint / "vision_encoder.pt").exists()
    assert (checkpoint / "checkpoint_manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 0, "epochs must be > 0"),
        ("per_device_batch_size", 0, "per_device_batch_size must be > 0"),
        ("gradient_accumulation_steps", 0, "gradient_accumulation_steps must be > 0"),
        ("learning_rate", 0.0, "learning_rate must be > 0"),
        ("eval_every_epochs", 0, "eval_every_epochs must be > 0"),
        ("visualize_every_epochs", -1, "visualize_every_epochs must be >= 0"),
    ],
)
def test_training_config_rejects_invalid_values(field: str, value, message: str):
    kwargs = {
        "epochs": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "eval_every_epochs": 1,
        "visualize_every_epochs": 0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_stage1_requirements_pin_numpy_below_2_3():
    requirements_path = Path(__file__).resolve().parents[1] / "requirements-stage1.txt"
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "numpy<2.3" in requirements
