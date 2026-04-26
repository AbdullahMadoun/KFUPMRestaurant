from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from stage1_kcfd import probe_batch_size


class TinyProbeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"value": float(index + 1)}


class TinyProbeCollator:
    def __call__(self, examples):
        values = torch.tensor([[example["value"]] for example in examples], dtype=torch.float32)
        return {"input_ids": values, "labels": values.clone()}


class TinyProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.25]], dtype=torch.float32))

    def forward(self, input_ids, labels=None):
        pred = input_ids @ self.weight
        return SimpleNamespace(loss=((pred - labels) ** 2).mean())


def test_try_batch_runs_forward_backward_with_fake_model_dataset_and_collator():
    result = probe_batch_size._try_batch(
        TinyProbeModel(),
        TinyProbeDataset(),
        TinyProbeCollator(),
        batch_size=2,
    )

    assert result["ok"] is True
    assert result["batch_size"] == 2
    assert result["loss"] > 0.0
    assert result["peak_allocated_gb"] == 0.0


def test_probe_main_falls_back_to_next_candidate_and_keeps_effective_batch_exact(monkeypatch, capsys):
    attempts: list[int] = []

    monkeypatch.setattr(
        probe_batch_size,
        "parse_args",
        lambda: argparse.Namespace(
            export_root="unused",
            model_id="fake-qwen",
            reference_policy="exclude",
            split_seed=420,
            expected_hash=None,
            candidate_batches="4,2,1",
            num_samples=4,
            min_pixels=None,
            max_pixels=1280 * 28 * 28,
            freeze_vision=False,
            output_json=None,
        ),
    )
    monkeypatch.setattr(probe_batch_size, "Stage1KCFDDataset", lambda config: TinyProbeDataset())
    monkeypatch.setattr(probe_batch_size, "Stage1Collator", lambda processor, prompt: TinyProbeCollator())
    monkeypatch.setattr(
        probe_batch_size,
        "build_model_and_processor",
        lambda config: (TinyProbeModel(), object(), 0),
    )

    def fake_try_batch(model, dataset, collator, batch_size):
        attempts.append(batch_size)
        if batch_size == 4:
            raise RuntimeError("CUDA out of memory during fake probe")
        return {"ok": True, "batch_size": batch_size, "loss": 1.0}

    monkeypatch.setattr(probe_batch_size, "_try_batch", fake_try_batch)

    probe_batch_size.main()
    payload = json.loads(capsys.readouterr().out)

    assert attempts == [4, 2]
    assert payload["selected_per_device_batch_size"] == 2
    assert payload["recommended_gradient_accumulation_steps"] == 8
    assert payload["effective_batch_size"] == 16
    assert payload["results"] == [
        {"ok": False, "batch_size": 4, "error": "OOM", "detail": "CUDA out of memory during fake probe"},
        {"ok": True, "batch_size": 2, "loss": 1.0},
    ]


def test_probe_main_skips_candidates_larger_than_dataset(monkeypatch, capsys):
    attempts: list[int] = []

    monkeypatch.setattr(
        probe_batch_size,
        "parse_args",
        lambda: argparse.Namespace(
            export_root="unused",
            model_id="fake-qwen",
            reference_policy="exclude",
            split_seed=420,
            expected_hash=None,
            candidate_batches="8,4",
            num_samples=4,
            min_pixels=None,
            max_pixels=1280 * 28 * 28,
            freeze_vision=False,
            output_json=None,
        ),
    )
    monkeypatch.setattr(probe_batch_size, "Stage1KCFDDataset", lambda config: TinyProbeDataset())
    monkeypatch.setattr(probe_batch_size, "Stage1Collator", lambda processor, prompt: TinyProbeCollator())
    monkeypatch.setattr(
        probe_batch_size,
        "build_model_and_processor",
        lambda config: (TinyProbeModel(), object(), 0),
    )

    def fake_try_batch(model, dataset, collator, batch_size):
        attempts.append(batch_size)
        return {"ok": True, "batch_size": batch_size, "loss": 1.0}

    monkeypatch.setattr(probe_batch_size, "_try_batch", fake_try_batch)

    probe_batch_size.main()
    payload = json.loads(capsys.readouterr().out)

    assert attempts == [4]
    assert payload["selected_per_device_batch_size"] == 4
    assert payload["effective_batch_size"] == 16
