from __future__ import annotations

from pathlib import Path

import torch

from stage1_kcfd.model import load_vision_encoder_state, save_vision_encoder_state, trainable_parameter_summary


class FakeQwenLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(2, 2)
        self.text = torch.nn.Linear(2, 1)


def test_vision_encoder_state_round_trips_for_unfrozen_checkpoint(tmp_path: Path):
    model = FakeQwenLike()
    with torch.no_grad():
        model.visual.weight.fill_(3.0)
    path = tmp_path / "vision_encoder.pt"

    saved = save_vision_encoder_state(model, path)
    with torch.no_grad():
        model.visual.weight.fill_(0.0)
    result = load_vision_encoder_state(model, path)

    assert path.exists()
    assert saved == sum(parameter.numel() for parameter in model.visual.state_dict().values())
    assert not result.missing_keys
    assert torch.allclose(model.visual.weight, torch.full_like(model.visual.weight, 3.0))


def test_trainable_parameter_summary_counts_requires_grad_flags():
    model = FakeQwenLike()
    for parameter in model.text.parameters():
        parameter.requires_grad = False

    summary = trainable_parameter_summary(model)

    assert summary["total_params"] > summary["trainable_params"] > 0
    assert 0 < summary["trainable_pct"] < 100
