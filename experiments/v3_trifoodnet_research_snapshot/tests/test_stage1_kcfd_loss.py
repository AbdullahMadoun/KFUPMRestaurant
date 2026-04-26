from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from stage1_kcfd.loss import ce_loss_from_outputs, eval_loss_summary


def test_eval_loss_summary_combines_ce_and_clamped_giou_penalty():
    summary = eval_loss_summary(ce_loss=1.25, mean_matched_giou=0.40, lambda_giou=2.5)

    assert summary == {
        "loss": 2.75,
        "ce_loss": 1.25,
        "giou_loss": 0.60,
    }


def test_eval_loss_summary_clamps_negative_giou_penalty_to_zero():
    summary = eval_loss_summary(ce_loss=0.75, mean_matched_giou=1.20, lambda_giou=10.0)

    assert summary["loss"] == 0.75
    assert summary["giou_loss"] == 0.0


def test_ce_loss_from_outputs_returns_model_loss_tensor():
    loss = torch.tensor(3.5)

    assert ce_loss_from_outputs(SimpleNamespace(loss=loss)) is loss


@pytest.mark.parametrize("outputs", [SimpleNamespace(), SimpleNamespace(loss=None)])
def test_ce_loss_from_outputs_rejects_missing_or_none_loss(outputs):
    with pytest.raises(ValueError, match="CE loss"):
        ce_loss_from_outputs(outputs)
