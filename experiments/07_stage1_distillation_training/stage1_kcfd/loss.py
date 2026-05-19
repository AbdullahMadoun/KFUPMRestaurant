from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class LossBreakdown:
    loss: float
    ce_loss: float
    giou_eval_loss: float


def ce_loss_from_outputs(outputs) -> torch.Tensor:
    if not hasattr(outputs, "loss") or outputs.loss is None:
        raise ValueError("model outputs must expose a CE loss when labels are provided")
    return outputs.loss


def eval_loss_summary(ce_loss: float, mean_matched_giou: float, *, lambda_giou: float = 2.0) -> Dict[str, float]:
    """Report CE + eval-only GIoU loss.

    This is intentionally not used for backpropagation because generated JSON
    boxes are discrete text and parsed GIoU is not differentiable to Qwen tokens.
    """
    giou_eval_loss = max(0.0, 1.0 - float(mean_matched_giou))
    return {
        "loss": float(ce_loss) + float(lambda_giou) * giou_eval_loss,
        "ce_loss": float(ce_loss),
        "giou_loss": giou_eval_loss,
    }
