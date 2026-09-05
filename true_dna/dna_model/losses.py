"""Losses with DNA-specific target semantics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dna_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_smoothing: float,
    valid_target_ids: list[int] | tuple[int, ...] | None,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy whose smoothing mass excludes control/special tokens."""
    smoothing = float(label_smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if smoothing == 0.0:
        return F.cross_entropy(logits, targets, reduction="mean", weight=class_weights)
    if not valid_target_ids:
        raise ValueError("valid_target_ids are required when DNA-only label smoothing is enabled")

    valid_ids = torch.as_tensor(valid_target_ids, dtype=torch.long, device=logits.device)
    if valid_ids.ndim != 1 or valid_ids.numel() == 0:
        raise ValueError("valid_target_ids must be a non-empty one-dimensional sequence")
    if int(valid_ids.min().item()) < 0 or int(valid_ids.max().item()) >= logits.size(-1):
        raise ValueError("valid_target_ids contain an ID outside the logits vocabulary")

    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.index_select(1, valid_ids).mean(dim=1)
    losses = (1.0 - smoothing) * nll + smoothing * smooth
    if class_weights is None:
        return losses.mean()
    target_weights = class_weights.to(logits.device)[targets]
    return (losses * target_weights).sum() / target_weights.sum().clamp_min(torch.finfo(losses.dtype).eps)
