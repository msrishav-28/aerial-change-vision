from __future__ import annotations

import torch


@torch.no_grad()
def threshold_scores(pairs: torch.Tensor) -> torch.Tensor:
    """Baseline score: mean absolute difference between the two dates."""
    diff = pairs[:, 1].float() - pairs[:, 0].float()
    return diff.abs().mean(dim=tuple(range(1, diff.dim())))
