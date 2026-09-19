from __future__ import annotations

import torch
from torch import nn


class PairDiffBaseline(nn.Module):
    """Cheap stand-in for |t1-t0| + tiny conv. Not a paper model."""

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        # pair: B, 2, C, H, W
        b, two, c, h, w = pair.shape
        x = pair.reshape(b, two * c, h, w)
        return self.net(x)


def build_model(name: str) -> nn.Module:
    if name in {"pair", "siamese", "diff"}:
        return PairDiffBaseline()
    raise ValueError(name)
