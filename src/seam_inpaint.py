"""Neighbour scoring by seam inpainting (after Bridger et al., CVPR 2020).

Every measure tried so far compares the two facing border strips directly, and
all of them saturate near R@1 0.17 because those strips are the noisiest part of
the tile (border error 15.1 versus 12.0 in the interior).

This scores a pair differently: the strip at the join is REMOVED, a network
predicts it from the surrounding context of both pieces, and compatibility is
how well that prediction agrees with what was actually observed there.  For a
true neighbour the two pieces constrain the missing strip consistently; for a
stranger they do not.  The prediction is formed from the cleaner interior, so
the comparison is not noise-against-noise.

Bridger et al. reuse a GAN discriminator for both inpainting and classification.
This is the regression-only core of that idea, which is what can be trained and
evaluated inside our budget.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FS


class SeamInpainter(nn.Module):
    """(pair with a hole at the join) -> the missing strip.

    Input  (B,4,20,40): joined pair, the central `hole` columns zeroed, plus a
                        mask plane marking what was removed.
    Output (B,3,20,hole).
    """

    def __init__(self, ch: int = 64, blocks: int = 5, hole: int = 4):
        super().__init__()
        self.hole = hole
        self.stem = nn.Conv2d(4, ch, 3, padding=1, padding_mode="reflect")
        dil = [1, 2, 3, 2, 1][:blocks] + [1] * max(0, blocks - 5)
        self.body = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(ch, ch, 3, padding=d, dilation=d, padding_mode="reflect"),
                          nn.GroupNorm(8, ch), nn.GELU()) for d in dil])
        self.head = nn.Conv2d(ch, 3, 3, padding=1, padding_mode="reflect")

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        m = pair.mean(dim=(1, 2, 3), keepdim=True)
        s = pair.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
        x = (pair - m) / s
        mask = torch.zeros_like(x[:, :1])
        lo = FS - self.hole // 2
        mask[:, :, :, lo:lo + self.hole] = 1.0
        h = self.body(self.stem(torch.cat([x * (1 - mask), mask], dim=1)))
        out = self.head(h)[:, :, :, lo:lo + self.hole]
        return out * s + m


def join_pair(a: torch.Tensor, b: torch.Tensor, axis: str) -> torch.Tensor:
    """a,b: (B,3,20,20) -> (B,3,20,40) with b to the right of / below a."""
    if axis == "h":
        return torch.cat([a, b], dim=3)
    return torch.cat([a.transpose(2, 3), b.transpose(2, 3)], dim=3)


def observed_strip(pair: torch.Tensor, hole: int) -> torch.Tensor:
    lo = FS - hole // 2
    return pair[:, :, :, lo:lo + hole]
