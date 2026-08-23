"""Predict the target's coarse colour field from the UNORDERED bag of tiles.

Why this exists
---------------
M137 changed the scoreboard: absolute SSIM on this task mostly reports how close
the output is to a constant, so every arm is now quoted as a gain over the flat
fill at our own tiles' mean colour.  On that scale the deployed submission is
-0.141, our best layout is -0.002, and the true layout is +0.131.  The prize is
real but it is locked behind an arrangement we cannot recover.

M138 found a second door to the same kind of structure.  A correct 3x3 version
of the target is worth +0.032 -- more than the leader's estimated +0.02 -- and a
4x4 is worth +0.046.  That is 27 to 48 numbers rather than a permutation of 576,
and crucially it does not require placing anything.  The tile-mean SNR is 1.60
(M135): enough signal for a few dozen numbers, nowhere near enough for a 576-way
matching problem, which is exactly why every assembly route has stalled.

What the model may legitimately use
-----------------------------------
A photograph is not a constant.  It has sky above and ground below, and the
palette of a photograph says a great deal about how it is laid out: a bag of
pale blue and dark green tiles is a landscape, and a landscape is bright at the
top.  That is a real regularity of the data, and reading it is what every
restoration prior does.

Two design choices keep the result honest and legible:

* The head is zero-initialised and the output is `flat + delta`, so an untrained
  model emits EXACTLY the flat fill and scores a gain of 0.000.  Any number
  above zero is something the model found, not something the rendering did.
  (M131 used the same trick to make "no effect" unarguable.)
* Aggregation over tiles is by order statistics, not just a mean.  The mean of
  576 tiles is nearly the flat fill itself and carries almost nothing extra;
  the quantiles describe the palette's SHAPE, which is what distinguishes a
  landscape from an interior.

The nuisance this has to survive is the same one M129-M132 fought in the
matcher: every tile carries an independent brightness offset and contrast gain.
Averaged over 576 tiles those largely cancel, which is why a coarse target is
the right size for this signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RING_SIGMA = 13.4
QUANTILES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)


class TileEncoder(nn.Module):
    """(B*T, 3, 20, 20) -> (B*T, dim). Content, not position."""

    def __init__(self, ch=48, dim=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(6, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.GroupNorm(8, ch * 2), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1), nn.GroupNorm(8, ch * 2), nn.GELU(),
        )
        self.proj = nn.Linear(ch * 2 * 3 * 3, dim)

    def prep(self, x):
        """Raw view plus a noise-aware normalised one.

        The same pair the matcher uses.  M130 measured that dropping the raw
        view costs real signal -- neighbouring tiles' true brightness is
        correlated because they are pieces of one photograph -- so both are
        supplied and the model weighs them itself.
        """
        s = x.flatten(2)
        mu = s.mean(-1)[:, :, None, None]
        var = s.var(-1)[:, :, None, None] - RING_SIGMA ** 2
        sd = torch.sqrt(torch.clamp(var, min=(0.25 * RING_SIGMA) ** 2))
        return torch.cat([x / 255.0 - 0.5, (x - mu) / sd / 4.0], 1)

    def forward(self, x):
        h = self.body(self.prep(x))
        return self.proj(h.flatten(1))


class CoarseField(nn.Module):
    """Bag of tiles -> n x n x 3 colour field, as a delta from the flat fill."""

    def __init__(self, n=8, ch=48, dim=128, hidden=512):
        super().__init__()
        self.n = n
        self.enc = TileEncoder(ch, dim)
        self.register_buffer("qs", torch.tensor(QUANTILES))
        # per-tile summary statistics enter directly as well: the palette is the
        # single most informative thing here and should not have to survive a
        # convolutional bottleneck to reach the head
        stat_dim = 4 * len(QUANTILES) + 8
        agg_dim = dim * (2 + len(QUANTILES))
        self.head = nn.Sequential(
            nn.LayerNorm(agg_dim + stat_dim),
            nn.Linear(agg_dim + stat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n * n * 3),
        )
        # zero output: an untrained model emits the flat fill exactly
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def stats(self, tiles):
        """(B, T, 3, 20, 20) -> per-board palette statistics, order-invariant."""
        mu = tiles.mean((3, 4))                             # (B, T, 3)
        sd = tiles.std((3, 4)).mean(2, keepdim=True)        # (B, T, 1)
        f = torch.cat([mu, sd], 2)                          # (B, T, 4)
        # quantile() refuses half precision, and under autocast these arrive
        # as fp16; the statistics are cheap enough to compute in fp32
        f = f.float()
        q = torch.quantile(f, self.qs.to(f.dtype), dim=1)   # (Q, B, 4)
        q = q.permute(1, 0, 2).flatten(1)
        extra = torch.cat([f.mean(1), f.std(1)], 1)         # (B, 8)
        return torch.cat([q, extra], 1) / 128.0

    def aggregate(self, e):
        """(B, T, D) -> (B, D * (2 + Q)); mean, max and per-dimension quantiles."""
        e = e.float()
        q = torch.quantile(e, self.qs.to(e.dtype), dim=1).permute(1, 0, 2).flatten(1)
        return torch.cat([e.mean(1), e.amax(1), q], 1)

    def forward(self, tiles):
        """tiles: (B, T, 3, 20, 20) in 0..255. Returns (B, 3, n, n)."""
        b, t = tiles.shape[:2]
        e = self.enc(tiles.reshape(b * t, *tiles.shape[2:])).reshape(b, t, -1)
        z = torch.cat([self.aggregate(e), self.stats(tiles)], 1).to(e.dtype)
        delta = self.head(z).reshape(b, 3, self.n, self.n)
        flat = tiles.mean((1, 3, 4))[:, :, None, None] / 255.0
        return flat + delta


def render(field, size=480, mode="bicubic"):
    """n x n field -> full-resolution image, differentiably."""
    return F.interpolate(field, size=(size, size), mode=mode,
                         align_corners=False).clamp(0.0, 1.0)
