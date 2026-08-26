"""A model that sees five candidate seams at once and picks one.

Why this shape, and not another re-ranker
-----------------------------------------
M409 leaves one door and M407 sizes it. The shipping roster finds the true
partner first 0.299 of the time and inside its top five 0.475 of the time; the
percolation knee is at 450 to 500 correct bonds and 0.475 of 1104 is 524. So
choosing perfectly inside the top five solves the board. We currently choose
right 66 per cent of the time there and need 86.

Everything that has failed at this task shares one property: it scored
candidates ONE AT A TIME and compared the numbers afterwards. The matcher does,
the selector does, M157 and M164's re-rankers did, M404's best-square-per-
fragment did. A margin between two scores is a SUMMARY of a comparison; the
comparison itself -- five seams side by side, where one continues a line and
four merely match colour -- has never been shown to a model.

And the other family is closed from the opposite side. M410 measured that no
global objective beats plain per-fragment top-1: the Hungarian assignment on
seam scores reaches 332.4 correct bonds, square-closure search 335.6, mutual
best 315.7, plain top-1 348.4. Optimising over arrangements does not help, so
the remaining move is to make the per-fragment CHOICE better.

The design follows from that
----------------------------
* The five candidates go in together and attend to each other, so the score of
  one is computed in the presence of its rivals. That is the whole point; a
  per-candidate tower with a softmax on top would be the selector again.
* A sixth NONE option, because the truth is outside the top five for 0.525 of
  fragments and a model forced to choose would learn to guess. Abstaining is
  worth more than guessing here: M409 measured that at our operating point
  precision protects the four hundred fragments outside the block.
* The seam patch is pixels from BOTH fragments across the join, `strip` columns
  each side. M34 measured that the signal decays fast with distance from the
  edge -- inset 0/1/2/3 gives R@1 0.159/0.084/0.059/0.040 -- so a wide patch
  would mostly add noise.
* The fused score and the rank come in as scalars beside the pixels, because
  the model should improve on the matcher rather than relearn it from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

K = 5


class SeamEncoder(nn.Module):
    """(B, 3, 20, 2*strip) -> (B, dim). One candidate's join, as pixels."""

    def __init__(self, ch=48, dim=128, strip=4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, ch * 2), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Linear(ch * 2, dim)

    def forward(self, x):
        h = self.body(x)
        return self.head(h.mean((2, 3)))


class Choose5(nn.Module):
    """Five joins in, one choice out, with a NONE option.

    The candidates are encoded independently and then attend to one another
    before any of them is scored, which is the one thing a per-pair model
    cannot do.
    """

    def __init__(self, ch=48, dim=128, strip=4, layers=2, heads=4):
        super().__init__()
        self.strip = strip
        self.enc = SeamEncoder(ch, dim, strip)
        self.scalars = nn.Sequential(
            nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        enc_layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 2, dropout=0.0, batch_first=True,
            norm_first=True, activation="gelu")
        self.mix = nn.TransformerEncoder(enc_layer, layers)
        self.none = nn.Parameter(torch.zeros(1, 1, dim))
        self.score = nn.Linear(dim, 1)

    def forward(self, patch, scalars):
        """patch (B, K, 3, 20, 2*strip); scalars (B, K, 4) -> logits (B, K+1)."""
        b, k = patch.shape[:2]
        z = self.enc(patch.reshape(b * k, *patch.shape[2:])).reshape(b, k, -1)
        z = z + self.scalars(scalars)
        z = torch.cat([z, self.none.expand(b, 1, z.shape[-1])], 1)
        z = self.mix(z)
        return self.score(z).squeeze(-1)


def seam_patch(tiles, src, dst, axis, strip=4):
    """The join between two fragments, as one image.

    `tiles` is (N, 20, 20, 3). For a horizontal join the last `strip` columns of
    the source meet the first `strip` of the destination; a vertical join is the
    same picture transposed, so one encoder serves both.
    """
    a, b = tiles[src], tiles[dst]
    if axis == "h":
        p = torch.cat([a[:, :, -strip:], b[:, :, :strip]], 2)
    else:
        p = torch.cat([a[:, -strip:, :], b[:, :strip, :]], 1)
        p = p.transpose(1, 2)
    return p.permute(0, 3, 1, 2).contiguous()


def choose_loss(logits, label):
    """Cross-entropy over the five candidates plus NONE.

    `label` is the index of the true candidate, or K when the truth is outside
    the shortlist. Rows with no true partner at all -- the last column and the
    bottom row of the board -- are dropped by the caller.
    """
    return F.cross_entropy(logits, label)
