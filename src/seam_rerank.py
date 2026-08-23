"""Second stage: re-rank a shortlist by looking at both edges at once.

The descriptor stage retrieves well and decides badly -- R@20 0.568 against R@1
0.200 at the same checkpoint.  The true neighbour is usually IN the shortlist;
what fails is picking it.  That is exactly the shape of problem a joint model
fixes and a siamese one cannot: a dot product compares two summaries, while
continuity across a seam is a local, spatial property of the two edges TOGETHER
-- an edge continuing at row 7 with a slight vertical drift is visible in the
joint patch and invisible after each side has been pooled into a vector.

A joint scorer over all 576 candidates is quadratic and untrainable, which is
what sank the earlier pair CNN (M20).  Over a shortlist of 20 it is 30x cheaper
than the retriever it sits behind.

The two tiles are placed side by side so convolutions straddle the seam.  Each
is normalised by its OWN tile's statistics, with the noise variance removed
(std(dirty)^2 = a^2 s^2 + n^2), so the per-tile affine cannot be used as a cue
and flat tiles are not divided by their own noise.

Width matters more than it looks.  A first version fed only 4 columns either
side of the seam, on the reasoning that adjacency lives in a 2-pixel ring (M28).
It scored 0.188 within a shortlist where the retriever scores 0.48 -- because
the retriever reads the WHOLE tile and uses the interior to clean the ring,
while a narrow strip is pure noise with no context to denoise it.  A second
stage must not see less than the stage it corrects.

The retriever's own score enters as a constant input plane for the same reason:
starting from its opinion means the head refines rather than competes, so the
fused result cannot be structurally worse than what it re-ranks.

That plane is dropped at random during training.  Handed the retriever's score
unconditionally the model simply copies it -- pick accuracy pinned at 0.486
against the retriever's own 0.484, flat from step 200 to 2000, which is the
signature of a shortcut rather than a plateau.  Dropping it half the time forces
the pixels to carry the decision, while still letting the score help when it is
there.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RING_SIGMA = 13.4


def _norm_stats(tiles):
    """Per-tile mean and signal std, noise variance subtracted. tiles: (N,3,20,20)."""
    s = tiles.flatten(2)
    mu = s.mean(-1)
    var = s.var(-1) - RING_SIGMA ** 2
    sd = torch.sqrt(torch.clamp(var, min=(0.25 * RING_SIGMA) ** 2))
    return mu, sd


def build_patches(tiles, left_idx, right_idx, axis, width=20, score=None,
                  score_drop=0.0):
    """Facing tiles of the given pairs, laid out with the seam down the middle.

    left_idx/right_idx are flat index tensors of equal length; for axis "v" the
    tile is transposed first so the seam is vertical in both cases and one
    convolution stack serves both axes.  `width` counts columns kept from each
    side, 20 meaning the whole tile.  `score` optionally supplies the retriever's
    log-score per pair, broadcast as a seventh input plane.
    """
    t = tiles if axis == "h" else tiles.transpose(2, 3)
    mu, sd = _norm_stats(tiles)
    mu = mu[:, :, None, None]; sd = sd[:, :, None, None]
    a = t[left_idx][:, :, :, -width:]
    b = t[right_idx][:, :, :, :width]
    an = (a - mu[left_idx]) / sd[left_idx]
    bn = (b - mu[right_idx]) / sd[right_idx]
    out = torch.cat([torch.cat([an, bn], 3),
                     torch.cat([a, b], 3) / 255.0 - 0.5], 1)
    if score is not None:
        s = score.reshape(-1, 1, 1, 1).expand(-1, 1, out.shape[2], out.shape[3])
        s = s.to(out.dtype)
        if score_drop > 0.0:
            keep = (torch.rand(out.shape[0], 1, 1, 1, device=out.device)
                    >= score_drop).to(out.dtype)
            s = s * keep
        out = torch.cat([out, s], 1)
    return out


class SeamRerank(nn.Module):
    """(B, 6, 20, 2*width) -> one score per pair."""

    def __init__(self, ch=64, blocks=3, width=20, use_score=True):
        super().__init__()
        self.width = width
        self.use_score = use_score
        layers = [nn.Conv2d(7 if use_score else 6, ch, 3, padding=1), nn.GELU()]
        for _ in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.GELU()]
        self.body = nn.Sequential(*layers)
        # pooling only along the seam: the across-seam axis is where the evidence
        # lives, so it is kept and flattened rather than averaged away
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch * 2 * width, 128),
                                  nn.GELU(), nn.Linear(128, 1))

    def forward(self, patch):
        h = self.body(patch).mean(2)            # average over the 20 seam positions
        return self.head(h.unsqueeze(2)).squeeze(-1)
