"""A joint verifier of one seam, trained for PRECISION rather than retrieval.

Why this and not another matcher
--------------------------------
M456 measured the constraint that governs everything: holding the true edges
fixed and adding false ones, the connected block runs 350 at precision 1.00,
186 at 0.99, 65 at 0.95 and 18 at the 0.746 our harvest delivers. One wrong edge
in a hundred halves the block, because a wrong edge WELDS two islands at a false
offset and destroys correct structure. So the quantity to optimise is not how
many correct edges we win but how few wrong ones we keep -- and no scorer in
this project has ever been trained for that.

Every scorer here is a BI-ENCODER: it embeds two fragments separately and takes
a dot product of pooled descriptors (M107 says so of all of them). A pooled dot
product cannot express continuity ACROSS a join, only similarity of two
summaries. And all of them are trained by retrieval -- which of 576 -- which
optimises the whole ranking, not its extreme tail.

M438 rejected the five-candidate chooser at 2900 boards, and it is not this: it
asked "which of these five" under cross-entropy, which is again a ranking
objective. This asks "is this pair adjacent, and how sure are you", and it is
scored on the only thing that matters, PRECISION AT THE VOLUME WE HARVEST.

The design points that are not guesses
--------------------------------------
The matcher's own score enters as a feature, so the verifier starts from what is
already known and spends its capacity on what a dot product cannot say -- M107's
rule that a second stage must be strictly stronger than the one it corrects, and
that giving it the first stage's answer makes copying cheap unless the head is
built to need more. The score head is zero-initialised on top of that feature,
so an untrained verifier reproduces the matcher exactly (M412 measured that this
is what keeps a second stage from collapsing at step one).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeamVerifier(nn.Module):
    """Both sides of one join in a single tensor, convolved ACROSS it."""

    def __init__(self, ch=64, blocks=4, feats=6, strip=4):
        super().__init__()
        self.strip = strip
        layers = [nn.Conv2d(3, ch, 3, padding=1), nn.GELU()]
        for k in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, padding=1, dilation=1),
                       nn.BatchNorm2d(ch), nn.GELU()]
            if k % 2 == 1:
                layers += [nn.Conv2d(ch, ch, (3, 1), padding=(1, 0)), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(ch * 2 + feats, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU())
        self.out = nn.Linear(64, 1)
        # zero-initialised, so an untrained verifier scores exactly the
        # matcher's own feature and cannot start below it
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.prior = nn.Parameter(torch.ones(1))

    def forward(self, patch, feats):
        """patch: (B, 3, 20, 2*strip) centred on the join. feats: (B, F)."""
        h = self.trunk(patch)
        # the join is the middle column pair; pool it separately from the rest
        s = self.strip
        joint = h[:, :, :, s - 1:s + 1].mean((2, 3))
        whole = h.mean((2, 3))
        z = self.head(torch.cat([joint, whole, feats], 1))
        return self.out(z).squeeze(1) + self.prior * feats[:, 0]


def precision_at_k(score, label, k):
    """Precision among the k highest-scoring pairs -- the harvest's own metric."""
    if k <= 0 or not len(score):
        return float("nan")
    idx = torch.argsort(score, descending=True)[:k]
    return float(label[idx].float().mean())


def topk_hinge(logit, label, k, margin=1.0):
    """Push positives above the k-th ranked score, and negatives below it.

    The metric is PRECISION AT 430 EDGES A BOARD (M456), which is a property of
    where the threshold falls, and a pointwise loss does not optimise that: the
    first verifier's focal BCE moved from 0.0873 to 0.0763 over three epochs
    while precision moved 0.0086, because the gradient goes to the ninety-odd
    per cent of pairs that are settled either way. This puts the loss exactly at
    the operating point -- the k-th score is the threshold the harvest will use,
    so a positive below it and a negative above it are the only mistakes that
    cost anything.
    """
    k = min(int(k), logit.numel() - 1)
    if k <= 0:
        return logit.mean() * 0.0
    thr = torch.kthvalue(-logit.detach().flatten(), k).values.neg()
    pos = label > 0.5
    below = torch.relu(margin - (logit - thr))[pos]
    above = torch.relu(margin + (logit - thr))[~pos]
    n = below.numel() + above.numel()
    if n == 0:
        return logit.mean() * 0.0
    return (below.sum() + above.sum()) / n


def focal_bce(logit, label, gamma=2.0, pos_weight=1.0):
    """Binary loss that concentrates on the pairs the model gets wrong.

    Plain cross-entropy spends most of its gradient on the easy negatives --
    about 95% of a shortlist is not adjacent -- and what decides the harvest is
    the ordering of the top few per cent. The focal form down-weights what is
    already settled, which is the closest thing to training for precision at a
    low recall without a ranking loss.
    """
    p = torch.sigmoid(logit)
    w = torch.where(label > 0.5, torch.full_like(p, pos_weight),
                    torch.ones_like(p))
    mod = torch.where(label > 0.5, (1 - p) ** gamma, p ** gamma)
    return (w * mod * F.binary_cross_entropy_with_logits(
        logit, label, reduction="none")).mean()
