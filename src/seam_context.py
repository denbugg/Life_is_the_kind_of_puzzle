"""Give each tile's descriptors the context of the whole board before matching.

The idea, from SuperGlue
------------------------
SuperGlue's gain over raw descriptor matching comes from two attention passes:
self-attention widens each descriptor's receptive field, cross-attention lets
the two sets talk, and only then are the scores turned into an assignment by
Sinkhorn.  We already do the Sinkhorn half (M86) and have never done the first.

Why it should matter here specifically.  A flat patch of sky is not ambiguous
because the matcher is weak -- it is ambiguous because nothing about the patch
ALONE distinguishes it from the other hundred patches of sky on the same board.
Context can: attending over all 576 tiles lets a descriptor encode "the sky that
is slightly darker than most" rather than "sky".  That is exactly the failure
mode measured here: 27.9% of rows have a true neighbour with a visual twin
(M83), and edge precision sits at 0.44 while the assembly threshold is 0.72.

Why this is not the transformer that failed
-------------------------------------------
M89 put attention over the same 576 tiles and asked it to output each tile's
POSITION.  That task is ill-posed without an anchor, so the model settled on the
one thing it could learn, the content prior, and never used the relational
structure.  Here the output is still a descriptor and the loss is still the
576-way ranking loss, so there is no such shortcut: every candidate lives on the
same board, and "which board is this" tells the model nothing.

The trunk stays frozen; only the context layers are learned, so this starts from
a matcher that already works rather than rediscovering one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from assemble_net import BiasedAttention


class Layer(nn.Module):
    def __init__(self, d, heads, n_bias, ff, mix_init):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = BiasedAttention(d, heads, n_bias, mix_init)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, planes):
        x = x + self.att(self.n1(x), planes)
        return x + self.ff(self.n2(x))


class SeamContext(nn.Module):
    """Frozen descriptors in, board-contextualised descriptors of the same shape out."""

    def __init__(self, embed, d=256, heads=8, layers=4, ff=1024, n_bias=4,
                 mix_init=1.0):
        super().__init__()
        self.embed = embed
        for p in self.embed.parameters():
            p.requires_grad_(False)
        self.dim = embed.heads[0][-1].out_features
        self.inp = nn.Sequential(nn.LayerNorm(4 * self.dim + 6),
                                 nn.Linear(4 * self.dim + 6, d))
        self.layers = nn.ModuleList([Layer(d, heads, n_bias, ff, mix_init)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 4 * self.dim)
        # start as a no-op: the refinement is added to the original descriptors,
        # so at initialisation the model matches exactly as well as the retriever
        # it wraps and can only be pushed away from that by evidence
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.logit_scale = nn.Parameter(embed.logit_scale.detach().clone())

    def base(self, tiles):
        with torch.no_grad():
            return [t.float() for t in self.embed(tiles)[:4]]

    def forward(self, tiles, planes):
        desc = self.base(tiles)
        s = tiles.flatten(2)
        tok = torch.cat(desc + [s.mean(-1) / 255.0, s.std(-1) / 255.0], 1)
        h = self.inp(tok).unsqueeze(0)
        planes = planes.unsqueeze(0)
        for lay in self.layers:
            h = lay(h, planes)
        delta = self.out(self.norm(h).squeeze(0)).reshape(-1, 4, self.dim)
        return [F.normalize(desc[i] + delta[:, i], dim=-1) for i in range(4)]
