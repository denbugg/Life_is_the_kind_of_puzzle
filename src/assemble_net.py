"""Whole-board assembly as a set-to-permutation problem.

Why a global model at all
-------------------------
Everything before this reasons pairwise and then hands the result to a
combinatorial solver.  M87 measured where that ends: even a core of 28 edges at
precision 0.986 leaves placement at chance, because 28 edges over 576 tiles is a
scatter of fragments and nothing pins a fragment's absolute position.  Assembly
needs a spanning structure, so pairwise precision has to be near-perfect almost
everywhere -- R@1 around 0.7 against the 0.24 in hand.

A model that sees all 576 tiles at once is not bound by that.  It can settle
fragments against each other, use the fact that exactly one tile occupies each
position, and read the board's borders, none of which any pairwise score can
express.  M64's belief propagation failed at the same task precisely because it
enforced no exclusivity; here Sinkhorn does, by construction.

Grounding rather than starting over
-----------------------------------
The seam costs we already have are good information -- 590 mutual edges, a core
at precision 0.96 -- so they enter as an additive attention bias instead of
being discarded.  Attention then starts from the measured pairwise structure and
spends its capacity on what pairwise scoring cannot do.  Tokens likewise come
from the trained matcher's descriptors rather than raw pixels.

M67 rejected predicting a tile's absolute position from its own content (row
band 0.21 against chance 0.167), and that stands: nothing here asks a tile where
it belongs.  Position is decided collectively.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasedAttention(nn.Module):
    """Self-attention with a per-head additive bias built from the seam costs."""

    def __init__(self, d, heads, n_bias, mix_init=0.0):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        # Each head mixes the supplied cost planes its own way.  Starting at
        # zero looks safe and is not: after 2000 diffusion steps the weights had
        # crawled to 0.003-0.010 against attention logits of order one, so the
        # cost graph was contributing about a percent and the model was solving
        # the problem the way the single-shot transformer did -- from the
        # trivial content prior (M89).  The planes are standardised and the
        # costs are informative, so the sane starting point is to USE them and
        # let training modulate, not to rediscover that they exist.
        self.mix = nn.Parameter(torch.full((heads, n_bias), float(mix_init))
                                + 0.05 * torch.randn(heads, n_bias))

    def forward(self, x, planes):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att + torch.einsum("hp,bpij->bhij", self.mix, planes)
        return self.out((att.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d))


class Layer(nn.Module):
    def __init__(self, d, heads, n_bias, ff, mix_init=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = BiasedAttention(d, heads, n_bias, mix_init)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, planes):
        x = x + self.att(self.n1(x), planes)
        return x + self.ff(self.n2(x))


class AssembleNet(nn.Module):
    """(tokens, cost planes) -> logits of shape (n_tiles, n_positions)."""

    def __init__(self, in_dim, d=256, heads=8, layers=6, ff=1024, grid=24,
                 n_bias=4):
        super().__init__()
        self.grid = grid
        n = grid * grid
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d))
        self.layers = nn.ModuleList([Layer(d, heads, n_bias, ff) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        # Row and column are predicted separately rather than as one of 576
        # slots.  A flat slot vocabulary makes the model discover, from scratch,
        # that slot s+1 lies right of slot s -- the very relation the cost bias
        # is already telling it about -- and it does not: 200 steps left the
        # loss at ln(576).  Factorising gives 24+24 classes with the grid
        # geometry built in, and loses nothing, since a grid position IS its
        # row and column.
        self.row = nn.Linear(d, grid)
        self.col = nn.Linear(d, grid)

    def forward(self, tokens, planes):
        x = self.inp(tokens).unsqueeze(0)
        planes = planes.unsqueeze(0)
        for lay in self.layers:
            x = lay(x, planes)
        x = self.norm(x).squeeze(0)
        return self.row(x), self.col(x)

    def slot_logits(self, row_lg, col_lg):
        """Combine into (n_tiles, n_positions) for Hungarian assignment."""
        lr = F.log_softmax(row_lg, -1)
        lc = F.log_softmax(col_lg, -1)
        return (lr[:, :, None] + lc[:, None, :]).reshape(row_lg.shape[0], -1)


def sinkhorn_log(logits, tau=1.0, iters=20):
    L = logits / tau
    for _ in range(iters):
        L = L - torch.logsumexp(L, 1, keepdim=True)
        L = L - torch.logsumexp(L, 0, keepdim=True)
    return L


def cost_planes(cost_h, cost_v):
    """Four planes: right-of, left-of, below, above, each standardised.

    Both orientations are supplied because attention is directed and the model
    should be able to ask either question without transposing anything itself.
    """
    out = []
    for C in (cost_h, cost_v):
        Z = (C - C.mean()) / (C.std() + 1e-9)
        out += [-Z, -Z.t()]          # negated: a cheap seam should ATTRACT
    return torch.stack(out)
