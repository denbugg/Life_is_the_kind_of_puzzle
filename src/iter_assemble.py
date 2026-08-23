"""Iterative discrete assembly: fill in positions, most confident first.

Why not coordinate regression
-----------------------------
Positional diffusion predicts each piece's (x, y) and is trained with a
regression loss, so for any piece it cannot resolve it returns the conditional
MEAN -- the board centre, or whatever the weak content prior suggests.  Measured
at 6000 steps: the relative-structure ratio sat at 0.88 / 0.77 / 0.81 across
evaluations without trending, row correlation climbed to +0.202 while column
stayed at +0.037, which is the "sky is at the top" prior (M67) and nothing
relational.  The same collapse sank the L1-trained restorer, which lowered pixel
error by smoothing away the very microstructure matching depends on (M23).

Resolution makes it worse.  True neighbours sit 0.087 apart on the [-1, 1] grid
and the model's positional error was 0.212 at the EASIEST noise level, so the
Hungarian step scrambles whatever structure existed: greedy's adjacency 0.383
became 0.006 after refinement (M115).

Discrete positions do not have either problem.  A distribution over 24 rows can
say "either row 3 or row 19" instead of averaging to row 11, and its argmax is
exact by construction, so grid resolution stops mattering.

The mechanism that made diffusion attractive is kept: iteration.  Training masks
a random fraction of the pieces' positions and asks for all of them; inference
starts with everything masked, commits the most confident predictions, and
re-runs with those revealed.  Constraints propagate one hop per round exactly as
they did per denoising step, which is what a single-shot transformer could not do
(M89).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from assemble_net import BiasedAttention


class Layer(nn.Module):
    def __init__(self, d, heads, n_bias, ff, mix_init=1.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = BiasedAttention(d, heads, n_bias, mix_init)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, planes):
        x = x + self.att(self.n1(x), planes)
        return x + self.ff(self.n2(x))


class IterAssemble(nn.Module):
    """(features, revealed positions, cost planes) -> row and column logits."""

    def __init__(self, feat_dim, d=256, heads=8, layers=8, ff=1024, grid=24,
                 n_bias=4, mix_init=1.0):
        super().__init__()
        self.grid = grid
        self.feat = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
        # one extra index per axis is the "not yet revealed" symbol
        self.row_emb = nn.Embedding(grid + 1, d)
        self.col_emb = nn.Embedding(grid + 1, d)
        self.layers = nn.ModuleList([Layer(d, heads, n_bias, ff, mix_init)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.row = nn.Linear(d, grid)
        self.col = nn.Linear(d, grid)

    def forward(self, feats, rows, cols, planes):
        h = self.feat(feats) + self.row_emb(rows) + self.col_emb(cols)
        h = h.unsqueeze(0)
        planes = planes.unsqueeze(0)
        for lay in self.layers:
            h = lay(h, planes)
        h = self.norm(h).squeeze(0)
        return self.row(h), self.col(h)

    def slot_logits(self, row_lg, col_lg):
        lr = F.log_softmax(row_lg, -1)
        lc = F.log_softmax(col_lg, -1)
        return (lr[:, :, None] + lc[:, None, :]).reshape(row_lg.shape[0], -1)


@torch.no_grad()
def decode(model, feats, planes, rounds=8, grid=24, device="cuda", seed_rows=None,
           seed_cols=None):
    """Reveal the most confident predictions round by round.

    Returns per-tile row and column log-probabilities taken AT THE MOMENT each
    tile was committed, not from a final pass over a fully revealed board.  The
    loss only ever supervises hidden tiles, so the model is untrained on the
    all-revealed input and its output there is noise -- re-running it at the end
    overwrote every decision the decode had made and cost adjacency 0.38 -> 0.02.

    seed_rows/seed_cols optionally pin part of the board (masked entries are
    `grid`), which is how an existing solver's layout is handed in.
    """
    n = feats.shape[0]
    mask = grid
    rows = (torch.full((n,), mask, dtype=torch.long, device=device)
            if seed_rows is None else seed_rows.clone())
    cols = (torch.full((n,), mask, dtype=torch.long, device=device)
            if seed_cols is None else seed_cols.clone())
    out_r = torch.zeros(n, grid, device=device)
    out_c = torch.zeros(n, grid, device=device)
    done = rows != mask
    if done.any():                       # seeded tiles keep their given position
        out_r[done] = F.one_hot(rows[done], grid).float() * 20.0
        out_c[done] = F.one_hot(cols[done], grid).float() * 20.0

    for r in range(rounds):
        hidden = (~done).nonzero(as_tuple=True)[0]
        if hidden.numel() == 0:
            break
        with torch.autocast("cuda", torch.float16):
            rl, cl = model(feats, rows, cols, planes)
        lr = F.log_softmax(rl.float(), -1)
        lc = F.log_softmax(cl.float(), -1)
        conf = lr.max(1).values + lc.max(1).values
        take_n = max(1, hidden.numel() // max(1, rounds - r))
        take = hidden[conf[hidden].topk(take_n).indices]
        rows[take] = lr[take].argmax(1)
        cols[take] = lc[take].argmax(1)
        out_r[take] = lr[take]
        out_c[take] = lc[take]
        done[take] = True
    return out_r, out_c
