"""A weak per-tile position prior, used only to break the toroidal ambiguity.

The problem it solves
---------------------
Greedy assembly is correct only up to a cyclic shift, and it picks that shift
badly: at severity 0.3 its layout is worth place_acc 0.305 at the best shift and
scores 0.109 at the one it chooses (M102).  Two thirds of a finished assembly is
thrown away at the last step.

Why the existing fix cannot do better.  `fix_origin` takes the highest-cost
toroidal cut, which is exactly optimal for the summed-cost objective -- rolling
a torus changes nothing except WHICH 48 seams stop being counted, so total cost
differs between shifts only by the excluded cuts.  The trouble is that the true
image border is not reliably the most expensive seam (M50: exact 33% of the
time).  No better statistic exists inside the cost matrix, because the cost
matrix genuinely contains no other information about the shift.

Why a position prior is a different kind of evidence.  It does not come from the
seams at all.  M67 measured it and rejected it as a MATCHING cue -- a single
tile predicts its own row band at 0.21 against a chance 0.167, and fusing that
into MGC only hurt.  But choosing among 576 shifts is a different question:
every tile votes, the votes are conditionally independent given the layout, and
576 weak votes decide a 576-way choice comfortably where one weak vote decides
nothing.  Columns carry almost no signal in photographs, which is fine -- the
row half alone pins the vertical shift, and the horizontal one can be left to
the cut statistic.

Trained on synthetic boards, where every tile's true row and column are known
exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RING_SIGMA = 13.4


class RowPrior(nn.Module):
    """(B,3,20,20) tiles -> (row logits over `grid`, column logits over `grid`)."""

    def __init__(self, ch=48, blocks=3, grid=24):
        super().__init__()
        self.grid = grid
        layers = [nn.Conv2d(6, ch, 3, padding=1), nn.GELU()]
        for _ in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, stride=2, padding=1),
                       nn.GroupNorm(8, ch), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch * 3 * 3, 128), nn.GELU())
        self.row = nn.Linear(128, grid)
        self.col = nn.Linear(128, grid)

    def prep(self, x):
        s = x.flatten(2)
        mu = s.mean(-1)[:, :, None, None]
        var = s.var(-1)[:, :, None, None] - RING_SIGMA ** 2
        sd = torch.sqrt(torch.clamp(var, min=(0.25 * RING_SIGMA) ** 2))
        return torch.cat([x / 255.0 - 0.5, (x - mu) / sd / 4.0], 1)

    def forward(self, x):
        h = self.head(self.body(self.prep(x)))
        return self.row(h), self.col(h)


@torch.no_grad()
def best_shift(board, tiles, model, grid=24, device="cuda", use_col=True):
    """Roll `board` to the cyclic shift its tiles' own content votes for.

    board[p] = tile index at grid position p, correct up to a cyclic shift.
    Returns the rolled board.
    """
    n = grid * grid
    x = torch.as_tensor(tiles, dtype=torch.float32, device=device)
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    with torch.autocast("cuda", torch.float16):
        rl, cl = model(x)
    lr = F.log_softmax(rl.float(), -1)
    lc = F.log_softmax(cl.float(), -1)

    b = torch.as_tensor(board, dtype=torch.long, device=device).reshape(grid, grid)
    # A column shift leaves every tile in its row and a row shift leaves every
    # tile in its column, so the two votes separate: 24 + 24 evaluations decide
    # the shift, not 576.
    rows = torch.arange(grid, device=device)[:, None].expand(grid, grid).reshape(-1)
    cols = torch.arange(grid, device=device)[None, :].expand(grid, grid).reshape(-1)
    r_scores = torch.stack([lr[torch.roll(b, -d, dims=0).reshape(-1), rows].sum()
                            for d in range(grid)])
    dr = int(r_scores.argmax())
    if use_col:
        c_scores = torch.stack([lc[torch.roll(b, -d, dims=1).reshape(-1), cols].sum()
                                for d in range(grid)])
        dc = int(c_scores.argmax())
    else:
        dc = 0
    return torch.roll(b, (-dr, -dc), dims=(0, 1)).reshape(-1).cpu().numpy()
