"""Find a SMALL correct block among all 576 fragments, instead of a whole board.

The idea, and the correction it needs
-------------------------------------
The recurrent discrete assembler reaches placement 0.2222 at 6x6 against a
chance of 0.0278 and collapses at 24x24. That success is real but it is measured
on CROPS: `crop_batch` hands the model exactly the 36 fragments of one 6x6
square and asks it to arrange them, so its pool is 36 and not 576. The number
does not transfer to a model that must also CHOOSE which 36 of 576 belong
together, which is the harder and more useful problem.

Why it is worth building anyway
-------------------------------
M472 closed the loop constraint with a diagnosis rather than a plateau: a board
admits about 135000 closed 2x2 squares, so at our shortlist purity four edges
agreeing is free, and the count of supporting squares is actually INVERTED
against correctness because it concentrates on degenerate fragments. A 4x4 block
needs 24 edges to be simultaneously consistent, not 4, and such configurations
are exponentially rarer -- so the evidence that carries nothing at 2x2 may carry
a great deal at 4x4. This is the instrument for finding out.

And it never welds. Growth fails because one false edge fuses two correct
islands at a wrong offset (M456: the block runs 350 fragments at edge precision
1.00 and 18 at the 0.746 we harvest). A block emitted whole by an assignment is
either right or wrong on its own; it cannot corrupt another one.

Two design decisions that are not obvious
-----------------------------------------
RECTANGULAR ASSIGNMENT. There are k*k cells and 576 fragments, so the transport
is unbalanced: every cell takes exactly one fragment, every fragment is taken at
most once. Sinkhorn therefore normalises rows to one and clamps columns at one
rather than normalising both.

THE OBJECTIVE IS ADJACENCY, NOT A LABEL. Supervising "reproduce this particular
crop" would tie the model to one square when ANY internally correct square is a
success. Instead the loss maximises the expected number of TRUE adjacencies
realised inside whatever block the model emits, which is label-free over crops,
invariant to which square is found, and is exactly the quantity M474 showed to
be binding -- with a complete permutation 576 fragments exactly fill 24x24, so
placement follows from adjacency and nothing else.

The gate is not how many blocks come out but what fraction of them are correct
ENTIRELY, at a stated yield per board.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from render_assign import (SUB, BagEncoder, Block, cell_coords,
                           timestep_embedding)


def log_sinkhorn_rect(logits, iterations=10):
    """Unbalanced: each cell takes one fragment, each fragment at most one.

    Rows -- the cells -- are normalised to sum to one. Columns are only pushed
    DOWN to one, never up, because 576 - k*k fragments must be free to go
    unused. Clamping the column correction at zero is what expresses that.
    """
    z = logits
    for _ in range(iterations):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True).clamp(min=0.0)
    return z


class BlockFinder(nn.Module):
    """k*k cell queries against 576 fragment keys, plus seam relations."""

    def __init__(self, side=4, d=192, bag_layers=4, blocks=6, heads=6,
                 tdim=256):
        super().__init__()
        self.side = side
        self.d = d
        self.bag = BagEncoder(d, bag_layers, heads)
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(),
                                  nn.Linear(tdim, tdim))
        self.cell = nn.Parameter(torch.zeros(1, side * side, d))
        nn.init.normal_(self.cell, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, tdim)
                                     for _ in range(blocks)])
        self.qn = nn.LayerNorm(d)
        self.kn = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.rel_gain = nn.Parameter(torch.tensor(1.0))
        self.canvas = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        nn.init.zeros_(self.canvas[1].weight)
        nn.init.zeros_(self.canvas[1].bias)

    def messages(self, p, wr, wd):
        """Cell k+1 should hold whatever sits to the right of cell k's tile."""
        s = self.side
        m = p.shape[1]
        out = torch.zeros_like(p)
        idx = torch.arange(m, device=p.device).reshape(s, s)
        sr, dr = idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)
        sd, dd = idx[:-1].reshape(-1), idx[1:].reshape(-1)
        eps = 1e-9
        out[:, dr] += torch.log(p[:, sr] @ wr + eps)
        out[:, sr] += torch.log(p[:, dr] @ wr.transpose(1, 2) + eps)
        out[:, dd] += torch.log(p[:, sd] @ wd + eps)
        out[:, sd] += torch.log(p[:, dd] @ wd.transpose(1, 2) + eps)
        return out

    def forward(self, view, stats, seam, rounds=12, damp=0.5, iters=10,
                assign_rounds=2, grad_rounds=2):
        ctx = self.bag(view, stats)
        b = ctx.shape[0]
        h = self.cell.expand(b, -1, -1) + cell_coords(
            self.side, ctx.device, self.d)[None]
        t = torch.zeros(b, dtype=torch.long, device=ctx.device)
        e = self.temb(timestep_embedding(t, self.tdim))
        wr = torch.softmax(seam[0].float(), dim=2)
        wd = torch.softmax(seam[1].float(), dim=2)

        def settle(base):
            base = base.float().clamp(-30.0, 30.0)
            lp = log_sinkhorn_rect(base, iters)
            acc = torch.zeros_like(base)
            cut = max(rounds - grad_rounds, 0)
            with torch.no_grad():
                for _ in range(cut):
                    acc = ((1 - damp) * acc
                           + damp * self.messages(lp.exp(), wr, wd)
                           ).clamp(-40.0, 40.0)
                    lp = log_sinkhorn_rect(base + self.rel_gain * acc, iters)
            for _ in range(rounds - cut):
                acc = ((1 - damp) * acc
                       + damp * self.messages(lp.exp(), wr, wd)
                       ).clamp(-40.0, 40.0)
                lp = log_sinkhorn_rect(base + self.rel_gain * acc, iters)
            return lp

        for blk in self.blocks:
            h = blk(h, ctx, e)
        lg = ((self.qn(h) @ self.kn(ctx).transpose(1, 2))
              * (self.scale.exp() / math.sqrt(self.d)))
        logp = settle(lg)
        for _ in range(assign_rounds):
            h = h + self.canvas(logp.exp() @ ctx)
            for blk in self.blocks:
                h = blk(h, ctx, e)
            lg = ((self.qn(h) @ self.kn(ctx).transpose(1, 2))
                  * (self.scale.exp() / math.sqrt(self.d)))
            logp = settle(lg)
        return logp


def adjacency_reward(p, side, right, ok_r, down, ok_d):
    """Expected TRUE adjacencies inside the emitted block.

    `p` is (B, side*side, N) over bag SLOTS, and `right[b, j]` is the slot
    holding the true right neighbour of slot j, with `ok_r` masking the
    fragments that have none. Passing the maps in rather than deriving them
    from the index keeps the bag shuffled, so the fragments' storage order --
    which in the caches is the true cell order -- cannot leak.

    Label-free over crops: any internally correct square scores the same, which
    is the point, since there is no single right answer to supervise.
    """
    b, m, _n = p.shape
    idx = torch.arange(m, device=p.device).reshape(side, side)
    sr, dr = idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)
    sd, dd = idx[:-1].reshape(-1), idx[1:].reshape(-1)

    def pull(q, nb):
        return torch.gather(q, 2, nb[:, None, :].expand(-1, q.shape[1], -1))

    r = (p[:, sr] * ok_r[:, None, :] * pull(p[:, dr], right)).sum()
    d = (p[:, sd] * ok_d[:, None, :] * pull(p[:, dd], down)).sum()
    return (r + d) / b


def decode_block(logp):
    """The block the model actually emits: one fragment a cell, all distinct."""
    c = -logp.detach().float().cpu().numpy()
    if not np.isfinite(c).all():
        c = np.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6)
    rows, cols = linear_sum_assignment(c)
    out = np.empty(c.shape[0], np.int64)
    out[rows] = cols
    return out


def block_is_perfect(order, side, grid):
    """True when the emitted cells are a genuine square of the board."""
    g = np.asarray(order).reshape(side, side)
    base = g[0, 0]
    want = (base // grid + np.arange(side)[:, None]) * grid \
        + (base % grid + np.arange(side)[None, :])
    if base % grid + side > grid or base // grid + side > grid:
        return False
    return bool((g == want).all())


def block_bonds(order, side, grid):
    """How many of the block's internal joins are true, out of 2*side*(side-1)."""
    g = np.asarray(order).reshape(side, side)
    ok = int(((g[:, 1:] - g[:, :-1]) == 1).sum())
    ok += int(((g[1:] - g[:-1]) == grid).sum())
    return ok, 2 * side * (side - 1)
