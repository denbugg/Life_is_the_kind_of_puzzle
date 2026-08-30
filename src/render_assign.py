"""Assignment that is RENDERED from the bag, so the prior cannot cheat.

Why this shape and not the two that failed
------------------------------------------
G2 rejected the plain conditional DDPM with an exact mechanism: `bag delta`
reached 11.66 RMSE, so the conditioning did move the pixels, yet placement with
the CORRECT bag and with a WRONG bag were the same and both at chance. Given a
free pixel output the model learns an unconditional photograph prior faster than
it learns this photograph's dependence on this bag, and an unconditional prior
places nothing -- M471 measured a stranger's picture at 0.0030 against a chance
of 0.0017.

`SinkhornAssembler` has the opposite problem. It reasons in permutation space,
so it cannot cheat, and it reaches 8 to 9 times chance on small boards -- 0.2222
at 6x6 against 0.0278, 0.0625 at 12x12 against 0.0069 -- but collapses at 24x24.
Its supervision is `permutation_loss` plus `edge_loss`: 576 class labels a board,
which is a very thin gradient for a 576x576 decision.

This takes the constraint from one and the gradient from the other. The model
emits cell-by-tile logits, Sinkhorn makes them doubly stochastic, and the clean
estimate is RENDERED as P @ bag -- every value it can possibly output is a value
the bag already contains. An unconditional photograph prior is therefore worth
nothing to it, which is the G2 failure removed by construction rather than by a
penalty. And the loss lands on the rendered picture, which is 96x96x3 dense
values a board instead of 576 labels.

The soft P is a TRAINING device for the gradient. Inference decodes a hard
permutation by Hungarian and moves the original fragments once; nothing averaged
ever reaches a result, in keeping with the project decision that the submission
is a permutation of untouched tiles.

Size agnostic on purpose: cells are queries carrying sinusoidal coordinates
normalised to the board, tiles are keys, so one model runs at 6x6, 12x12 and
24x24 and the curriculum that already works can be used.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

SUB = 4          # values a cell side in the rendered picture -- M428/M471


def log_sinkhorn(logits, iterations=10):
    """Doubly stochastic in log space; rows are cells, columns are tiles."""
    z = logits
    for _ in range(iterations):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    return z


def cell_coords(side, device, dim):
    """Sinusoidal 2-D position normalised by the board, so it transfers sizes."""
    y, x = torch.meshgrid(torch.arange(side, device=device),
                          torch.arange(side, device=device), indexing="ij")
    p = torch.stack([y.reshape(-1), x.reshape(-1)], 1).float() / max(side - 1, 1)
    half = dim // 4
    f = torch.exp(torch.arange(half, device=device) * (-math.log(1e4) / half))
    a = p[:, :, None] * f[None, None]
    return torch.cat([a.sin(), a.cos()], -1).reshape(side * side, -1)


def timestep_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.cos(), a.sin()], -1)


class Block(nn.Module):
    """Cells attend to each other, then to the bag, then think. FiLM by time."""

    def __init__(self, d, heads, tdim):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                nn.Linear(4 * d, d))
        self.film = nn.Linear(tdim, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, ctx, t):
        s, b = self.film(t)[:, None].chunk(2, -1)
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.cross(h, ctx, ctx, need_weights=False)[0]
        x = x + self.ff(self.n3(x) * (1 + s) + b)
        return x


class BagEncoder(nn.Module):
    """The tiles as an unordered set. No positional encoding anywhere."""

    def __init__(self, d, layers, heads, view=8):
        super().__init__()
        self.view = view
        self.inp = nn.Linear(view * view * 3 + 6, d)
        layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, batch_first=True, norm_first=True, dropout=0.0,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.LayerNorm(d)

    def forward(self, view, stats):
        """view (B, M, 8, 8, 3) and stats (B, M, 6), both in [0, 255]."""
        b, m = view.shape[:2]
        v = view.permute(0, 1, 4, 2, 3).reshape(b, m, -1) / 127.5 - 1.0
        s = torch.cat([stats[..., :3] / 127.5 - 1.0, stats[..., 3:] / 127.5], -1)
        return self.out(self.enc(self.inp(torch.cat([v, s], -1))))


def neighbour_messages(p, e_right, e_down, side):
    """Turn tile-tile relations into evidence about neighbouring CELLS.

    If cell k holds tile i with probability p[k, i], and tile j sits to the
    right of tile i with score e_right[i, j], then cell k+1 should hold tile j.
    Summing over i is one round of message passing on the grid, and it is the
    only place relative evidence enters an otherwise absolute model.

    The seam signal is the strongest measurement this project has -- R@1 about
    0.32 a pair -- and it fails on its own because greedy growth welds islands
    at a false offset. Inside a permutation it cannot weld: every message is
    reweighted by a doubly stochastic assignment that already spends each tile
    exactly once.
    """
    b, m, _ = p.shape
    out = p.new_zeros(b, m, m)
    idx = torch.arange(m, device=p.device).reshape(side, side)
    src_r, dst_r = idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)
    src_d, dst_d = idx[:-1].reshape(-1), idx[1:].reshape(-1)
    # each message is a DISTRIBUTION over tiles, so the several arriving at one
    # cell combine as a product, which is a sum of logs. Adding the raw
    # probabilities instead makes every message about 1/576 in size and it
    # vanishes against the cell logits -- measured, and it is why the frozen
    # seam evidence appeared to do nothing at initialisation
    eps = 1e-9
    out[:, dst_r] += torch.log(p[:, src_r] @ e_right + eps)
    out[:, dst_d] += torch.log(p[:, src_d] @ e_down + eps)
    # and backwards, so evidence flows both ways along each axis
    out[:, src_r] += torch.log(p[:, dst_r] @ e_right.transpose(1, 2) + eps)
    out[:, src_d] += torch.log(p[:, dst_d] @ e_down.transpose(1, 2) + eps)
    return out


class RenderAssign(nn.Module):
    def __init__(self, d=192, bag_layers=4, blocks=6, heads=6, tdim=256):
        super().__init__()
        self.d = d
        self.bag = BagEncoder(d, bag_layers, heads)
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(),
                                  nn.Linear(tdim, tdim))
        self.tdim = tdim
        self.patch = nn.Linear(SUB * SUB * 3, d)
        self.blocks = nn.ModuleList([Block(d, heads, tdim)
                                     for _ in range(blocks)])
        self.qn = nn.LayerNorm(d)
        self.kn = nn.LayerNorm(d)
        # the absolute head starts quiet so the relation evidence is not
        # buried under random cell logits before either has been learned
        self.scale = nn.Parameter(torch.tensor(-2.0))
        # relative head: directed tile-to-tile compatibility, right and down
        self.edge_q = nn.ModuleList([nn.Linear(d, d, bias=False)
                                     for _ in range(2)])
        self.edge_k = nn.ModuleList([nn.Linear(d, d, bias=False)
                                     for _ in range(2)])
        self.edge_n = nn.LayerNorm(d)
        # the assignment fed back as context. Each cell reads the tile feature
        # it currently believes it holds, so the next round reasons about a
        # board rather than about 576 independent cells -- the mechanism the
        # recurrent discrete assembler uses to reach eight times chance at 6x6,
        # which a single forward pass does not have.
        self.canvas = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        nn.init.zeros_(self.canvas[1].weight)
        nn.init.zeros_(self.canvas[1].bias)
        # zero-initialised, so a model built with relations starts exactly at
        # the absolute-only baseline and can only depart from it by learning
        self.rel_gain = nn.Parameter(torch.tensor(0.0))
        # when frozen seam evidence is supplied it is the BASE of the relation
        # head and the learned part is a zero-initialised residual on top, so
        # the model starts exactly at the evidence M475 decoded to adjacency
        # 0.205 and can only depart from it by learning. The pooled 8x8 view the
        # learned head reads cannot carry the seam signal at all: M461 measured
        # it living in about four columns at the boundary
        self.seam_gain = nn.Parameter(torch.tensor(1.0))
        self.edge_res = nn.Parameter(torch.tensor(0.0))

    def cells_of(self, x, side):
        b = x.shape[0]
        return x.reshape(b, side, SUB, side, SUB, 3).permute(
            0, 1, 3, 2, 4, 5).reshape(b, side * side, -1)

    def logits(self, x, t, view, stats, side, ctx=None):
        """Cell-by-tile log compatibility, (B, side*side, M)."""
        if ctx is None:
            ctx = self.bag(view, stats)
        h = self.patch(self.cells_of(x, side)) + cell_coords(
            side, x.device, self.d)[None]
        e = self.temb(timestep_embedding(t, self.tdim))
        for blk in self.blocks:
            h = blk(h, ctx, e)
        return self.pair(h, ctx), h, e, ctx

    def pair(self, h, ctx):
        return ((self.qn(h) @ self.kn(ctx).transpose(1, 2))
                * (self.scale.exp() / math.sqrt(self.d)))

    def edges(self, ctx):
        """Directed tile-by-tile logits for the right and down relations."""
        h = self.edge_n(ctx)
        return [(self.edge_q[i](h) @ self.edge_k[i](h).transpose(1, 2))
                / math.sqrt(self.d) for i in range(2)]

    def forward(self, x, t, view, stats, bag_cells, side, iters=10, ctx=None,
                rel_rounds=0, assign_rounds=0, seam=None, damp=0.5, grad_rounds=2):
        """Rendered clean estimate, log assignment, and the relation logits.

        `bag_cells` is (B, M, SUB*SUB*3) on the same [-1, 1] scale as x: the
        picture is assembled ONLY from these, which is what stops the model
        inventing a photograph the bag does not contain.
        """
        lg, h, e, ctx = self.logits(x, t, view, stats, side, ctx)
        e_r, e_d = self.edges(ctx)
        lg = lg.float()
        if seam is not None:
            e_r = self.seam_gain * seam[0] + self.edge_res * e_r
            e_d = self.seam_gain * seam[1] + self.edge_res * e_d

        # The settle loop is a fixed-point iteration over log-probabilities.
        # Under fp16 autocast p @ W underflows to zero, log(0 + eps) returns
        # -20.7 every round, the accumulator walks to -inf and the Hungarian
        # is handed NaN -- which is exactly how the first run died, at the
        # evaluation after its first epoch. It runs in float32 and bounded.
        e_r = e_r.float()
        e_d = e_d.float()
        wr = torch.softmax(e_r, dim=2)
        wd = torch.softmax(e_d, dim=2)

        def settle(base):
            base = base.float().clamp(-30.0, 30.0)
            # messages ACCUMULATE with damping rather than being recomputed
            # from scratch each round. Without the accumulation the iteration
            # does not converge -- measured at 8, 20 and 40 rounds giving
            # adjacency 0.0201, 0.0186 and 0.0162, drifting the wrong way --
            # where the same evidence under damped accumulation decodes to
            # 0.205 (M475)
            lp = log_sinkhorn(base, iters)
            acc = torch.zeros_like(base)
            # truncated backpropagation: the iteration is a fixed point solver,
            # so the settled beliefs are computed without gradient and only the
            # last rounds are differentiated. Twenty rounds of full autograd
            # costs about ten seconds a batch, which is six hours an epoch
            cut = max(rel_rounds - grad_rounds, 0)
            with torch.no_grad():
                for _ in range(cut):
                    acc = ((1.0 - damp) * acc
                           + damp * neighbour_messages(lp.exp(), wr, wd, side)
                           ).clamp(-40.0, 40.0)
                    lp = log_sinkhorn(base + self.rel_gain * acc, iters)
            for _ in range(rel_rounds - cut):
                acc = ((1.0 - damp) * acc
                       + damp * neighbour_messages(lp.exp(), wr, wd, side)
                       ).clamp(-40.0, 40.0)
                lp = log_sinkhorn(base + self.rel_gain * acc, iters)
            return lp

        logp = settle(lg)
        for _ in range(assign_rounds):
            # hand the board back to the cells and think again
            h = h + self.canvas(logp.exp() @ ctx)
            for blk in self.blocks:
                h = blk(h, ctx, e)
            lg = self.pair(h, ctx)
            logp = settle(lg)

        p = logp.exp()
        cells = p @ bag_cells
        b = cells.shape[0]
        x0 = cells.reshape(b, side, side, SUB, SUB, 3).permute(
            0, 1, 3, 2, 4, 5).reshape(b, side * SUB, side * SUB, 3)
        return x0, logp, (e_r, e_d)


def decode(logp):
    """Hungarian on one board's log assignment: cell -> tile, a bijection."""
    c = -logp.detach().float().cpu().numpy()
    if not np.isfinite(c).all():
        # never let one divergent board end a long run; the layout it produces
        # is meaningless and will read as chance, which is the honest answer
        c = np.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6)
    r, t = linear_sum_assignment(c)
    out = np.empty(c.shape[0], np.int64)
    out[r] = t
    return out
