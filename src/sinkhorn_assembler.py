"""Recurrent discrete puzzle assembler.

Unlike coordinate regression or image regression, this model always reasons in
the space of permutations.  A doubly-stochastic assignment renders an internal
feature canvas; a shared 2-D CNN reads local context from that canvas and
updates the assignment.  Hungarian makes the final prediction exactly
bijective.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def log_sinkhorn(logits: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    z = logits
    for _ in range(iterations):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    return z


class ConvBlock(nn.Module):
    def __init__(self, d: int, dilation: int = 1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, d)
        self.c1 = nn.Conv2d(d, d, 3, padding=dilation, dilation=dilation)
        self.n2 = nn.GroupNorm(8, d)
        self.c2 = nn.Conv2d(d, d, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(F.gelu(self.n2(self.c1(F.gelu(self.n1(x))))))


class SinkhornAssembler(nn.Module):
    def __init__(self, d: int = 96, rounds: int = 6, blocks: int = 3):
        super().__init__()
        # raw 4x4 RGB and the same descriptor standardised per tile.  The raw
        # half keeps absolute colour; the normalised half survives the
        # independent affine corruption.
        self.tile = nn.Sequential(nn.Linear(8 * 8 * 3 * 2, d), nn.GELU(),
                                  nn.Linear(d, d), nn.LayerNorm(d))
        self.pos = nn.Sequential(nn.Linear(10, d), nn.GELU(), nn.Linear(d, d))
        self.canvas = nn.ModuleList([
            nn.Sequential(*[ConvBlock(d, 2 ** (k % 3)) for k in range(blocks)])
            for _ in range(rounds)
        ])
        self.query = nn.ModuleList([nn.Linear(d, d, bias=False)
                                    for _ in range(rounds + 1)])
        self.key = nn.ModuleList([nn.Linear(d, d, bias=False)
                                  for _ in range(rounds + 1)])
        self.mix = nn.Parameter(torch.zeros(rounds))
        self.edge_q = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.edge_k = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.edge_mix = nn.Parameter(torch.ones(rounds))
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.rounds = rounds

    @staticmethod
    def tile_features(x):
        # x: B,N,8,8,3 in 0..255
        raw = x.float().reshape(*x.shape[:2], -1) / 127.5 - 1.0
        mu = raw.mean(-1, keepdim=True)
        sd = raw.std(-1, keepdim=True).clamp_min(0.05)
        return torch.cat([raw, (raw - mu) / sd], -1)

    @staticmethod
    def coordinates(side: int, device):
        a = torch.linspace(-1, 1, side, device=device)
        y, x = torch.meshgrid(a, a, indexing="ij")
        return torch.stack([x, y, x * y, x * x, y * y,
                            torch.sin(math.pi * x), torch.cos(math.pi * x),
                            torch.sin(math.pi * y), torch.cos(math.pi * y),
                            torch.ones_like(x)], -1).reshape(side * side, 10)

    def pair_logits(self, cells, tiles, index):
        q = F.normalize(self.query[index](cells), dim=-1)
        k = F.normalize(self.key[index](tiles), dim=-1)
        return torch.einsum("bcd,btd->bct", q, k) * self.scale.exp().clamp(1, 50)

    def edge_logits(self, tiles):
        out = []
        scale = self.scale.exp().clamp(1, 50)
        for q, k in zip(self.edge_q, self.edge_k):
            a = F.normalize(q(tiles), dim=-1)
            b = F.normalize(k(tiles), dim=-1)
            out.append(torch.bmm(a, b.transpose(1, 2)) * scale)
        return torch.stack(out, 1)  # B,2(right/down),tile,tile

    @staticmethod
    def relation_message(p, edges, side):
        """Expected four-neighbour compatibility for every (cell,tile)."""
        b, n, _ = p.shape
        pg = p.reshape(b, side, side, n)
        msg = torch.zeros_like(pg)
        right, down = edges[:, 0], edges[:, 1]
        # candidate t at (y,x), neighbour distribution over u beside it.
        msg[:, :, :-1] += torch.einsum("byxu,btu->byxt", pg[:, :, 1:], right)
        msg[:, :, 1:] += torch.einsum("byxu,but->byxt", pg[:, :, :-1], right)
        msg[:, :-1] += torch.einsum("byxu,btu->byxt", pg[:, 1:], down)
        msg[:, 1:] += torch.einsum("byxu,but->byxt", pg[:, :-1], down)
        return msg.reshape(b, n, n) / 4.0

    def forward(self, x, side: int, temperature: float = 1.0,
                teacher: torch.Tensor | None = None,
                teacher_weight: float = 0.0,
                edge_override: torch.Tensor | None = None):
        b, n = x.shape[:2]
        if n != side * side:
            raise ValueError("tile count and side disagree")
        tiles = self.tile(self.tile_features(x))
        edges = self.edge_logits(tiles) if edge_override is None else edge_override
        if edges.shape != (b, 2, n, n):
            raise ValueError(f"edge_override must have shape {(b, 2, n, n)}")
        cells0 = self.pos(self.coordinates(side, x.device))[None].expand(b, -1, -1)
        logits = self.pair_logits(cells0, tiles, 0)
        history = [logits]
        for r in range(self.rounds):
            p = log_sinkhorn(logits / temperature).exp()
            if teacher is not None and teacher_weight > 0:
                p = (1.0 - teacher_weight) * p + teacher_weight * teacher
            feat = torch.bmm(p, tiles).transpose(1, 2).reshape(b, -1, side, side)
            context = self.canvas[r](feat).flatten(2).transpose(1, 2)
            cells = cells0 + context
            update = self.pair_logits(cells, tiles, r + 1)
            relation = self.relation_message(p, edges, side)
            logits = (logits + torch.sigmoid(self.mix[r]) * update
                      + self.edge_mix[r] * relation)
            history.append(logits)
        return logits, history, edges


def permutation_loss(history, target, decay: float = 0.7):
    total = 0.0
    weights = 0.0
    for age, logits in enumerate(reversed(history)):
        w = decay ** age
        row = F.cross_entropy(logits.flatten(0, 1), target.flatten())
        # Inverse direction explicitly trains exclusivity before Sinkhorn.
        inv = torch.empty_like(target)
        ids = torch.arange(target.shape[1], device=target.device)[None].expand_as(target)
        inv.scatter_(1, target, ids)
        col = F.cross_entropy(logits.transpose(1, 2).flatten(0, 1), inv.flatten())
        total = total + w * 0.5 * (row + col)
        weights += w
    return total / weights


def edge_loss(edges, target, side):
    """Directed right/down neighbour supervision in input-tile indices."""
    b = target.shape[0]
    grid = target.reshape(b, side, side)
    losses = []
    right_anchor = grid[:, :, :-1].reshape(-1)
    right_target = grid[:, :, 1:].reshape(-1)
    er = edges[:, 0].reshape(b * side * side, -1)
    batch_r = torch.arange(b, device=target.device)[:, None, None].expand(
        b, side, side - 1).reshape(-1)
    losses.append(F.cross_entropy(edges[batch_r, 0, right_anchor], right_target))
    down_anchor = grid[:, :-1].reshape(-1)
    down_target = grid[:, 1:].reshape(-1)
    batch_d = torch.arange(b, device=target.device)[:, None, None].expand(
        b, side - 1, side).reshape(-1)
    losses.append(F.cross_entropy(edges[batch_d, 1, down_anchor], down_target))
    return torch.stack(losses).mean()


@torch.no_grad()
def decode(logits: torch.Tensor) -> np.ndarray:
    out = []
    for score in logits.float().cpu().numpy():
        rows, cols = linear_sum_assignment(-score)
        lay = np.empty(len(rows), np.int64)
        lay[rows] = cols
        out.append(lay)
    return np.stack(out)
