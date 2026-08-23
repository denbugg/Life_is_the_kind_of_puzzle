"""Positional diffusion: denoise the pieces' coordinates instead of predicting them.

Why iterative, when the single-shot transformer failed
------------------------------------------------------
M89 built a transformer over all 576 tiles that predicted each tile's row and
column in one pass.  It learned only the trivial content prior -- row accuracy
0.065 against chance 0.042, columns nothing -- because a tile's absolute
position is defined solely by the global arrangement, so deriving it means
effectively counting hops from a board edge, and six attention layers do not
count.

Diffusion turns depth into time.  Each denoising step sees every piece's CURRENT
position estimate, so a constraint travels one more hop per step; fifty steps
cross a 24-wide board comfortably where six layers could not.  The 2025 corrupted
puzzle benchmark found the same thing empirically: at erosion levels where
Gallagher, Paikin-Tal and Yu misplace every piece, fine-tuned positional
diffusion is the only solver left standing.

Two departures from the published method, both because we have things it did not
------------------------------------------------------------------------------
Tokens carry the trained matcher's descriptors rather than raw pixels, which
inherits 17000 steps of 576-way contrastive training.  And the calibrated seam
costs enter as a per-head additive attention bias, so the message passing runs
along the graph our costs already describe -- edge precision 0.50 is poor for a
solver that must commit to individual edges (M102), but it is a great deal of
information for a process that only has to bias attention.

x0-prediction, not eps: the target is two numbers per piece, the signal is small
and the geometry matters more than the noise, so predicting the clean coordinate
is both better conditioned and directly supervisable against the true grid.
"""
from __future__ import annotations

import math

import numpy as np
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


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.cos(), a.sin()], -1)


class PosDiffusion(nn.Module):
    """(features, noisy positions, t, cost planes) -> predicted clean positions."""

    def __init__(self, feat_dim, d=256, heads=8, layers=8, ff=1024, n_bias=4,
                 mix_init=1.0):
        super().__init__()
        self.d = d
        self.feat = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
        self.pos_in = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.t_in = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([Layer(d, heads, n_bias, ff, mix_init)
                                     for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, feats, xt, t, planes):
        h = self.feat(feats) + self.pos_in(xt)
        h = h + self.t_in(timestep_embedding(t.expand(1), self.d))
        h = h.unsqueeze(0)
        planes = planes.unsqueeze(0)
        for lay in self.layers:
            h = lay(h, planes)
        return self.out(h.squeeze(0))


def cosine_alphas(T, s=0.008, device="cuda"):
    """Cumulative alphas on the cosine schedule (Nichol and Dhariwal)."""
    t = torch.arange(T + 1, device=device, dtype=torch.float32) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a = f / f[0]
    return torch.clamp(a, 1e-4, 1.0)


@torch.no_grad()
def sample(model, feats, planes, alphas, steps=50, grid=24, seed=0, device="cuda"):
    """DDIM sampling from pure noise down to clean positions."""
    n = feats.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 2, generator=g).to(device)
    T = alphas.shape[0] - 1
    ts = torch.linspace(T, 1, steps, device=device).long()
    for i, t in enumerate(ts):
        a_t = alphas[t]
        with torch.autocast("cuda", torch.float16):
            x0 = model(feats, x, t, planes).float()
        x0 = x0.clamp(-1.2, 1.2)
        if i == len(ts) - 1:
            return x0
        a_prev = alphas[ts[i + 1]]
        eps = (x - a_t.sqrt() * x0) / (1 - a_t).clamp_min(1e-8).sqrt()
        x = a_prev.sqrt() * x0 + (1 - a_prev).clamp_min(0).sqrt() * eps
    return x0


def grid_targets(n, grid=24, device="cuda"):
    """True positions of grid slots, scaled to [-1, 1]."""
    idx = torch.arange(n, device=device)
    r = (idx // grid).float()
    c = (idx % grid).float()
    return torch.stack([r, c], 1) / (grid - 1) * 2.0 - 1.0


@torch.no_grad()
def refine(model, feats, planes, alphas, layout, t_start=0.5, steps=25, grid=24,
           seed=0, device="cuda"):
    """Denoise from an EXISTING layout instead of from pure noise.

    Sampling from noise asks the first prediction to solve the whole board from
    features and costs alone -- exactly the single-shot problem that failed
    (M89) -- and only then lets later steps refine.  Starting partway down the
    schedule hands the process a layout that is already partly right (greedy
    recovers 20% of true adjacencies on real boards, M104) and asks only for a
    correction, which is the regime the intermediate timesteps were trained on.

    layout[p] = tile at grid position p, as the solvers return it.
    """
    n = feats.shape[0]
    cells = grid_targets(n, grid, device)
    x0 = torch.empty(n, 2, device=device)
    lay = torch.as_tensor(np.asarray(layout), dtype=torch.long, device=device)
    x0[lay] = cells                              # tile at slot p sits at cell p

    T = alphas.shape[0] - 1
    t0 = max(1, int(round(T * t_start)))
    g = torch.Generator(device="cpu").manual_seed(seed)
    a0 = alphas[t0]
    x = a0.sqrt() * x0 + (1 - a0).sqrt() * torch.randn(n, 2, generator=g).to(device)

    ts = torch.linspace(t0, 1, steps, device=device).long()
    for i, t in enumerate(ts):
        a_t = alphas[t]
        with torch.autocast("cuda", torch.float16):
            pred = model(feats, x, t, planes).float()
        pred = pred.clamp(-1.2, 1.2)
        if i == len(ts) - 1:
            return pred
        a_prev = alphas[ts[i + 1]]
        eps = (x - a_t.sqrt() * pred) / (1 - a_t).clamp_min(1e-8).sqrt()
        x = a_prev.sqrt() * pred + (1 - a_prev).clamp_min(0).sqrt() * eps
    return pred
