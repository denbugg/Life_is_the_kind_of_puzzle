"""Soft-permutation relaxation of the layout problem.

Why this rather than the solvers already here
---------------------------------------------
Every earlier solver reduces the cost matrix to a SET of edges and then reasons
about that set: LP synchronises translations over selected matches, greedy grows
components from loop-verified ones, BP passes messages on a graph built from
them.  M87 measured where that ends: even a core of 28 edges at precision 0.986
leaves place_acc at chance, because 28 edges over 576 tiles is a scatter of
fragments and translation synchronisation slides each one freely.  Selection
throws away the very thing that could connect them -- the weak but non-zero
preference expressed by every other entry of the matrix.

This keeps all of it.  The layout is a doubly stochastic X, X[t, p] being the
belief that tile t sits at position p, and the same summed seam cost is
minimised directly in that continuous space, so all 576x576 entries push on
every position at once.  Sinkhorn keeps X doubly stochastic, which is the
assignment constraint the answer must satisfy anyway, and lowering its
temperature drives X towards a hard permutation.  Hungarian rounds whatever is
left.

This only became worth building once M82 showed the true layout is the minimum
of this objective under the learned cost.  Under MGC it was not -- annealing
reached 0.858x the true cost -- and a better optimiser would have found a worse
answer faster.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _sinkhorn(logits, tau, iters):
    L = logits / tau
    for _ in range(iters):
        L = L - torch.logsumexp(L, dim=1, keepdim=True)
        L = L - torch.logsumexp(L, dim=0, keepdim=True)
    return L.exp()


def solve_soft(cost_h, cost_v, grid=24, steps=600, lr=0.15, tau_hi=1.0,
               tau_lo=0.03, sink_iters=20, init=None, seed=0, device="cuda",
               restarts=1):
    """Return lay with lay[position] = tile index, and the final hard cost.

    cost_h[i, j] scores tile j immediately right of tile i; cost_v likewise for
    below.  Costs are standardised internally, so any scale works.
    """
    n = grid * grid
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    CH = torch.as_tensor(np.ascontiguousarray(cost_h), dtype=torch.float32, device=dev)
    CV = torch.as_tensor(np.ascontiguousarray(cost_v), dtype=torch.float32, device=dev)
    CH = (CH - CH.mean()) / (CH.std() + 1e-9)
    CV = (CV - CV.mean()) / (CV.std() + 1e-9)

    pos = torch.arange(n, device=dev)
    hp = pos[pos % grid != grid - 1]          # positions with a right neighbour
    vp = pos[pos < n - grid]                  # positions with a lower neighbour

    best_lay, best_cost = None, float("inf")
    for r in range(restarts):
        g = torch.Generator(device="cpu").manual_seed(seed + r)
        logits = torch.randn(n, n, generator=g).to(dev) * 0.01
        if init is not None:
            # a construction to start from: put most of the mass on it, but keep
            # the rest soft so the relaxation can still move tiles around
            lay0 = torch.as_tensor(np.asarray(init), dtype=torch.long, device=dev)
            logits[lay0, pos] += 3.0
        logits.requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=lr)

        for s in range(steps):
            tau = tau_hi * (tau_lo / tau_hi) ** (s / max(1, steps - 1))
            X = _sinkhorn(logits, tau, sink_iters)
            # X[:, p] is the tile distribution at position p, so the expected
            # cost of one seam is a bilinear form and the whole board is a sum
            # of two such traces
            e = ((X[:, hp] * (CH @ X[:, hp + 1])).sum()
                 + (X[:, vp] * (CV @ X[:, vp + grid])).sum())
            opt.zero_grad(set_to_none=True)
            e.backward()
            opt.step()

        with torch.no_grad():
            X = _sinkhorn(logits.detach(), tau_lo, sink_iters)
        tile, position = linear_sum_assignment(-X.double().cpu().numpy())
        lay = np.empty(n, np.int64)
        lay[position] = tile
        c = float(cost_h[lay[:-1][(np.arange(n - 1) % grid != grid - 1)],
                         lay[1:][(np.arange(n - 1) % grid != grid - 1)]].sum()
                  + cost_v[lay[:n - grid], lay[grid:]].sum())
        if c < best_cost:
            best_cost, best_lay = c, lay
    return best_lay, best_cost
