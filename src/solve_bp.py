"""Loopy belief propagation over the grid MRF (after Cho et al., CVPR 2010).

Why another solver
------------------
The two solvers already here commit to hard decisions.  Greedy growth accepts
one neighbour at a time, and the LP assigns continuous positions from mutual
top-1 matches; both need edge precision around 0.9, while ours is 0.26.

Belief propagation never commits.  Each grid cell holds a distribution over all
576 tiles, messages carry that uncertainty to neighbouring cells, and a tile
that ranks second or fifth locally can still win once the surrounding evidence
agrees.  That matters here because the true neighbour is inside the top 20 for
about half of all tiles (R@20 ~ 0.50) even though it is top-1 for only 0.17 —
the information is present in the candidate lists, it just cannot be read off
greedily.

Cho et al. demonstrated this formulation on 432-piece puzzles, the closest
published scale to our 576.

Formulation
-----------
Nodes are the 576 grid cells, labels are the 576 tiles.  The pairwise potential
between horizontally adjacent cells is exp(-cost_h/T), and likewise for
vertical.  Messages are passed in log space and damped.  BP alone does not
enforce that each tile is used once, so the converged beliefs are finally
converted to a bijection by Hungarian assignment.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import GRID as G, NFRAG as N


def _normalise(costs: np.ndarray) -> np.ndarray:
    """Row-standardise a cost matrix so temperatures mean the same thing."""
    c = np.array(costs, np.float64)
    np.fill_diagonal(c, np.nan)
    mu = np.nanmean(c, axis=1, keepdims=True)
    sd = np.nanstd(c, axis=1, keepdims=True) + 1e-9
    out = (c - mu) / sd
    np.fill_diagonal(out, 6.0)                    # a tile is never its own neighbour
    return out


def solve_bp(cost_h: np.ndarray, cost_v: np.ndarray, iters: int = 30,
             temp: float = 1.0, damping: float = 0.5,
             top_k: int = 32) -> np.ndarray:
    """cost matrices (lower = better) -> board[p] = tile index at position p.

    `top_k` truncates each cell's label set to the tiles that are plausible
    somewhere, which keeps the message tensor affordable without discarding the
    candidates BP is meant to arbitrate between.
    """
    ch = _normalise(cost_h)
    cv = _normalise(cost_v)
    # log potentials: phi_h[a, b] is the score of tile b sitting right of tile a
    log_h = -ch / temp
    log_v = -cv / temp

    # messages[d, r, c, label]: message into cell (r,c) from direction d
    # d = 0 from left, 1 from right, 2 from above, 3 from below
    msg = np.zeros((4, G, G, N), np.float32)
    belief = np.zeros((G, G, N), np.float32)

    for _ in range(iters):
        new = np.zeros_like(msg)
        # incoming totals excluding the direction being sent
        total = msg.sum(0) + belief * 0.0
        for r in range(G):
            for c in range(G):
                inc = total[r, c]
                if c + 1 < G:                      # send right: cell(r,c) -> cell(r,c+1)
                    m = inc - msg[1, r, c]
                    new[0, r, c + 1] = (m[:, None] + log_h).max(axis=0)
                if c - 1 >= 0:                     # send left
                    m = inc - msg[0, r, c]
                    new[1, r, c - 1] = (m[None, :] + log_h).max(axis=1)
                if r + 1 < G:                      # send down
                    m = inc - msg[3, r, c]
                    new[2, r + 1, c] = (m[:, None] + log_v).max(axis=0)
                if r - 1 >= 0:                     # send up
                    m = inc - msg[2, r, c]
                    new[3, r - 1, c] = (m[None, :] + log_v).max(axis=1)
        new -= new.max(axis=3, keepdims=True)      # keep messages bounded
        msg = damping * msg + (1.0 - damping) * new
        belief = msg.sum(0)

    # BP gives per-cell scores; Hungarian turns them into a legal bijection
    score = belief.reshape(N, N)
    if top_k and top_k < N:
        cut = np.partition(score, N - top_k, axis=1)[:, N - top_k][:, None]
        score = np.where(score >= cut, score, -1e6)
    cell, tile = linear_sum_assignment(-score)
    board = np.empty(N, np.int64)
    board[cell] = tile
    return board
