"""Robust global placement by linear programming (Yu, Russell & Agapito, BMVC 2016).

Why this exists
---------------
Every solver in this repository is greedy: best-buddy growth, Kruskal over
loop-verified edges, Paikin-Tal frontier growth.  All of them commit to
decisions locally, so a single wrong edge drags an entire component out of
alignment -- measured on CLEAN tiles, greedy assembly reached only 0.19 even
with 90% correct edges.

The LP formulation instead uses ALL pairwise matches at once and solves for
every piece position globally.  Its robustness comes from the objective: the
non-convex L0 count is relaxed to a weighted L1 penalty, under which a minority
of wrong constraints is absorbed rather than propagated.  On the corrupted-puzzle
benchmark (arXiv 2507.07828) this solver ranks first under eroded edges, which
is exactly our degradation.

Formulation
-----------
Variables are continuous piece positions (x_i, y_i).  Each candidate match
(i, j, dx, dy) with confidence w contributes

    w * ( |x_j - x_i - dx| + |y_j - y_i - dy| )

linearised with slack variables.  One piece is pinned to remove the global
translation ambiguity.  Successive rounds re-solve after discarding matches
whose residual under the current solution is large, which is the paper's
U(k)/A(k) reweighting.  Positions are finally snapped to the integer grid by
Hungarian assignment, which restores the bijection the LP relaxation dropped.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import csr_matrix

from config import GRID as G, NFRAG as N


def build_matches(cost_h: np.ndarray, cost_v: np.ndarray, top_k: int = 1,
                  mutual: bool = True, margin_pow: float = 2.0
                  ) -> list[tuple[int, int, int, int, float]]:
    """Candidate matches (i, j, dx, dy, weight).

    Two rules matter, both measured.  First, feed the LP only MUTUAL argmin
    edges: taking the top-4 of every row yields 4608 candidates of which at most
    1104 can be correct, i.e. precision below 0.24, and the LP drowns.  Second,
    weight them: with weights the LP tolerates 10% outliers exactly, without
    them it already fails at 5%.  The relative margin is a calibrated
    confidence -- the top 5% of edges by margin carry precision 0.922 against
    0.263 for the full mutual set -- so weight rises steeply with it.
    """
    out = []
    # positions are (row, col): a horizontal neighbour differs in COLUMN,
    # a vertical one in ROW.  Swapping these silently yields a valid LP whose
    # solution is transposed nonsense.
    for mat, d_row, d_col in ((cost_h, 0, 1), (cost_v, 1, 0)):
        C = mat.copy()
        np.fill_diagonal(C, np.inf)
        best = C.argmin(1)
        rev = C.argmin(0)
        srt = np.sort(C, axis=1)
        for i in range(N):
            for r in range(top_k):
                j = int(np.argsort(C[i])[r]) if top_k > 1 else int(best[i])
                if mutual and rev[j] != i:
                    continue
                margin = (srt[i, r + 1] - srt[i, r]) / (abs(srt[i, r]) + 1e-9)
                out.append((i, j, d_row, d_col, float(max(margin, 1e-4) ** margin_pow)))
    return out


def solve_positions(matches, pin: int = 0) -> np.ndarray | None:
    """Weighted-L1 translation synchronisation. Returns (N,2) float positions."""
    m = len(matches)
    # unknowns: x(N), y(N), tx(m), ty(m)
    n_var = 2 * N + 2 * m
    c = np.zeros(n_var)
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for e, (i, j, dx, dy, w) in enumerate(matches):
        c[2 * N + e] = w
        c[2 * N + m + e] = w
        for sign, off, d, base in ((1, 0, dx, 0), (-1, 0, dx, 0),
                                   (1, m, dy, N), (-1, m, dy, N)):
            #  sign*(x_j - x_i - d) - t <= 0
            rows += [r, r, r]
            cols += [base + j, base + i, 2 * N + off + e]
            vals += [sign, -sign, -1.0]
            rhs.append(sign * d)
            r += 1
    A = csr_matrix((vals, (rows, cols)), shape=(r, n_var))
    bounds = [(None, None)] * (2 * N) + [(0, None)] * (2 * m)
    bounds[pin] = (0.0, 0.0)
    bounds[N + pin] = (0.0, 0.0)
    res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method="highs")
    if not res.success:
        return None
    return np.stack([res.x[:N], res.x[N:2 * N]], axis=1)


def snap_to_grid(pos: np.ndarray) -> np.ndarray:
    """Assign continuous positions to the 24x24 lattice, one piece per cell."""
    pos = pos - pos.min(axis=0)
    span = np.maximum(pos.max(axis=0), 1e-6)
    scaled = pos / span * (G - 1)
    cells = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(N, 2)
    d = ((scaled[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    piece, cell = linear_sum_assignment(d)
    board = np.empty(N, np.int64)
    board[cells[cell, 0] * G + cells[cell, 1]] = piece
    return board


def solve_lp(cost_h: np.ndarray, cost_v: np.ndarray, top_k: int = 1,
             rounds: int = 3, keep_frac: float = 0.7, margin_pow: float = 2.0) -> np.ndarray:
    """cost matrices (lower = better) -> board[p] = piece index at position p."""
    matches = build_matches(cost_h, cost_v, top_k, True, margin_pow)
    pos = None
    for _ in range(rounds):
        got = solve_positions(matches)
        if got is None:
            break
        pos = got
        # U(k)/A(k): keep the matches the current solution actually explains
        resid = np.array([abs(pos[j, 0] - pos[i, 0] - dx) + abs(pos[j, 1] - pos[i, 1] - dy)
                          for i, j, dx, dy, _ in matches])
        thr = np.quantile(resid, keep_frac)
        kept = [mt for mt, rr in zip(matches, resid) if rr <= thr]
        if len(kept) < 4 * N or len(kept) == len(matches):
            break
        matches = kept
    if pos is None:
        return np.arange(N)
    return snap_to_grid(pos)
