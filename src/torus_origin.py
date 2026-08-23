"""Recover the absolute image origin after a torus-ambiguous solve.

The buddies solver only ever sees RELATIVE adjacency, so its board is correct
up to a cyclic shift: on oracle scores it reproduces the true arrangement
exactly but rolled by (4,0), which scores place_acc 0.0000 while being a
perfect solve.  Every `place_acc ~= chance` reading in this repo is suspect for
this reason.

Fix: the real image border is the one seam that has no continuity behind it, so
the correct origin is the row/column cut with the LARGEST seam cost.  24 + 24
line scores, no search over 576 placements.
"""
from __future__ import annotations

import numpy as np

from config import GRID as G, NFRAG as N
from restore_tile import ridge_cost


def _cut_costs(board: np.ndarray, cost_h: np.ndarray, cost_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total seam cost of each of the 24 toroidal column / row cuts."""
    grid = board.reshape(G, G)
    col = np.empty(G, np.float64)
    row = np.empty(G, np.float64)
    for c in range(G):
        left, right = grid[:, c], grid[:, (c + 1) % G]
        col[c] = cost_h[left, right].sum()
    for r in range(G):
        top, bottom = grid[r, :], grid[(r + 1) % G, :]
        row[r] = cost_v[top, bottom].sum()
    return row, col


def fix_origin(board: np.ndarray, tiles: np.ndarray, w: float = 0.03,
               cols: int = 3, metric: str = "mgc") -> np.ndarray:
    """Roll `board` so the highest-cost toroidal cuts become the outer border.

    board[p] = tile index at grid position p.  Returns a board of the same
    convention with the absolute origin restored.

    The cut must be scored with the SAME measure that produced the layout: with
    an MGC-solved board (0.9965 correct up to shift on clean tiles) a ridge-based
    cut recovered only 0.6655, because the two disagree about which seams are
    expensive.
    """
    if metric == "mgc":
        from mgc import mgc_cost
        cost_h, cost_v = mgc_cost(tiles, "h"), mgc_cost(tiles, "v")
    else:
        cost_h = ridge_cost(tiles, w, cols, "h")
        cost_v = ridge_cost(tiles, w, cols, "v")
    row_cut, col_cut = _cut_costs(board, cost_h, cost_v)
    # cut index k means "the boundary sits between line k and k+1", so the tile
    # that must become line 0 is k+1.
    shift_r = (int(np.argmax(row_cut)) + 1) % G
    shift_c = (int(np.argmax(col_cut)) + 1) % G
    grid = board.reshape(G, G)
    return np.roll(np.roll(grid, -shift_r, axis=0), -shift_c, axis=1).reshape(N)


def best_possible_shift(board: np.ndarray, truth: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Diagnostic upper bound: accuracy of the best cyclic shift, using labels."""
    grid = board.reshape(G, G)
    best = (-1.0, (0, 0))
    for sr in range(G):
        for sc in range(G):
            cand = np.roll(np.roll(grid, -sr, axis=0), -sc, axis=1).reshape(N)
            acc = float(np.mean(cand == truth))
            if acc > best[0]:
                best = (acc, (sr, sc))
    return best
