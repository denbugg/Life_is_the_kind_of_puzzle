"""Paikin-Tal style greedy growth: place one tile at a time, most confident first.

Why the graph-first approach cannot work here
---------------------------------------------
Collecting a high-precision edge set and gluing it is hopeless at this scale.
Measured edges available per board per axis, ranked by mutual-best margin:

    tiles         p>=0.99  p>=0.95  p>=0.90  p>=0.80
    clean            10.2     16.8    114.5    210.8
    restored real     7.9      8.9     14.0     23.3

while feeding synthetic edge sets to a Kruskal builder shows it needs roughly
900 edges at precision >=0.95 to span the board.  Even PERFECT tiles are two
orders of magnitude short, so no scorer improvement rescues that formulation.
Conflict pruning does not help either: at 900 edges over 576 tiles the graph has
degree 1.5 and almost no cycles to cross-check.

What changes here
-----------------
Growth is sequential and context-aware.  A tile is never judged by a single
seam: once a free cell already touches two or three placed tiles, the candidate
is scored against ALL of them at once, so the effective evidence per decision
grows as the board fills.  Decisions are taken globally in confidence order --
the most certain (cell, tile) pair anywhere on the frontier goes first -- so
ambiguous regions are resolved last, after their context exists.
"""
from __future__ import annotations

import numpy as np

from config import GRID as G, NFRAG as N

_NEIGH = ((-1, 0), (1, 0), (0, -1), (0, 1))


def normalise(score: np.ndarray) -> np.ndarray:
    """Row+column standardisation of a compatibility matrix.

    Raw seam costs are not comparable across tiles: a textured tile produces
    costs an order of magnitude larger than a flat one.  Greedy growth compares
    confidences across DIFFERENT cells, so without this the frontier is ranked
    by tile texture rather than by match quality.
    """
    s = np.array(score, np.float64)
    finite = np.isfinite(s)
    fill = s[finite].min() if finite.any() else 0.0
    s = np.where(finite, s, fill)
    r = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)
    c = (s - s.mean(0, keepdims=True)) / (s.std(0, keepdims=True) + 1e-6)
    out = 0.5 * (r + c)
    np.fill_diagonal(out, -1e9)
    return out


def _seed(score_h: np.ndarray, score_v: np.ndarray) -> tuple[int, int]:
    """Most confident mutual pair anywhere; returns (tile, its right/below tile)."""
    best = (-np.inf, 0, 0, "h")
    for mat, tag in ((score_h, "h"), (score_v, "v")):
        s = mat.copy()
        np.fill_diagonal(s, -np.inf)
        part = np.partition(s, -2, axis=1)
        margin = part[:, -1] - part[:, -2]
        col_best = s.argmax(0)
        for a in range(N):
            b = int(s[a].argmax())
            if col_best[b] == a and margin[a] > best[0]:
                best = (margin[a], a, b, tag)
    return best[1], best[2], best[3]


def solve_grow(score_h: np.ndarray, score_v: np.ndarray,
               margin_weight: float = 1.0) -> np.ndarray:
    """score matrices (higher = better) -> board[p] = tile index at position p.

    score_h[i,j] scores j placed to the right of i; score_v[i,j] j below i.
    """
    score_h = normalise(score_h)
    score_v = normalise(score_v)
    board = np.full((G, G), -1, np.int64)
    used = np.zeros(N, bool)

    a, b, tag = _seed(score_h, score_v)
    r0, c0 = G // 2, G // 2
    board[r0, c0] = a
    used[a] = True
    if tag == "h" and c0 + 1 < G:
        board[r0, c0 + 1] = b
    else:
        board[r0 + 1, c0] = b
    used[b] = True

    for _ in range(N - 2):
        free_tiles = np.flatnonzero(~used)
        if not len(free_tiles):
            break
        # candidate cells: empty, adjacent to at least one placed tile, and
        # inside a 24x24 window that still fits every remaining tile
        rows, cols = np.nonzero(board >= 0)
        rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()
        best = (-np.inf, None, None)
        for r in range(G):
            for c in range(G):
                if board[r, c] >= 0:
                    continue
                if max(rmax, r) - min(rmin, r) >= G or max(cmax, c) - min(cmin, c) >= G:
                    continue
                total = np.zeros(len(free_tiles), np.float64)
                n_ctx = 0
                for dr, dc in _NEIGH:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < G and 0 <= nc < G) or board[nr, nc] < 0:
                        continue
                    other = board[nr, nc]
                    n_ctx += 1
                    if (dr, dc) == (0, -1):        # placed tile is to the LEFT
                        total += score_h[other, free_tiles]
                    elif (dr, dc) == (0, 1):       # placed tile is to the RIGHT
                        total += score_h[free_tiles, other]
                    elif (dr, dc) == (-1, 0):      # placed tile is ABOVE
                        total += score_v[other, free_tiles]
                    else:                          # placed tile is BELOW
                        total += score_v[free_tiles, other]
                if n_ctx == 0:
                    continue
                order = np.argsort(-total)
                cand = int(free_tiles[order[0]])
                top = total[order[0]]
                second = total[order[1]] if len(order) > 1 else top
                # Best-buddy priority (Paikin & Tal): a candidate that also picks
                # this cell's context as ITS own best match is placed before any
                # merely-high-scoring one, however large the raw score.  Without
                # this the frontier is driven by whichever cell happens to have
                # the loudest neighbours.
                buddy = False
                for dr, dc in _NEIGH:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < G and 0 <= nc < G) or board[nr, nc] < 0:
                        continue
                    o = board[nr, nc]
                    if (dr, dc) == (0, -1):
                        buddy |= int(np.argmax(score_h[o])) == cand and int(np.argmax(score_h[:, cand])) == o
                    elif (dr, dc) == (0, 1):
                        buddy |= int(np.argmax(score_h[cand])) == o and int(np.argmax(score_h[:, o])) == cand
                    elif (dr, dc) == (-1, 0):
                        buddy |= int(np.argmax(score_v[o])) == cand and int(np.argmax(score_v[:, cand])) == o
                    else:
                        buddy |= int(np.argmax(score_v[cand])) == o and int(np.argmax(score_v[:, o])) == cand
                conf = (1e6 if buddy else 0.0) + top / n_ctx + margin_weight * (top - second)
                if conf > best[0]:
                    best = (conf, (r, c), cand)
        if best[1] is None:
            break
        (r, c), tile = best[1], best[2]
        board[r, c] = tile
        used[tile] = True

    # any leftovers go into whatever cells remain, in index order
    leftovers = list(np.flatnonzero(~used))
    for r in range(G):
        for c in range(G):
            if board[r, c] < 0 and leftovers:
                board[r, c] = leftovers.pop()
    return board.reshape(N)
