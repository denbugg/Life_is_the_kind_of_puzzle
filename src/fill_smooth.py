"""Fill the cells no component claims by colour continuity, not by seam score.

The problem this solves
-----------------------
A component packer places the pieces it trusts and then has to put roughly four
hundred leftover tiles somewhere.  Today they go by the same raw seam-score sum
that M171 showed is misaligned -- and those particular tiles are the ones the
matcher cannot read at all: M160 measured edge precision among the leftovers at
0.349 against 0.46 over the whole board, because assembly takes the easy tiles
first and what remains is flat sky and texture-free regions.

So their seam scores are noise, and arranging them by noise scatters the colour
field.  That matters more than it sounds: M137 established that the metric on
this task is paid almost entirely by the low-frequency colour field, and M171
saw exactly the predicted symptom -- the packer raised place_acc elevenfold and
LOST adjacency, netting +0.003.

What is left in a tile whose seams are unreadable is its mean colour, and a
photograph is locally continuous.  Filling by that is an assembly criterion, not
a restoration one: every fragment is placed exactly once and none is altered.

Order matters.  Cells are filled most-constrained-first -- the cell with the
most already-placed neighbours is decided first -- so each decision is made with
the most evidence available, and the frontier grows inward from what is trusted.
"""
from __future__ import annotations

import numpy as np

GRID = 24
NEIGH = ((-1, 0), (1, 0), (0, -1), (0, 1))


def fill_by_colour(board, tiles, grid=GRID):
    """Complete a partial board, choosing each tile by local colour continuity.

    `board` is a (grid*grid,) int array with -1 in unfilled cells and a tile
    index elsewhere.  Returns a full permutation.
    """
    board = np.asarray(board, np.int64).copy()
    n = grid * grid
    mu = tiles.reshape(tiles.shape[0], -1, 3).mean(1)
    free = [t for t in range(n) if t not in set(board[board >= 0].tolist())]
    free = np.array(free, np.int64)
    if free.size == 0:
        return board

    def neighbours(cell):
        r, c = divmod(cell, grid)
        out = []
        for dr, dc in NEIGH:
            rr, cc = r + dr, c + dc
            if 0 <= rr < grid and 0 <= cc < grid and board[rr * grid + cc] >= 0:
                out.append(board[rr * grid + cc])
        return out

    empty = [i for i in range(n) if board[i] < 0]
    while empty and free.size:
        # most-constrained cell first: decide where the evidence is strongest
        counts = [(len(neighbours(cell)), cell) for cell in empty]
        counts.sort(key=lambda kv: -kv[0])
        k, cell = counts[0]
        if k == 0:
            # nothing placed anywhere near; take any cell and any tile, the
            # choice is genuinely uninformed
            cell = empty[0]
            pick = 0
        else:
            ref = mu[neighbours(cell)].mean(0)
            d = np.abs(mu[free] - ref[None, :]).sum(1)
            pick = int(np.argmin(d))
        board[cell] = free[pick]
        free = np.delete(free, pick)
        empty.remove(cell)
    return board


def board_from_layout(layout, keep, grid=GRID):
    """Blank every cell whose tile is not in `keep`, so it can be refilled."""
    out = np.full(grid * grid, -1, np.int64)
    keep = set(int(t) for t in keep)
    for cell, tile in enumerate(np.asarray(layout, np.int64)):
        if int(tile) in keep:
            out[cell] = int(tile)
    return out
