"""Fill the most-constrained cell first, using every neighbour it already has.

The measurement this exists for: with the deployed cost path, choosing the tile
for a cell whose neighbours are known scores

    1 neighbour   top-1 0.324      the ordinary matching problem
    2 neighbours  top-1 0.511      above M102's activation threshold
    3 neighbours  top-1 0.608
    4 neighbours  top-1 0.669

and the candidate set is 576 at every k -- only the evidence grows.  This is
M150's object-size effect taken on the evidence side rather than the object
side, which is the side that does not pay for it: M180 killed merging because
498 components times hundreds of offsets is a candidate set two orders of
magnitude larger, and here the set never grows at all.

The packer does not use this.  It places components against their contacts and
then fills every leftover cell -- about four hundred per board -- with a single
seam score, which M160 measured at precision 0.349.

Growth order is the whole design.  Always fill the empty cell with the MOST
placed neighbours, so the weakest decisions are made last, when the most
evidence is available; among cells tied on neighbour count, take the one whose
best candidate wins by the largest margin, so confident decisions come first and
shape the ones after them.
"""
from __future__ import annotations

import numpy as np

from config import GRID as G

N = G * G


def _neighbours(cell):
    """(neighbour cell, axis, this cell's role) for each of the four sides.

    role 'after' means this cell sits after the neighbour along that axis, so
    the score to read is M[neighbour_tile, candidate]; 'before' means this cell
    comes first and the score is M[candidate, neighbour_tile].
    """
    r, c = divmod(cell, G)
    out = []
    if c > 0:
        out.append((cell - 1, "h", "after"))
    if c < G - 1:
        out.append((cell + 1, "h", "before"))
    if r > 0:
        out.append((cell - G, "v", "after"))
    if r < G - 1:
        out.append((cell + G, "v", "before"))
    return out


def grow(H, V, seeds=None, min_neighbours=1):
    """Grow a full layout from `seeds`. H[i, j]: j sits right of i, higher better.

    `seeds` maps cell -> tile and may be empty; with no seeds the first cell is
    chosen by the single most confident edge on the board.  Returns
    `layout[cell] = tile` and, per cell, how many neighbours were known when it
    was decided, which is the honest confidence label for that decision.
    """
    H = np.ascontiguousarray(H, np.float64)
    V = np.ascontiguousarray(V, np.float64)
    layout = np.full(N, -1, np.int64)
    known = np.zeros(N, np.int8)
    score = np.zeros((N, N), np.float64)          # cell -> candidate evidence
    count = np.zeros(N, np.int8)
    used = np.zeros(N, bool)
    decided_with = np.zeros(N, np.int8)

    def place(cell, tile):
        layout[cell] = tile
        used[tile] = True
        for nb, axis, role in _neighbours(cell):
            if layout[nb] >= 0:
                continue
            M = H if axis == "h" else V
            # from the neighbour's point of view, WE are the placed one
            score[nb] += M[tile] if role == "before" else M[:, tile]
            count[nb] += 1

    if seeds:
        for cell, tile in seeds.items():
            place(int(cell), int(tile))
    else:
        best = np.unravel_index(int(np.argmax(H)), H.shape)
        place(0, int(best[0]))

    while (layout < 0).any():
        empty = np.nonzero(layout < 0)[0]
        ready = empty[count[empty] >= min(min_neighbours, count[empty].max())]
        if count[empty].max() == 0:
            # a disconnected remainder: seed it at the first free cell with the
            # globally best remaining tile, then carry on
            cell = int(empty[0])
            cand = np.nonzero(~used)[0]
            place(cell, int(cand[0]))
            decided_with[cell] = 0
            continue
        k = count[ready].max()
        ready = ready[count[ready] == k]
        # among equally constrained cells, take the most decisive one
        best_cell, best_tile, best_margin = -1, -1, -np.inf
        for cell in ready:
            s = score[cell].copy()
            s[used] = -np.inf
            top = np.argpartition(s, -2)[-2:]
            top = top[np.argsort(-s[top])]
            margin = s[top[0]] - s[top[1]]
            if margin > best_margin:
                best_cell, best_tile, best_margin = int(cell), int(top[0]), margin
        decided_with[best_cell] = k
        place(best_cell, best_tile)
    return layout, decided_with
