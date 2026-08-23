"""Agglomerative assembly: grow islands, and re-score at the island level.

The measurement this is built on
--------------------------------
M150 found that the wall this project has spent its life against is a property
of the OBJECT being matched, not of the matcher.  Judging one tile against one
tile gives mutual-edge precision 0.438 against a knee of 0.72.  Judging a 2x2
block against a 2x2 block gives 0.709; 3x3 gives 0.840; 4x4 gives 0.882 with
R@1 0.847.  A k x k island presents a seam 20k pixels long, the errors along it
are largely independent, and the decision sharpens roughly as sqrt(k).

Every solver already in this repo -- greedy loop construction, the LP, the
Sholomon genetic crossover, relaxation labelling -- consumes tile-to-tile costs.
Sholomon's kernel growth comes closest, but it still SELECTS with a single
edge's score.  The whole point here is to recompute the score over the entire
shared boundary once the pieces are bigger than one tile, which is the step that
turns 0.438 into 0.88.

M151 confirmed the entry point exists: from the real cost matrices, mutual loop
closure yields about 13 non-overlapping 2x2 blocks at precision 0.878, covering
9% of the board.  Small, but they are objects, and objects match at 0.709.

What this deliberately does not do
----------------------------------
It does not place anything absolutely.  Seams give relative structure only, and
M149 measured that our predicted colour field cannot supply the missing
coordinate (island placement 0.000 against an oracle field's 0.911, because our
field's RMSE is 53 where about 30 is needed).  An island that reaches the full
width or height of the board localises itself along that axis and nothing else
does, so `absolute_span` reports how far each island got.

Callers that want a shippable answer should follow M148: place only the islands
they trust and leave the rest to the restorer.  Coverage 0.20 at precision 0.9
is worth about the same as a flat blob while carrying seventy times the visible
detail; coverage 0.20 at precision 0.7 is worth LESS than the blob.  Placing
something we are unsure of is the expensive mistake.
"""
from __future__ import annotations

import numpy as np

GRID = 24


class Island:
    """A set of tiles with fixed relative coordinates.

    `cells` maps (row, col) -> tile index, in the island's own frame, which is
    normalised so the top-left occupied cell is (0, 0).
    """

    __slots__ = ("cells", "height", "width")

    def __init__(self, cells):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        self.cells = {(r - r0, c - c0): t for (r, c), t in cells.items()}
        self.height = max(rows) - r0 + 1
        self.width = max(cols) - c0 + 1

    def __len__(self):
        return len(self.cells)

    def tiles(self):
        return set(self.cells.values())


def seed_quads(cost_h, cost_v, mutual=True, keep_fraction=0.5, grid=GRID):
    """Harvest 2x2 blocks by loop closure; the M151 entry point.

    Four independent judgements have to agree, which filters far harder than any
    single edge: mutual closure keeps its cheapest half at precision 0.878 where
    the underlying edges run at 0.438.
    """
    n = cost_h.shape[0]
    H, V = cost_h.copy(), cost_v.copy()
    np.fill_diagonal(H, np.inf)
    np.fill_diagonal(V, np.inf)
    right, below = H.argmin(1), V.argmin(1)
    right_back, below_back = H.argmin(0), V.argmin(0)

    found = []
    for a in range(n):
        b, c = int(right[a]), int(below[a])
        if b == c:
            continue
        d = int(right[c])
        if d != int(below[b]) or len({a, b, c, d}) != 4:
            continue
        if mutual and not (right_back[b] == a and below_back[c] == a
                           and right_back[d] == c and below_back[d] == b):
            continue
        found.append((H[a, b] + H[c, d] + V[a, c] + V[b, d], (a, b, c, d)))

    found.sort(key=lambda kv: kv[0])
    used, islands = set(), []
    for _, (a, b, c, d) in found:
        if used & {a, b, c, d}:
            continue
        used |= {a, b, c, d}
        islands.append(Island({(0, 0): a, (0, 1): b, (1, 0): c, (1, 1): d}))
    if keep_fraction < 1.0:
        islands = islands[: max(1, int(round(keep_fraction * len(islands))))]
    placed = {t for isl in islands for t in isl.tiles()}
    islands += [Island({(0, 0): t}) for t in range(n) if t not in placed]
    return islands


def islands_from_edges(edges, n_tiles, grid=GRID):
    """Build islands directly from a trusted edge set, geometry-checked.

    `edges` are (i, j, (dr, dc), confidence): tile j sits at offset (dr, dc)
    from tile i.  They are consumed in descending confidence, and an edge is
    accepted only if it does not contradict what has already been built --
    either by implying a different relative position for two tiles that are
    already in one island, or by putting two tiles in the same cell, or by
    pushing an island past the board.

    This exists because routing a trusted edge set back through a cost matrix
    destroys it: M158 filled the non-agreed entries with a constant and the
    growth rule, which compares a merge against its runner-up, then had nothing
    to compare against and lost a quarter of its trusted coverage.  Edges are
    the evidence; islands should be built from them directly.
    """
    parent = list(range(n_tiles))
    offset = [(0, 0)] * n_tiles          # position of the tile within its root

    def find(x):
        root, shift = x, (0, 0)
        while parent[root] != root:
            shift = (shift[0] + offset[root][0], shift[1] + offset[root][1])
            root = parent[root]
        return root, shift

    members = {i: {(0, 0): i} for i in range(n_tiles)}
    for i, j, (dr, dc), _conf in edges:
        ri, si = find(i)
        rj, sj = find(j)
        want = (si[0] + dr - sj[0], si[1] + dc - sj[1])
        if ri == rj:
            continue                     # already related; contradictions ignored
        a, b = members[ri], members[rj]
        moved = {(r + want[0], c + want[1]): t for (r, c), t in b.items()}
        if set(moved) & set(a):
            continue                     # two tiles would share a cell
        union = {**a, **moved}
        rows = [r for r, _ in union]
        cols = [c for _, c in union]
        if max(rows) - min(rows) >= grid or max(cols) - min(cols) >= grid:
            continue                     # wider or taller than the board
        parent[rj] = ri
        offset[rj] = want
        members[ri] = union
        del members[rj]
    return [Island(cells) for cells in members.values()]


def merge_cost(a, b, offset, cost_h, cost_v, min_boundary=1):
    """Mean tile cost along the seam when `b` sits at `offset` from `a`.

    Returns (cost, boundary_length) or None when the two overlap or do not
    touch.  Averaging over the whole boundary is the entire point: it is what
    takes precision from 0.438 to 0.88 once the pieces are larger than a tile.
    """
    dr, dc = offset
    shifted = {(r + dr, c + dc): t for (r, c), t in b.cells.items()}
    if set(shifted) & set(a.cells):
        return None
    total, count = 0.0, 0
    for (r, c), t in a.cells.items():
        u = shifted.get((r, c + 1))
        if u is not None:
            total += cost_h[t, u]
            count += 1
        u = shifted.get((r + 1, c))
        if u is not None:
            total += cost_v[t, u]
            count += 1
    for (r, c), t in shifted.items():
        u = a.cells.get((r, c + 1))
        if u is not None:
            total += cost_h[t, u]
            count += 1
        u = a.cells.get((r + 1, c))
        if u is not None:
            total += cost_v[t, u]
            count += 1
    if count < min_boundary:
        return None
    return total / count, count


def _offsets(a, b, grid):
    """Relative placements of b against a that keep the union inside the grid."""
    for dr in range(-b.height + 1 - 1, a.height + 1):
        for dc in range(-b.width + 1 - 1, a.width + 1):
            h = max(a.height, dr + b.height) - min(0, dr)
            w = max(a.width, dc + b.width) - min(0, dc)
            if h <= grid and w <= grid:
                yield dr, dc


def merge(a, b, offset):
    dr, dc = offset
    cells = dict(a.cells)
    for (r, c), t in b.cells.items():
        cells[(r + dr, c + dc)] = t
    return Island(cells)


def _propose(islands, cost_h, cost_v, topk):
    """Candidate (i, j, offset) triples suggested by tile-level edges.

    Considering every pair at every offset is O(n^2 * offsets) and does not run
    at 533 islands.  Proposals are cheap and only have to be RECALLED, not
    judged: a tile on island i's boundary names its top-k partners under the
    relevant cost, the island holding that partner is j, and the two tiles'
    coordinates fix the offset exactly.  The proposal is then scored over the
    whole shared boundary, which is where the precision actually comes from.
    """
    owner, where = {}, {}
    for idx, isl in enumerate(islands):
        for rc, t in isl.cells.items():
            owner[t] = idx
            where[t] = rc
    order_h = np.argsort(cost_h, axis=1)[:, :topk]
    order_v = np.argsort(cost_v, axis=1)[:, :topk]
    order_h_back = np.argsort(cost_h, axis=0)[:topk].T
    order_v_back = np.argsort(cost_v, axis=0)[:topk].T

    seen = set()
    for i, isl in enumerate(islands):
        for (r, c), t in isl.cells.items():
            # (direction from t to the partner, table, partner's cell offset)
            for table, (dr, dc) in ((order_h[t], (0, 1)), (order_v[t], (1, 0)),
                                    (order_h_back[t], (0, -1)),
                                    (order_v_back[t], (-1, 0))):
                for u in table:
                    u = int(u)
                    j = owner.get(u)
                    if j is None or j == i:
                        continue
                    ur, uc = where[u]
                    off = (r + dr - ur, c + dc - uc)
                    key = (i, j, off)
                    if key not in seen:
                        seen.add(key)
                        yield key


def grow(islands, cost_h, cost_v, rounds=6, min_boundary=1, grid=GRID,
         mutual=True, topk=3, strong_boundary=2, margin=1.0):
    """Repeatedly merge mutually-best island pairs, re-scoring every round.

    Boundary length is the whole currency here, because it is what separates the
    0.438 regime from the 0.88 one (M150).  A merge whose shared boundary is
    `strong_boundary` cells or longer is decided in the good regime and is
    accepted on its cost alone.  A merge across a boundary of one cell carries
    exactly the tile-level evidence we already know is not enough, so it is
    accepted only when it also clears a MARGIN: the runner-up for the same
    island must cost at least `margin` times as much.

    The margin exists because refusing single-cell merges outright does not
    work either -- M153 measured that, with them excluded, only the seed blocks
    can merge with each other, coverage stays at 9% and trusted coverage falls.
    Single tiles have to be absorbed somehow, and a high-margin single-cell
    merge is the only door available at the start: it makes L-shapes, L-shapes
    make notches, and a notch is a boundary of 2 to 4, which is back in the
    good regime.  `margin = 1.0` disables the check.
    """
    for _ in range(rounds):
        best_pair = {}
        for i, j, off in _propose(islands, cost_h, cost_v, topk):
            got = merge_cost(islands[i], islands[j], off, cost_h, cost_v,
                             min_boundary)
            if got is None:
                continue
            if (i, j) not in best_pair or got[0] < best_pair[(i, j)][0]:
                best_pair[(i, j)] = (got[0], off, got[1])
        cands = [(v[0], i, j, v[1], v[2]) for (i, j), v in best_pair.items()]
        if not cands:
            break
        # best and runner-up per island, so a weak-boundary merge can be asked
        # to prove it is not merely the least bad of several equals
        best_for, second_for = {}, {}
        for score, i, j, off, blen in cands:
            if i not in best_for or score < best_for[i][0]:
                second_for[i] = best_for.get(i)
                best_for[i] = (score, j, off, blen)
            elif i not in second_for or second_for[i] is None or score < second_for[i][0]:
                second_for[i] = (score, j, off, blen)
        cands.sort(key=lambda kv: (-kv[4], kv[0]))   # long boundaries first
        used, new, dead = set(), [], set()
        for score, i, j, off, blen in cands:
            if i in used or j in used:
                continue
            if best_for.get(i, (None, None))[1] != j:
                continue
            if mutual and best_for.get(j, (None, None))[1] != i:
                continue
            if blen < strong_boundary and margin > 1.0:
                runner = second_for.get(i)
                if runner is None or runner[0] < margin * max(score, 1e-9):
                    continue
                runner = second_for.get(j)
                if runner is None or runner[0] < margin * max(score, 1e-9):
                    continue
            new.append(merge(islands[i], islands[j], off))
            used |= {i, j}
            dead |= {i, j}
        if not new:
            break
        islands = new + [isl for k, isl in enumerate(islands) if k not in dead]
        islands.sort(key=len, reverse=True)
    return islands


def absolute_span(island, grid=GRID):
    """Whether the island is wide or tall enough to localise itself.

    An island spanning the full width must occupy every column, which fixes its
    columns exactly; the same holds for height.  Nothing else in this pipeline
    supplies an absolute coordinate (M149).
    """
    return island.width >= grid, island.height >= grid
