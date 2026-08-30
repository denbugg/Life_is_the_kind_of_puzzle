"""High-precision relative islands anchored by an absolute unary field."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


class _Components:
    def __init__(self, side):
        self.side = int(side)
        self.owner: dict[int, int] = {}
        self.items: list[dict[int, tuple[int, int]]] = []

    def _valid(self, comp):
        cells = list(comp.values())
        if len(cells) != len(set(cells)):
            return False
        ys, xs = zip(*cells)
        return max(ys) - min(ys) < self.side and max(xs) - min(xs) < self.side

    def add(self, a, b, dy, dx):
        ca, cb = self.owner.get(a), self.owner.get(b)
        if ca is None and cb is None:
            k = len(self.items)
            self.items.append({a: (0, 0), b: (dy, dx)})
            self.owner[a] = self.owner[b] = k
            return True
        if ca is not None and cb is None:
            ay, ax = self.items[ca][a]
            trial = dict(self.items[ca]); trial[b] = (ay + dy, ax + dx)
            if not self._valid(trial):
                return False
            self.items[ca] = trial; self.owner[b] = ca
            return True
        if ca is None and cb is not None:
            by, bx = self.items[cb][b]
            trial = dict(self.items[cb]); trial[a] = (by - dy, bx - dx)
            if not self._valid(trial):
                return False
            self.items[cb] = trial; self.owner[a] = cb
            return True
        if ca == cb:
            ay, ax = self.items[ca][a]; by, bx = self.items[ca][b]
            return (by - ay, bx - ax) == (dy, dx)
        aa, bb = self.items[ca], self.items[cb]
        ay, ax = aa[a]; by, bx = bb[b]
        sy, sx = ay + dy - by, ax + dx - bx
        moved = {tile: (y + sy, x + sx) for tile, (y, x) in bb.items()}
        trial = dict(aa); trial.update(moved)
        if not self._valid(trial):
            return False
        self.items[ca] = trial; self.items[cb] = {}
        for tile in moved:
            self.owner[tile] = ca
        return True

    def result(self):
        return [c for c in self.items if len(c) > 1]


def select_islands(right, down, side, keep):
    """Select a board-wide score tail and make conflict-safe rigid islands."""
    n = side * side
    mats = np.stack([np.asarray(right, np.float64),
                     np.asarray(down, np.float64)]).copy()
    diag = np.arange(n)
    mats[:, diag, diag] = -np.inf
    flat = mats.reshape(-1)
    # Conflict checks reject some claims.  A bounded oversampled tail avoids a
    # Python sort of all 2*N^2 pairs (662k on a full board) while leaving ample
    # replacements for rejected claims.
    pool = min(len(flat), max(int(keep) * 16, int(keep) + 512))
    ids = np.argpartition(flat, -pool)[-pool:]
    ids = ids[np.argsort(flat[ids])[::-1]]
    builder = _Components(side)
    accepted = 0
    plane = n * n
    for index in ids:
        direction, rem = divmod(int(index), plane)
        i, j = divmod(rem, n)
        dy, dx = ((0, 1), (1, 0))[direction]
        if builder.add(i, j, dy, dx):
            accepted += 1
            if accepted >= keep:
                break
    return builder.result()


def border_score(right, down, side):
    """Cell×tile score for the four missing-neighbour constraints.

    A weak best outward match is evidence that the tile belongs on that side of
    the bag's frame.  Directional values are rank-normalised, making the score
    insensitive to matcher temperature and horizontal/vertical scale.
    """
    right = np.asarray(right, np.float64).copy()
    down = np.asarray(down, np.float64).copy()
    np.fill_diagonal(right, -np.inf); np.fill_diagonal(down, -np.inf)
    outward = np.stack([right.max(0), right.max(1),
                        down.max(0), down.max(1)])  # left,right,up,down
    rank = np.empty_like(outward)
    n = side * side
    for d in range(4):
        order = np.argsort(np.argsort(outward[d]))
        rank[d] = (order + 0.5) / n * 2.0 - 1.0
    # Maximised score: weak match (low rank) is good on the corresponding edge.
    score = np.zeros((n, n), np.float64)
    for p in range(n):
        y, x = divmod(p, side)
        if x == 0: score[p] -= rank[0]
        if x == side - 1: score[p] -= rank[1]
        if y == 0: score[p] -= rank[2]
        if y == side - 1: score[p] -= rank[3]
    return score


@dataclass
class _State:
    used: np.ndarray
    fixed: list[tuple[np.ndarray, np.ndarray]]
    cost: float


def _options(component, unary_cost, side, limit):
    tiles = np.asarray(list(component), np.int64)
    coords = np.asarray([component[t] for t in tiles], np.int64)
    coords -= coords.min(0, keepdims=True)
    h, w = coords.max(0) + 1
    out = []
    for y in range(side - h + 1):
        for x in range(side - w + 1):
            cells = (coords[:, 0] + y) * side + coords[:, 1] + x
            out.append((float(unary_cost[cells, tiles].sum()), cells, tiles))
    out.sort(key=lambda z: z[0])
    return out[:limit]


def anchor_islands(unary_score, components, side, beam=128, offsets=36):
    """Place rigid islands under unary evidence, then assign all free tiles."""
    unary_cost = -np.asarray(unary_score, np.float64)
    n = side * side
    # Large/high-margin islands first; tiny islands cannot consume the beam.
    opts = [(_options(c, unary_cost, side, offsets), c) for c in components]
    opts.sort(key=lambda z: (-len(z[1]), z[0][0][0] / len(z[1])))
    states = [_State(np.zeros(n, bool), [], 0.0)]
    for choices, _ in opts:
        nxt = []
        for state in states:
            for value, cells, tiles in choices:
                if state.used[cells].any():
                    continue
                used = state.used.copy(); used[cells] = True
                nxt.append(_State(used, state.fixed + [(cells, tiles)],
                                  state.cost + value))
        if not nxt:
            # A low-value island may be geometrically incompatible with all
            # stronger ones; dropping it is safer than welding the solution.
            continue
        nxt.sort(key=lambda s: s.cost)
        states = nxt[:beam]

    best_layout, best_cost = None, np.inf
    for state in states:
        layout = np.full(n, -1, np.int64)
        fixed_tiles = np.zeros(n, bool)
        fixed_cost = 0.0
        for cells, tiles in state.fixed:
            layout[cells] = tiles; fixed_tiles[tiles] = True
            fixed_cost += float(unary_cost[cells, tiles].sum())
        cells = np.flatnonzero(layout < 0)
        tiles = np.flatnonzero(~fixed_tiles)
        rr, cc = linear_sum_assignment(unary_cost[np.ix_(cells, tiles)])
        layout[cells[rr]] = tiles[cc]
        value = fixed_cost + float(unary_cost[cells[rr], tiles[cc]].sum())
        if value < best_cost:
            best_layout, best_cost = layout, value
    return best_layout, best_cost


def solve_island_field(right, down, unary_score, side, keep,
                       beam=128, offsets=36, border_weight=0.0):
    components = select_islands(right, down, side, keep)
    unary = np.asarray(unary_score) + float(border_weight) * border_score(
        right, down, side)
    return (*anchor_islands(unary, components, side, beam, offsets),
            components)
