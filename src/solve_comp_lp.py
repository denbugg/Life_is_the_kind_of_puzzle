"""Place greedy's components by solving for their offsets, not by dropping them.

The loss this targets
---------------------
Greedy assembles well and positions badly.  At severity 0.3 its layout is worth
place_acc 0.305 at the best possible origin and scores 0.109 at the one it
picks; at severity 0.0, 0.347 against 0.086.  Two thirds of a finished assembly
is lost at the placement step (M102, M108).

Why the existing repairs cannot help.  `fix_origin` maximises the cost of the
excluded toroidal cuts, which is exactly optimal for the summed-cost objective,
but that objective is nearly blind to a cyclic shift -- rolling a torus changes
only WHICH 48 seams stop counting.  Selecting between solver arms by total cost
inherits the same blindness and picked the worse arm at severity 0.0.  A learned
per-tile position prior is the other kind of evidence and is too smooth to help:
row accuracy 0.073 against chance 0.042, exact shift recovery 0.000 alone, and
no gain at any mixing weight when added to the cut statistic.

What is left unused is the geometry between components.  `place_components`
drops them one at a time, largest first, best fit wins -- a greedy choice made
once per component with no way to revisit.  But every mutual edge whose ends lie
in DIFFERENT components is a constraint on their relative offset, and there are
hundreds of them against about a hundred components.  Solving those jointly is
the same weighted-L1 translation synchronisation the tile-level LP uses, on a
problem an order of magnitude smaller and correspondingly better conditioned:
roughly seven constraints per unknown instead of one.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import csr_matrix

from config import GRID as G, NFRAG as N


def _sync(n_units, cons, pin=0):
    """Weighted-L1 translation synchronisation over `n_units` unknown offsets.

    cons: iterable of (i, j, dr, dc, w) meaning offset_j - offset_i ~= (dr, dc).
    Returns (n_units, 2) offsets, or None if the LP fails.
    """
    m = len(cons)
    if m == 0:
        return np.zeros((n_units, 2))
    n_var = 2 * n_units + 2 * m
    c = np.zeros(n_var)
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for e, (i, j, dr, dc, w) in enumerate(cons):
        c[2 * n_units + e] = w
        c[2 * n_units + m + e] = w
        for sign, off, d, base in ((1, 0, dr, 0), (-1, 0, dr, 0),
                                   (1, m, dc, n_units), (-1, m, dc, n_units)):
            rows += [r, r, r]
            cols += [base + j, base + i, 2 * n_units + off + e]
            vals += [sign, -sign, -1.0]
            rhs.append(sign * d)
            r += 1
    A = csr_matrix((vals, (rows, cols)), shape=(r, n_var))
    bounds = [(None, None)] * (2 * n_units) + [(0, None)] * (2 * m)
    bounds[pin] = (0.0, 0.0)
    bounds[n_units + pin] = (0.0, 0.0)
    res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method="highs")
    if not res.success:
        return None
    return np.stack([res.x[:n_units], res.x[n_units:2 * n_units]], axis=1)


def _snap(pos):
    """Hungarian assignment of continuous positions onto the 24x24 lattice."""
    cells = np.stack([np.arange(N) // G, np.arange(N) % G], 1).astype(np.float64)
    cost = ((pos[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    tile, cell = linear_sum_assignment(cost)
    lay = np.empty(N, np.int64)
    lay[cell] = tile
    return lay


def _place_rigid(off, owner, internal, members, cost_h=None, cost_v=None):
    """Drop each component whole at its solved offset, largest first.

    Snapping tiles one at a time tears components apart -- it cost adjacency
    0.183 against greedy's 0.384 and place_acc 0.007 against 0.131 -- because
    the assignment is free to send neighbours to opposite corners.  The offsets
    are what the LP actually solved for; the internal geometry must be carried
    across intact, so a component moves as one piece and only the leftovers are
    assigned individually.
    """
    # The LP fixes offsets only up to one global translation, since it pins
    # component 0 arbitrarily.  Unlike the torus that freedom is NOT free here:
    # a translation moves tiles off the board and forces them to be re-seated
    # elsewhere, which changes the summed seam cost.  Choosing the cheapest
    # translation is therefore a real decision on the objective, not the blind
    # one the toroidal cut faces.  Counting how many tiles merely FIT is not --
    # a compact layout fits at almost any shift, and picking by fit dropped
    # place_acc from 0.129 to 0.001 while leaving adjacency untouched.
    pos = off[owner] + internal
    lo = np.floor(np.percentile(pos, 2, axis=0)).astype(np.int64)
    cand = [(dr, dc) for dr in range(-3, 4) for dc in range(-3, 4)]

    best_lay, best_cost = None, np.inf
    for shift in cand:
        lay = _drop(off - lo + np.array(shift), owner, internal, members)
        if cost_h is None:
            return lay
        b = lay.reshape(G, G)
        c = float(cost_h[b[:, :-1], b[:, 1:]].sum() + cost_v[b[:-1], b[1:]].sum())
        if c < best_cost:
            best_cost, best_lay = c, lay
    return best_lay


def _drop(off, owner, internal, members):
    """Seat every component whole at its offset, largest first."""
    taken = np.zeros(N, bool)
    lay = np.full(N, -1, np.int64)
    order = sorted(range(len(members)), key=lambda c: -len(members[c]))
    leftover = []
    for ci in order:
        tiles = members[ci]
        if not tiles:
            continue
        base = np.rint(off[ci]).astype(np.int64)
        best, best_cells = -1, None
        # try the solved offset and small nudges; keep whichever fits most tiles
        for dr in (0, -1, 1, -2, 2):
            for dc in (0, -1, 1, -2, 2):
                cells, fit = [], 0
                for t in tiles:
                    r = int(base[0] + internal[t][0]) + dr
                    c = int(base[1] + internal[t][1]) + dc
                    if 0 <= r < G and 0 <= c < G and not taken[r * G + c]:
                        cells.append((t, r * G + c))
                        fit += 1
                if fit > best:
                    best, best_cells = fit, cells
                if best == len(tiles):
                    break
            if best == len(tiles):
                break
        placed = set()
        for t, cell in best_cells or []:
            if not taken[cell]:
                taken[cell] = True
                lay[cell] = t
                placed.add(t)
        leftover += [t for t in tiles if t not in placed]
    free = [p for p in range(N) if lay[p] < 0]
    for t, p in zip(leftover, free):
        lay[p] = t
    return lay


def solve_comp_lp(comps, matches, grid=G, rigid=True,
                  cost_h=None, cost_v=None):
    """comps from solve_loop, matches from build_matches -> board[p] = tile.

    Tiles in no component become singletons, so every tile has an offset and the
    Hungarian step always produces a complete board.
    """
    owner = np.full(N, -1, np.int64)
    internal = np.zeros((N, 2), np.float64)
    members = []
    for ci, comp in enumerate(comps):
        members.append(list(comp.pos.keys()))
        for tile, (r, c) in comp.pos.items():
            owner[tile] = ci
            internal[tile] = (r, c)
    n_comp = len(comps)
    for tile in range(N):
        if owner[tile] < 0:                    # a singleton is its own component
            owner[tile] = n_comp
            members.append([tile])
            n_comp += 1
    # components carry their own origin, so shift each to start at (0, 0)
    for ci, ts in enumerate(members):
        if len(ts) > 1:
            base = internal[ts].min(0)
            internal[ts] -= base

    cons = []
    for i, j, dr, dc, w in matches:
        ci, cj = int(owner[i]), int(owner[j])
        if ci == cj:
            continue                            # already fixed inside the component
        d = internal[i] + np.array((dr, dc)) - internal[j]
        cons.append((ci, cj, float(d[0]), float(d[1]), float(w)))

    off = _sync(n_comp, cons)
    if off is None:
        return None
    if rigid:
        return _place_rigid(off, owner, internal, members, cost_h, cost_v)
    return _snap(off[owner] + internal)
