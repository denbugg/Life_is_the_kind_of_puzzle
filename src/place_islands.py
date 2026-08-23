"""Place a handful of trusted islands JOINTLY, not one at a time.

M154 reduced the whole task to one sentence: we hold about thirteen fragments we
know are internally correct, and we do not know where they go.  Placed at their
true positions they are worth +0.0527 over the flat fill -- roughly 0.434
absolute, above the leaderboard -- and placed by any rule we actually have they
are worth less than pasting nothing at all.

Placing each island on its own throws away the two constraints that make the
problem small:

* islands cannot overlap, so every placement rules out others;
* islands have seams with each other, and M150 measured island-to-island
  matching at precision 0.709 for 2x2 blocks rising to 0.882 for 4x4 -- far
  above the 0.438 that tile-level matching supplies.

Thirteen objects on a 24x24 board is a search that fits in a few seconds, where
576 tiles never did.  The objective is a sum of three terms, each already
measured rather than guessed:

  seam        cost of every island pair that ends up touching, averaged over
              the shared boundary; this is the strong term (M150)
  field       agreement between an island's mean colours and the predicted
              field; weak on its own (M149 places a 2x2 block at 0.077 even
              against an ORACLE field) so it enters with a small weight, as a
              tie-breaker rather than a driver
  overlap     forbidden outright

The field term is kept because it is the only term that carries any absolute
information at all; the seam term decides relative structure and would
otherwise leave the whole arrangement free to translate.
"""
from __future__ import annotations

import numpy as np

GRID = 24


def island_arrays(islands, mu):
    """Precompute per-island cell offsets, tile ids and mean colours."""
    out = []
    for isl in islands:
        cells = sorted(isl.cells.items())
        rc = np.array([c for c, _ in cells], np.int64)
        ids = np.array([t for _, t in cells], np.int64)
        out.append((rc, ids, mu[ids], isl.height, isl.width))
    return out


def _seam_between(rc_a, ids_a, pos_a, rc_b, ids_b, pos_b, cost_h, cost_v):
    """Mean seam cost over cells of a and b that end up adjacent."""
    a = rc_a + pos_a
    b = rc_b + pos_b
    index = {(int(r), int(c)): int(t) for (r, c), t in zip(b, ids_b)}
    total, count = 0.0, 0
    for (r, c), t in zip(a, ids_a):
        u = index.get((int(r), int(c) + 1))
        if u is not None:
            total += cost_h[t, u]
            count += 1
        u = index.get((int(r), int(c) - 1))
        if u is not None:
            total += cost_h[u, t]
            count += 1
        u = index.get((int(r) + 1, int(c)))
        if u is not None:
            total += cost_v[t, u]
            count += 1
        u = index.get((int(r) - 1, int(c)))
        if u is not None:
            total += cost_v[u, t]
            count += 1
    return (total / count, count) if count else (0.0, 0)


def _field_cost(rc, colours, pos, field_grid):
    """Squared error of the island's colours against the field, offset removed.

    The per-tile photometric bias is unknown and uninformative (M129), so only
    the SHAPE of the island's colour map may decide anything.
    """
    cells = rc + pos
    if (cells < 0).any() or (cells[:, 0] >= GRID).any() or (cells[:, 1] >= GRID).any():
        return np.inf
    ref = field_grid[cells[:, 0], cells[:, 1]]
    d = colours - ref
    d = d - d.mean(0, keepdims=True)
    return float((d ** 2).mean())


def seam_baseline(cost_h, cost_v):
    """Cost of an arbitrary, meaningless contact.

    Without this the seam term is a pure penalty -- every contact adds a
    non-negative number -- so the cheapest arrangement is the one where nothing
    touches anything, and M155 measured exactly that: zero islands placed
    correctly under all three weightings.  Subtracting the baseline turns the
    term into evidence: a contact better than chance is rewarded, one worse is
    charged, and an arrangement is judged on whether its seams are GOOD rather
    than on whether it has any.
    """
    return 0.5 * (float(np.median(cost_h)) + float(np.median(cost_v)))


def energy(positions, arrays, cost_h, cost_v, field_grid, w_field, w_seam,
           grid=GRID, baseline=0.0):
    """Total cost of one arrangement; np.inf when anything overlaps."""
    occupied = {}
    total = 0.0
    for k, (rc, ids, colours, h, w) in enumerate(arrays):
        cells = rc + positions[k]
        if (cells < 0).any() or (cells[:, 0] >= grid).any() or (cells[:, 1] >= grid).any():
            return np.inf
        for (r, c) in cells:
            key = (int(r), int(c))
            if key in occupied:
                return np.inf
            occupied[key] = k
        total += w_field * _field_cost(rc, colours, positions[k], field_grid)
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            cost, count = _seam_between(arrays[i][0], arrays[i][1], positions[i],
                                        arrays[j][0], arrays[j][1], positions[j],
                                        cost_h, cost_v)
            if count:
                total += w_seam * (cost - baseline) * count
    return total


def solve(islands, mu, cost_h, cost_v, field_grid, w_field=1.0, w_seam=1.0,
          iters=20000, restarts=8, seed=0, grid=GRID, baseline=None):
    """Simulated annealing over island top-left positions.

    Returns (positions, energy).  Positions are (row, col) of each island's
    own (0, 0) cell.
    """
    arrays = island_arrays(islands, mu)
    if baseline is None:
        baseline = seam_baseline(cost_h, cost_v)
    rng = np.random.default_rng(seed)
    n = len(arrays)
    if n == 0:
        return [], 0.0

    best_pos, best_e = None, np.inf
    for _ in range(restarts):
        pos = []
        for (rc, ids, colours, h, w) in arrays:
            pos.append(np.array([rng.integers(0, grid - h + 1),
                                 rng.integers(0, grid - w + 1)], np.int64))
        e = energy(pos, arrays, cost_h, cost_v, field_grid, w_field, w_seam, grid, baseline)
        tries = 0
        while not np.isfinite(e) and tries < 200:
            for k, (rc, ids, colours, h, w) in enumerate(arrays):
                pos[k] = np.array([rng.integers(0, grid - h + 1),
                                   rng.integers(0, grid - w + 1)], np.int64)
            e = energy(pos, arrays, cost_h, cost_v, field_grid, w_field, w_seam, grid, baseline)
            tries += 1
        if not np.isfinite(e):
            continue
        t0, t1 = 1.0, 0.01
        for step in range(iters):
            temp = t0 * (t1 / t0) ** (step / max(1, iters - 1))
            k = int(rng.integers(0, n))
            h, w = arrays[k][3], arrays[k][4]
            old = pos[k].copy()
            if rng.random() < 0.5:
                pos[k] = np.array([rng.integers(0, grid - h + 1),
                                   rng.integers(0, grid - w + 1)], np.int64)
            else:
                pos[k] = old + rng.integers(-2, 3, 2)
            cand = energy(pos, arrays, cost_h, cost_v, field_grid, w_field,
                          w_seam, grid, baseline)
            if np.isfinite(cand) and (cand < e
                                      or rng.random() < np.exp((e - cand) / temp)):
                e = cand
            else:
                pos[k] = old
        if e < best_e:
            best_e, best_pos = e, [p.copy() for p in pos]
    return best_pos, best_e
