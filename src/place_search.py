"""Search over component positions, on an objective that points at the truth.

M241 corrected M232: read on the seams that join two different components, the
learned cost puts the TRUE arrangement two to four times below the packer's on
every board where the comparison means anything.  The objective is honest; the
search is not.  `solve_components_from_scores` places components greedily and
receives R = -CH with CH non-negative, so every contact subtracts and its optimum
is components that never touch -- half the boards come back with zero contacts
where the truth has thirteen to fifty-nine.

This searches instead.  Components keep their internal geometry and only their
position moves; the score of an arrangement is the total agreement over the
seams where two components meet, measured against a BASELINE so that a
better-than-typical contact pays and a poor one is still refused.  Coordinate
descent: repeatedly take one component and put it at its best position given
where all the others currently are, until nothing moves.

The leftover cells are filled afterwards and how they are filled does not matter
(M226): seam, colour and RANDOM agree to within one ten-thousandth, because SSIM
has no tolerance for displacement and a wrong tile is a wrong tile.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import GRID as G

N = G * G


def _cells(comp):
    """(dy, dx) -> tile, normalised so the top-left of the bounding box is (0,0)."""
    ys = [dy for dy, dx in comp.values()]
    xs = [dx for dy, dx in comp.values()]
    oy, ox = min(ys), min(xs)
    return ({(dy - oy, dx - ox): int(t) for t, (dy, dx) in comp.items()},
            max(ys) - oy + 1, max(xs) - ox + 1)


def _contact_score(cells, r0, c0, board, owner, R, D, baseline, prior=None,
                   lam=0.0):
    """Agreement gained by putting these cells at (r0, c0), minus the baseline.

    Only seams against ALREADY PLACED components count; empty cells contribute
    nothing, which is what makes the baseline necessary -- without it the best
    move is always to touch nothing.

    `prior` is an optional (N, G, G) bonus for putting a tile in a cell, weighted
    by `lam`.  It exists because the seam objective is nearly FLAT near its
    maximum (M245): the search already reaches 88% of the value the true
    arrangement attains while placing thirty times fewer tiles correctly, so
    what it lacks is not altitude but a way to tell near-ties apart.  A prior
    read off the border detector is absolute and independent of every seam,
    which is exactly the tie-break a flat objective needs.
    """
    total = 0.0
    if prior is not None and lam:
        for (dy, dx), t in cells.items():
            total += lam * prior[t, r0 + dy, c0 + dx]
    for (dy, dx), t in cells.items():
        r, c = r0 + dy, c0 + dx
        for (er, ec), fwd, M in (((r, c - 1), False, R), ((r, c + 1), True, R),
                                 ((r - 1, c), False, D), ((r + 1, c), True, D)):
            if not (0 <= er < G and 0 <= ec < G):
                continue
            u = board[er, ec]
            if u < 0 or owner[er, ec] == -2:
                continue
            total += (baseline - (M[t, u] if fwd else M[u, t]))
    return total


def search(components, CH, CV, rounds=6, baseline_q=0.05, seed=0, prior=None,
           lam=0.0, fixed=None):
    """Place components by coordinate descent. Returns board (G, G) of tile ids.

    `components` are the builder's dicts; cells left at -1 are for the caller to
    fill. `baseline_q` sets what counts as a contact worth making: the quantile
    of the cost distribution below which a seam pays for itself.
    """
    rng = np.random.default_rng(seed)
    base_h = float(np.quantile(CH, baseline_q))
    base_v = float(np.quantile(CV, baseline_q))
    baseline = 0.5 * (base_h + base_v)

    comps = [c for c in components if len(c) > 1]
    comps.sort(key=len, reverse=True)
    shapes = [_cells(c) for c in comps]
    # `fixed` maps a component's identity -- the smallest tile it holds, which
    # survives the sort -- to a position it is placed at and never lifted from.
    # Staged assembly needs this: once part of the board is committed, the
    # components still loose gain contacts against it, and M245's flat objective
    # is flat only because so little is committed for them to touch.
    ident = [min(int(t) for t in c) for c in comps]
    pinned = {i: fixed[ident[i]] for i in range(len(comps))
              if fixed and ident[i] in fixed}

    board = -np.ones((G, G), np.int64)
    owner = -np.ones((G, G), np.int64)
    pos = [None] * len(comps)

    def put(i, r0, c0):
        cells, _, _ = shapes[i]
        for (dy, dx), t in cells.items():
            board[r0 + dy, c0 + dx] = t
            owner[r0 + dy, c0 + dx] = i
        pos[i] = (r0, c0)

    def lift(i):
        if pos[i] is None:
            return
        cells, _, _ = shapes[i]
        r0, c0 = pos[i]
        for (dy, dx) in cells:
            board[r0 + dy, c0 + dx] = -1
            owner[r0 + dy, c0 + dx] = -1
        pos[i] = None

    def best_position(i):
        cells, h, w = shapes[i]
        best, score = None, -np.inf
        for r0 in range(G - h + 1):
            for c0 in range(G - w + 1):
                ok = True
                for (dy, dx) in cells:
                    if board[r0 + dy, c0 + dx] >= 0:
                        ok = False
                        break
                if not ok:
                    continue
                s = _contact_score(cells, r0, c0, board, owner, CH, CV,
                                   baseline, prior, lam)
                if s > score:
                    best, score = (r0, c0), s
        return best, score

    for i, p in pinned.items():
        cells, h, w = shapes[i]
        if 0 <= p[0] <= G - h and 0 <= p[1] <= G - w and all(
                board[p[0] + dy, p[1] + dx] < 0 for (dy, dx) in cells):
            put(i, *p)
    for i in range(len(comps)):
        if pos[i] is not None:
            continue
        p, _ = best_position(i)
        if p is None:                      # nowhere left: leave it out
            continue
        put(i, *p)

    identity = ident
    for _ in range(rounds):
        moved = False
        for i in rng.permutation(len(comps)):
            i = int(i)
            if pos[i] is None or i in pinned:
                continue
            old = pos[i]
            lift(i)
            p, _ = best_position(i)
            if p is None:
                put(i, *old)
                continue
            put(i, *p)
            if p != old:
                moved = True
        # Coordinate descent alone stalls: a component cannot move to a good
        # place while another sits there, and moving one at a time never clears
        # the way.  Lifting a PAIR and re-placing both escapes exactly that.
        for _ in range(len(comps) if len(comps) > 1 else 0):
            a, b = (int(x) for x in rng.choice(len(comps), 2, replace=False))
            if pos[a] is None or pos[b] is None or a in pinned or b in pinned:
                continue
            oa, ob = pos[a], pos[b]
            lift(a)
            lift(b)
            pa, sa = best_position(a)
            if pa is None:
                put(a, *oa)
                put(b, *ob)
                continue
            put(a, *pa)
            pb, sb = best_position(b)
            if pb is None:
                lift(a)
                put(a, *oa)
                put(b, *ob)
                continue
            put(b, *pb)
            if (pa, pb) != (oa, ob):
                moved = True
        if not moved:
            break
    placement = {identity[i]: pos[i] for i in range(len(comps))
                 if pos[i] is not None}
    return board, owner, placement


ANNEAL_STATS = {}


def anneal(components, CH, CV, iters=40000, baseline_q=0.05, t0=6.0, t1=0.05,
           seed=0, init=None, prior=None, lam=0.0, jump=1.0, step=3.0,
           swap=0.0):
    """Simulated annealing over component positions.

    Coordinate descent reaches only half to four fifths of the objective value
    the true arrangement attains (M243), so the limit there is the search and
    not the score.  Annealing accepts a worse move with probability exp(-d/T)
    and cools geometrically, which is the standard way out of the basin a
    descent settles into.

    The move is deliberately plain -- lift one component, drop it somewhere it
    fits -- because the pair move that seemed obviously needed did not help
    (M244), and a plain move under a temperature explores the same escapes
    without the cost of evaluating two placements at once.
    """
    rng = np.random.default_rng(seed)
    baseline = 0.5 * (float(np.quantile(CH, baseline_q))
                      + float(np.quantile(CV, baseline_q)))
    comps = [c for c in components if len(c) > 1]
    if not comps:
        return -np.ones((G, G), np.int64), -np.ones((G, G), np.int64)
    shapes = [_cells(c) for c in comps]
    order = sorted(range(len(comps)), key=lambda i: -len(shapes[i][0]))

    board = -np.ones((G, G), np.int64)
    owner = -np.ones((G, G), np.int64)
    pos = [None] * len(comps)

    def put(i, r0, c0):
        cells = shapes[i][0]
        for (dy, dx), t in cells.items():
            board[r0 + dy, c0 + dx] = t
            owner[r0 + dy, c0 + dx] = i
        pos[i] = (r0, c0)

    def lift(i):
        cells = shapes[i][0]
        r0, c0 = pos[i]
        for (dy, dx) in cells:
            board[r0 + dy, c0 + dx] = -1
            owner[r0 + dy, c0 + dx] = -1
        pos[i] = None

    def fits(i, r0, c0):
        cells, h, w = shapes[i]
        if r0 < 0 or c0 < 0 or r0 + h > G or c0 + w > G:
            return False
        return all(board[r0 + dy, c0 + dx] < 0 for (dy, dx) in cells)

    def gain(i, r0, c0):
        return _contact_score(shapes[i][0], r0, c0, board, owner, CH, CV,
                              baseline, prior, lam)

    if init is not None:
        for i, p in enumerate(init):
            if p is not None and fits(i, *p):
                put(i, *p)
    for i in order:
        if pos[i] is not None:
            continue
        best, sc = None, -np.inf
        cells, h, w = shapes[i]
        for r0 in range(G - h + 1):
            for c0 in range(G - w + 1):
                if not fits(i, r0, c0):
                    continue
                s = gain(i, r0, c0)
                if s > sc:
                    best, sc = (r0, c0), s
        if best is not None:
            put(i, *best)

    total = sum(gain(i, *pos[i]) for i in range(len(comps))
                if pos[i] is not None) / 2.0
    best_total, best_pos = total, list(pos)
    ratio = (t1 / t0) ** (1.0 / max(iters, 1))
    T = t0
    ANNEAL_STATS.update(proposed=0, unfit=0, uphill=0, accepted=0, improved=0)
    for _ in range(iters):
        ANNEAL_STATS["proposed"] += 1
        T *= ratio
        if swap > 0 and len(comps) > 1 and rng.random() < swap:
            # M361: single-component relocation cannot escape the greedy
            # initialisation. Of the thousands of proposals that FIT, not one
            # was uphill -- greedy already put each component at its best spot
            # given the others, so moving one alone only breaks its own
            # contacts. A swap changes two at once, which is the smallest move
            # that can trade one component's contacts for another's.
            i = int(rng.integers(len(comps)))
            j = int(rng.integers(len(comps)))
            if i == j or pos[i] is None or pos[j] is None:
                continue
            pi, pj = pos[i], pos[j]
            before = gain(i, *pi) + gain(j, *pj)
            lift(i)
            lift(j)
            if not (fits(i, *pj) and fits(j, *pi)):
                ANNEAL_STATS["unfit"] += 1
                put(i, *pi)
                put(j, *pj)
                continue
            put(i, *pj)
            put(j, *pi)
            d = gain(i, *pj) + gain(j, *pi) - before
            if d >= 0:
                ANNEAL_STATS["uphill"] += 1
            if d >= 0 or rng.random() < np.exp(d / max(T, 1e-6)):
                ANNEAL_STATS["accepted"] += 1
                total += d
                if total > best_total:
                    ANNEAL_STATS["improved"] += 1
                    best_total, best_pos = total, list(pos)
            else:
                lift(i)
                lift(j)
                put(i, *pi)
                put(j, *pj)
            continue
        i = int(rng.integers(len(comps)))
        if pos[i] is None:
            continue
        old = pos[i]
        before = gain(i, *old)
        lift(i)
        cells, h, w = shapes[i]
        if rng.random() < jump:
            r0 = int(rng.integers(G - h + 1))
            c0 = int(rng.integers(G - w + 1))
        else:
            # a LOCAL displacement. M360 measured why this matters: after the
            # greedy initialisation the board is 43 to 72 per cent occupied, so
            # a uniform relocation must find a free rectangle the size of the
            # component's bounding box and almost never does -- across 32 boards
            # and five seeds the annealer never once improved on its own
            # initialisation, making 20000 iterations a no-op.
            r0 = int(np.clip(old[0] + rng.normal(0, step), 0, G - h))
            c0 = int(np.clip(old[1] + rng.normal(0, step), 0, G - w))
        if not fits(i, r0, c0):
            ANNEAL_STATS["unfit"] += 1
            put(i, *old)
            continue
        after = gain(i, r0, c0)
        d = after - before
        if d >= 0:
            ANNEAL_STATS["uphill"] += 1
        if d >= 0 or rng.random() < np.exp(d / max(T, 1e-6)):
            ANNEAL_STATS["accepted"] += 1
            put(i, r0, c0)
            total += d
            if total > best_total:
                ANNEAL_STATS["improved"] += 1
                best_total, best_pos = total, list(pos)
        else:
            put(i, *old)

    board[:] = -1
    owner[:] = -1
    pos = [None] * len(comps)
    for i, p in enumerate(best_pos):
        if p is not None and fits(i, *p):
            put(i, *p)
    return board, owner


def fill_seams(board, CH, CV, seed=0, contrast=None, dead_q=0.0,
               rounds=1):
    """Assign the leftover fragments to the leftover cells by seam evidence.

    M226 called the fill rule irrelevant and it is -- for SSIM, where a wrong
    fragment is a wrong fragment and seam, colour and random agree to one
    ten-thousandth.  It is not irrelevant for PLACEMENT, which is what the
    organisers check: the components cover a quarter of the board, so this
    decides 330 of the 576 cells, and a random permutation gets essentially
    none of them right.

    The candidate order is shuffled before solving.  A cell with no placed
    neighbour has an all-zero cost row and `linear_sum_assignment` breaks that
    tie by index, which on a board indexed by true position hands over the
    answer (M264) -- and, on a real board, silently prefers whatever the tile
    numbering happens to be.
    """
    rng = np.random.default_rng(seed)
    lay = board.reshape(-1).copy()
    free = np.nonzero(lay < 0)[0]
    unused = np.setdiff1d(np.arange(N), lay[lay >= 0])
    if not len(free) or not len(unused):
        return fill_rest(board, rng)
    unused = unused[rng.permutation(len(unused))]
    if dead_q > 0 and contrast is not None and len(unused) > 1:
        # The generator leaves a share of the fragments with no information in
        # them -- crushed to black or blown out, visible as solid squares in any
        # render. Their seam scores are noise, so in one joint assignment they
        # take cells on meaningless evidence, and M69 measured that this is the
        # expensive way round: misplacing a FLAT fragment costs 2.6x less SSIM
        # than misplacing a textured one. So let the textured fragments choose
        # first, over every free cell, and let the dead ones absorb what is
        # left. The holes stay open for the fragments that can use them.
        live = unused[np.argsort(-contrast[unused])[:max(
            int(round((1 - dead_q) * len(unused))), 1)]]
        lay = _seam_assign(lay, free, live, CH, CV)
        free = np.nonzero(lay < 0)[0]
        unused = np.setdiff1d(np.arange(N), lay[lay >= 0])
        if not len(free) or not len(unused):
            return lay if not len(free) else fill_rest(lay.reshape(G, G), rng)
    lay = _seam_assign(lay, free, unused, CH, CV)
    # A SECOND pass has evidence the first could not: the components cover
    # about two hundred fragments, so most free cells start with no placed
    # neighbour at all and their cost row is flat -- M264 caught a tie in
    # exactly that situation handing over the answer. After one pass every
    # cell has four neighbours, so the assignment can be re-solved against
    # them. The placed components are never disturbed; only the leftovers
    # are re-assigned among themselves.
    for _ in range(max(rounds - 1, 0)):
        # the context is the PREVIOUS fill, not an empty board. Blanking every
        # free cell again restored the original situation exactly and the
        # second pass returned the first pass's answer to four decimals.
        lay = _seam_assign(lay, free, unused, CH, CV, context=lay)
    return lay


def _seam_assign(lay, free, unused, CH, CV, context=None):
    """One Hungarian solve of `unused` fragments into `free` cells by seams.

    `context` supplies the neighbours to score against. On the first pass it is
    the board with the free cells still empty, so a cell with no placed
    neighbour has a flat row; on a later pass it is the previous fill, which
    gives every cell four neighbours to be judged by.
    """
    ctx = lay if context is None else context
    C = np.zeros((len(free), len(unused)))
    for k, c in enumerate(free):
        r, q = divmod(int(c), G)
        for (rr, qq), M, fwd in (((r, q - 1), CH, False), ((r, q + 1), CH, True),
                                 ((r - 1, q), CV, False), ((r + 1, q), CV, True)):
            if not (0 <= rr < G and 0 <= qq < G):
                continue
            v = ctx[rr * G + qq]
            if v >= 0:
                C[k] += M[unused, v] if fwd else M[v, unused]
    a, b = linear_sum_assignment(C)
    lay[free[a]] = unused[b]
    return lay


def fill_rest(board, rng=None):
    """Put the unused tiles into the empty cells; the rule is irrelevant (M226)."""
    rng = rng or np.random.default_rng(0)
    lay = board.reshape(-1).copy()
    free = np.nonzero(lay < 0)[0]
    unused = np.setdiff1d(np.arange(N), lay[lay >= 0])
    lay[free] = rng.permutation(unused)
    return lay


def corroborate(components, pool, CH, CV, boost=4.0, min_votes=3):
    """Discount every seam whose implied component offset another seam confirms.

    The objective sums (baseline - cost) over the seams two components make and
    so cannot see AGREEMENT: an offset proposed by one plausible seam and one
    proposed by three independent seams that concur are scored by cost alone,
    though M248 measured those two cases at precision 0.111 and 0.938.  This
    puts the difference where the search can read it, by lowering the cost of
    every seam that takes part in a corroborated offset.

    `boost` is in units of the cost spread between its fifth and fiftieth
    percentile, so it means the same thing on any board.
    """
    comps = [c for c in components if len(c) > 1]
    own = {}
    for ci, c in enumerate(comps):
        for t in c:
            own[int(t)] = ci
    groups = {}
    for (i, j, o), n in pool.items():
        if n < min_votes:
            continue
        a, b = own.get(i, -1), own.get(j, -1)
        if a < 0 or b < 0 or a == b:
            continue
        dy, dx = comps[a][i]
        ey, ex = comps[b][j]
        off = (dy + o[0] - ey, dx + o[1] - ex)
        key, val = ((a, b), off) if a < b else ((b, a), (-off[0], -off[1]))
        groups.setdefault(key, {}).setdefault(val, []).append((i, j, o))
    scale = boost * float(np.quantile(CH, 0.5) - np.quantile(CH, 0.05))
    H, V = CH.copy(), CV.copy()
    for offs in groups.values():
        for links in offs.values():
            if len(links) < 2:
                continue
            for (i, j, o) in links:
                (H if o == (0, 1) else V)[i, j] -= scale * (len(links) - 1)
    return H, V
