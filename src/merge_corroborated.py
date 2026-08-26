"""Merge two components when independent tile pairs agree on their offset.

`place_search.corroborate` already reads the same signal and uses it differently:
it DISCOUNTS the seams that take part in a corroborated offset so the placement
search can see agreement it would otherwise average away. That leaves the two
components separate, and separateness is what costs -- M321 measured that
placement follows the largest coherent block alone, blocks of 19.6, 27.2 and
37.7 fragments all paying about 0.002 while one of 194 pays 0.40.

So do the other thing with the same evidence and actually join them. M248
measured the rule: an offset between two components that two independent tile
pairs imply is right far more often than one a single pair proposes, 0.938
against 0.111 -- though that number was measured between components already
known to be internally correct, and against the components we really build the
rule scores 0.221 (M378). What survives the correction is the end state rather
than the rule's own precision, and M381 measured it: seeded from the shipping
harvest, corroborated merging lifts the largest coherent block from 33.7 to 42.9
and true adjacencies from 254.2 to 271.5 over 24 boards.

The pool is `voted_pool`'s -- every pair ANY scorer called mutually best, with
how many did -- so the second witness is usually an edge the vote threshold
discarded, which is what M248 found it needed and could not reach.
"""
import numpy as np

from config import GRID as G


def _span_ok(comp):
    ys = [p[0] for p in comp.values()]
    xs = [p[1] for p in comp.values()]
    return max(ys) - min(ys) < G and max(xs) - min(xs) < G


def merge_corroborated(components, pool, support=2, min_votes=1, rounds=60):
    """Join component pairs whose relative offset `support` tile pairs imply.

    One merge per round, the best-supported first, so that every later round
    sees the geometry the earlier ones established. Returns a new component
    list; the input is not modified.
    """
    comps = [dict(c) for c in components]
    if not comps or not pool:
        return comps
    owner = {}
    for ci, c in enumerate(comps):
        for t in c:
            owner[int(t)] = ci

    for _ in range(rounds):
        sup = {}
        for (i, j, o), n in pool.items():
            if n < min_votes:
                continue
            ca, cb = owner.get(int(i), -1), owner.get(int(j), -1)
            if ca < 0 or cb < 0 or ca == cb:
                continue
            yi, xi = comps[ca][int(i)]
            yj, xj = comps[cb][int(j)]
            key = (ca, cb, (yi + o[0] - yj, xi + o[1] - xj))
            v = sup.setdefault(key, [0, 0])
            v[0] += 1
            v[1] += n
        best = sorted(((v[0], v[1], k) for k, v in sup.items()
                       if v[0] >= support), reverse=True)
        joined = False
        for _n, _w, (ca, cb, shift) in best:
            A, B = comps[ca], comps[cb]
            if not A or not B:
                continue
            moved = {f: (p[0] + shift[0], p[1] + shift[1]) for f, p in B.items()}
            if set(moved.values()) & set(A.values()):
                continue
            m = dict(A)
            m.update(moved)
            if not _span_ok(m):
                continue
            comps[ca] = m
            comps[cb] = {}
            for f in moved:
                owner[int(f)] = ca
            joined = True
            break
        if not joined:
            break
    return [c for c in comps if c]


def largest_block(components):
    """Fragments of the biggest component that share one offset from the truth.

    Diagnostic only -- it needs the true index, so it is for validation runs.
    """
    best = 1
    for c in components:
        shifts = {}
        for f, (y, x) in c.items():
            k = (y - int(f) // G, x - int(f) % G)
            shifts[k] = shifts.get(k, 0) + 1
        if shifts:
            best = max(best, max(shifts.values()))
    return best


def merge_by_contact(components, CH, CV, rounds=40, keep=6, min_seam=None):
    """Join the two components whose contact carries the single BEST seam.

    M415 measured the statistic. Ranked by the mean seam -- the rule M379 and
    M381 used, and the one the whole island branch stood on -- growth reaches a
    clean block of 30.9; ranked by the MAXIMUM it reaches 38.9, with more true
    adjacencies, for a one-line change. The minimum is worst of the three at
    27.9, which confirms the reading: a pair of islands belongs together when
    ONE seam across the join is convincing, and the average is dragged down by
    the weak seams every contact contains.

    Offsets are seeded from the best cross-component fragment pairs rather than
    enumerated, for the same reason and at a fraction of the cost.
    """
    H, V = -np.asarray(CH, np.float64), -np.asarray(CV, np.float64)
    np.fill_diagonal(H, -1e9)
    np.fill_diagonal(V, -1e9)
    comps = [dict(c) for c in components if c]

    def offers(A, B):
        cand = []
        for fa, pa in A.items():
            for fb, pb in B.items():
                a, b = int(fa), int(fb)
                for dy, dx, sc in ((0, 1, H[a, b]), (0, -1, H[b, a]),
                                   (1, 0, V[a, b]), (-1, 0, V[b, a])):
                    cand.append((sc, (pa[0] + dy - pb[0], pa[1] + dx - pb[1])))
        cand.sort(key=lambda t: -t[0])
        occ = set(A.values())
        seen, out = set(), []
        for _sc, s in cand:
            if s in seen:
                continue
            seen.add(s)
            moved = {f: (p[0] + s[0], p[1] + s[1]) for f, p in B.items()}
            if occ & set(moved.values()):
                continue
            m = dict(A)
            m.update(moved)
            if not _span_ok(m):
                continue
            best = -np.inf
            occ_map = {p: int(f) for f, p in A.items()}
            for f, (y, x) in moved.items():
                g = occ_map.get((y, x - 1))
                if g is not None:
                    best = max(best, H[g, int(f)])
                g = occ_map.get((y, x + 1))
                if g is not None:
                    best = max(best, H[int(f), g])
                g = occ_map.get((y - 1, x))
                if g is not None:
                    best = max(best, V[g, int(f)])
                g = occ_map.get((y + 1, x))
                if g is not None:
                    best = max(best, V[int(f), g])
            if np.isfinite(best):
                out.append((float(best), s, m))
            if len(out) >= keep:
                break
        return out

    for _ in range(rounds):
        best = None
        for ia in range(len(comps)):
            if not comps[ia]:
                continue
            for ib in range(ia + 1, len(comps)):
                if not comps[ib]:
                    continue
                for sc, _s, m in offers(comps[ia], comps[ib]):
                    if best is None or sc > best[0]:
                        best = (sc, ia, ib, m)
                    break
        if best is None or (min_seam is not None and best[0] < min_seam):
            break
        _sc, ia, ib, m = best
        comps[ia] = m
        comps[ib] = {}
    return [c for c in comps if c]
