"""Join islands that several independent PATHS agree about.

The owner's formulation, implemented end to end.

The idea and what it generalises
--------------------------------
M248 measured corroboration on DIRECT contacts: a relative offset between two
islands that two independent touching pairs imply is right 0.938 of the time
against 0.111 when a single pair proposes it -- and it fired 1.3 times a board,
because it required the islands to touch already.

A path through free fragments implies a relative offset too and needs no
contact, so the same corroboration reaches pairs M248 could not. Measured, the
shape reproduces: with vertex-disjoint paths, the geometric veto and a
confidence filter -- M248's three conditions, which it says are needed together
-- precision runs 0.034 at one path, 0.177 at two and 0.373 at three, measured
on pairs of islands that are both internally correct.

What the measurement also showed
--------------------------------
Volume is the wall, as it was for M248. At three paths there are about two
offers a board. What the global reconciliation adds is real and was reported as
a null first: an assignment that accepts a hypothesis only when it agrees with
everything already placed runs 0.383 at the two-path bar against the pairwise
0.177, and 0.778 at three. It cannot GROW, though -- 5.7 correct fragments a
board at two paths and 5.9 at three -- so the extra volume a loose bar admits is
entirely wrong, and against M386's 235 correctly placed fragments this is noise.

The reason it cannot grow is structural: thirty hypotheses over thirty islands
form nearly a FOREST, and "A to B plus B to C must equal A to C" only bites on
cycles. That is M318's diagnosis in a new setting -- consistency can only reject
an edge whose two ends already stand in a known relation.
"""
from collections import defaultdict

import numpy as np

from config import GRID as G

N = G * G


def _neighbours(H, V, k, min_margin):
    """Top-k candidates per direction, kept only where the choice was clear."""
    nb = defaultdict(list)
    for M, (dy, dx) in ((H, (0, 1)), (V, (1, 0))):
        top = np.argsort(-M, axis=1)[:, :k]
        part = np.partition(M, -2, axis=1)
        lead = part[:, -1] - part[:, -2]
        for i in range(N):
            if lead[i] < min_margin:
                continue
            for j in top[i]:
                nb[i].append((int(j), dy, dx))
                nb[int(j)].append((i, -dy, -dx))
    return nb


def _hypotheses(nb, isl, owner, pos, free, maxlen):
    """(ia, ib, shift) -> how many VERTEX-DISJOINT paths support it.

    Two paths through the same free fragment are one witness, not two, which is
    what separates this from the first version: without the disjointness test
    precision sat at 0.002 however many "witnesses" agreed.
    """
    pair = defaultdict(lambda: defaultdict(list))
    for ia, A in enumerate(isl):
        front = {}
        for f, (y, x) in A.items():
            for g, dy, dx in nb[int(f)]:
                if g in free:
                    front[(g, (y + dy, x + dx))] = frozenset({g})
        seen = set(front)
        for _ in range(maxlen):
            nxt = {}
            for (g, gp), used in front.items():
                for h, dy, dx in nb[g]:
                    hp = (gp[0] + dy, gp[1] + dx)
                    ib = owner.get(h)
                    if ib is not None and ib != ia:
                        pair[(ia, ib)][(hp[0] - pos[h][0],
                                        hp[1] - pos[h][1])].append(used)
                    elif h in free and h not in used and (h, hp) not in seen:
                        nxt[(h, hp)] = used | {h}
                        seen.add((h, hp))
            front = nxt
            if not front:
                break
    out = {}
    for (ia, ib), offs in pair.items():
        for shift, wits in offs.items():
            chosen, taken = 0, set()
            for w in sorted(wits, key=len):
                if not (w & taken):
                    chosen += 1
                    taken |= w
            if chosen:
                out[(ia, ib, shift)] = chosen
    return out


def _grow(isl, order, anchor):
    """Place islands one at a time, accepting only what agrees with the rest.

    Returns the placement and the SUPPORT it rests on -- the paths behind the
    hypotheses it actually used. Size alone is the wrong thing to maximise over
    anchors: at three paths there are about four hypotheses a board and nearly
    every anchor reaches the same count, so ranking by size picks an arbitrary
    one of them and throws away the corroboration that is the whole signal.
    """
    place = {anchor: (0, 0)}
    cells = {p: anchor for p in isl[anchor].values()}
    support = 0
    changed = True
    while changed:
        changed = False
        for (ia, ib, shift), _n in order:
            if (ia in place) == (ib in place):
                continue
            if ia in place:
                k, at = ib, (place[ia][0] - shift[0], place[ia][1] - shift[1])
            else:
                k, at = ia, (place[ib][0] + shift[0], place[ib][1] + shift[1])
            put = [(p[0] + at[0], p[1] + at[1]) for p in isl[k].values()]
            if any(q in cells for q in put):
                continue
            ys = [q[0] for q in list(cells) + put]
            xs = [q[1] for q in list(cells) + put]
            if max(ys) - min(ys) >= G or max(xs) - min(xs) >= G:
                continue
            place[k] = at
            support += _n
            for q in put:
                cells[q] = k
            changed = True
    return place, support


def merge_by_paths(components, CH, CV, min_paths=2, k=2, maxlen=2,
                   min_margin=0.5):
    """Islands joined where several independent paths agree on the offset.

    Returns a new component list: the islands the assignment placed become one
    component, and everything it could not place is returned untouched.
    """
    H, V = -np.asarray(CH, np.float64), -np.asarray(CV, np.float64)
    np.fill_diagonal(H, -1e9)
    np.fill_diagonal(V, -1e9)
    isl = [dict(c) for c in components if len(c) >= 2]
    if len(isl) < 2:
        return [dict(c) for c in components if c]
    owner = {int(f): i for i, c in enumerate(isl) for f in c}
    pos = {int(f): p for c in isl for f, p in c.items()}
    free = set(range(N)) - set(owner)

    nb = _neighbours(H, V, k, min_margin)
    hyp = _hypotheses(nb, isl, owner, pos, free, maxlen)
    hyp = {kk: v for kk, v in hyp.items() if v >= min_paths}
    if not hyp:
        return [dict(c) for c in components if c]

    order = sorted(hyp.items(), key=lambda kv: -kv[1])
    best, rank = None, None
    for anchor in range(len(isl)):
        place, support = _grow(isl, order, anchor)
        here = (len(place), support)
        if rank is None or here > rank:
            best, rank = place, here
    if best is None or len(best) < 2:
        return [dict(c) for c in components if c]

    joined = {}
    for kk, at in best.items():
        for f, p in isl[kk].items():
            joined[f] = (p[0] + at[0], p[1] + at[1])
    rest = [dict(c) for i, c in enumerate(isl) if i not in best]
    singles = [dict(c) for c in components if len(c) < 2 and c]
    return [joined] + rest + singles
