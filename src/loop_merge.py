"""Joining islands on a closed loop, which is the first merge rule that holds.

Two attempts closed this thread before.  M180 merged components on their best
single contact and measured precision 0.145; M233 required longer contacts,
expecting the object-size effect, and watched precision FALL from 0.203 to 0.000
because two islands in their true relative position often touch along a short
edge, so a long-contact rule excludes the truth and keeps the interlocking
mistakes.

Both judged a merge by one number.  This judges it by an agreement.  If two
DIFFERENT tile pairs linking island A to island B imply the SAME relative
offset, a loop through A and B closes; neither link need be trustworthy, because
what is unlikely is two independently-drawn errors agreeing.  A wrong link picks
its offset from hundreds of possibilities, while two right ones agree always.

Measured over twelve boards, precision by filter: any single contact 0.111, two
independent links 0.317, three 0.609.  Adding the two conditions that cost
nothing -- the scorers must be confident, and two islands cannot OVERLAP, which
is impossible whatever any matcher says and which nothing here ever checked --
takes two links to 0.938 on sixteen merges.  Triangles A to B to C were tested
and give nothing: built from untrusted legs they close by accident (0.184), and
built from trusted ones they never close at all.
"""
from __future__ import annotations

from collections import defaultdict

from config import GRID as G

RIGHT, DOWN = (0, 1), (1, 0)


def _cells(comp):
    return {(dy, dx) for dy, dx in comp.values()}


def pair_offsets(pool, clean):
    """Candidate relative offsets between clean components, with their support.

    Returns (a, b, offset, links, votes): the offset places b relative to a,
    `links` counts how many distinct tile pairs imply it -- the loop support --
    and `votes` is the most scorers any of them convinced.
    """
    comp_of = {}
    for ci, c in enumerate(clean):
        for t in c:
            comp_of[int(t)] = ci
    links = defaultdict(lambda: defaultdict(set))
    votes = defaultdict(lambda: defaultdict(int))
    for (i, j, o), n in pool.items():
        a, b = comp_of.get(i, -1), comp_of.get(j, -1)
        if a < 0 or b < 0 or a == b:
            continue
        dy, dx = clean[a][i]
        ey, ex = clean[b][j]
        off = (dy + o[0] - ey, dx + o[1] - ex)
        key, val = ((a, b), off) if a < b else ((b, a), (-off[0], -off[1]))
        links[key][val].add((i, j, o))
        votes[key][val] = max(votes[key][val], n)
    return [(a, b, off, len(ev), votes[(a, b)][off])
            for (a, b), offs in links.items() for off, ev in offs.items()]


def merge(clean, pool, min_links=2, min_votes=12):
    """Fuse islands whose offset closes a loop. Returns a new component list.

    Proposals are taken in order of confidence and each is checked against what
    has already been accepted: a pair already joined must agree with the offset
    it was joined by, and no two tiles may land in the same cell.  Rejecting on
    overlap is what makes this safe to apply greedily -- a wrong merge that
    survives the vote is usually one that would place two islands on top of each
    other, and geometry refuses it for free.
    """
    if len(clean) < 2:
        return list(clean)
    cand = [c for c in pair_offsets(pool, clean)
            if c[3] >= min_links and c[4] >= min_votes]
    cand.sort(key=lambda c: (-c[4], -c[3]))

    root = list(range(len(clean)))
    shift = [(0, 0)] * len(clean)          # position of this island inside its group

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    groups = {i: {i} for i in range(len(clean))}
    occupied = {i: {(dy, dx): t for t, (dy, dx) in clean[i].items()}
                for i in range(len(clean))}

    for a, b, off, _, _ in cand:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # b must land at a's position plus the offset, both read in group frames
        delta = (shift[a][0] + off[0] - shift[b][0],
                 shift[a][1] + off[1] - shift[b][1])
        moved = {(dy + delta[0], dx + delta[1]): t
                 for (dy, dx), t in occupied[rb].items()}
        if set(moved) & set(occupied[ra]):
            continue
        occupied[ra].update(moved)
        for i in groups[rb]:
            shift[i] = (shift[i][0] + delta[0], shift[i][1] + delta[1])
        groups[ra] |= groups[rb]
        del groups[rb], occupied[rb]
        root[rb] = ra

    out = []
    for r, cellmap in occupied.items():
        oy = min(dy for dy, dx in cellmap)
        ox = min(dx for dy, dx in cellmap)
        out.append({int(t): (dy - oy, dx - ox) for (dy, dx), t in cellmap.items()})
    return out


def is_clean(comp):
    """True when every tile of the component sits at its true relative place."""
    return len({(t // G - dy, t % G - dx) for t, (dy, dx) in comp.items()}) == 1


def grow(clean, pool, min_links=1, min_votes=39, rounds=3):
    """Attach loose fragments to islands on the same loop rule.

    Island-to-island merging is right and worth nothing: it fires 2.3 times a
    board against 42 components, and M231 spreads the whole relative-placement
    prize across all of them.  One level down the supply is ten times larger --
    islands hold about 140 of 576 fragments, so 430 sit loose -- and a loose
    fragment is attached by the same test: tiles of the island linking to it,
    agreeing on which cell it goes in.  Measured over twelve boards, 35
    attachments a board at precision 0.632 and 10 at 0.754, both above the 0.496
    the shipping edge harvest runs at.

    Repeated, because an attached fragment becomes an anchor for the next one.
    """
    comps = [dict(c) for c in clean]
    for _ in range(rounds):
        taken = {int(t) for c in comps for t in c}
        cand = defaultdict(lambda: defaultdict(set))
        votes = defaultdict(lambda: defaultdict(int))
        owner = {}
        for ci, c in enumerate(comps):
            for t in c:
                owner[int(t)] = ci
        for (i, j, o), n in pool.items():
            for src, dst, sign in ((i, j, 1), (j, i, -1)):
                a = owner.get(src, -1)
                if a < 0 or dst in taken:
                    continue
                dy, dx = comps[a][src]
                off = (dy + o[0], dx + o[1]) if sign > 0 else (dy - o[0], dx - o[1])
                cand[(a, dst)][off].add((i, j, o))
                votes[(a, dst)][off] = max(votes[(a, dst)][off], n)

        prop = []
        for (a, t), offs in cand.items():
            for off, ev in offs.items():
                if len(ev) >= min_links and votes[(a, t)][off] >= min_votes:
                    prop.append((votes[(a, t)][off], len(ev), a, t, off))
        if not prop:
            break
        prop.sort(key=lambda p: (-p[0], -p[1]))
        used = set()
        cells = [{(dy, dx) for dy, dx in c.values()} for c in comps]
        for _, _, a, t, off in prop:
            if t in used or off in cells[a]:
                continue
            comps[a][int(t)] = off
            cells[a].add(off)
            used.add(t)
    return [{t: (dy - min(y for y, _ in c.values()),
                 dx - min(x for _, x in c.values()))
             for t, (dy, dx) in c.items()} for c in comps]
