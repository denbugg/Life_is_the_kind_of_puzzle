"""Loop-verified component solver for the 24x24 fixed-orientation board.

Why not solve_buddies
---------------------
solve_buddies is not robust to false edges.  Measured on PERFECTLY CLEAN tiles
(mutual-edge precision 0.883) it produced 16 components sized 226/128/124/...
of which only 8 were geometrically pure, holding 36 of 576 tiles; end-to-end
placement was 0.19 even allowing the best cyclic shift.  On oracle scores it is
exact, so the failure is entirely its tolerance to a ~10% false-edge rate: one
bad merge drags a whole component out of alignment.

Two fixes, both measured:

1. Loop verification (Son et al., loop constraints).  Keeping only mutual edges
   that close a 2x2 cycle lifts precision 0.883 -> 0.985 on clean tiles,
   0.801 -> 0.942 blurred, 0.633 -> 0.852 at noise 8, retaining 7-29% of edges.
   Components are grown from those first, so the skeleton is trustworthy.

2. Explicit geometry.  Merges are rejected on tile collision or on a bounding
   box exceeding 24x24, and edges that contradict an existing relative offset
   are dropped rather than averaged.

Torus origin
------------
Relative adjacency fixes the board only up to a cyclic shift: on oracle scores
the old solver reproduced the arrangement exactly but rolled by (4,0), scoring
place_acc 0.0000.  Any component that does not span all 24 lines leaves that
freedom, so the origin is chosen afterwards by the maximum-cost border cut.
"""
from __future__ import annotations

import numpy as np

from config import GRID as G, NFRAG as N


def mutual_edges(cost_h: np.ndarray, cost_v: np.ndarray) -> list[tuple]:
    """Mutual-argmin edges as (priority, score, a, b, dr, dc).

    Loop-verified edges get priority 0 and are consumed before the rest, so the
    component skeleton is built from the high-precision subset.
    """
    ch, cv = cost_h.copy(), cost_v.copy()
    np.fill_diagonal(ch, np.inf)
    np.fill_diagonal(cv, np.inf)
    best_r, best_d = ch.argmin(1), cv.argmin(1)
    left_of, up_of = ch.argmin(0), cv.argmin(0)

    def margin(mat, a, b):
        row = mat[a].copy()
        row[b] = np.inf
        return float(row.min() - mat[a, b])

    out = []
    for a in range(N):
        b = int(best_r[a])
        if left_of[b] == a:
            # 2x2 closure: a-R->b, a-D->c, c-R->d, b-D->d
            closed = int(best_r[int(best_d[a])]) == int(best_d[b])
            out.append((0 if closed else 1, -margin(ch, a, b), a, b, 0, 1))
        b = int(best_d[a])
        if up_of[b] == a:
            closed = int(best_d[int(best_r[a])]) == int(best_r[b])
            out.append((0 if closed else 1, -margin(cv, a, b), a, b, 1, 0))
    out.sort()
    return out


class _Comp:
    """A rigid group of tiles with integer relative coordinates."""

    __slots__ = ("pos", "occ")

    def __init__(self, tile: int):
        self.pos: dict[int, tuple[int, int]] = {tile: (0, 0)}
        self.occ: dict[tuple[int, int], int] = {(0, 0): tile}

    def span(self) -> tuple[int, int]:
        rs = [r for r, _ in self.pos.values()]
        cs = [c for _, c in self.pos.values()]
        return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def build_components(edges: list[tuple]) -> list[_Comp]:
    """Kruskal over the edge list with hard collision and 24x24 span checks."""
    owner = list(range(N))
    comps: dict[int, _Comp] = {t: _Comp(t) for t in range(N)}

    def find(t: int) -> int:
        while owner[t] != t:
            owner[t] = owner[owner[t]]
            t = owner[t]
        return t

    for _, _, a, b, dr, dc in edges:
        ra, rb = find(a), find(b)
        ca, cb = comps[ra], comps[rb]
        ar, ac = ca.pos[a]
        br, bc = cb.pos[b]
        # b must sit at a + (dr,dc): shift component B by this delta
        off_r = ar + dr - br
        off_c = ac + dc - bc
        if ra == rb:
            continue                      # cycle: relative geometry already fixed
        moved = {t: (r + off_r, c + off_c) for t, (r, c) in cb.pos.items()}
        if any(p in ca.occ for p in moved.values()):
            continue                      # collision
        merged = dict(ca.pos)
        merged.update(moved)
        rs = [r for r, _ in merged.values()]
        cs = [c for _, c in merged.values()]
        if max(rs) - min(rs) >= G or max(cs) - min(cs) >= G:
            continue                      # cannot fit the board
        ca.pos = merged
        ca.occ = {p: t for t, p in merged.items()}
        owner[rb] = ra
        comps.pop(rb, None)
    seen = {find(t) for t in range(N)}
    return sorted((comps[r] for r in seen), key=lambda c: -len(c.pos))


def _fit_score(comp: _Comp, board: np.ndarray, used: np.ndarray,
               sr: int, sc: int, score_h: np.ndarray, score_v: np.ndarray) -> float | None:
    """Score placing `comp` with its min corner at (sr,sc); None if illegal."""
    total = 0.0
    for tile, (r, c) in comp.pos.items():
        rr, cc = r + sr, c + sc
        if not (0 <= rr < G and 0 <= cc < G) or board[rr * G + cc] >= 0:
            return None
        if used[tile]:
            return None
    for tile, (r, c) in comp.pos.items():
        rr, cc = r + sr, c + sc
        for dr, dc, mat, flip in ((0, 1, score_h, False), (0, -1, score_h, True),
                                  (1, 0, score_v, False), (-1, 0, score_v, True)):
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < G and 0 <= nc < G):
                continue
            other = board[nr * G + nc]
            if other < 0:
                continue
            total += mat[other, tile] if flip else mat[tile, other]
    return total


def place_components(comps: list[_Comp], score_h: np.ndarray, score_v: np.ndarray) -> np.ndarray:
    """Greedily drop components onto the board, largest first, best fit wins."""
    board = np.full(N, -1, np.int64)
    used = np.zeros(N, bool)
    for comp in comps:
        rs = [r for r, _ in comp.pos.values()]
        cs = [c for _, c in comp.pos.values()]
        h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        norm = _Comp.__new__(_Comp)
        norm.pos = {t: (r - min(rs), c - min(cs)) for t, (r, c) in comp.pos.items()}
        norm.occ = {}
        best = (None, -np.inf)
        for sr in range(G - h + 1):
            for sc in range(G - w + 1):
                val = _fit_score(norm, board, used, sr, sc, score_h, score_v)
                if val is not None and val > best[1]:
                    best = ((sr, sc), val)
        if best[0] is None:
            continue
        sr, sc = best[0]
        for tile, (r, c) in norm.pos.items():
            board[(r + sr) * G + (c + sc)] = tile
            used[tile] = True
    # fill any remaining cells with the leftover tiles, best local score first
    free_cells = [p for p in range(N) if board[p] < 0]
    free_tiles = [t for t in range(N) if not used[t]]
    for p in free_cells:
        if not free_tiles:
            break
        r, c = divmod(p, G)
        vals = []
        for t in free_tiles:
            s = 0.0
            for dr, dc, mat, flip in ((0, 1, score_h, False), (0, -1, score_h, True),
                                      (1, 0, score_v, False), (-1, 0, score_v, True)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < G and 0 <= nc < G and board[nr * G + nc] >= 0:
                    o = board[nr * G + nc]
                    s += mat[o, t] if flip else mat[t, o]
            vals.append(s)
        pick = free_tiles[int(np.argmax(vals))]
        board[p] = pick
        free_tiles.remove(pick)
    return board


def prune_conflicts(comp: _Comp, edges: list[tuple], min_ratio: float = 1.0) -> _Comp:
    """Drop tiles whose placement is contradicted by more edges than support it.

    Kruskal currently throws away every edge whose endpoints already share a
    component (`if ra == rb: continue`), but those edges are free validation:
    each one either confirms the established relative offset or contradicts it.
    Measured need is precision >=0.95, yet at precision 0.85 with 900 edges the
    graph still holds 765 correct edges - far more than the 575 a spanning tree
    needs.  The information is there; this pass is what stops a few outliers
    from dragging the component out of alignment.
    """
    support: dict[int, int] = {}
    conflict: dict[int, int] = {}
    for _ in range(4):
        support.clear(); conflict.clear()
        for _, _, a, b, dr, dc in edges:
            if a not in comp.pos or b not in comp.pos:
                continue
            ar, ac = comp.pos[a]
            br, bc = comp.pos[b]
            hit = (br - ar, bc - ac) == (dr, dc)
            for t in (a, b):
                (support if hit else conflict)[t] = (support if hit else conflict).get(t, 0) + 1
        bad = [t for t in comp.pos
               if conflict.get(t, 0) > min_ratio * support.get(t, 0) and conflict.get(t, 0) > 0]
        if not bad:
            break
        for t in bad:
            comp.pos.pop(t, None)
        comp.occ = {p: t for t, p in comp.pos.items()}
    return comp


def _contact(a: _Comp, b: _Comp, off: tuple[int, int],
             score_h: np.ndarray, score_v: np.ndarray) -> tuple[int, float] | None:
    """Contacts and total seam score of placing b at `off` relative to a."""
    orr, occ = off
    moved = {t: (r + orr, c + occ) for t, (r, c) in b.pos.items()}
    if any(p in a.occ for p in moved.values()):
        return None
    rs = [r for r, _ in list(a.pos.values()) + list(moved.values())]
    cs = [c for _, c in list(a.pos.values()) + list(moved.values())]
    if max(rs) - min(rs) >= G or max(cs) - min(cs) >= G:
        return None
    n, total = 0, 0.0
    for tile, (r, c) in moved.items():
        for dr, dc, mat, flip in ((0, 1, score_h, False), (0, -1, score_h, True),
                                  (1, 0, score_v, False), (-1, 0, score_v, True)):
            other = a.occ.get((r + dr, c + dc))
            if other is None:
                continue
            n += 1
            total += mat[other, tile] if flip else mat[tile, other]
    return (n, total) if n else None


def merge_components(comps: list[_Comp], score_h: np.ndarray, score_v: np.ndarray,
                     min_contacts: int = 2) -> list[_Comp]:
    """Grow the largest component by absorbing others along multi-contact joins.

    A single seam is the weakest possible evidence; two already-rigid groups
    meeting along several seams at one consistent offset is far stronger, and
    the border between two ~150-tile groups offers many contacts at once.  This
    is what lets the skeleton pass the ~160-edge ceiling of loop-verified pairs.
    """
    comps = sorted(comps, key=lambda c: -len(c.pos))
    anchor, rest = comps[0], comps[1:]
    changed = True
    while changed and rest:
        changed = False
        best = (None, -np.inf, 0)
        ar = [r for r, _ in anchor.pos.values()]
        ac = [c for _, c in anchor.pos.values()]
        for idx, comp in enumerate(rest):
            br = [r for r, _ in comp.pos.values()]
            bc = [c for _, c in comp.pos.values()]
            for orr in range(min(ar) - max(br) - 1, max(ar) - min(br) + 2):
                for occ in range(min(ac) - max(bc) - 1, max(ac) - min(bc) + 2):
                    got = _contact(anchor, comp, (orr, occ), score_h, score_v)
                    if got is None or got[0] < min_contacts:
                        continue
                    mean = got[1] / got[0]
                    if mean > best[1]:
                        best = (idx, mean, (orr, occ))
        if best[0] is None:
            break
        idx, _, off = best
        comp = rest.pop(idx)
        for t, (r, c) in comp.pos.items():
            anchor.pos[t] = (r + off[0], c + off[1])
        anchor.occ = {p: t for t, p in anchor.pos.items()}
        changed = True
    return [anchor] + rest


def solve(cost_h: np.ndarray, cost_v: np.ndarray, min_contacts: int = 2,
          merge: bool = False) -> tuple[np.ndarray, list[_Comp]]:
    """cost matrices (lower = better) -> board[p] = tile index at position p.

    `merge` is off by default: absorbing components along multi-contact joins
    grew the largest component (196 -> 367 tiles on clean input) but wrecked it,
    dropping purity from 1.00 to 0.13-0.26 at low contact thresholds while
    higher thresholds simply refused to merge.  It also cost 78-158 s per board.
    """
    comps = build_components(mutual_edges(cost_h, cost_v))
    if merge:
        comps = merge_components(comps, -cost_h, -cost_v, min_contacts)
    board = place_components(comps, -cost_h, -cost_v)
    return board, comps
