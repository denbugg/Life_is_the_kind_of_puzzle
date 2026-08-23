"""Genetic solver after Sholomon, David and Netanyahu (CVPR 2013).

Why this, given everything else here failed
-------------------------------------------
Two measurements point at it.

First, component PURITY.  Greedy assembly produces one component holding 168
tiles whose relative offsets are right for only 4.6% of them (M104): a single
bad merge drags the rest out of alignment and nothing can revisit it.  The
crossover here is built for exactly that -- it grows a child from two parents
and takes a placement only when both parents AGREE on it, falling back to
best-buddy relations and only then to raw compatibility.  Segments that both
parents assembled the same way survive; segments one of them botched do not.

Second, the ANCHOR.  Every solver here works on a torus, so its layout is right
only up to a cyclic shift, and the shift cannot be recovered: rolling a torus
changes total cost only by which 48 seams stop counting, and three separate
attempts to break the tie failed (M108).  A kernel grown inside a bounded 24x24
board has no such freedom.  A cyclically shifted solution would have to join the
image's left column to its right column, and those pieces are not compatible, so
compatibility-driven growth does not produce one in the first place.

Scope, honestly.  The published results are on clean puzzles, up to 30745
pieces.  A 2025 benchmark of corrupted puzzles found that at 30% content erosion
on 144 pieces the heuristic solvers -- Gallagher, Paikin-Tal and Yu, all three of
which this repo implements -- misplace every piece, so nothing guarantees a GA
survives our corruption either.  What it does is use the edges we already have
more carefully, which is the one lever left while edge precision sits at 0.50
against the 0.72 assembly needs.
"""
from __future__ import annotations

import heapq

import numpy as np

from config import GRID as G, NFRAG as N

# right, left, down, up as (dr, dc), with the index of the opposite direction
DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))
OPP = (1, 0, 3, 2)


def relations(lay, grid=G):
    """rel[d][i] = the piece sitting in direction d from piece i, else -1."""
    b = np.asarray(lay).reshape(grid, grid)
    rel = np.full((4, N), -1, np.int64)
    rel[0, b[:, :-1].ravel()] = b[:, 1:].ravel()
    rel[1, b[:, 1:].ravel()] = b[:, :-1].ravel()
    rel[2, b[:-1].ravel()] = b[1:].ravel()
    rel[3, b[1:].ravel()] = b[:-1].ravel()
    return rel


def _cost(cost_h, cost_v, a, b, d):
    """Cost of putting piece b in direction d from piece a."""
    if d == 0:
        return cost_h[a, b]
    if d == 1:
        return cost_h[b, a]
    if d == 2:
        return cost_v[a, b]
    return cost_v[b, a]


def best_buddies(cost_h, cost_v):
    """bb[d][i] = j when i and j are each other's cheapest partner across d."""
    bb = np.full((4, N), -1, np.int64)
    for d, C in ((0, cost_h), (2, cost_v)):
        D = C.copy()
        np.fill_diagonal(D, np.inf)
        fwd, back = D.argmin(1), D.argmin(0)
        for i in range(N):
            j = int(fwd[i])
            if int(back[j]) == i:
                bb[d, i] = j
                bb[OPP[d], j] = i
    return bb


def _shortlists(cost_h, cost_v, k):
    """The k cheapest partners of each piece in each direction."""
    out = np.empty((4, N, k), np.int64)
    for d, C in ((0, cost_h), (2, cost_v)):
        D = C.copy()
        np.fill_diagonal(D, np.inf)
        out[d] = np.argsort(D, axis=1)[:, :k]
        out[OPP[d]] = np.argsort(D, axis=0).T[:, :k]
    return out


class _Kernel:
    """A partial assembly that must stay inside a grid x grid bounding box."""

    __slots__ = ("pos", "at", "r0", "r1", "c0", "c1", "grid")

    def __init__(self, seed, grid=G):
        self.pos = {seed: (0, 0)}
        self.at = {(0, 0): seed}
        self.r0 = self.r1 = self.c0 = self.c1 = 0
        self.grid = grid

    def free(self, cell):
        r, c = cell
        if cell in self.at:
            return False
        return (max(self.r1, r) - min(self.r0, r) < self.grid
                and max(self.c1, c) - min(self.c0, c) < self.grid)

    def add(self, piece, cell):
        r, c = cell
        self.pos[piece] = cell
        self.at[cell] = piece
        self.r0, self.r1 = min(self.r0, r), max(self.r1, r)
        self.c0, self.c1 = min(self.c0, c), max(self.c1, c)

    def board(self):
        """Absolute layout; the bounding box IS the board, so no shift is free."""
        lay = np.full(N, -1, np.int64)
        for piece, (r, c) in self.pos.items():
            lay[(r - self.r0) * self.grid + (c - self.c0)] = piece
        missing = [p for p in range(N) if p not in self.pos]
        holes = [p for p in range(N) if lay[p] < 0]
        for piece, hole in zip(missing, holes):
            lay[hole] = piece
        return lay


def crossover(pa, pb, cost_h, cost_v, bb, short, rng, mutation=0.05, grid=G):
    """Grow a child from two parents: agreed placements, then buddies, then greedy."""
    rel_a, rel_b = relations(pa, grid), relations(pb, grid)
    k = _Kernel(int(rng.integers(N)), grid)
    used = np.zeros(N, bool)
    used[next(iter(k.pos))] = True

    q_agree, q_buddy, q_greedy = [], [], []
    tick = [0]

    def offer(piece):
        r, c = k.pos[piece]
        for d, (dr, dc) in enumerate(DIRS):
            cell = (r + dr, c + dc)
            if not k.free(cell):
                continue
            ja, jb = int(rel_a[d, piece]), int(rel_b[d, piece])
            if ja >= 0 and ja == jb and not used[ja]:
                heapq.heappush(q_agree, (0.0, tick[0], ja, cell))
                tick[0] += 1
                continue
            for j in (ja, jb, int(bb[d, piece])):
                if j >= 0 and not used[j]:
                    heapq.heappush(q_buddy,
                                   (float(_cost(cost_h, cost_v, piece, j, d)),
                                    tick[0], j, cell))
                    tick[0] += 1
            for j in short[d, piece]:
                j = int(j)
                if not used[j]:
                    heapq.heappush(q_greedy,
                                   (float(_cost(cost_h, cost_v, piece, j, d)),
                                    tick[0], j, cell))
                    tick[0] += 1
                    break

    offer(next(iter(k.pos)))
    while len(k.pos) < N:
        got = None
        for q in (q_agree, q_buddy, q_greedy):
            while q:
                _, _, piece, cell = heapq.heappop(q)
                if not used[piece] and k.free(cell):
                    got = (piece, cell)
                    break
            if got:
                break
        if got is None:
            # no queued candidate survives; seat any spare piece next to the kernel
            spare = [p for p in range(N) if not used[p]]
            if not spare:
                break
            cells = [(r + dr, c + dc) for (r, c) in k.at for dr, dc in DIRS]
            cells = [c for c in cells if k.free(c)]
            if not cells:
                break
            got = (spare[0], cells[0])
        piece, cell = got
        if rng.random() < mutation:
            spare = [p for p in range(N) if not used[p]]
            piece = int(spare[int(rng.integers(len(spare)))])
        used[piece] = True
        k.add(piece, cell)
        offer(piece)
    return k.board()


def fitness(lay, cost_h, cost_v, grid=G):
    b = np.asarray(lay).reshape(grid, grid)
    return float(cost_h[b[:, :-1], b[:, 1:]].sum() + cost_v[b[:-1], b[1:]].sum())


def solve_ga(cost_h, cost_v, population=24, generations=20, elite=4, k_short=4,
             mutation=0.05, seed=0, grid=G, init=None):
    """Return the fittest layout found. init seeds the population if supplied."""
    rng = np.random.default_rng(seed)
    bb = best_buddies(cost_h, cost_v)
    short = _shortlists(cost_h, cost_v, k_short)

    pop = []
    for i in range(population):
        if init is not None and i < 2:
            pop.append(np.asarray(init, np.int64).copy())
        else:
            pop.append(rng.permutation(N).astype(np.int64))
    scored = sorted(((fitness(p, cost_h, cost_v, grid), i) for i, p in enumerate(pop)))
    pop = [pop[i] for _, i in scored]

    for _ in range(generations):
        children = pop[:elite]
        while len(children) < population:
            ia, ib = rng.integers(0, max(2, population // 2), 2)
            children.append(crossover(pop[ia], pop[ib], cost_h, cost_v, bb, short,
                                      rng, mutation, grid))
        scored = sorted(((fitness(c, cost_h, cost_v, grid), i)
                         for i, c in enumerate(children)))
        pop = [children[i] for _, i in scored]
    return pop[0], fitness(pop[0], cost_h, cost_v, grid)
