"""Features for LEARNING which candidate edges to trust.

Why a learned confidence and not another threshold
--------------------------------------------------
M166 priced the target exactly.  The island builder is not wasting material --
on synthetic edges it extracts what the count and precision allow -- but the
curve it sits on is steep and narrow: 244 edges at precision 0.84 yield trusted
coverage 0.13, while 400 edges at 0.90 yield 0.285, and 900 at 0.90 collapse to
0.02 because surplus wrong edges glue everything into one bad island.

So the goal is no longer "better matching" in the abstract.  It is 400 edges per
board at precision 0.88 or better, from a pool that currently gives 244 at 0.84.

Every scheme tried so far combined two signals by hand: vote count across models
(M163) and the runner-up cost ratio (M158).  Those are two features.  A cost
matrix carries many more -- where the partner ranks in the row AND in the
column, how far the cost sits from that row's distribution, whether the pair
closes a 2x2 loop, how textured the two tiles are -- and the labels are free,
because on a training board we know which adjacencies are real.

Nothing here looks at the target image or the true layout; the label does, the
features do not.
"""
from __future__ import annotations

import numpy as np

GRID = 24


def _rank_and_margin(C):
    """Per row: the order of candidates, and each row's cost distribution."""
    D = C.copy()
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    rank = np.empty_like(order)
    rows = np.arange(D.shape[0])[:, None]
    rank[rows, order] = np.arange(D.shape[1])[None, :]
    part = np.sort(D, axis=1)
    best = part[:, 0]
    second = part[:, 1]
    med = np.median(D[np.isfinite(D)].reshape(D.shape[0], -1), axis=1)
    return rank, best, second, med


def candidate_pool(costs, topk=4):
    """Union of each model's top-k partners, per axis. Returns {axis: set}."""
    pool = {"h": set(), "v": set()}
    for CH, CV in costs.values():
        for axis, C in (("h", CH), ("v", CV)):
            D = C.copy()
            np.fill_diagonal(D, np.inf)
            idx = np.argsort(D, axis=1)[:, :topk]
            for i in range(C.shape[0]):
                for j in idx[i]:
                    pool[axis].add((i, int(j)))
    return pool


def _quad_partners(CH, CV):
    """Tiles whose four best-neighbour choices close a 2x2 loop.

    Loop closure is the single strongest hand-made signal in this project
    (precision 0.878 against 0.438 for a bare mutual edge), so it enters as a
    feature rather than as a separate pipeline.
    """
    H, V = CH.copy(), CV.copy()
    np.fill_diagonal(H, np.inf)
    np.fill_diagonal(V, np.inf)
    right, below = H.argmin(1), V.argmin(1)
    closed_h, closed_v = set(), set()
    for a in range(CH.shape[0]):
        b, c = int(right[a]), int(below[a])
        if b == c:
            continue
        d = int(right[c])
        if d != int(below[b]) or len({a, b, c, d}) != 4:
            continue
        closed_h.add((a, b))
        closed_h.add((c, d))
        closed_v.add((a, c))
        closed_v.add((b, d))
    return closed_h, closed_v


def board_features(costs, tiles, topk=4):
    """Return (X, pairs, axes) for every candidate edge on one board.

    `costs` maps a model name to (cost_h, cost_v).  Feature order is fixed by
    the sorted model names so the matrix means the same thing on every board.
    """
    names = sorted(costs)
    n = tiles.shape[0]
    prep = {}
    for nm in names:
        CH, CV = costs[nm]
        prep[nm] = {
            "h": _rank_and_margin(CH) + (_rank_and_margin(CH.T)[0],),
            "v": _rank_and_margin(CV) + (_rank_and_margin(CV.T)[0],),
            "C": {"h": CH, "v": CV},
        }
    quads = {nm: dict(zip(("h", "v"), _quad_partners(*costs[nm]))) for nm in names}

    mu = tiles.mean((1, 2))
    sd = tiles.std((1, 2)).mean(1)
    pool = candidate_pool(costs, topk)

    X, pairs, axes = [], [], []
    for axis in ("h", "v"):
        for (i, j) in sorted(pool[axis]):
            f = []
            votes = 0
            for nm in names:
                rank, best, second, med, rank_back = prep[nm][axis]
                C = prep[nm]["C"][axis]
                c = float(C[i, j])
                r = int(rank[i, j])
                rb = int(rank_back[j, i])
                spread = max(med[i] - best[i], 1e-6)
                f += [
                    r, rb,
                    (c - best[i]) / spread,          # how far past this row's best
                    second[i] / max(best[i], 1e-9),  # the row's own margin
                    float(r == 0 and rb == 0),       # mutual top-1
                    float((i, j) in quads[nm][axis]),
                ]
                votes += int(r == 0 and rb == 0)
            f += [votes,
                  float(sd[i]), float(sd[j]),
                  float(np.abs(mu[i] - mu[j]).mean()),
                  float(min(sd[i], sd[j]))]
            X.append(f)
            pairs.append((i, j))
            axes.append(axis)
    return np.asarray(X, np.float32), pairs, axes


def labels_for(pairs, axes, grid=GRID, n=None):
    n = n if n is not None else grid * grid
    out = np.zeros(len(pairs), np.float32)
    for k, ((i, j), ax) in enumerate(zip(pairs, axes)):
        if ax == "h":
            out[k] = float(j == i + 1 and i % grid != grid - 1)
        else:
            out[k] = float(j == i + grid and i < n - grid)
    return out


FEATURE_NOTE = (
    "6 features per model (rank forward, rank backward, cost above the row's "
    "best in units of its spread, the row's runner-up ratio, mutual top-1, "
    "closes a 2x2 loop) plus vote count and four tile statistics"
)
