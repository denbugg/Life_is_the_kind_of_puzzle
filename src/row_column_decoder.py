"""Discrete row/column decomposition for a fixed-orientation square puzzle.

The decoder never renders or alters pixels.  It converts directed horizontal
and vertical compatibility matrices into one bijection of the input tiles.
"""
from __future__ import annotations

import numpy as np


def _rank_scores(score: np.ndarray) -> np.ndarray:
    """Row-wise percentile scores, robust to matcher temperature."""
    score = np.asarray(score, np.float64)
    n = len(score)
    order = np.argsort(np.argsort(score, axis=1), axis=1)
    return order / max(1, n - 1)


def constrained_paths(score: np.ndarray, count: int, max_length: int,
                      mutual_weight: float = 0.15) -> list[list[int]]:
    """Greedy maximum-weight directed path cover with exact path count.

    A merge is legal only from the tail of one path to the head of another.
    Cycles, duplicate predecessors/successors and overlong paths are therefore
    impossible by construction.
    """
    raw = np.asarray(score, np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("score must be square")
    n = len(raw)
    if count <= 0 or max_length <= 0 or count * max_length < n:
        raise ValueError("infeasible path-cover dimensions")

    rank = _rank_scores(raw)
    # Reverse support rewards an edge which is also a strong predecessor
    # claim, without requiring brittle exact mutual-best agreement.
    fused = rank + float(mutual_weight) * _rank_scores(raw.T).T
    np.fill_diagonal(fused, -np.inf)

    paths: dict[int, list[int]] = {i: [i] for i in range(n)}
    owner = np.arange(n, dtype=np.int64)
    head = np.arange(n, dtype=np.int64)
    tail = np.arange(n, dtype=np.int64)
    active = set(range(n))

    flat = np.argsort(fused, axis=None)[::-1]
    for index in flat:
        if len(active) <= count:
            break
        a, b = divmod(int(index), n)
        ca, cb = int(owner[a]), int(owner[b])
        if ca == cb or ca not in active or cb not in active:
            continue
        if tail[ca] != a or head[cb] != b:
            continue
        if len(paths[ca]) + len(paths[cb]) > max_length:
            continue
        moved = paths.pop(cb)
        paths[ca].extend(moved)
        for tile in moved:
            owner[tile] = ca
        tail[ca] = tail[cb]
        active.remove(cb)

    # The complete graph normally reaches the requested cover.  Finish any
    # rare capacity dead-end deterministically using the best legal endpoint.
    while len(active) > count:
        best = None
        for ca in sorted(active):
            for cb in sorted(active):
                if ca == cb or len(paths[ca]) + len(paths[cb]) > max_length:
                    continue
                value = fused[tail[ca], head[cb]]
                key = (float(value), -ca, -cb)
                if best is None or key > best[0]:
                    best = (key, ca, cb)
        if best is None:
            raise RuntimeError("could not complete constrained path cover")
        _, ca, cb = best
        moved = paths.pop(cb)
        paths[ca].extend(moved)
        for tile in moved:
            owner[tile] = ca
        tail[ca] = tail[cb]
        active.remove(cb)

    result = [paths[c] for c in sorted(active)]
    if sorted(x for p in result for x in p) != list(range(n)):
        raise AssertionError("path cover is not bijective")
    return result


def balanced_paths(score: np.ndarray, count: int, length: int,
                   mutual_weight: float = 0.15) -> list[list[int]]:
    """Build equally sized paths by balanced endpoint growth.

    Unlike a capacity-only Kruskal pass, balanced growth cannot dead-end with
    too many nearly-full paths.  The globally strongest disjoint edges seed the
    paths, after which the currently shortest paths are extended first.
    """
    raw = np.asarray(score, np.float64)
    n = len(raw)
    if raw.shape != (n, n) or count * length != n:
        raise ValueError("score and requested balanced cover disagree")
    rank = _rank_scores(raw)
    fused = rank + float(mutual_weight) * _rank_scores(raw.T).T
    np.fill_diagonal(fused, -np.inf)

    unused = np.ones(n, bool)
    paths: list[list[int]] = []
    for index in np.argsort(fused, axis=None)[::-1]:
        if len(paths) == count:
            break
        a, b = divmod(int(index), n)
        if unused[a] and unused[b]:
            paths.append([a, b])
            unused[a] = unused[b] = False
    if len(paths) != count:
        raise RuntimeError("not enough disjoint seed edges")

    while unused.any():
        minimum = min(len(p) for p in paths)
        eligible = [i for i, p in enumerate(paths)
                    if len(p) == minimum and len(p) < length]
        tiles = np.flatnonzero(unused)
        best = None
        for i in eligible:
            path = paths[i]
            append_values = fused[path[-1], tiles]
            j = int(np.argmax(append_values))
            key = (float(append_values[j]), -i, -int(tiles[j]), 1)
            if best is None or key > best[0]:
                best = (key, i, int(tiles[j]), True)
            prepend_values = fused[tiles, path[0]]
            j = int(np.argmax(prepend_values))
            key = (float(prepend_values[j]), -i, -int(tiles[j]), 0)
            if best is None or key > best[0]:
                best = (key, i, int(tiles[j]), False)
        if best is None:
            raise RuntimeError("balanced path growth stalled")
        _, i, tile, append = best
        if append:
            paths[i].append(tile)
        else:
            paths[i].insert(0, tile)
        unused[tile] = False
    return paths


def _chain_score(chains: list[list[int]], cross: np.ndarray,
                 trim: float = 0.2) -> np.ndarray:
    """Compatibility of placing one complete chain after another."""
    m = len(chains)
    out = np.full((m, m), -np.inf, np.float64)
    for a in range(m):
        aa = np.asarray(chains[a], np.int64)
        for b in range(m):
            if a == b:
                continue
            bb = np.asarray(chains[b], np.int64)
            values = np.asarray(cross)[aa, bb]
            k = int(np.floor(len(values) * trim))
            if k and len(values) > 2 * k:
                values = np.sort(values)[k:-k]
            out[a, b] = float(values.mean())
    return out


def _one_chain(score: np.ndarray) -> list[int]:
    paths = constrained_paths(score, count=1, max_length=len(score),
                              mutual_weight=0.1)
    return paths[0]


def solve_rows_then_columns(right: np.ndarray, down: np.ndarray, side: int,
                            first: str = "rows", trim: float = 0.2,
                            mutual_weight: float = 0.15) -> np.ndarray:
    """Return a cell->tile bijection through two constrained 1-D problems."""
    n = side * side
    if np.shape(right) != (n, n) or np.shape(down) != (n, n):
        raise ValueError("compatibility matrices disagree with side")
    if first not in ("rows", "columns"):
        raise ValueError("first must be 'rows' or 'columns'")

    primary, cross = (right, down) if first == "rows" else (down, right)
    try:
        chains = constrained_paths(primary, count=side, max_length=side,
                                  mutual_weight=mutual_weight)
    except RuntimeError:
        chains = balanced_paths(primary, count=side, length=side,
                               mutual_weight=mutual_weight)
    if any(len(c) != side for c in chains):
        raise RuntimeError("path cover did not produce equal full chains")
    order = _one_chain(_chain_score(chains, cross, trim))
    grid = np.asarray([chains[i] for i in order], np.int64)
    if first == "columns":
        grid = grid.T
    layout = grid.reshape(-1)
    if len(np.unique(layout)) != n:
        raise AssertionError("decoder output is not bijective")
    return layout
