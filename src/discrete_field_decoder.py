"""Bijective unary+pairwise decoder for a fixed-orientation square puzzle."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def objective(layout, right, down, unary, side, unary_weight):
    n = side * side
    value = 0.0
    for p in range(n):
        value += unary_weight * unary[p, layout[p]]
        if p % side != side - 1:
            value += right[layout[p], layout[p + 1]]
        if p < n - side:
            value += down[layout[p], layout[p + side]]
    return value


@njit(cache=True, fastmath=True)
def _local(layout, p, skip, right, down, unary, side, unary_weight):
    n = side * side
    value = unary_weight * unary[p, layout[p]]
    if p % side != side - 1 and p + 1 != skip:
        value += right[layout[p], layout[p + 1]]
    if p % side != 0 and p - 1 != skip:
        value += right[layout[p - 1], layout[p]]
    if p < n - side and p + side != skip:
        value += down[layout[p], layout[p + side]]
    if p >= side and p - side != skip:
        value += down[layout[p - side], layout[p]]
    return value


@njit(cache=True, fastmath=True)
def _run(layout, right, down, unary, side, unary_weight, iters,
         temperature, seed, shortlist):
    np.random.seed(seed)
    n = side * side
    cur = objective(layout, right, down, unary, side, unary_weight)
    best = cur
    best_layout = layout.copy()
    decay = (0.002 / temperature) ** (1.0 / max(1, iters))
    t = temperature
    k = shortlist.shape[1]
    for _ in range(iters):
        p = np.random.randint(0, n)
        if p % side and np.random.random() < 0.55:
            wanted = shortlist[layout[p - 1], np.random.randint(0, k)]
            q = 0
            while layout[q] != wanted:
                q += 1
        else:
            q = np.random.randint(0, n)
        if p == q:
            t *= decay
            continue
        before = (_local(layout, p, -1, right, down, unary, side, unary_weight)
                  + _local(layout, q, p, right, down, unary, side, unary_weight))
        layout[p], layout[q] = layout[q], layout[p]
        after = (_local(layout, p, -1, right, down, unary, side, unary_weight)
                 + _local(layout, q, p, right, down, unary, side, unary_weight))
        delta = after - before
        if delta >= 0.0 or np.random.random() < np.exp(delta / max(t, 1e-9)):
            cur += delta
            if cur > best:
                best = cur
                best_layout[:] = layout
        else:
            layout[p], layout[q] = layout[q], layout[p]
        t *= decay
    layout[:] = best_layout
    return best


@njit(cache=True, fastmath=True)
def _polish(layout, right, down, unary, side, unary_weight, sweeps):
    n = side * side
    for _ in range(sweeps):
        moved = 0
        for p in range(n):
            for q in range(p + 1, n):
                before = (_local(layout, p, -1, right, down, unary, side, unary_weight)
                          + _local(layout, q, p, right, down, unary, side, unary_weight))
                layout[p], layout[q] = layout[q], layout[p]
                after = (_local(layout, p, -1, right, down, unary, side, unary_weight)
                         + _local(layout, q, p, right, down, unary, side, unary_weight))
                if after > before + 1e-9:
                    moved += 1
                else:
                    layout[p], layout[q] = layout[q], layout[p]
        if moved == 0:
            break
    return objective(layout, right, down, unary, side, unary_weight)


def solve_discrete(right, down, unary, init, side, unary_weight=0.1,
                   iters=100_000, restarts=2, seed=0, sweeps=8):
    """Maximise pairwise compatibility plus absolute unary evidence."""
    right = np.ascontiguousarray(right, np.float64)
    down = np.ascontiguousarray(down, np.float64)
    unary = np.ascontiguousarray(unary, np.float64)
    shortlist = np.argsort(-right, axis=1)[:, :min(8, len(right))].astype(np.int64)
    # A rank-normalised score makes the weight transferable between checkpoints.
    scale = np.median(np.abs(right - np.median(right)))
    temperature = max(float(scale) * 0.35, 1e-3)
    best_layout, best = None, -np.inf
    for r in range(restarts):
        layout = np.ascontiguousarray(init, np.int64).copy()
        if r:
            # Keep the learned anchor but give later restarts a small escape.
            rng = np.random.default_rng(seed + r)
            for _ in range(max(1, len(layout) // 8)):
                p, q = rng.integers(0, len(layout), 2)
                layout[p], layout[q] = layout[q], layout[p]
        _run(layout, right, down, unary, side, unary_weight, int(iters),
             temperature, seed + r, shortlist)
        value = _polish(layout, right, down, unary, side, unary_weight, sweeps)
        if value > best:
            best_layout, best = layout.copy(), float(value)
    return best_layout, best
