"""Learned missing-neighbour/frame likelihood from a bag's seam matrices."""
from __future__ import annotations

import numpy as np


def directional_matrices(right, down):
    """Scores indexed as tile -> candidate in left/right/up/down directions."""
    return (np.asarray(right).T, np.asarray(right),
            np.asarray(down).T, np.asarray(down))


def frame_features(right, down, tile_stats=None, topk=16):
    """Return (4*N,F) features, direction-major, for boundary classification."""
    mats = directional_matrices(right, down)
    n = len(right)
    all_best = []
    sorted_rows = []
    for matrix in mats:
        x = np.asarray(matrix, np.float64).copy()
        np.fill_diagonal(x, -np.inf)
        order = np.sort(x, axis=1)[:, ::-1]
        sorted_rows.append(order)
        all_best.append(order[:, 0])
    all_best = np.stack(all_best, 1)
    out = []
    for direction, order in enumerate(sorted_rows):
        k = min(topk, n - 1)
        top = order[:, :k]
        if k < topk:
            top = np.pad(top, ((0, 0), (0, topk - k)), mode="edge")
        # Absolute calibrated levels plus within-row shape.  The row shape is
        # invariant to a global logit offset; absolute maxima retain the useful
        # Sinkhorn evidence that a row has no convincing partner at all.
        shape = top - top[:, :1]
        rank = np.argsort(np.argsort(all_best[:, direction])).astype(np.float64)
        rank = ((rank + 0.5) / n * 2.0 - 1.0)[:, None]
        onehot = np.zeros((n, 4), np.float64); onehot[:, direction] = 1.0
        parts = [top, shape, all_best, rank, onehot]
        if tile_stats is not None:
            parts.append(np.asarray(tile_stats, np.float64))
        out.append(np.concatenate(parts, 1))
    return np.concatenate(out, 0).astype(np.float32)


def frame_labels(side):
    y, x = np.divmod(np.arange(side * side), side)
    # Must match directional_matrices: left, right, up, down.
    return np.concatenate([x == 0, x == side - 1, y == 0, y == side - 1]).astype(
        np.uint8)


def frame_unary(probability, side, eps=1e-4):
    """Bernoulli frame log-likelihood for every (cell,tile) assignment."""
    n = side * side
    p = np.clip(np.asarray(probability).reshape(4, n), eps, 1 - eps)
    out = np.zeros((n, n), np.float64)
    for cell in range(n):
        y, x = divmod(cell, side)
        border = (x == 0, x == side - 1, y == 0, y == side - 1)
        for d, yes in enumerate(border):
            out[cell] += np.log(p[d] if yes else 1.0 - p[d])
    return out
