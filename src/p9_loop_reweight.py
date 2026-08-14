"""P9 sparse 2x2-loop score reweighting for fixed-orientation jigsaw decoding.

This module is intentionally model-free.  It accepts a frozen candidate graph
and directional scores, reweights only edges that already exist in that graph,
and never generates or learns candidate edges.  It has no dataset, target,
restoration, or submission dependency.

Direction convention: 0=right, 1=down, 2=left, 3=up.  P9's canonical decoder
consumes only the right and down score matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

RIGHT, DOWN = 0, 1


@dataclass(frozen=True)
class LoopReweightReport:
    queried_edges: int
    supported_edges: int
    accepted_loops: int
    rejected_duplicate_loops: int
    candidate_k: int
    loop_k: int
    lambda_value: float


def _validate(candidates: np.ndarray, scores: np.ndarray) -> tuple[int, int]:
    candidates = np.asarray(candidates)
    scores = np.asarray(scores)
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be [N,K], got {candidates.shape}")
    n, k = candidates.shape
    if scores.shape != (4, n, k):
        raise ValueError(f"scores must be [4,N,K]={4,n,k}, got {scores.shape}")
    if candidates.dtype.kind not in "iu":
        raise TypeError("candidate IDs must be integer")
    if np.any(candidates < 0) or np.any(candidates >= n):
        raise ValueError("candidate ID outside [0,N)")
    return n, k


def _ranked_indices(scores: np.ndarray, anchor: int, direction: int, limit: int) -> np.ndarray:
    """Return usable candidate-list positions in decreasing frozen score order."""
    row = scores[direction, anchor]
    # Stable sort makes output independent of NumPy implementation tie behavior.
    order = np.argsort(-row, kind="mergesort")
    return order[:limit]


def _index_map(candidates: np.ndarray) -> list[dict[int, int]]:
    """Map candidate tile ID to its unique list position; reject ambiguous lists."""
    maps: list[dict[int, int]] = []
    for a, row in enumerate(candidates):
        if len(np.unique(row)) != row.size:
            raise ValueError(f"duplicate candidate ID in anchor {a}")
        maps.append({int(tile): int(pos) for pos, tile in enumerate(row)})
    return maps


def _safe_norm(values: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Per-query robust normalization preserving absent edges at exactly zero delta."""
    out = np.zeros_like(values, dtype=np.float64)
    x = values[finite]
    if x.size <= 1:
        return out
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(1.4826 * mad, 1.0e-6)
    out[finite] = np.clip((x - med) / scale, -5.0, 5.0)
    return out


def reweight_2x2_loops(
    candidates: np.ndarray,
    scores: np.ndarray,
    *,
    loop_k: int = 8,
    lambda_value: float = 0.0,
    sentinel: float = -1.0e9,
) -> tuple[np.ndarray, LoopReweightReport]:
    """Add sparse 2x2 cycle support to frozen right/down candidate scores.

    For each proposed right edge ``i -> j``, the function looks for
    ``i ->down k``, ``j ->down l``, and ``k ->right l`` edges within the top
    ``loop_k`` frozen candidates.  The support of a completed square is its
    bottleneck score (minimum of its four directed relations), which is then
    maximized over all squares containing the queried edge.  The same symmetric
    computation is applied to down edges.

    Returned scores retain shape ``[4,N,K]``.  Only directions 0 and 1 receive
    deltas.  A candidate absent from the frozen graph remains absent, and a
    zero lambda returns a bit-identical copy of the input score tensor.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n, k = _validate(candidates, scores)
    if not (1 <= loop_k <= k):
        raise ValueError(f"loop_k must be in [1,{k}], got {loop_k}")
    if not np.isfinite(lambda_value):
        raise ValueError("lambda_value must be finite")
    original = scores.copy()
    if lambda_value == 0.0:
        return original, LoopReweightReport(0, 0, 0, 0, k, loop_k, lambda_value)

    idx = _index_map(candidates)
    support = np.full((2, n, k), -np.inf, dtype=np.float64)
    queried = supported = loops = rejected_duplicates = 0

    # A completed loop adds its support to both top/left edges.  When processing
    # an edge as vertical, the same square is revisited but max aggregation makes
    # the operation deterministic and idempotent.
    for i in range(n):
        r_positions = _ranked_indices(scores, i, RIGHT, loop_k)
        d_positions = _ranked_indices(scores, i, DOWN, loop_k)

        # Horizontal edge i --R--> j.
        for pr in r_positions:
            j = int(candidates[i, pr])
            edge_score = scores[RIGHT, i, pr]
            if edge_score <= sentinel / 2.0:
                continue
            queried += 1
            best = -np.inf
            for pd_i in d_positions:
                k_tile = int(candidates[i, pd_i])
                if scores[DOWN, i, pd_i] <= sentinel / 2.0:
                    continue
                for pd_j in _ranked_indices(scores, j, DOWN, loop_k):
                    l = int(candidates[j, pd_j])
                    pos_kr = idx[k_tile].get(l)
                    if pos_kr is None:
                        continue
                    if len({i, j, k_tile, l}) != 4:
                        rejected_duplicates += 1
                        continue
                    vals = (
                        edge_score,
                        scores[DOWN, i, pd_i],
                        scores[DOWN, j, pd_j],
                        scores[RIGHT, k_tile, pos_kr],
                    )
                    if min(vals) <= sentinel / 2.0:
                        continue
                    val = float(min(vals))
                    best = max(best, val)
                    loops += 1
            if np.isfinite(best):
                support[RIGHT, i, pr] = best
                supported += 1

        # Vertical edge i --D--> k.
        for pd in d_positions:
            k_tile = int(candidates[i, pd])
            edge_score = scores[DOWN, i, pd]
            if edge_score <= sentinel / 2.0:
                continue
            queried += 1
            best = -np.inf
            for pr_i in r_positions:
                j = int(candidates[i, pr_i])
                if scores[RIGHT, i, pr_i] <= sentinel / 2.0:
                    continue
                for pd_j in _ranked_indices(scores, j, DOWN, loop_k):
                    l = int(candidates[j, pd_j])
                    pos_kr = idx[k_tile].get(l)
                    if pos_kr is None:
                        continue
                    if len({i, j, k_tile, l}) != 4:
                        rejected_duplicates += 1
                        continue
                    vals = (
                        edge_score,
                        scores[RIGHT, i, pr_i],
                        scores[DOWN, j, pd_j],
                        scores[RIGHT, k_tile, pos_kr],
                    )
                    if min(vals) <= sentinel / 2.0:
                        continue
                    val = float(min(vals))
                    best = max(best, val)
                    loops += 1
            if np.isfinite(best):
                support[DOWN, i, pd] = best
                supported += 1

    out = original.copy()
    for d in (RIGHT, DOWN):
        for i in range(n):
            valid = np.isfinite(support[d, i]) & (original[d, i] > sentinel / 2.0)
            if not valid.any():
                continue
            normalized = _safe_norm(support[d, i], valid)
            out[d, i, valid] = original[d, i, valid] + lambda_value * normalized[valid]
    return out, LoopReweightReport(queried, supported, loops, rejected_duplicates, k, loop_k, lambda_value)


def sparse_to_dense_rd(candidates: np.ndarray, scores: np.ndarray, *, sentinel: float = -1.0e9) -> tuple[np.ndarray, np.ndarray]:
    """Materialize canonical right/down dense score matrices without new edges."""
    candidates = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n, k = _validate(candidates, scores)
    r = np.full((n, n), sentinel, dtype=np.float64)
    d = np.full((n, n), sentinel, dtype=np.float64)
    for anchor in range(n):
        r[anchor, candidates[anchor]] = scores[RIGHT, anchor]
        d[anchor, candidates[anchor]] = scores[DOWN, anchor]
        r[anchor, anchor] = sentinel
        d[anchor, anchor] = sentinel
    return r, d


def assert_valid_loop_report(report: LoopReweightReport) -> None:
    if report.rejected_duplicate_loops < 0 or report.accepted_loops < 0:
        raise AssertionError("invalid loop counters")
    if report.candidate_k < report.loop_k:
        raise AssertionError("loop_k exceeds candidate_k")
