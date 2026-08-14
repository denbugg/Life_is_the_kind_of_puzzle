"""P9 directed sparse 2×2-loop reweighting over frozen rank96 score lists.

Unlike the preliminary shared-candidate structural module, this implementation
matches the canonical hard-list representation: each `(anchor, direction)` has
its own candidate IDs and frozen scores.  It consumes only these four arrays:
anchors, directions, members, and baseline scores.  Labels, permutations, tiles,
images, targets, model checkpoints, and calibration data are deliberately absent
from its interface.

Direction convention: 0=right, 1=down, 2=left, 3=up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3


@dataclass(frozen=True)
class DirectedLoopReport:
    query_count: int
    candidate_k: int
    loop_k: int
    horizontal_edges_queried: int
    vertical_edges_queried: int
    supported_horizontal_edges: int
    supported_vertical_edges: int
    accepted_loops: int
    rejected_duplicate_loops: int
    lambda_value: float


def validate_queries(
    anchors: np.ndarray,
    directions: np.ndarray,
    members: np.ndarray,
    scores: np.ndarray,
    *,
    n_tiles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    a = np.asarray(anchors, dtype=np.int64)
    d = np.asarray(directions, dtype=np.int64)
    m = np.asarray(members, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if a.ndim != 1 or d.shape != a.shape:
        raise ValueError(f"anchors/directions must be [Q], got {a.shape} and {d.shape}")
    if m.ndim != 2 or s.shape != m.shape or m.shape[0] != a.size:
        raise ValueError(f"members/scores must be [Q,K], got {m.shape} and {s.shape}")
    if np.any(a < 0) or np.any(a >= n_tiles) or np.any(m < 0) or np.any(m >= n_tiles):
        raise ValueError("tile index outside [0,n_tiles)")
    if np.any(d < 0) or np.any(d > 3):
        raise ValueError("direction outside [0,3]")
    lookup: dict[tuple[int, int], int] = {}
    for q, (anchor, direction) in enumerate(zip(a.tolist(), d.tolist())):
        key = (int(anchor), int(direction))
        if key in lookup:
            raise ValueError(f"duplicate directional query {key}")
        # Canonical rank96 candidate unions can repeat a tile.  This is not a
        # new edge: later lookup deterministically uses the strongest occurrence.
        lookup[key] = q
    return a, d, m, s, lookup


def _rank_positions(row: np.ndarray, limit: int) -> np.ndarray:
    return np.argsort(-row, kind="mergesort")[:limit]


def _candidate_pos(members: np.ndarray, scores: np.ndarray, query: int, tile: int) -> int | None:
    """Return the strongest frozen occurrence of tile in a possibly repeated list."""
    found = np.flatnonzero(members[query] == tile)
    if found.size == 0:
        return None
    values = scores[query, found]
    # Stable argmax resolves identical scores by earliest canonical list position.
    return int(found[int(np.argmax(values))])


def _normalize(support: np.ndarray, usable: np.ndarray) -> np.ndarray:
    out = np.zeros_like(support, dtype=np.float64)
    x = support[usable]
    if x.size <= 1:
        return out
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = max(1.4826 * mad, 1.0e-6)
    out[usable] = np.clip((x - median) / scale, -5.0, 5.0)
    return out


def reweight_directed_2x2_loops(
    anchors: np.ndarray,
    directions: np.ndarray,
    members: np.ndarray,
    scores: np.ndarray,
    *,
    n_tiles: int,
    loop_k: int = 8,
    lambda_value: float = 0.0,
    sentinel: float = -1.0e9,
) -> tuple[np.ndarray, DirectedLoopReport]:
    """Reweight frozen right/down hard-list scores through completed 2×2 loops.

    A horizontal candidate relation `i --R--> j` receives support only when
    there exists a tile `k` and a distinct tile `l` satisfying all four frozen
    directed relations: `i --R--> j`, `i --D--> k`, `j --D--> l`, and
    `k --R--> l`.  A vertical relation is handled symmetrically.  The loop value
    is the bottleneck (minimum) of the four input scores, maximized over valid
    completions.  Therefore P9 never manufactures an edge absent from the frozen
    score graph.
    """
    a, d, m, s, lookup = validate_queries(anchors, directions, members, scores, n_tiles=n_tiles)
    q_count, candidate_k = m.shape
    if not (1 <= loop_k <= candidate_k):
        raise ValueError(f"loop_k must be in [1,{candidate_k}], got {loop_k}")
    if not np.isfinite(lambda_value):
        raise ValueError("lambda_value must be finite")
    original = s.copy()
    if lambda_value == 0.0:
        return original, DirectedLoopReport(q_count, candidate_k, loop_k, 0, 0, 0, 0, 0, 0, lambda_value)

    support = np.full_like(s, -np.inf, dtype=np.float64)
    h_queried = v_queried = h_supported = v_supported = loops = duplicates = 0

    for i in range(n_tiles):
        q_r = lookup.get((i, RIGHT))
        q_d = lookup.get((i, DOWN))
        if q_r is None or q_d is None:
            continue
        r_pos = _rank_positions(s[q_r], loop_k)
        d_pos = _rank_positions(s[q_d], loop_k)

        for pos_r in r_pos:
            j = int(m[q_r, pos_r])
            s_ij = float(s[q_r, pos_r])
            if s_ij <= sentinel / 2.0:
                continue
            h_queried += 1
            q_jd = lookup.get((j, DOWN))
            if q_jd is None:
                continue
            best = -np.inf
            for pos_d in d_pos:
                k = int(m[q_d, pos_d])
                s_ik = float(s[q_d, pos_d])
                if s_ik <= sentinel / 2.0:
                    continue
                q_kr = lookup.get((k, RIGHT))
                if q_kr is None:
                    continue
                for pos_jd in _rank_positions(s[q_jd], loop_k):
                    l = int(m[q_jd, pos_jd])
                    pos_kr = _candidate_pos(m, s, q_kr, l)
                    if pos_kr is None:
                        continue
                    if len({i, j, k, l}) != 4:
                        duplicates += 1
                        continue
                    values = (s_ij, s_ik, float(s[q_jd, pos_jd]), float(s[q_kr, pos_kr]))
                    if min(values) <= sentinel / 2.0:
                        continue
                    best = max(best, float(min(values)))
                    loops += 1
            if np.isfinite(best):
                support[q_r, pos_r] = best
                h_supported += 1

        for pos_d in d_pos:
            k = int(m[q_d, pos_d])
            s_ik = float(s[q_d, pos_d])
            if s_ik <= sentinel / 2.0:
                continue
            v_queried += 1
            q_kr = lookup.get((k, RIGHT))
            if q_kr is None:
                continue
            best = -np.inf
            for pos_r in r_pos:
                j = int(m[q_r, pos_r])
                s_ij = float(s[q_r, pos_r])
                if s_ij <= sentinel / 2.0:
                    continue
                q_jd = lookup.get((j, DOWN))
                if q_jd is None:
                    continue
                for pos_jd in _rank_positions(s[q_jd], loop_k):
                    l = int(m[q_jd, pos_jd])
                    pos_kr = _candidate_pos(m, s, q_kr, l)
                    if pos_kr is None:
                        continue
                    if len({i, j, k, l}) != 4:
                        duplicates += 1
                        continue
                    values = (s_ik, s_ij, float(s[q_jd, pos_jd]), float(s[q_kr, pos_kr]))
                    if min(values) <= sentinel / 2.0:
                        continue
                    best = max(best, float(min(values)))
                    loops += 1
            if np.isfinite(best):
                support[q_d, pos_d] = best
                v_supported += 1

    out = original.copy()
    for q in range(q_count):
        if int(d[q]) not in (RIGHT, DOWN):
            continue
        usable = np.isfinite(support[q]) & (original[q] > sentinel / 2.0)
        if usable.any():
            z = _normalize(support[q], usable)
            out[q, usable] = original[q, usable] + lambda_value * z[usable]
    report = DirectedLoopReport(
        q_count,
        candidate_k,
        loop_k,
        h_queried,
        v_queried,
        h_supported,
        v_supported,
        loops,
        duplicates,
        lambda_value,
    )
    return out, report


def directed_to_dense_rd(
    anchors: np.ndarray,
    directions: np.ndarray,
    members: np.ndarray,
    scores: np.ndarray,
    *,
    n_tiles: int,
    sentinel: float = -1.0e9,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense canonical right/down matrices using only supplied candidate lists."""
    a, d, m, s, _ = validate_queries(anchors, directions, members, scores, n_tiles=n_tiles)
    r = np.full((n_tiles, n_tiles), sentinel, dtype=np.float64)
    down = np.full((n_tiles, n_tiles), sentinel, dtype=np.float64)
    for q in range(a.size):
        anchor = int(a[q])
        direction = int(d[q])
        if direction == RIGHT:
            r[anchor, m[q]] = s[q]
        elif direction == DOWN:
            down[anchor, m[q]] = s[q]
    np.fill_diagonal(r, sentinel)
    np.fill_diagonal(down, sentinel)
    return r, down
