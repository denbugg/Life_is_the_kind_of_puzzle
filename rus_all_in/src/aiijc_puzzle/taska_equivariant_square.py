"""Permutation-equivariant centred 2x2 reranking for TASKA seam scores.

The historical ``quad_rerank`` added a useful second-order signal: a proposed
right/down neighbour is preferred when it participates in many strong 2x2
squares.  Its implementation was not legal on a raw unordered bag, however,
because it skipped rows selected by ``tile_id % 24`` and ``tile_id // 24``.
Those ids happened to be target positions in the historical validation path.

This module keeps the measured historical recipe -- top-16 square supports,
temperature 0.5, one round, a centred top-20 update -- but applies it to every
source row.  Consequently the only role of an integer tile id is to index the
current input bag.  Simultaneously relabelling both axes of both score matrices
simply relabels both output axes in the same way.

For ordinary continuous model logits the implementation is exactly the
historical centred calculation, apart from removal of the boundary masks.  At
an exact top-k boundary tie it includes the complete tie block.  This is the
permutation-equivariant extension of top-k: choosing an arbitrary subset of
indistinguishable tied ids would itself reintroduce an id-dependent rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SQUARE_SUPPORT_K = 16
SQUARE_TEMPERATURE = 0.5
SQUARE_ROUNDS = 1
SQUARE_SHORTLIST = 20
DEFAULT_SQUARE_WEIGHT = 0.4


def _score_pair(right: Any, down: Any) -> tuple[np.ndarray, np.ndarray]:
    right_array = np.asarray(right, dtype=np.float64)
    down_array = np.asarray(down, dtype=np.float64)
    if right_array.ndim != 2 or right_array.shape[0] != right_array.shape[1]:
        raise ValueError("right_log must be one square matrix")
    if down_array.shape != right_array.shape:
        raise ValueError("down_log must have the same square shape as right_log")
    if right_array.shape[0] < SQUARE_SHORTLIST:
        raise ValueError(
            f"score matrices need at least {SQUARE_SHORTLIST} tiles for the fixed recipe"
        )
    if not np.isfinite(right_array).all() or not np.isfinite(down_array).all():
        raise ValueError("right_log and down_log must contain only finite values")
    return (
        np.array(right_array, dtype=np.float64, copy=True, order="C"),
        np.array(down_array, dtype=np.float64, copy=True, order="C"),
    )


def _top_with_boundary_ties(values: np.ndarray, limit: int) -> np.ndarray:
    """Return nominal top-``limit`` ids plus the complete cutoff tie block."""

    order = np.argsort(-values, kind="stable")
    if limit >= len(order):
        return order
    cutoff = values[order[limit - 1]]
    selected = np.flatnonzero(values >= cutoff)
    # Score ordering matches historical argsort whenever ranks are strict.
    return selected[np.argsort(-values[selected], kind="stable")]


def _soft_square_score(values: np.ndarray) -> float:
    scaled = values.reshape(-1) / 3.0
    maximum = float(np.max(scaled))
    return maximum + SQUARE_TEMPERATURE * float(
        np.log(np.exp((scaled - maximum) / SQUARE_TEMPERATURE).sum())
    )


def _square_side(
    axis: np.ndarray,
    cross: np.ndarray,
    source: int,
    target: int,
    first: np.ndarray,
    second: np.ndarray,
    *,
    backwards: bool,
) -> float:
    if backwards:
        values = (
            cross[first, source][:, None]
            + axis[np.ix_(first, second)]
            + cross[second, target][None, :]
        )
    else:
        values = (
            cross[source, first][:, None]
            + axis[np.ix_(first, second)]
            + cross[target, second][None, :]
        )
    return _soft_square_score(values)


def _bonus_all_rows(
    axis: np.ndarray,
    cross: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    count = len(axis)
    forward = tuple(
        _top_with_boundary_ties(cross[source], SQUARE_SUPPORT_K)
        for source in range(count)
    )
    backward = tuple(
        _top_with_boundary_ties(cross[:, target], SQUARE_SUPPORT_K)
        for target in range(count)
    )
    candidates = tuple(
        _top_with_boundary_ties(axis[source], SQUARE_SHORTLIST)
        for source in range(count)
    )
    output = axis.copy()
    for source, targets in enumerate(candidates):
        square = np.empty(len(targets), dtype=np.float64)
        for rank, target_value in enumerate(targets):
            target = int(target_value)
            square[rank] = _square_side(
                axis,
                cross,
                source,
                target,
                forward[source],
                forward[target],
                backwards=False,
            ) + _square_side(
                axis,
                cross,
                source,
                target,
                backward[source],
                backward[target],
                backwards=True,
            )
        square -= square.mean()
        output[source, targets] += weight * square
    return output


def equivariant_square_rerank(
    right_log: Any,
    down_log: Any,
    *,
    weight: float = DEFAULT_SQUARE_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank two high-is-good score matrices using centred 2x2 evidence.

    The fixed candidate uses ``weight=0.4``.  Weight remains an argument so a
    caller can explicitly recover the identity arm with zero, but the measured
    structural recipe itself is deliberately not configurable.
    """

    if isinstance(weight, bool) or not np.isscalar(weight):
        raise ValueError("weight must be one finite non-negative scalar")
    numeric_weight = float(weight)
    if not np.isfinite(numeric_weight) or numeric_weight < 0.0:
        raise ValueError("weight must be one finite non-negative scalar")
    right, down = _score_pair(right_log, down_log)
    right_diagonal = np.diag(right).copy()
    down_diagonal = np.diag(down).copy()
    np.fill_diagonal(right, -1e9)
    np.fill_diagonal(down, -1e9)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for _ in range(SQUARE_ROUNDS):
                next_right = _bonus_all_rows(right, down, weight=numeric_weight)
                next_down = _bonus_all_rows(down, right, weight=numeric_weight)
                right, down = next_right, next_down
        except FloatingPointError as error:
            raise ValueError("square rerank overflowed on the supplied scores") from error
    np.fill_diagonal(right, right_diagonal)
    np.fill_diagonal(down, down_diagonal)
    if not np.isfinite(right).all() or not np.isfinite(down).all():
        raise ValueError("square rerank produced non-finite scores")
    return np.ascontiguousarray(right), np.ascontiguousarray(down)


__all__ = [
    "DEFAULT_SQUARE_WEIGHT",
    "SQUARE_ROUNDS",
    "SQUARE_SHORTLIST",
    "SQUARE_SUPPORT_K",
    "SQUARE_TEMPERATURE",
    "equivariant_square_rerank",
]
