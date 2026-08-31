"""Scale-free two-sided rank selector for complete TASKA layouts.

This research primitive scores every realised right/down bond by both its
outgoing row rank and incoming column rank in the frozen dense TASKA matrix.
It contains no fitted parameter and never inspects a target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _strict_layout(value: Any, *, grid: int) -> np.ndarray:
    count = grid * grid
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (count,) or not np.array_equal(
        np.sort(layout), np.arange(count)
    ):
        raise ValueError("layout must be a strict permutation")
    return layout


def _matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    matrix = np.ascontiguousarray(value, dtype=np.float64)
    if matrix.shape != (count, count) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite square cost matrix")
    return matrix


def _row_and_column_ranks(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(cost)
    row_order = np.argsort(cost, axis=1, kind="stable")
    row_rank = np.empty_like(row_order)
    row_rank[np.arange(count)[:, None], row_order] = np.arange(count)
    column_order = np.argsort(cost, axis=0, kind="stable")
    column_rank = np.empty_like(column_order)
    column_rank[column_order, np.arange(count)[None, :]] = np.arange(count)[:, None]
    return row_rank, column_rank


def two_sided_log_rank_score(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    *,
    grid: int = 24,
) -> float:
    """Return the parameter-free sum of log outgoing/incoming seam ranks."""

    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    count = grid * grid
    current = _strict_layout(layout, grid=grid).reshape(grid, grid)
    right = _matrix(cost_right, count=count, name="cost_right")
    down = _matrix(cost_down, count=count, name="cost_down")
    right_out, right_in = _row_and_column_ranks(right)
    down_out, down_in = _row_and_column_ranks(down)
    left = current[:, :-1].ravel()
    right_tile = current[:, 1:].ravel()
    above = current[:-1, :].ravel()
    below = current[1:, :].ravel()
    values = (
        np.log1p(right_out[left, right_tile]).sum(dtype=np.float64)
        + np.log1p(right_in[left, right_tile]).sum(dtype=np.float64)
        + np.log1p(down_out[above, below]).sum(dtype=np.float64)
        + np.log1p(down_in[above, below]).sum(dtype=np.float64)
    )
    return float(values)


def select_two_sided_log_rank_layout(
    layouts: Mapping[str, Any],
    roster: Sequence[str],
    cost_right: Any,
    cost_down: Any,
    *,
    grid: int = 24,
) -> tuple[str, np.ndarray, dict[str, float]]:
    """Select the minimum-score layout; roster order resolves exact ties."""

    names = tuple(roster)
    if not names or tuple(layouts) != names or len(set(names)) != len(names):
        raise ValueError("layouts must follow a non-empty unique roster")
    strict = {name: _strict_layout(layouts[name], grid=grid) for name in names}
    scores = {
        name: two_sided_log_rank_score(
            strict[name], cost_right, cost_down, grid=grid
        )
        for name in names
    }
    choice = min(names, key=scores.__getitem__)
    result = strict[choice].copy()
    result.setflags(write=False)
    return choice, result, scores


__all__ = ["select_two_sided_log_rank_layout", "two_sided_log_rank_score"]
