"""Reusable tile-to-true-position distance metrics for strict puzzle layouts.

The absolute metric keeps the board origin visible.  A separately named
best-global-cyclic-aligned diagnostic evaluates the same metric suite after a
single whole-board cyclic roll; it must not be substituted for the absolute
metric because it intentionally removes one class of origin error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TilePositionDistance:
    """Distance summary over every tile identity in one strict layout."""

    tile_count: int
    exact_tile_count: int
    mean_manhattan_cells: float
    median_manhattan_cells: float
    p90_manhattan_cells: float
    normalized_mean_l1: float
    mean_euclidean_cells: float
    within_radius_0_recall: float
    within_radius_1_recall: float
    within_radius_2_recall: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CyclicAlignedTilePositionDistance:
    """Best whole-board cyclic roll and its absolute distance summary."""

    selected_row_roll: int
    selected_column_roll: int
    candidates_evaluated: int
    changed: bool
    metrics: TilePositionDistance

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = self.metrics.as_dict()
        return payload


def _grid_count(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _strict_layout(value: Any, *, grid: int, name: str) -> np.ndarray:
    count = _grid_count(grid)
    layout = np.asarray(value)
    if layout.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}, got {layout.shape}")
    if not np.issubdtype(layout.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype")
    layout = np.asarray(layout, dtype=np.int32)
    if not np.array_equal(np.sort(layout), np.arange(count, dtype=np.int32)):
        raise ValueError(f"{name} must contain every tile identity exactly once")
    return np.ascontiguousarray(layout)


def _positions(layout: np.ndarray, *, grid: int) -> tuple[np.ndarray, np.ndarray]:
    position_of_tile = np.empty(len(layout), dtype=np.int32)
    position_of_tile[layout] = np.arange(len(layout), dtype=np.int32)
    return divmod(position_of_tile, grid)


def evaluate_tile_position_distance(
    layout: Any,
    exact_reference: Any,
    *,
    grid: int = 24,
) -> TilePositionDistance:
    """Evaluate absolute per-tile position errors in board-cell units."""

    candidate = _strict_layout(layout, grid=grid, name="layout")
    reference = _strict_layout(exact_reference, grid=grid, name="exact_reference")
    predicted_row, predicted_column = _positions(candidate, grid=grid)
    true_row, true_column = _positions(reference, grid=grid)
    row_error = np.abs(predicted_row - true_row)
    column_error = np.abs(predicted_column - true_column)
    manhattan = row_error + column_error
    euclidean = np.hypot(row_error, column_error)
    count = len(candidate)
    exact = int(np.count_nonzero(manhattan == 0))
    mean_l1 = float(manhattan.mean(dtype=np.float64))
    return TilePositionDistance(
        tile_count=count,
        exact_tile_count=exact,
        mean_manhattan_cells=mean_l1,
        median_manhattan_cells=float(np.median(manhattan)),
        p90_manhattan_cells=float(np.quantile(manhattan, 0.90, method="higher")),
        normalized_mean_l1=mean_l1 / (2.0 * (grid - 1)),
        mean_euclidean_cells=float(euclidean.mean(dtype=np.float64)),
        within_radius_0_recall=exact / count,
        within_radius_1_recall=float(np.count_nonzero(manhattan <= 1) / count),
        within_radius_2_recall=float(np.count_nonzero(manhattan <= 2) / count),
    )


def evaluate_best_cyclic_aligned_tile_position_distance(
    layout: Any,
    exact_reference: Any,
    *,
    grid: int = 24,
    minimum_gain: float = 1e-12,
) -> CyclicAlignedTilePositionDistance:
    """Evaluate metrics after the best single global cyclic board roll.

    Selection minimizes absolute mean Manhattan distance.  Row and column
    contributions are separable, so all ``grid**2`` candidates are represented
    exactly without materializing 576 full layouts.  Zero is the incumbent and
    only a strict improvement larger than ``minimum_gain`` can replace it.
    """

    if not np.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("minimum_gain must be finite and non-negative")
    candidate = _strict_layout(layout, grid=grid, name="layout")
    reference = _strict_layout(exact_reference, grid=grid, name="exact_reference")
    predicted_row, predicted_column = _positions(candidate, grid=grid)
    true_row, true_column = _positions(reference, grid=grid)

    row_cost = np.asarray(
        [
            np.abs((predicted_row + roll) % grid - true_row).mean(dtype=np.float64)
            for roll in range(grid)
        ],
        dtype=np.float64,
    )
    column_cost = np.asarray(
        [
            np.abs((predicted_column + roll) % grid - true_column).mean(
                dtype=np.float64
            )
            for roll in range(grid)
        ],
        dtype=np.float64,
    )

    def choose(costs: np.ndarray) -> int:
        selected = 0
        for index in range(1, grid):
            if costs[index] < costs[selected] - minimum_gain:
                selected = index
        return selected

    row_roll = choose(row_cost)
    column_roll = choose(column_cost)
    rolled = np.roll(
        candidate.reshape(grid, grid),
        shift=(row_roll, column_roll),
        axis=(0, 1),
    ).reshape(-1)
    return CyclicAlignedTilePositionDistance(
        selected_row_roll=row_roll,
        selected_column_roll=column_roll,
        candidates_evaluated=grid * grid,
        changed=(row_roll, column_roll) != (0, 0),
        metrics=evaluate_tile_position_distance(rolled, reference, grid=grid),
    )


__all__ = [
    "CyclicAlignedTilePositionDistance",
    "TilePositionDistance",
    "evaluate_best_cyclic_aligned_tile_position_distance",
    "evaluate_tile_position_distance",
]
