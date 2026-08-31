"""Target-free portfolio choice between two strict TASKA layouts.

The selector compares the sum of the original TASKA right/down costs on all
1,104 realised board bonds.  It sees neither a target nor an absolute tile
identity feature: both layouts and both cost matrices are derived from the
same current shuffled tile bag.  Exact score ties retain the raw-priority
layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


def _grid_count(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _strict_layout(value: Any, *, count: int, name: str) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}, got {layout.shape}")
    if not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError(f"{name} must contain every tile exactly once")
    return np.ascontiguousarray(layout)


def _cost_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def total_taska_adjacent_seam_cost(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    *,
    grid: int = 24,
) -> float:
    """Sum original TASKA costs over every directed adjacent board bond."""

    count = _grid_count(grid)
    current = _strict_layout(layout, count=count, name="layout").reshape(grid, grid)
    right = _cost_matrix(cost_right, count=count, name="cost_right")
    down = _cost_matrix(cost_down, count=count, name="cost_down")
    total = right[current[:, :-1], current[:, 1:]].sum(dtype=np.float64)
    total += down[current[:-1, :], current[1:, :]].sum(dtype=np.float64)
    result = float(total)
    if not np.isfinite(result):
        raise RuntimeError("total seam cost is not finite")
    return result


@dataclass(frozen=True)
class TaskaPortfolioSelection:
    """One strict selected layout plus auditable target-free decision values."""

    layout: np.ndarray
    choice: Literal["raw", "calibrated"]
    raw_total_cost: float
    calibrated_total_cost: float


@dataclass(frozen=True)
class TaskaMultiPortfolioSelection:
    """One selected layout and all named all-bond seam costs."""

    layout: np.ndarray
    choice: str
    total_costs: tuple[tuple[str, float], ...]


def select_lower_taska_seam_cost_layout(
    raw_layout: Any,
    calibrated_layout: Any,
    cost_right: Any,
    cost_down: Any,
    *,
    grid: int = 24,
) -> TaskaPortfolioSelection:
    """Choose the lower-cost strict layout, retaining ``raw`` on exact ties."""

    count = _grid_count(grid)
    raw = _strict_layout(raw_layout, count=count, name="raw_layout")
    calibrated = _strict_layout(
        calibrated_layout,
        count=count,
        name="calibrated_layout",
    )
    right = _cost_matrix(cost_right, count=count, name="cost_right")
    down = _cost_matrix(cost_down, count=count, name="cost_down")
    raw_cost = total_taska_adjacent_seam_cost(raw, right, down, grid=grid)
    calibrated_cost = total_taska_adjacent_seam_cost(
        calibrated,
        right,
        down,
        grid=grid,
    )
    choice: Literal["raw", "calibrated"]
    if calibrated_cost < raw_cost:
        choice = "calibrated"
        selected = calibrated
    else:
        choice = "raw"
        selected = raw
    frozen = selected.copy()
    frozen.setflags(write=False)
    return TaskaPortfolioSelection(
        layout=frozen,
        choice=choice,
        raw_total_cost=raw_cost,
        calibrated_total_cost=calibrated_cost,
    )


def select_lowest_taska_seam_cost_layout(
    layouts: Mapping[str, Any],
    cost_right: Any,
    cost_down: Any,
    *,
    grid: int = 24,
) -> TaskaMultiPortfolioSelection:
    """Choose the lowest-cost named strict layout; insertion order breaks ties."""

    if not isinstance(layouts, Mapping) or not layouts:
        raise ValueError("layouts must be a non-empty mapping")
    if not all(isinstance(name, str) and name for name in layouts):
        raise ValueError("layout names must be non-empty strings")
    count = _grid_count(grid)
    right = _cost_matrix(cost_right, count=count, name="cost_right")
    down = _cost_matrix(cost_down, count=count, name="cost_down")
    validated = tuple(
        (name, _strict_layout(layout, count=count, name=f"layouts[{name!r}]"))
        for name, layout in layouts.items()
    )
    costs = tuple(
        (name, total_taska_adjacent_seam_cost(layout, right, down, grid=grid))
        for name, layout in validated
    )
    # ``min`` is stable, so an exact cost tie retains the first named arm.
    choice, _ = min(costs, key=lambda item: item[1])
    selected = next(layout for name, layout in validated if name == choice).copy()
    selected.setflags(write=False)
    return TaskaMultiPortfolioSelection(
        layout=selected,
        choice=choice,
        total_costs=costs,
    )


__all__ = [
    "TaskaMultiPortfolioSelection",
    "TaskaPortfolioSelection",
    "select_lower_taska_seam_cost_layout",
    "select_lowest_taska_seam_cost_layout",
    "total_taska_adjacent_seam_cost",
]
