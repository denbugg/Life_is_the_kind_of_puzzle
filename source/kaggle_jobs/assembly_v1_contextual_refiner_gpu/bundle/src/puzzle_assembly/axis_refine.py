"""Deterministic whole-row/whole-column refinement for a 24x24 layout.

The ordinary 576-node objective can preserve useful horizontal/vertical strips
while placing those strips in the wrong global order.  This module collapses a
fixed layout into 24 atomic bands, aggregates all aligned seams between each
ordered band pair, solves the resulting directed Hamiltonian-path problem, and
optionally alternates rows and columns.  It never consumes target pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE_COUNT, validate_permutation
from .solvers import outside_evidence


@dataclass(frozen=True)
class AxisRefineResult:
    """Refined layout plus input-only diagnostics."""

    position_to_slot: np.ndarray
    objective_before: float
    objective_after: float
    accepted_steps: int
    row_orders: tuple[tuple[int, ...], ...]
    column_orders: tuple[tuple[int, ...], ...]


def _finite_values(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.asarray([1e6], dtype=np.float64)
    return finite


def _aggregate(values: np.ndarray, mode: str, best_fraction: float) -> float:
    finite = np.sort(_finite_values(values))
    if mode == "mean":
        return float(finite.mean())
    if mode == "median":
        return float(np.median(finite))
    if mode == "best_mean":
        if not 0.0 < best_fraction <= 1.0:
            raise ValueError("best_fraction must lie in (0,1]")
        keep = max(2, int(np.ceil(len(finite) * best_fraction)))
        return float(finite[:keep].mean())
    raise ValueError(f"unknown aggregation mode: {mode!r}")


def _rank_cost(values: np.ndarray, reciprocal_weight: float) -> np.ndarray:
    if not 0.0 <= reciprocal_weight <= 1.0:
        raise ValueError("reciprocal_weight must lie in [0,1]")
    matrix = np.asarray(values, dtype=np.float64)
    count = len(matrix)
    row_order = np.argsort(matrix, axis=1, kind="stable")
    column_order = np.argsort(matrix, axis=0, kind="stable")
    row_rank = np.empty((count, count), dtype=np.int16)
    column_rank = np.empty((count, count), dtype=np.int16)
    ranks = np.arange(count, dtype=np.int16)
    row_rank[np.arange(count)[:, None], row_order] = ranks[None, :]
    column_rank[column_order, np.arange(count)[None, :]] = ranks[:, None]
    normalizer = float(np.log1p(max(count - 1, 1)))
    outgoing = np.log1p(row_rank.astype(np.float64)) / normalizer
    incoming = np.log1p(column_rank.astype(np.float64)) / normalizer
    result = (1.0 - reciprocal_weight) * outgoing + reciprocal_weight * incoming
    np.fill_diagonal(result, np.inf)
    return result


def band_cost_matrix(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    axis: str,
    aggregation: str = "mean",
    best_fraction: float = 1.0 / 3.0,
    rank_normalize: bool = True,
    reciprocal_weight: float = 0.35,
) -> np.ndarray:
    """Return a directed 24x24 atomic-band successor cost matrix."""

    layout = validate_permutation(position_to_slot, name="position_to_slot")
    grid = layout.reshape(GRID, GRID)
    if axis not in {"row", "column"}:
        raise ValueError("axis must be 'row' or 'column'")
    result = np.full((GRID, GRID), np.inf, dtype=np.float64)
    for first in range(GRID):
        for second in range(GRID):
            if first == second:
                continue
            if axis == "row":
                seams = compatibility.down[grid[first, :], grid[second, :]]
            else:
                seams = compatibility.right[grid[:, first], grid[:, second]]
            result[first, second] = _aggregate(seams, aggregation, best_fraction)
    return (
        _rank_cost(result, reciprocal_weight)
        if rank_normalize
        else result.astype(np.float64, copy=False)
    )


def _boundary_costs(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    grid = layout.reshape(GRID, GRID)
    outside = outside_evidence(compatibility)
    if axis == "row":
        start = 1.0 - outside[grid, 2].mean(axis=1)
        end = 1.0 - outside[grid, 3].mean(axis=1)
    elif axis == "column":
        start = 1.0 - outside[grid, 0].mean(axis=0)
        end = 1.0 - outside[grid, 1].mean(axis=0)
    else:
        raise ValueError("axis must be 'row' or 'column'")
    return start.astype(np.float64), end.astype(np.float64)


def _path_objective(
    order: np.ndarray,
    costs: np.ndarray,
    start_cost: np.ndarray,
    end_cost: np.ndarray,
    boundary_weight: float,
) -> float:
    path = costs[order[:-1], order[1:]].sum(dtype=np.float64)
    boundary = start_cost[order[0]] + end_cost[order[-1]]
    return float(path + boundary_weight * boundary)


def _greedy_seed(
    first: int,
    costs: np.ndarray,
    end_cost: np.ndarray,
    boundary_weight: float,
) -> np.ndarray:
    order = [int(first)]
    unused = set(range(len(costs))) - {int(first)}
    while unused:
        current = order[-1]
        candidate = min(
            unused,
            key=lambda value: (
                costs[current, value]
                + (boundary_weight * end_cost[value] if len(unused) == 1 else 0.0),
                value,
            ),
        )
        order.append(int(candidate))
        unused.remove(candidate)
    return np.asarray(order, dtype=np.int32)


def _local_search(
    initial: np.ndarray,
    costs: np.ndarray,
    start_cost: np.ndarray,
    end_cost: np.ndarray,
    boundary_weight: float,
    *,
    passes: int,
) -> tuple[np.ndarray, float]:
    order = np.asarray(initial, dtype=np.int32).copy()
    best = _path_objective(order, costs, start_cost, end_cost, boundary_weight)
    for _ in range(passes):
        selected: np.ndarray | None = None
        selected_value = best
        selected_key: tuple[int, int, int] | None = None
        for first in range(GRID):
            for second in range(first + 1, GRID):
                candidates: list[tuple[int, np.ndarray]] = []
                swapped = order.copy()
                swapped[first], swapped[second] = swapped[second], swapped[first]
                candidates.append((0, swapped))
                reversed_segment = order.copy()
                reversed_segment[first : second + 1] = reversed_segment[
                    first : second + 1
                ][::-1]
                candidates.append((1, reversed_segment))
                moved = np.delete(order, first)
                moved = np.insert(moved, second, order[first]).astype(np.int32)
                candidates.append((2, moved))
                for move_kind, candidate in candidates:
                    value = _path_objective(
                        candidate, costs, start_cost, end_cost, boundary_weight
                    )
                    key = (first, second, move_kind)
                    if value < selected_value - 1e-12 or (
                        abs(value - selected_value) <= 1e-12
                        and selected_key is not None
                        and key < selected_key
                    ):
                        selected = candidate
                        selected_value = value
                        selected_key = key
        if selected is None:
            break
        order = selected
        best = selected_value
    return order, best


def solve_band_path(
    costs: np.ndarray,
    start_cost: np.ndarray,
    end_cost: np.ndarray,
    *,
    boundary_weight: float = 0.05,
    random_restarts: int = 8,
    local_passes: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Solve a deterministic multi-start 24-node directed path problem."""

    values = np.asarray(costs, dtype=np.float64)
    if values.shape != (GRID, GRID):
        raise ValueError("costs must have shape 24x24")
    if np.asarray(start_cost).shape != (GRID,) or np.asarray(end_cost).shape != (GRID,):
        raise ValueError("boundary costs must have shape (24,)")
    if boundary_weight < 0.0 or random_restarts < 0 or local_passes <= 0:
        raise ValueError("invalid path solver configuration")
    rng = np.random.default_rng(seed)
    seeds = [np.arange(GRID, dtype=np.int32)]
    start_order = np.argsort(start_cost, kind="stable")
    # At 24 nodes every possible start is cheap to evaluate.  Keeping all of
    # them is important when the seam objective identifies a path but provides
    # little or no evidence for which cyclic cut is the true image boundary.
    for first in start_order.tolist():
        seeds.append(_greedy_seed(first, values, end_cost, boundary_weight))
    for _ in range(random_restarts):
        seeds.append(rng.permutation(GRID).astype(np.int32))
    best_order: np.ndarray | None = None
    best_value = np.inf
    for initial in seeds:
        candidate, value = _local_search(
            initial,
            values,
            np.asarray(start_cost, dtype=np.float64),
            np.asarray(end_cost, dtype=np.float64),
            boundary_weight,
            passes=local_passes,
        )
        key = (value, tuple(candidate.tolist()))
        best_key = (
            best_value,
            tuple(best_order.tolist()) if best_order is not None else (),
        )
        if best_order is None or key < best_key:
            best_order = candidate.copy()
            best_value = float(value)
    assert best_order is not None
    return best_order, best_value


def apply_band_order(
    position_to_slot: np.ndarray, order: np.ndarray, *, axis: str
) -> np.ndarray:
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    permutation = np.asarray(order, dtype=np.int32)
    if permutation.shape != (GRID,) or set(permutation.tolist()) != set(range(GRID)):
        raise ValueError("band order must be a permutation of range(24)")
    grid = layout.reshape(GRID, GRID)
    if axis == "row":
        result = grid[permutation, :]
    elif axis == "column":
        result = grid[:, permutation]
    else:
        raise ValueError("axis must be 'row' or 'column'")
    return validate_permutation(result.ravel(), name="band_refined_position_to_slot")


def layout_seam_mean(
    position_to_slot: np.ndarray, compatibility: CompatibilityMatrices
) -> float:
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    grid = layout.reshape(GRID, GRID)
    values = np.concatenate(
        [
            compatibility.right[grid[:, :-1], grid[:, 1:]].ravel(),
            compatibility.down[grid[:-1, :], grid[1:, :]].ravel(),
        ]
    )
    return float(_finite_values(values).mean())


def alternating_axis_refine(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    cycles: int = 2,
    aggregation: str = "mean",
    best_fraction: float = 1.0 / 3.0,
    rank_normalize: bool = True,
    reciprocal_weight: float = 0.35,
    boundary_weight: float = 0.05,
    random_restarts: int = 8,
    local_passes: int = 12,
    seam_guard_ratio: float = 1.02,
    axis_order: tuple[str, str] = ("row", "column"),
    seed: int = 0,
) -> AxisRefineResult:
    """Alternately reorder atomic rows and columns without target access."""

    if cycles <= 0 or seam_guard_ratio < 1.0:
        raise ValueError("cycles must be positive and seam_guard_ratio >= 1")
    if len(axis_order) != 2 or set(axis_order) != {"row", "column"}:
        raise ValueError("axis_order must contain row and column exactly once")
    layout = validate_permutation(position_to_slot, name="position_to_slot").copy()
    before = layout_seam_mean(layout, compatibility)
    current = before
    row_orders: list[tuple[int, ...]] = []
    column_orders: list[tuple[int, ...]] = []
    accepted = 0
    for cycle in range(cycles):
        for axis_index, axis in enumerate(axis_order):
            costs = band_cost_matrix(
                layout,
                compatibility,
                axis=axis,
                aggregation=aggregation,
                best_fraction=best_fraction,
                rank_normalize=rank_normalize,
                reciprocal_weight=reciprocal_weight,
            )
            start_cost, end_cost = _boundary_costs(layout, compatibility, axis)
            order, _ = solve_band_path(
                costs,
                start_cost,
                end_cost,
                boundary_weight=boundary_weight,
                random_restarts=random_restarts,
                local_passes=local_passes,
                seed=seed + 1009 * cycle + 9176 * axis_index,
            )
            if axis == "row":
                row_orders.append(tuple(int(value) for value in order))
            else:
                column_orders.append(tuple(int(value) for value in order))
            candidate = apply_band_order(layout, order, axis=axis)
            candidate_cost = layout_seam_mean(candidate, compatibility)
            if candidate_cost <= current * seam_guard_ratio + 1e-12:
                layout = candidate
                current = candidate_cost
                accepted += 1
    return AxisRefineResult(
        position_to_slot=layout,
        objective_before=before,
        objective_after=current,
        accepted_steps=accepted,
        row_orders=tuple(row_orders),
        column_orders=tuple(column_orders),
    )


__all__ = [
    "AxisRefineResult",
    "alternating_axis_refine",
    "apply_band_order",
    "band_cost_matrix",
    "layout_seam_mean",
    "solve_band_path",
]
