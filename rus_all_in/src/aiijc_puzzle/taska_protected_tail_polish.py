"""Target-free seam polish for the unprotected tail of a TASKA layout.

Every tile participating in a harvested edge already realised by the input
layout is frozen.  Among the remaining positions, the routine greedily accepts
the globally best non-adjacent two-tile swap under the original TASKA
right/down cost matrices.  Thus it cannot break any initially realised
harvested relation, while still repairing contacts created by the Hungarian
tail fill.

Adjacent swap positions are excluded deliberately: the vectorised placement
unary delta is exact only when the two positions do not share a bond.  The
output remains a strict permutation of the original upright fragments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost


def _count(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _strict_layout(value: Any, *, count: int) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (count,):
        raise ValueError(f"layout must have shape {(count,)}, got {layout.shape}")
    if not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError("layout must contain every tile exactly once")
    return np.ascontiguousarray(layout)


def _cost_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _validated_edges(
    edges: Sequence[RawTailEdge],
    *,
    count: int,
) -> tuple[RawTailEdge, ...]:
    result: list[RawTailEdge] = []
    seen: set[tuple[int, int, str]] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, RawTailEdge):
            raise TypeError(f"candidate_edges[{index}] must be a RawTailEdge")
        if edge.axis not in {"right", "down"}:
            raise ValueError(f"candidate_edges[{index}] has an invalid axis")
        if not 0 <= edge.source < count or not 0 <= edge.target < count:
            raise ValueError(f"candidate_edges[{index}] has an out-of-range tile")
        if edge.source == edge.target:
            raise ValueError(f"candidate_edges[{index}] is a self-edge")
        key = (edge.source, edge.target, edge.axis)
        if key in seen:
            raise ValueError(f"candidate_edges[{index}] duplicates an earlier edge")
        seen.add(key)
        result.append(edge)
    return tuple(result)


def _realised_edge_mask(
    layout: np.ndarray,
    edges: tuple[RawTailEdge, ...],
    *,
    grid: int,
) -> np.ndarray:
    if not edges:
        return np.zeros(0, dtype=bool)
    position = np.empty(len(layout), dtype=np.int32)
    position[layout] = np.arange(len(layout), dtype=np.int32)
    source = np.fromiter((edge.source for edge in edges), dtype=np.int32)
    target = np.fromiter((edge.target for edge in edges), dtype=np.int32)
    right = np.fromiter((edge.axis == "right" for edge in edges), dtype=bool)
    source_position = position[source]
    target_position = position[target]
    right_realised = (target_position == source_position + 1) & (
        target_position // grid == source_position // grid
    )
    down_realised = target_position == source_position + grid
    return np.where(right, right_realised, down_realised)


def _placement_costs(
    layout: np.ndarray,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    *,
    grid: int,
) -> np.ndarray:
    """Return the incident seam cost of every tile at every current position."""

    count = len(layout)
    board = layout.reshape(grid, grid)
    positions = np.arange(count, dtype=np.int32).reshape(grid, grid)
    result = np.zeros((count, count), dtype=np.float64)

    current = positions[:, 1:].ravel()
    left = board[:, :-1].ravel()
    result[:, current] += cost_right[left, :].T

    current = positions[:, :-1].ravel()
    right = board[:, 1:].ravel()
    result[:, current] += cost_right[:, right]

    current = positions[1:, :].ravel()
    above = board[:-1, :].ravel()
    result[:, current] += cost_down[above, :].T

    current = positions[:-1, :].ravel()
    below = board[1:, :].ravel()
    result[:, current] += cost_down[:, below]
    return result


@dataclass(frozen=True)
class TaskaProtectedTailDiagnostics:
    """Auditable target-free measurements from one bounded polish."""

    protected_tile_count: int
    free_tile_count: int
    initial_realised_edge_count: int
    final_realised_edge_count: int
    accepted_swap_count: int
    initial_total_cost: float
    final_total_cost: float


@dataclass(frozen=True)
class TaskaProtectedTailResult:
    """A strict read-only layout and polish diagnostics."""

    layout: np.ndarray
    diagnostics: TaskaProtectedTailDiagnostics


def polish_unprotected_taska_tail(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    grid: int = 24,
    max_swaps: int = 24,
    minimum_gain: float = 1e-9,
) -> TaskaProtectedTailResult:
    """Greedily reduce full-board seam cost without moving protected tiles."""

    count = _count(grid)
    if isinstance(max_swaps, bool) or not isinstance(max_swaps, int) or max_swaps < 0:
        raise ValueError("max_swaps must be a non-negative integer")
    if not np.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("minimum_gain must be finite and non-negative")
    current = _strict_layout(layout, count=count).copy()
    right = _cost_matrix(cost_right, count=count, name="cost_right")
    down = _cost_matrix(cost_down, count=count, name="cost_down")
    edges = _validated_edges(candidate_edges, count=count)

    initial_realised = _realised_edge_mask(current, edges, grid=grid)
    if edges and initial_realised.any():
        source = np.fromiter((edge.source for edge in edges), dtype=np.int32)
        target = np.fromiter((edge.target for edge in edges), dtype=np.int32)
        protected = np.unique(
            np.concatenate((source[initial_realised], target[initial_realised]))
        )
    else:
        protected = np.empty(0, dtype=np.int32)
    tile_is_free = np.ones(count, dtype=bool)
    tile_is_free[protected] = False
    free_positions = np.flatnonzero(tile_is_free[current])

    rows, columns = divmod(free_positions, grid)
    adjacent = (np.abs(rows[:, None] - rows[None, :]) + np.abs(
        columns[:, None] - columns[None, :]
    )) == 1
    initial_cost = total_taska_adjacent_seam_cost(current, right, down, grid=grid)
    accepted = 0

    for _ in range(max_swaps):
        if len(free_positions) < 2:
            break
        placement = _placement_costs(current, right, down, grid=grid)
        free_tiles = current[free_positions]
        cross = placement[free_tiles[None, :], free_positions[:, None]]
        old = placement[free_tiles, free_positions]
        delta = cross + cross.T - old[:, None] - old[None, :]
        delta[adjacent] = np.inf
        delta[np.tril_indices(len(free_positions))] = np.inf
        first, second = np.unravel_index(int(np.argmin(delta)), delta.shape)
        best_delta = float(delta[first, second])
        if not best_delta < -minimum_gain:
            break
        first_position = int(free_positions[first])
        second_position = int(free_positions[second])
        current[first_position], current[second_position] = (
            current[second_position],
            current[first_position],
        )
        accepted += 1

    final_realised = _realised_edge_mask(current, edges, grid=grid)
    if not np.all(final_realised[initial_realised]):
        raise RuntimeError("tail polish broke an initially realised harvested edge")
    final_cost = total_taska_adjacent_seam_cost(current, right, down, grid=grid)
    tolerance = 1e-9 * max(1.0, abs(initial_cost))
    if final_cost > initial_cost + tolerance:
        raise RuntimeError("tail polish increased its declared seam objective")
    if not np.array_equal(np.sort(current), np.arange(count)):
        raise RuntimeError("tail polish emitted a non-permutation")

    current.setflags(write=False)
    return TaskaProtectedTailResult(
        layout=current,
        diagnostics=TaskaProtectedTailDiagnostics(
            protected_tile_count=len(protected),
            free_tile_count=len(free_positions),
            initial_realised_edge_count=int(initial_realised.sum()),
            final_realised_edge_count=int(final_realised.sum()),
            accepted_swap_count=accepted,
            initial_total_cost=initial_cost,
            final_total_cost=final_cost,
        ),
    )


__all__ = [
    "TaskaProtectedTailDiagnostics",
    "TaskaProtectedTailResult",
    "polish_unprotected_taska_tail",
]
