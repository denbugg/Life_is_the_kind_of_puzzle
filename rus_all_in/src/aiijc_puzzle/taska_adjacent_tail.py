"""Adjacent-aware protected TASKA tail polish.

This is one bounded experimental extension of the retained protected tail.
It keeps the same realised-edge protection, original TASKA right/down
objective, swap budget, and deterministic global-best greedy rule.  The only
change is that horizontally or vertically adjacent free board positions are
legal swap candidates.

For non-adjacent positions the retained vectorised placement-unary formula is
exact.  For adjacent positions this module replaces that entry with an exact
before/after sum over the union of affected directed board bonds, so the bond
shared by the two positions is counted exactly once.  Outputs remain strict
permutations of the original upright fragments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_protected_tail_polish import (
    _cost_matrix,
    _count,
    _placement_costs,
    _realised_edge_mask,
    _strict_layout,
    _validated_edges,
)


def _incident_directed_bonds(position: int, *, grid: int) -> tuple[tuple[int, int, int], ...]:
    """Return ``(axis, source_position, target_position)`` incident bonds."""

    row, column = divmod(position, grid)
    bonds: list[tuple[int, int, int]] = []
    if column > 0:
        bonds.append((0, position - 1, position))
    if column + 1 < grid:
        bonds.append((0, position, position + 1))
    if row > 0:
        bonds.append((1, position - grid, position))
    if row + 1 < grid:
        bonds.append((1, position, position + grid))
    return tuple(bonds)


def _exact_swap_delta_validated(
    layout: np.ndarray,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    first_position: int,
    second_position: int,
    *,
    grid: int,
) -> float:
    bonds = sorted(
        set(_incident_directed_bonds(first_position, grid=grid)).union(
            _incident_directed_bonds(second_position, grid=grid)
        )
    )
    first_tile = int(layout[first_position])
    second_tile = int(layout[second_position])

    def tile_after(position: int) -> int:
        if position == first_position:
            return second_tile
        if position == second_position:
            return first_tile
        return int(layout[position])

    before = 0.0
    after = 0.0
    for axis, source_position, target_position in bonds:
        matrix = cost_right if axis == 0 else cost_down
        before += float(matrix[layout[source_position], layout[target_position]])
        after += float(matrix[tile_after(source_position), tile_after(target_position)])
    return after - before


def exact_taska_swap_delta(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    first_position: int,
    second_position: int,
    *,
    grid: int = 24,
) -> float:
    """Return the exact full-board cost delta for swapping two positions."""

    count = _count(grid)
    current = _strict_layout(layout, count=count)
    right = _cost_matrix(cost_right, count=count, name="cost_right")
    down = _cost_matrix(cost_down, count=count, name="cost_down")
    for name, position in (
        ("first_position", first_position),
        ("second_position", second_position),
    ):
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(f"{name} must be an integer position")
        if not 0 <= position < count:
            raise ValueError(f"{name} is out of range")
    if first_position == second_position:
        return 0.0
    return _exact_swap_delta_validated(
        current,
        right,
        down,
        first_position,
        second_position,
        grid=grid,
    )


def _swap_delta_matrix(
    layout: np.ndarray,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    free_positions: np.ndarray,
    adjacent_pairs: np.ndarray,
    *,
    grid: int,
) -> np.ndarray:
    """Return upper-triangle deltas, exact for adjacent and non-adjacent swaps."""

    placement = _placement_costs(layout, cost_right, cost_down, grid=grid)
    free_tiles = layout[free_positions]
    cross = placement[free_tiles[None, :], free_positions[:, None]]
    old = placement[free_tiles, free_positions]
    delta = cross + cross.T - old[:, None] - old[None, :]
    for first, second in adjacent_pairs:
        delta[first, second] = _exact_swap_delta_validated(
            layout,
            cost_right,
            cost_down,
            int(free_positions[first]),
            int(free_positions[second]),
            grid=grid,
        )
    delta[np.tril_indices(len(free_positions))] = np.inf
    return delta


def _globally_best_swap(delta: np.ndarray) -> tuple[int, int, float]:
    """Choose the row-major first global minimum for deterministic ties."""

    first, second = np.unravel_index(int(np.argmin(delta)), delta.shape)
    return int(first), int(second), float(delta[first, second])


@dataclass(frozen=True)
class TaskaAdjacentTailDiagnostics:
    """Auditable target-free measurements from one bounded polish."""

    protected_tile_count: int
    free_tile_count: int
    initial_realised_edge_count: int
    final_realised_edge_count: int
    accepted_swap_count: int
    accepted_adjacent_swap_count: int
    accepted_nonadjacent_swap_count: int
    initial_total_cost: float
    final_total_cost: float


@dataclass(frozen=True)
class TaskaAdjacentTailResult:
    """A strict read-only layout and adjacent-aware polish diagnostics."""

    layout: np.ndarray
    diagnostics: TaskaAdjacentTailDiagnostics


def polish_unprotected_taska_tail_with_adjacent_swaps(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    grid: int = 24,
    max_swaps: int = 96,
    minimum_gain: float = 1e-9,
) -> TaskaAdjacentTailResult:
    """Greedily reduce full-board cost while allowing exact adjacent swaps."""

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
    adjacent = (
        np.abs(rows[:, None] - rows[None, :])
        + np.abs(columns[:, None] - columns[None, :])
    ) == 1
    adjacent_pairs = np.argwhere(np.triu(adjacent, k=1))
    initial_cost = total_taska_adjacent_seam_cost(current, right, down, grid=grid)
    accepted = 0
    accepted_adjacent = 0

    for _ in range(max_swaps):
        if len(free_positions) < 2:
            break
        delta = _swap_delta_matrix(
            current,
            right,
            down,
            free_positions,
            adjacent_pairs,
            grid=grid,
        )
        first, second, best_delta = _globally_best_swap(delta)
        if not best_delta < -minimum_gain:
            break
        first_position = int(free_positions[first])
        second_position = int(free_positions[second])
        current[first_position], current[second_position] = (
            current[second_position],
            current[first_position],
        )
        accepted += 1
        accepted_adjacent += int(adjacent[first, second])

    final_realised = _realised_edge_mask(current, edges, grid=grid)
    if not np.all(final_realised[initial_realised]):
        raise RuntimeError("adjacent tail broke an initially realised harvested edge")
    final_cost = total_taska_adjacent_seam_cost(current, right, down, grid=grid)
    tolerance = 1e-9 * max(1.0, abs(initial_cost))
    if final_cost > initial_cost + tolerance:
        raise RuntimeError("adjacent tail increased its declared seam objective")
    if not np.array_equal(np.sort(current), np.arange(count)):
        raise RuntimeError("adjacent tail emitted a non-permutation")

    current.setflags(write=False)
    return TaskaAdjacentTailResult(
        layout=current,
        diagnostics=TaskaAdjacentTailDiagnostics(
            protected_tile_count=len(protected),
            free_tile_count=len(free_positions),
            initial_realised_edge_count=int(initial_realised.sum()),
            final_realised_edge_count=int(final_realised.sum()),
            accepted_swap_count=accepted,
            accepted_adjacent_swap_count=accepted_adjacent,
            accepted_nonadjacent_swap_count=accepted - accepted_adjacent,
            initial_total_cost=initial_cost,
            final_total_cost=final_cost,
        ),
    )


__all__ = [
    "TaskaAdjacentTailDiagnostics",
    "TaskaAdjacentTailResult",
    "exact_taska_swap_delta",
    "polish_unprotected_taska_tail_with_adjacent_swaps",
]
