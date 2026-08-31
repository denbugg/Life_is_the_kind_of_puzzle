"""Rejected target-free block-Hungarian tail experiment for TASKA layouts.

The routine freezes every tile participating in a harvested relation realised
by the input layout.  It then proposes a simultaneous Hungarian reassignment
of all remaining tiles from their incident seam costs against the current
layout.  A proposal is accepted only when the exact full-board TASKA seam cost
strictly decreases.  This preserves the initially realised harvested edges and
the strict upright-tile permutation contract.

The experiment is retained for reproducibility, not used by the production
pair pipeline: its small opened-panel gain did not transfer to held data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost


def _count(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _layout(value: Any, *, count: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.int32)
    if result.shape != (count,):
        raise ValueError(f"layout must have shape {(count,)}, got {result.shape}")
    if not np.array_equal(np.sort(result), np.arange(count)):
        raise ValueError("layout must contain every tile exactly once")
    return np.ascontiguousarray(result)


def _cost(value: Any, *, count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _edges(
    values: Sequence[RawTailEdge],
    *,
    count: int,
) -> tuple[RawTailEdge, ...]:
    result = tuple(values)
    seen: set[tuple[int, int, str]] = set()
    for index, edge in enumerate(result):
        if not isinstance(edge, RawTailEdge):
            raise TypeError(f"candidate_edges[{index}] must be a RawTailEdge")
        if edge.axis not in {"right", "down"}:
            raise ValueError(f"candidate_edges[{index}] has an invalid axis")
        if not 0 <= edge.source < count or not 0 <= edge.target < count:
            raise ValueError(f"candidate_edges[{index}] is outside the input bag")
        if edge.source == edge.target:
            raise ValueError(f"candidate_edges[{index}] is a self-edge")
        identity = (edge.source, edge.target, edge.axis)
        if identity in seen:
            raise ValueError(f"candidate_edges[{index}] duplicates an earlier edge")
        seen.add(identity)
    return result


def _realised(
    layout: np.ndarray,
    edges: tuple[RawTailEdge, ...],
    *,
    grid: int,
) -> np.ndarray:
    position = np.empty(len(layout), dtype=np.int32)
    position[layout] = np.arange(len(layout), dtype=np.int32)
    result = np.empty(len(edges), dtype=bool)
    for index, edge in enumerate(edges):
        source = int(position[edge.source])
        target = int(position[edge.target])
        if edge.axis == "right":
            result[index] = target == source + 1 and target // grid == source // grid
        else:
            result[index] = target == source + grid
    return result


def _assignment_proposal(
    layout: np.ndarray,
    free_positions: np.ndarray,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    *,
    grid: int,
) -> np.ndarray:
    free_tiles = layout[free_positions].copy()
    assignment = np.zeros((len(free_positions), len(free_tiles)), dtype=np.float64)
    board = layout.reshape(grid, grid)
    for index, raw_position in enumerate(free_positions):
        row, column = divmod(int(raw_position), grid)
        if column:
            assignment[index] += cost_right[board[row, column - 1], free_tiles]
        if column + 1 < grid:
            assignment[index] += cost_right[free_tiles, board[row, column + 1]]
        if row:
            assignment[index] += cost_down[board[row - 1, column], free_tiles]
        if row + 1 < grid:
            assignment[index] += cost_down[free_tiles, board[row + 1, column]]
    rows, columns = linear_sum_assignment(assignment)
    proposal = layout.copy()
    proposal[free_positions[rows]] = free_tiles[columns]
    return proposal


@dataclass(frozen=True)
class TaskaMonotoneHungarianTailDiagnostics:
    """Auditable target-free measurements from the rejected block search."""

    protected_tile_count: int
    free_tile_count: int
    initial_realised_edge_count: int
    final_realised_edge_count: int
    accepted_round_count: int
    initial_total_cost: float
    final_total_cost: float


@dataclass(frozen=True)
class TaskaMonotoneHungarianTailResult:
    """One strict read-only layout and its block-search diagnostics."""

    layout: np.ndarray
    diagnostics: TaskaMonotoneHungarianTailDiagnostics


def repack_unprotected_taska_tail(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    grid: int = 24,
    max_rounds: int = 6,
    minimum_gain: float = 1e-9,
) -> TaskaMonotoneHungarianTailResult:
    """Accept only exact-cost-improving block assignments of the free tail."""

    count = _count(grid)
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 0:
        raise ValueError("max_rounds must be a non-negative integer")
    if not np.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("minimum_gain must be finite and non-negative")
    current = _layout(layout, count=count).copy()
    right = _cost(cost_right, count=count, name="cost_right")
    down = _cost(cost_down, count=count, name="cost_down")
    edges = _edges(candidate_edges, count=count)
    initial_realised = _realised(current, edges, grid=grid)
    if initial_realised.any():
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
    initial_cost = total_taska_adjacent_seam_cost(current, right, down, grid=grid)
    current_cost = initial_cost
    accepted = 0
    for _ in range(max_rounds):
        if len(free_positions) < 2:
            break
        proposal = _assignment_proposal(
            current,
            free_positions,
            right,
            down,
            grid=grid,
        )
        proposal_cost = total_taska_adjacent_seam_cost(
            proposal,
            right,
            down,
            grid=grid,
        )
        if proposal_cost >= current_cost - minimum_gain:
            break
        current = proposal
        current_cost = proposal_cost
        accepted += 1

    final_realised = _realised(current, edges, grid=grid)
    if not np.all(final_realised[initial_realised]):
        raise RuntimeError("block assignment broke an initially realised harvested edge")
    if not np.array_equal(np.sort(current), np.arange(count)):
        raise RuntimeError("block assignment emitted a non-permutation")
    current.setflags(write=False)
    return TaskaMonotoneHungarianTailResult(
        layout=current,
        diagnostics=TaskaMonotoneHungarianTailDiagnostics(
            protected_tile_count=len(protected),
            free_tile_count=len(free_positions),
            initial_realised_edge_count=int(initial_realised.sum()),
            final_realised_edge_count=int(final_realised.sum()),
            accepted_round_count=accepted,
            initial_total_cost=initial_cost,
            final_total_cost=current_cost,
        ),
    )


__all__ = [
    "TaskaMonotoneHungarianTailDiagnostics",
    "TaskaMonotoneHungarianTailResult",
    "repack_unprotected_taska_tail",
]
