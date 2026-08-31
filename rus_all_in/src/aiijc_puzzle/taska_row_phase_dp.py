"""Exact cyclic row-phase dynamic program for a strict TASKA layout.

The input grid's row membership and cyclic order within every row are fixed.
Each row may choose one of ``grid`` cyclic phases.  A Viterbi dynamic program
then finds the globally minimum original TASKA all-bond seam energy over this
``grid ** grid`` family.  This is a structured large-neighbourhood move, not a
tile-swap tail, arm selector, threshold, or budget sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost


def _strict_layout(value: Any, *, grid: int) -> np.ndarray:
    count = grid * grid
    result = np.ascontiguousarray(value, dtype=np.int32)
    if result.shape != (count,) or not np.array_equal(np.sort(result), np.arange(count)):
        raise ValueError("layout must be one strict original-tile permutation")
    return result


def _cost_matrix(value: Any, *, grid: int, name: str) -> np.ndarray:
    count = grid * grid
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != (count, count) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite square tile-pair matrix")
    return result


@dataclass(frozen=True)
class TaskaRowPhaseDiagnostics:
    """Auditable exact-DP result."""

    phases: tuple[int, ...]
    changed_row_count: int
    before_total_cost: float
    after_total_cost: float
    total_cost_improvement: float
    state_count: int
    transition_count: int
    objective_monotone: bool


@dataclass(frozen=True)
class TaskaRowPhaseResult:
    """One strict read-only layout and target-free diagnostics."""

    layout: np.ndarray
    diagnostics: TaskaRowPhaseDiagnostics


def solve_taska_row_phase_dp(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    *,
    grid: int = 24,
) -> TaskaRowPhaseResult:
    """Globally optimize all cyclic row phases under the original seam cost."""

    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    original = _strict_layout(layout, grid=grid)
    right = _cost_matrix(cost_right, grid=grid, name="cost_right")
    down = _cost_matrix(cost_down, grid=grid, name="cost_down")
    board = original.reshape(grid, grid)
    variants = np.stack(
        [[np.roll(board[row], phase) for phase in range(grid)] for row in range(grid)]
    ).astype(np.int32, copy=False)

    unary = np.empty((grid, grid), dtype=np.float64)
    for row in range(grid):
        for phase in range(grid):
            values = variants[row, phase]
            unary[row, phase] = right[values[:-1], values[1:]].sum(dtype=np.float64)
    transition = np.empty((grid - 1, grid, grid), dtype=np.float64)
    for row in range(grid - 1):
        for upper_phase in range(grid):
            upper = variants[row, upper_phase]
            for lower_phase in range(grid):
                lower = variants[row + 1, lower_phase]
                transition[row, upper_phase, lower_phase] = down[upper, lower].sum(
                    dtype=np.float64
                )

    best_cost = np.empty((grid, grid), dtype=np.float64)
    changed = np.empty((grid, grid), dtype=np.int16)
    back = np.zeros((grid, grid), dtype=np.int16)
    best_cost[0] = unary[0]
    changed[0] = np.arange(grid) != 0
    for row in range(1, grid):
        for phase in range(grid):
            values = best_cost[row - 1] + transition[row - 1, :, phase]
            minimum = float(np.min(values))
            tied = np.flatnonzero(values == minimum)
            tie_changes = changed[row - 1, tied]
            predecessor = int(tied[int(np.argmin(tie_changes))])
            back[row, phase] = predecessor
            best_cost[row, phase] = minimum + unary[row, phase]
            changed[row, phase] = changed[row - 1, predecessor] + (phase != 0)

    final_minimum = float(np.min(best_cost[-1]))
    tied_final = np.flatnonzero(best_cost[-1] == final_minimum)
    final_phase = int(tied_final[int(np.argmin(changed[-1, tied_final]))])
    phases = np.zeros(grid, dtype=np.int16)
    phases[-1] = final_phase
    for row in range(grid - 1, 0, -1):
        phases[row - 1] = back[row, phases[row]]
    candidate = np.ascontiguousarray(
        np.concatenate([variants[row, phases[row]] for row in range(grid)]),
        dtype=np.int32,
    )
    candidate = _strict_layout(candidate, grid=grid)
    before = total_taska_adjacent_seam_cost(original, right, down, grid=grid)
    after = total_taska_adjacent_seam_cost(candidate, right, down, grid=grid)
    tolerance = 1e-9 * max(1.0, abs(before))
    if after > before + tolerance:
        raise RuntimeError("exact row-phase DP increased its own objective")
    candidate.setflags(write=False)
    return TaskaRowPhaseResult(
        layout=candidate,
        diagnostics=TaskaRowPhaseDiagnostics(
            phases=tuple(int(value) for value in phases),
            changed_row_count=int(np.count_nonzero(phases)),
            before_total_cost=before,
            after_total_cost=after,
            total_cost_improvement=before - after,
            state_count=grid * grid,
            transition_count=(grid - 1) * grid * grid,
            objective_monotone=after <= before + tolerance,
        ),
    )


__all__ = [
    "TaskaRowPhaseDiagnostics",
    "TaskaRowPhaseResult",
    "solve_taska_row_phase_dp",
]
