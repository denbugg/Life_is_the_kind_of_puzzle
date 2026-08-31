"""Fixed alternate-objective protected tail for a TASKA layout.

Both arms start from the same strict upright-tile permutation and freeze every
tile participating in an already realised harvested edge.  The control arm
uses the original TASKA right/down costs.  The alternate arm uses exactly the
negative frozen matcher log scores, so accepting a swap maximises the summed
raw matcher log score on the affected board bonds.

The final target-free choice is made only by the original TASKA cost over all
board bonds.  Stable insertion-order ties retain the control arm.  This is an
alternate search trajectory, not a blend of score spaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_layout_portfolio import (
    TaskaMultiPortfolioSelection,
    select_lowest_taska_seam_cost_layout,
)
from aiijc_puzzle.taska_protected_tail_polish import (
    TaskaProtectedTailResult,
    polish_unprotected_taska_tail,
)


def _finite_log_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


@dataclass(frozen=True)
class TaskaRawLogTailResult:
    """Control, alternate trajectory, and original-cost final selection."""

    control: TaskaProtectedTailResult
    rawlog_tail: TaskaProtectedTailResult
    selection: TaskaMultiPortfolioSelection


def select_taska_rawlog_tail(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    right_log: Any,
    down_log: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    grid: int = 24,
    max_swaps: int = 96,
    minimum_gain: float = 1e-9,
) -> TaskaRawLogTailResult:
    """Run the fixed original-cost and raw-log tails, then select legally.

    ``right_log`` and ``down_log`` are higher-is-better matcher scores.  Their
    negatives are passed directly to the unchanged minimising protected-tail
    primitive.  No scale, blend, budget, or threshold is tuned here.
    """

    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    count = grid * grid
    raw_right_log = _finite_log_matrix(right_log, count=count, name="right_log")
    raw_down_log = _finite_log_matrix(down_log, count=count, name="down_log")

    control = polish_unprotected_taska_tail(
        layout,
        cost_right,
        cost_down,
        candidate_edges,
        grid=grid,
        max_swaps=max_swaps,
        minimum_gain=minimum_gain,
    )
    rawlog_tail = polish_unprotected_taska_tail(
        layout,
        -raw_right_log,
        -raw_down_log,
        candidate_edges,
        grid=grid,
        max_swaps=max_swaps,
        minimum_gain=minimum_gain,
    )

    protection_fields = (
        "protected_tile_count",
        "free_tile_count",
        "initial_realised_edge_count",
    )
    if any(
        getattr(control.diagnostics, field)
        != getattr(rawlog_tail.diagnostics, field)
        for field in protection_fields
    ):
        raise RuntimeError("alternate tails did not use the same protected set")

    selection = select_lowest_taska_seam_cost_layout(
        {
            "control": control.layout,
            "rawlog_tail": rawlog_tail.layout,
        },
        cost_right,
        cost_down,
        grid=grid,
    )
    return TaskaRawLogTailResult(
        control=control,
        rawlog_tail=rawlog_tail,
        selection=selection,
    )


__all__ = ["TaskaRawLogTailResult", "select_taska_rawlog_tail"]
