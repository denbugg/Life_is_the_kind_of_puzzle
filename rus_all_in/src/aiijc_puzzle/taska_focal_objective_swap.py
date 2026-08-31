"""One-step focal-objective diagnostic for a frozen TASKA layout.

The retained TASKA tail protects realised focal-positive candidate edges but
optimises the original dense seam cost.  This module isolates a different
hypothesis: among the remaining free tiles, choose one non-adjacent swap by a
sparse objective made only from the frozen focal logits.

This is a diagnostic primitive, not a promoted solver.  It never reads a
target and always returns a strict permutation of the original tile ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_protected_tail_polish import (
    _count,
    _placement_costs,
    _realised_edge_mask,
    _strict_layout,
    _validated_edges,
)

FocalSwapObjective = Literal["positive_softplus", "signed_logit"]


@dataclass(frozen=True)
class TaskaFocalObjectiveSwapResult:
    """Read-only candidate and target-free diagnostics."""

    layout: np.ndarray
    changed: bool
    objective_gain: float
    protected_tile_count: int
    free_tile_count: int
    first_position: int | None
    second_position: int | None


def _objective_costs(
    edges: tuple[RawTailEdge, ...],
    logits: np.ndarray,
    *,
    count: int,
    objective: FocalSwapObjective,
) -> tuple[np.ndarray, np.ndarray]:
    right = np.zeros((count, count), dtype=np.float64)
    down = np.zeros((count, count), dtype=np.float64)
    if objective == "positive_softplus":
        weights = np.logaddexp(0.0, logits)
        weights = np.where(logits >= 0.0, weights, 0.0)
    elif objective == "signed_logit":
        weights = logits
    else:
        raise ValueError(f"unsupported focal swap objective: {objective}")
    for edge, weight in zip(edges, weights, strict=True):
        matrix = right if edge.axis == "right" else down
        matrix[edge.source, edge.target] -= float(weight)
    return right, down


def propose_one_focal_objective_swap(
    layout: Any,
    candidate_edges: Sequence[RawTailEdge],
    focal_logits: Any,
    *,
    objective: FocalSwapObjective,
    grid: int = 24,
    minimum_gain: float = 1e-9,
) -> TaskaFocalObjectiveSwapResult:
    """Apply the globally best non-adjacent focal-objective swap, if any.

    Endpoints of every initially realised edge with focal logit ``>= 0`` are
    immutable, matching the retained focal-gated protection contract.
    """

    count = _count(grid)
    current = _strict_layout(layout, count=count).copy()
    edges = _validated_edges(candidate_edges, count=count)
    logits = np.ascontiguousarray(focal_logits, dtype=np.float64)
    if logits.shape != (len(edges),) or not np.isfinite(logits).all():
        raise ValueError("focal_logits must contain one finite value per edge")
    if not np.isfinite(minimum_gain) or minimum_gain < 0.0:
        raise ValueError("minimum_gain must be finite and non-negative")

    positive_edges = tuple(
        edge for edge, logit in zip(edges, logits, strict=True) if logit >= 0.0
    )
    realised = _realised_edge_mask(current, positive_edges, grid=grid)
    protected: np.ndarray
    if realised.any():
        source = np.fromiter((edge.source for edge in positive_edges), dtype=np.int32)
        target = np.fromiter((edge.target for edge in positive_edges), dtype=np.int32)
        protected = np.unique(
            np.concatenate((source[realised], target[realised]))
        )
    else:
        protected = np.empty(0, dtype=np.int32)
    tile_is_free = np.ones(count, dtype=bool)
    tile_is_free[protected] = False
    free_positions = np.flatnonzero(tile_is_free[current])
    if len(free_positions) < 2:
        current.setflags(write=False)
        return TaskaFocalObjectiveSwapResult(
            layout=current,
            changed=False,
            objective_gain=0.0,
            protected_tile_count=len(protected),
            free_tile_count=len(free_positions),
            first_position=None,
            second_position=None,
        )

    cost_right, cost_down = _objective_costs(
        edges, logits, count=count, objective=objective
    )
    placement = _placement_costs(
        current, cost_right, cost_down, grid=grid
    )
    free_tiles = current[free_positions]
    cross = placement[free_tiles[None, :], free_positions[:, None]]
    old = placement[free_tiles, free_positions]
    delta = cross + cross.T - old[:, None] - old[None, :]
    rows, columns = divmod(free_positions, grid)
    adjacent = (
        np.abs(rows[:, None] - rows[None, :])
        + np.abs(columns[:, None] - columns[None, :])
    ) == 1
    delta[adjacent] = np.inf
    delta[np.tril_indices(len(free_positions))] = np.inf
    first, second = np.unravel_index(int(np.argmin(delta)), delta.shape)
    best_delta = float(delta[first, second])
    if not best_delta < -minimum_gain:
        current.setflags(write=False)
        return TaskaFocalObjectiveSwapResult(
            layout=current,
            changed=False,
            objective_gain=0.0,
            protected_tile_count=len(protected),
            free_tile_count=len(free_positions),
            first_position=None,
            second_position=None,
        )

    first_position = int(free_positions[first])
    second_position = int(free_positions[second])
    current[first_position], current[second_position] = (
        current[second_position],
        current[first_position],
    )
    if not np.array_equal(np.sort(current), np.arange(count)):
        raise RuntimeError("focal-objective swap emitted a non-permutation")
    current.setflags(write=False)
    return TaskaFocalObjectiveSwapResult(
        layout=current,
        changed=True,
        objective_gain=-best_delta,
        protected_tile_count=len(protected),
        free_tile_count=len(free_positions),
        first_position=first_position,
        second_position=second_position,
    )


__all__ = [
    "FocalSwapObjective",
    "TaskaFocalObjectiveSwapResult",
    "propose_one_focal_objective_swap",
]
