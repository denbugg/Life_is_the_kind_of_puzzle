"""Target-free majority-bond TASKA component arm.

The fixed raw, logistic, focal-top5, and nonlinear solvers can disagree about
the whole board while still realising some of the same directed neighbour
relations.  This module converts only those repeated relations into a new
component-builder supply:

1. enumerate all 1104 directed right/down bonds in each strict layout;
2. retain bonds realised by at least two of the four fixed layouts;
3. order them lexicographically by support (4, 3, 2), untouched TASKA raw
   priority ``-cost``, then a stable ``(axis, source, target)`` identity;
4. run the existing translation-consistent component placement and Hungarian
   fill with the original TASKA cost matrices; and
5. apply the fixed 96-swap protected-tail polish, protecting the consensus
   relations initially realised by the new layout.

No target, clean image, source filename, or absolute source-grid coordinate is
accepted by the API.  Every intermediate and final layout is a strict
permutation of the original upright tile ids.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, RawTailGlobalConfig
from aiijc_puzzle.taska_edge_calibrator import (
    PrioritizedRawTailResult,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_protected_tail_polish import (
    TaskaProtectedTailResult,
    polish_unprotected_taska_tail,
)

CONSENSUS_ARM_NAMES = ("raw", "logistic", "focal", "nonlinear")
CONSENSUS_MINIMUM_SUPPORT = 2
CONSENSUS_TAIL_MAX_SWAPS = 96
CONSENSUS_TAIL_MINIMUM_GAIN = 1e-9


def _count(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _strict_layout(value: Any, *, count: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (count,) or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be one integer vector of length {count}")
    layout = np.ascontiguousarray(raw, dtype=np.int32)
    if not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError(f"{name} must contain every original tile exactly once")
    return layout


def _cost_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def taska_layout_bonds(layout: Any, *, grid: int = 24) -> tuple[RawTailEdge, ...]:
    """Return every directed right/down board bond in stable board order."""

    count = _count(grid)
    board = _strict_layout(layout, count=count, name="layout").reshape(grid, grid)
    bonds: list[RawTailEdge] = []
    for row in range(grid):
        for column in range(grid - 1):
            bonds.append(
                RawTailEdge(
                    int(board[row, column]),
                    int(board[row, column + 1]),
                    "right",
                )
            )
    for row in range(grid - 1):
        for column in range(grid):
            bonds.append(
                RawTailEdge(
                    int(board[row, column]),
                    int(board[row + 1, column]),
                    "down",
                )
            )
    return tuple(bonds)


@dataclass(frozen=True)
class TaskaConsensusBond:
    """One retained directed bond and its target-free ordering evidence."""

    edge: RawTailEdge
    support: int
    raw_priority: float


def build_taska_consensus_bonds(
    layouts: Mapping[str, Any],
    cost_right: Any,
    cost_down: Any,
    *,
    grid: int = 24,
) -> tuple[TaskaConsensusBond, ...]:
    """Build the fixed support>=2 bond supply in its exact solver order."""

    count = _count(grid)
    if not isinstance(layouts, Mapping):
        raise TypeError("layouts must be a mapping")
    if set(layouts) != set(CONSENSUS_ARM_NAMES) or len(layouts) != len(
        CONSENSUS_ARM_NAMES
    ):
        raise ValueError(
            "layouts must contain exactly raw, logistic, focal, and nonlinear"
        )
    right = _cost_matrix(cost_right, count=count, name="cost_right")
    down = _cost_matrix(cost_down, count=count, name="cost_down")

    support: Counter[RawTailEdge] = Counter()
    for name in CONSENSUS_ARM_NAMES:
        support.update(taska_layout_bonds(layouts[name], grid=grid))

    def raw_priority(edge: RawTailEdge) -> float:
        matrix = right if edge.axis == "right" else down
        return -float(matrix[edge.source, edge.target])

    retained = [
        TaskaConsensusBond(edge=edge, support=value, raw_priority=raw_priority(edge))
        for edge, value in support.items()
        if value >= CONSENSUS_MINIMUM_SUPPORT
    ]
    axis_order = {"right": 0, "down": 1}
    retained.sort(
        key=lambda bond: (
            -bond.support,
            -bond.raw_priority,
            axis_order[bond.edge.axis],
            bond.edge.source,
            bond.edge.target,
        )
    )
    return tuple(retained)


@dataclass(frozen=True)
class TaskaConsensusComponentDiagnostics:
    """Target-free evidence for the fixed consensus arm and its tail."""

    arm_names: tuple[str, ...]
    minimum_support: int
    consensus_edge_count: int
    support_counts: tuple[tuple[int, int], ...]
    tail_max_swaps: int


@dataclass(frozen=True)
class TaskaConsensusComponentResult:
    """Strict consensus component layout and its protected-tail continuation."""

    bonds: tuple[TaskaConsensusBond, ...]
    component: PrioritizedRawTailResult
    tail: TaskaProtectedTailResult
    diagnostics: TaskaConsensusComponentDiagnostics

    @property
    def layout(self) -> np.ndarray:
        return self.tail.layout


def solve_taska_consensus_component_arm(
    layouts: Mapping[str, Any],
    cost_right: Any,
    cost_down: Any,
    *,
    grid: int = 24,
    solver_config: RawTailGlobalConfig | None = None,
) -> TaskaConsensusComponentResult:
    """Run the one fixed majority-bond component arm followed by tail-96."""

    if solver_config is None:
        solver_config = RawTailGlobalConfig(
            baseline_quantile=0.15,
            search_rounds=6,
            border_weight=0.0,
            random_seed=0,
            component_cap=0,
            fill_rounds=1,
        )
    bonds = build_taska_consensus_bonds(
        layouts,
        cost_right,
        cost_down,
        grid=grid,
    )
    edges = tuple(bond.edge for bond in bonds)
    # The bond tuple is already in the complete lexicographic order.  Unique
    # descending ranks make the existing external-priority solver preserve it
    # exactly, including raw-cost and stable-identity tie breaks.
    priorities = np.arange(len(edges), 0, -1, dtype=np.float64)
    component = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        edges,
        priorities,
        border_unary=None,
        grid=grid,
        config=solver_config,
    )
    tail = polish_unprotected_taska_tail(
        component.layout,
        cost_right,
        cost_down,
        edges,
        grid=grid,
        max_swaps=CONSENSUS_TAIL_MAX_SWAPS,
        minimum_gain=CONSENSUS_TAIL_MINIMUM_GAIN,
    )
    support_counts = Counter(bond.support for bond in bonds)
    return TaskaConsensusComponentResult(
        bonds=bonds,
        component=component,
        tail=tail,
        diagnostics=TaskaConsensusComponentDiagnostics(
            arm_names=CONSENSUS_ARM_NAMES,
            minimum_support=CONSENSUS_MINIMUM_SUPPORT,
            consensus_edge_count=len(bonds),
            support_counts=tuple(sorted(support_counts.items(), reverse=True)),
            tail_max_swaps=CONSENSUS_TAIL_MAX_SWAPS,
        ),
    )


__all__ = [
    "CONSENSUS_ARM_NAMES",
    "CONSENSUS_MINIMUM_SUPPORT",
    "CONSENSUS_TAIL_MAX_SWAPS",
    "TaskaConsensusBond",
    "TaskaConsensusComponentDiagnostics",
    "TaskaConsensusComponentResult",
    "build_taska_consensus_bonds",
    "solve_taska_consensus_component_arm",
    "taska_layout_bonds",
]
