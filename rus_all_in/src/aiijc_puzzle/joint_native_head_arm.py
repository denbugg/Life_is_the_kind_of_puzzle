"""One target-blind layout arm built directly from a frozen reciprocal head.

The joint head supplies only directed tile-bag relations and confidences.  It
does not select among an existing layout portfolio.  The relations are offered
to the frozen raw-tail component builder in global confidence order.  Component
placement and the Hungarian fill use dense raw boundary costs reconstructed
from the already frozen RGB-plus-gradient side sequences.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, RawTailGlobalConfig
from aiijc_puzzle.structured_decoder_fit_oracle import (
    DirectedEdge,
    strict_layout,
    validate_fixed_reciprocal_head,
)
from aiijc_puzzle.taska_edge_calibrator import solve_prioritized_raw_tail_global

FROZEN_SOLVER_CONFIG = RawTailGlobalConfig(
    baseline_quantile=0.15,
    search_rounds=6,
    border_weight=0.0,
    random_seed=0,
    component_cap=0,
    fill_rounds=1,
)


@dataclass(frozen=True)
class JointNativeHeadArmResult:
    """Strict layout and target-free construction diagnostics."""

    layout: np.ndarray
    layout_sha256: str
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        value = np.ascontiguousarray(self.layout, dtype=np.int32)
        value.setflags(write=False)
        object.__setattr__(self, "layout", value)


def dense_raw_side_costs(raw_sides: Any, *, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return full right/down mean-square RGB-plus-gradient seam costs.

    The frozen side order is right, left, bottom, top.  A matrix-product form
    avoids materialising an ``N x N x 120`` difference tensor.
    """

    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    count = grid * grid
    sides = np.asarray(raw_sides, dtype=np.float64)
    if sides.ndim != 4 or sides.shape[0] != 4 or sides.shape[1] != count:
        raise ValueError("raw_sides must have shape 4 x grid**2 x length x channels")
    if sides.shape[2] <= 0 or sides.shape[3] <= 0 or not np.isfinite(sides).all():
        raise ValueError("raw_sides must contain finite non-empty side sequences")

    flattened = np.ascontiguousarray(sides.reshape(4, count, -1))

    def squared_distance(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        dimension = source.shape[1]
        source_norm = np.square(source).sum(axis=1)[:, None]
        target_norm = np.square(target).sum(axis=1)[None, :]
        result = (source_norm + target_norm - 2.0 * (source @ target.T)) / dimension
        # Roundoff from the quadratic expansion can make exact zeros tiny negatives.
        return np.ascontiguousarray(np.maximum(result, 0.0), dtype=np.float64)

    return (
        squared_distance(flattened[0], flattened[1]),
        squared_distance(flattened[2], flattened[3]),
    )


def frozen_head_edges(
    sources: tuple[Any, Any] | list[Any],
    targets: tuple[Any, Any] | list[Any],
    confidences: tuple[Any, Any] | list[Any],
    *,
    grid: int,
    requested_per_axis: int,
) -> tuple[DirectedEdge, ...]:
    """Validate and globally order one fixed right/down reciprocal head."""

    if len(sources) != 2 or len(targets) != 2 or len(confidences) != 2:
        raise ValueError("sources, targets and confidences must contain right/down arrays")
    edges: list[DirectedEdge] = []
    for axis in range(2):
        current_sources = np.asarray(sources[axis])
        current_targets = np.asarray(targets[axis])
        current_confidences = np.asarray(confidences[axis], dtype=np.float64)
        if (
            current_sources.shape != (requested_per_axis,)
            or current_targets.shape != current_sources.shape
            or current_confidences.shape != current_sources.shape
        ):
            raise ValueError("fixed head arrays have the wrong directional shape")
        if not np.issubdtype(current_sources.dtype, np.integer) or not np.issubdtype(
            current_targets.dtype, np.integer
        ):
            raise ValueError("fixed head sources and targets must be integer arrays")
        edges.extend(
            DirectedEdge(axis, int(source), int(target), float(confidence))
            for source, target, confidence in zip(
                current_sources,
                current_targets,
                current_confidences,
                strict=True,
            )
        )
    validated = validate_fixed_reciprocal_head(
        edges,
        grid=grid,
        requested_per_axis=requested_per_axis,
    )
    return tuple(
        sorted(
            validated,
            key=lambda edge: (
                -edge.confidence,
                edge.axis,
                edge.source,
                edge.target,
            ),
        )
    )


def solve_joint_native_head_arm(
    raw_sides: Any,
    sources: tuple[Any, Any] | list[Any],
    targets: tuple[Any, Any] | list[Any],
    confidences: tuple[Any, Any] | list[Any],
    *,
    grid: int,
    requested_per_axis: int,
    solver_config: RawTailGlobalConfig = FROZEN_SOLVER_CONFIG,
) -> JointNativeHeadArmResult:
    """Build one strict joint-native arm without labels or whole-arm selection."""

    if solver_config != FROZEN_SOLVER_CONFIG:
        raise ValueError("joint-native arm solver config differs from its frozen contract")
    edges = frozen_head_edges(
        sources,
        targets,
        confidences,
        grid=grid,
        requested_per_axis=requested_per_axis,
    )
    cost_right, cost_down = dense_raw_side_costs(raw_sides, grid=grid)
    raw_edges = tuple(
        RawTailEdge(
            source=edge.source,
            target=edge.target,
            axis="right" if edge.axis == 0 else "down",
        )
        for edge in edges
    )
    priorities = np.asarray([edge.confidence for edge in edges], dtype=np.float64)
    solved = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        raw_edges,
        priorities,
        grid=grid,
        config=solver_config,
    )
    layout = strict_layout(solved.layout, grid=grid, name="joint_native_head_layout")
    accepted = [
        decision for decision in solved.decisions if decision.status.startswith("accepted_")
    ]
    return JointNativeHeadArmResult(
        layout=layout,
        layout_sha256=hashlib.sha256(layout.tobytes()).hexdigest(),
        diagnostics={
            "head_edge_count": len(edges),
            "requested_per_axis": requested_per_axis,
            "accepted_build_decision_count": len(accepted),
            "solver": solved.diagnostics.as_dict(),
            "solver_config": asdict(solver_config),
            "priority_order": "confidence-desc-axis-right-before-down-source-target",
            "dense_cost": "mean-square-frozen-rgb-plus-gradient-opposite-side",
        },
    )


def reference_from_target_slots(
    candidates: Any,
    target_slots: Any,
    *,
    grid: int,
) -> np.ndarray:
    """Reconstruct the exact tile-at-position reference after a freeze is verified."""

    count = grid * grid
    ids = np.asarray(candidates)
    slots = np.asarray(target_slots)
    if ids.ndim != 3 or ids.shape[:2] != (2, count) or slots.shape != (2, count):
        raise ValueError("candidates/target_slots schema changed")
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("candidates/target_slots must be integer arrays")
    truth = np.full((2, count), -1, dtype=np.int32)
    for axis in range(2):
        present = slots[axis] >= 0
        source = np.flatnonzero(present)
        selected = slots[axis, present].astype(np.int64)
        if np.any(selected >= ids.shape[2]):
            raise ValueError("target_slots points outside candidate width")
        truth[axis, source] = ids[axis, source, selected]
    if np.any((truth < -1) | (truth >= count)):
        raise ValueError("reconstructed neighbour identities are invalid")

    incoming_right = set(int(value) for value in truth[0] if value >= 0)
    incoming_down = set(int(value) for value in truth[1] if value >= 0)
    top_left = [
        tile for tile in range(count) if tile not in incoming_right and tile not in incoming_down
    ]
    if len(top_left) != 1:
        raise ValueError("exact neighbour graph does not have one top-left tile")
    board = np.full((grid, grid), -1, dtype=np.int32)
    board[0, 0] = top_left[0]
    for row in range(grid):
        if row > 0:
            board[row, 0] = truth[1, board[row - 1, 0]]
        for column in range(1, grid):
            board[row, column] = truth[0, board[row, column - 1]]
    layout = strict_layout(board.ravel(), grid=grid, name="reconstructed_reference")
    for row in range(grid):
        for column in range(grid):
            tile = int(board[row, column])
            expected_right = int(board[row, column + 1]) if column + 1 < grid else -1
            expected_down = int(board[row + 1, column]) if row + 1 < grid else -1
            if truth[0, tile] != expected_right or truth[1, tile] != expected_down:
                raise ValueError("exact neighbour graph is not a consistent square grid")
    return layout


__all__ = [
    "FROZEN_SOLVER_CONFIG",
    "JointNativeHeadArmResult",
    "dense_raw_side_costs",
    "frozen_head_edges",
    "reference_from_target_slots",
    "solve_joint_native_head_arm",
]
