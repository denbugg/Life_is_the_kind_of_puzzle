"""Experimental calibrated ordering for Socket decoder component constraints.

This module does not define another complete decoder.  It converts frozen
dirty-visible hard-edge probabilities into the opt-in priority matrices
accepted by :func:`aiijc_puzzle.socket_decoder.decode_socket_assignments` and
rebuilds the same greedy component trace for exact-synthetic diagnostics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle import socket_decoder
from aiijc_puzzle.socket_confidence_calibration import HardEdgeFeatures
from aiijc_puzzle.socket_decoder import (
    SocketEdge,
    hard_partial_axis_matching,
    prioritise_component_edges,
)


@dataclass(frozen=True)
class ComponentConstraintTrace:
    """One hard edge and the greedy translation-builder decision."""

    edge: SocketEdge
    status: str


@dataclass(frozen=True)
class ComponentBuildTrace:
    """Frozen component graph before any exact synthetic label is inspected."""

    constraints: tuple[ComponentConstraintTrace, ...]
    components: tuple[dict[int, tuple[int, int]], ...]
    status_counts: dict[str, int]


def calibrated_priority_matrices(
    features: HardEdgeFeatures,
    probabilities: Any,
    *,
    grid: int,
) -> dict[str, np.ndarray]:
    """Map calibrated probability rows back to the two hard-edge matrices."""

    count = grid * grid
    expected = 2 * grid * (grid - 1)
    probability = np.asarray(probabilities, dtype=np.float64)
    if features.values.shape[0] != expected:
        raise ValueError(f"expected {expected} hard-edge feature rows")
    if probability.shape != (expected,) or not np.isfinite(probability).all():
        raise ValueError(f"probabilities must have finite shape {(expected,)}")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if any(len(value) != expected for value in (features.source, features.target, features.axis)):
        raise ValueError("hard-edge identity arrays have inconsistent lengths")

    result = {
        "right": np.zeros((count, count), dtype=np.float64),
        "down": np.zeros((count, count), dtype=np.float64),
    }
    seen: set[tuple[int, int, int]] = set()
    for source, target, axis, value in zip(
        features.source,
        features.target,
        features.axis,
        probability,
        strict=True,
    ):
        key = (int(axis), int(source), int(target))
        if key in seen:
            raise ValueError("hard-edge identities contain a duplicate")
        seen.add(key)
        name = "down" if axis else "right"
        result[name][source, target] = value
    if sum(axis == 0 for axis in features.axis) != grid * (grid - 1) or sum(
        axis == 1 for axis in features.axis
    ) != grid * (grid - 1):
        raise ValueError("hard-edge axes do not preserve exact projection cardinality")
    return result


def _normalise_component(
    component: dict[int, tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    minimum_row = min(row for row, _ in component.values())
    minimum_column = min(column for _, column in component.values())
    return {
        tile: (row - minimum_row, column - minimum_column)
        for tile, (row, column) in component.items()
    }


def build_component_trace(
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int,
    edge_budget_per_axis: int,
    component_edge_priority: dict[str, np.ndarray] | None = None,
) -> ComponentBuildTrace:
    """Rebuild the decoder's exact selected edge order and greedy components."""

    count = grid * grid
    right = hard_partial_axis_matching(right_log_assignment, grid=grid, axis="right")
    down = hard_partial_axis_matching(down_log_assignment, grid=grid, axis="down")
    edges = prioritise_component_edges(
        right,
        down,
        edge_budget_per_axis=edge_budget_per_axis,
        tile_count=count,
        component_edge_priority=component_edge_priority,
    )
    builder = socket_decoder._TranslationComponents(count=count, grid=grid)
    counts = {name: 0 for name in ("added", "consistent", "contradiction", "collision", "span")}
    constraints: list[ComponentConstraintTrace] = []
    for edge in edges:
        status = builder.add(edge)
        counts[status] += 1
        constraints.append(ComponentConstraintTrace(edge=edge, status=status))
    components = tuple(
        _normalise_component(component) for component in builder.complete_components()
    )
    components = tuple(sorted(components, key=lambda value: (-len(value), min(value))))
    return ComponentBuildTrace(tuple(constraints), components, counts)


def _strict_reference(value: Any, *, grid: int) -> np.ndarray:
    count = grid * grid
    reference = np.asarray(value, dtype=np.int64)
    if reference.shape != (count,) or not np.array_equal(np.sort(reference), np.arange(count)):
        raise ValueError("reference layout must be a strict tile-at-position permutation")
    return reference


def edge_is_reference_correct(
    edge: SocketEdge,
    reference_tile_at_position: Any,
    *,
    grid: int,
) -> bool:
    """Return exact directed-neighbour correctness for one hard edge."""

    reference = _strict_reference(reference_tile_at_position, grid=grid)
    position = np.empty(grid * grid, dtype=np.int32)
    position[reference] = np.arange(grid * grid, dtype=np.int32)
    source = int(position[edge.source])
    target = int(position[edge.target])
    if edge.axis == "right":
        return bool(target == source + 1 and source % grid != grid - 1)
    if edge.axis == "down":
        return bool(target == source + grid)
    raise ValueError(f"unsupported edge axis {edge.axis!r}")


def exact_component_metrics(
    trace: ComponentBuildTrace,
    reference_tile_at_position: Any,
    *,
    grid: int,
) -> dict[str, float | int]:
    """Measure exact selected-edge correctness, false bridges and component purity."""

    reference = _strict_reference(reference_tile_at_position, grid=grid)
    count = grid * grid
    position = np.empty(count, dtype=np.int32)
    position[reference] = np.arange(count, dtype=np.int32)
    correct = np.asarray(
        [
            edge_is_reference_correct(constraint.edge, reference, grid=grid)
            for constraint in trace.constraints
        ],
        dtype=bool,
    )
    added = np.asarray(
        [constraint.status == "added" for constraint in trace.constraints],
        dtype=bool,
    )
    fully_exact_tiles = 0
    mode_support_total = 0
    correct_pairs = 0
    total_pairs = 0
    component_purities: list[float] = []
    for component in trace.components:
        shifts = Counter(
            (
                int(position[tile] // grid) - relative_row,
                int(position[tile] % grid) - relative_column,
            )
            for tile, (relative_row, relative_column) in component.items()
        )
        supports = tuple(shifts.values())
        mode_support = max(supports)
        size = len(component)
        mode_support_total += mode_support
        component_purities.append(mode_support / size)
        if mode_support == size:
            fully_exact_tiles += size
        correct_pairs += sum(support * (support - 1) // 2 for support in supports)
        total_pairs += size * (size - 1) // 2
    largest_size = len(trace.components[0])
    return {
        "selected_edges": len(trace.constraints),
        "correct_selected_edges": int(correct.sum()),
        "selected_edge_precision": float(correct.mean()),
        "added_constraints": int(added.sum()),
        "correct_added_constraints": int(np.count_nonzero(added & correct)),
        "false_added_bridges": int(np.count_nonzero(added & ~correct)),
        "added_constraint_precision": float(correct[added].mean()) if added.any() else 0.0,
        "component_count": len(trace.components),
        "largest_component": largest_size,
        "largest_component_translation_purity": component_purities[0],
        "tile_weighted_translation_purity": mode_support_total / count,
        "pairwise_relative_accuracy": correct_pairs / total_pairs if total_pairs else 1.0,
        "fully_exact_component_tiles": fully_exact_tiles,
        "rigid_component_tiles": sum(len(value) for value in trace.components if len(value) > 1),
    }


def edge_set_overlap(
    first: ComponentBuildTrace,
    second: ComponentBuildTrace,
) -> dict[str, float | int]:
    """Describe component-budget membership and greedy ordering overlap."""

    def key(constraint: ComponentConstraintTrace) -> tuple[str, int, int]:
        edge = constraint.edge
        return edge.axis, edge.source, edge.target

    first_keys = tuple(map(key, first.constraints))
    second_keys = tuple(map(key, second.constraints))
    first_set = set(first_keys)
    second_set = set(second_keys)
    intersection = first_set & second_set
    same_position = sum(left == right for left, right in zip(first_keys, second_keys, strict=True))
    prefix = 0
    for left, right in zip(first_keys, second_keys, strict=True):
        if left != right:
            break
        prefix += 1
    return {
        "edge_count_each": len(first_keys),
        "membership_intersection": len(intersection),
        "membership_jaccard": len(intersection) / len(first_set | second_set),
        "same_order_position_count": same_position,
        "identical_prefix_length": prefix,
    }


__all__ = [
    "ComponentBuildTrace",
    "ComponentConstraintTrace",
    "build_component_trace",
    "calibrated_priority_matrices",
    "edge_is_reference_correct",
    "edge_set_overlap",
    "exact_component_metrics",
]
