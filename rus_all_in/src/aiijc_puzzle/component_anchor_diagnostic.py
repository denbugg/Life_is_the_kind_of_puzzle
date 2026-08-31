"""Target-assisted diagnostics for SocketGlue translation components.

This module deliberately does not propose a new layout heuristic.  It rebuilds
the exact high-confidence components consumed by :mod:`socket_decoder` and,
after predictions have been frozen, measures whether each component is a
correct rigid fragment and whether its absolute translation was chosen well.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.socket_decoder import (
    SocketEdge,
    build_translation_components,
)


@dataclass(frozen=True)
class ComponentConstraint:
    """One decoder constraint and the translation-builder decision for it."""

    edge: SocketEdge
    status: str


@dataclass(frozen=True)
class DecoderComponentBuild:
    """Components reconstructed with the decoder's exact edge order and rules."""

    components: tuple[dict[int, tuple[int, int]], ...]
    constraints: tuple[ComponentConstraint, ...]
    status_counts: dict[str, int]


@dataclass(frozen=True)
class AnchorObservation:
    """One predicted translation relative to a target-assisted true shift."""

    shift: tuple[int, int]
    shift_support: int
    rigidity: float
    row_error: int
    column_error: int
    l1_error: int
    euclidean_error: float
    exact: bool
    within_two_cells: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentTranslationDiagnostic:
    """Internal geometry and absolute-anchor evidence for one component."""

    component_id: int
    size: int
    evidence_size: int
    height: int
    width: int
    true_shift: tuple[int, int]
    true_shift_support: int
    translation_purity: float
    pairwise_relative_accuracy: float
    internally_exact: bool
    true_centroid: tuple[float, float]
    true_centroid_distance_from_board_centre: float
    true_centroid_near_board_centre: bool
    texture_strength: float
    anchors: dict[str, AnchorObservation]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["anchors"] = {
            name: observation.as_dict() for name, observation in self.anchors.items()
        }
        return payload


def _normalise_component(
    component: dict[int, tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    minimum_row = min(row for row, _ in component.values())
    minimum_column = min(column for _, column in component.values())
    return {
        tile: (row - minimum_row, column - minimum_column)
        for tile, (row, column) in component.items()
    }


def rebuild_decoder_components(
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int,
    edge_budget_per_axis: int,
) -> DecoderComponentBuild:
    """Rebuild precisely the components used by ``decode_socket_assignments``.

    Diagnostics delegate to the decoder's public builder so collision, span,
    merge and edge tie-breaking behaviour cannot silently diverge.
    """

    build = build_translation_components(
        right_log_assignment,
        down_log_assignment,
        grid=grid,
        edge_budget_per_axis=edge_budget_per_axis,
    )
    constraints = tuple(
        ComponentConstraint(edge=decision.edge, status=decision.status)
        for decision in build.decisions
    )
    components = tuple(
        _normalise_component(component) for component in build.components
    )
    components = tuple(sorted(components, key=lambda value: (-len(value), min(value))))
    return DecoderComponentBuild(components, constraints, dict(build.status_counts))


def _positions_from_layout(layout: Any, *, grid: int) -> np.ndarray:
    count = grid * grid
    value = np.asarray(layout, dtype=np.int64)
    if value.shape != (count,) or not np.array_equal(np.sort(value), np.arange(count)):
        raise ValueError("layout must be a strict tile-at-position permutation")
    positions = np.empty((count, 2), dtype=np.int32)
    positions[value, 0], positions[value, 1] = divmod(np.arange(count), grid)
    return positions


def _mode_shift(
    component: dict[int, tuple[int, int]],
    positions: np.ndarray,
) -> tuple[tuple[int, int], int]:
    shifts = Counter(
        (
            int(positions[tile, 0]) - relative_row,
            int(positions[tile, 1]) - relative_column,
        )
        for tile, (relative_row, relative_column) in component.items()
    )
    shift, support = min(shifts.items(), key=lambda item: (-item[1], item[0]))
    return shift, support


def _centered_shift(
    component: dict[int, tuple[int, int]],
    *,
    grid: int,
) -> tuple[int, int]:
    coordinates = np.asarray(tuple(component.values()), dtype=np.float64)
    height = int(coordinates[:, 0].max()) + 1
    width = int(coordinates[:, 1].max()) + 1
    centre = 0.5 * (grid - 1)
    best: tuple[float, int, int] | None = None
    for row_shift in range(grid - height + 1):
        for column_shift in range(grid - width + 1):
            row_error = float(coordinates[:, 0].mean() + row_shift - centre)
            column_error = float(coordinates[:, 1].mean() + column_shift - centre)
            candidate = (
                row_error * row_error + column_error * column_error,
                row_shift,
                column_shift,
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("component has no feasible grid translation")
    return best[1], best[2]


def _anchor_observation(
    shift: tuple[int, int],
    support: int,
    *,
    component_size: int,
    true_shift: tuple[int, int],
) -> AnchorObservation:
    row_error = shift[0] - true_shift[0]
    column_error = shift[1] - true_shift[1]
    l1_error = abs(row_error) + abs(column_error)
    return AnchorObservation(
        shift=shift,
        shift_support=support,
        rigidity=support / component_size,
        row_error=row_error,
        column_error=column_error,
        l1_error=l1_error,
        euclidean_error=math.hypot(row_error, column_error),
        exact=l1_error == 0,
        within_two_cells=l1_error <= 2,
    )


def diagnose_component_translation(
    component: dict[int, tuple[int, int]],
    reference_layout: Any,
    predicted_layouts: dict[str, Any],
    *,
    grid: int,
    component_id: int,
    texture_unary: Any | None = None,
    evidence_tiles: Any | None = None,
    centre_radius: float = 4.0,
) -> ComponentTranslationDiagnostic:
    """Measure internal relative geometry and several absolute anchor choices."""

    if not component:
        raise ValueError("component must not be empty")
    if not np.isfinite(centre_radius) or centre_radius < 0:
        raise ValueError("centre_radius must be finite and non-negative")
    component = _normalise_component(component)
    evidence_component = component
    if evidence_tiles is not None:
        evidence = np.asarray(evidence_tiles, dtype=np.int64)
        evidence_component = {
            tile: coordinate for tile, coordinate in component.items() if tile in evidence
        }
        if not evidence_component:
            raise ValueError("evidence_tiles do not intersect the component")
    reference_positions = _positions_from_layout(reference_layout, grid=grid)
    true_shift, true_support = _mode_shift(evidence_component, reference_positions)
    size = len(component)
    shift_counts = Counter(
        (
            int(reference_positions[tile, 0]) - relative_row,
            int(reference_positions[tile, 1]) - relative_column,
        )
        for tile, (relative_row, relative_column) in evidence_component.items()
    )
    correct_pairs = sum(support * (support - 1) // 2 for support in shift_counts.values())
    evidence_size = len(evidence_component)
    total_pairs = evidence_size * (evidence_size - 1) // 2
    pairwise_accuracy = correct_pairs / total_pairs if total_pairs else 1.0

    relative_coordinates = np.asarray(tuple(component.values()), dtype=np.float64)
    true_centroid = (
        float(relative_coordinates[:, 0].mean() + true_shift[0]),
        float(relative_coordinates[:, 1].mean() + true_shift[1]),
    )
    board_centre = 0.5 * (grid - 1)
    centre_distance = math.hypot(
        true_centroid[0] - board_centre,
        true_centroid[1] - board_centre,
    )

    anchors: dict[str, AnchorObservation] = {}
    centre_shift = _centered_shift(component, grid=grid)
    anchors["geometric_centre"] = _anchor_observation(
        centre_shift,
        size,
        component_size=size,
        true_shift=true_shift,
    )
    for name, layout in predicted_layouts.items():
        positions = _positions_from_layout(layout, grid=grid)
        shift, support = _mode_shift(component, positions)
        anchors[name] = _anchor_observation(
            shift,
            support,
            component_size=size,
            true_shift=true_shift,
        )

    texture_strength = 0.0
    if texture_unary is not None:
        unary = np.asarray(texture_unary, dtype=np.float64)
        count = grid * grid
        if unary.shape != (count, count) or not np.isfinite(unary).all():
            raise ValueError(f"texture_unary must have finite shape {(count, count)}")
        tile_rows = unary[np.asarray(tuple(component), dtype=np.int64)]
        texture_strength = float(np.mean(np.ptp(tile_rows, axis=1)))

    coordinates = np.asarray(tuple(component.values()), dtype=np.int32)
    return ComponentTranslationDiagnostic(
        component_id=component_id,
        size=size,
        evidence_size=evidence_size,
        height=int(coordinates[:, 0].max()) + 1,
        width=int(coordinates[:, 1].max()) + 1,
        true_shift=true_shift,
        true_shift_support=true_support,
        translation_purity=true_support / evidence_size,
        pairwise_relative_accuracy=pairwise_accuracy,
        internally_exact=true_support == evidence_size,
        true_centroid=true_centroid,
        true_centroid_distance_from_board_centre=centre_distance,
        true_centroid_near_board_centre=centre_distance <= centre_radius,
        texture_strength=texture_strength,
        anchors=anchors,
    )


def constraint_is_reference_correct(
    constraint: ComponentConstraint,
    reference_layout: Any,
    *,
    grid: int,
) -> bool:
    """Return whether one proposed socket edge has its exact true displacement."""

    positions = _positions_from_layout(reference_layout, grid=grid)
    edge = constraint.edge
    observed = positions[edge.target] - positions[edge.source]
    return bool(
        observed[0] == edge.delta_row and observed[1] == edge.delta_column
    )


__all__ = [
    "AnchorObservation",
    "ComponentConstraint",
    "ComponentTranslationDiagnostic",
    "DecoderComponentBuild",
    "constraint_is_reference_correct",
    "diagnose_component_translation",
    "rebuild_decoder_components",
]
