"""Precision-first decoder variant for SocketGlue partial assignments.

The default decoder deliberately remains unchanged.  This variant admits an
edge only when dirty-visible evidence says that the hard OT projection chose a
clear real match over row, column, and dustbin alternatives.  Components are
then capped during incremental construction so one weak bridge cannot create a
large percolated island.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle import socket_decoder
from aiijc_puzzle.socket_decoder import (
    SocketEdge,
    hard_partial_axis_matching,
    socket_border_unary,
    socket_layout_objective,
)


@dataclass(frozen=True)
class PrecisionFirstDecoderConfig:
    """One frozen precision-first edge policy and component cap."""

    minimum_edge_confidence: float = -1.0
    minimum_real_row_margin: float = 0.0
    minimum_real_column_margin: float = 0.0
    minimum_dustbin_margin: float = 0.5
    maximum_component_size: int = 8
    border_weight: float = 0.20

    def validate(self, *, tile_count: int) -> None:
        for name, value in (
            ("minimum_edge_confidence", self.minimum_edge_confidence),
            ("minimum_real_row_margin", self.minimum_real_row_margin),
            ("minimum_real_column_margin", self.minimum_real_column_margin),
            ("minimum_dustbin_margin", self.minimum_dustbin_margin),
            ("border_weight", self.border_weight),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 2 <= self.maximum_component_size <= tile_count:
            raise ValueError(f"maximum_component_size must be in [2, {tile_count}]")
        if self.border_weight < 0:
            raise ValueError("border_weight must be non-negative")


@dataclass(frozen=True)
class PrecisionEdgeEvidence:
    """Dirty-only confidence features for one globally projected socket edge."""

    edge: SocketEdge
    real_row_margin: float
    real_column_margin: float
    dustbin_margin: float
    eligible: bool


@dataclass(frozen=True)
class PrecisionFirstDecoderDiagnostics:
    """JSON-ready evidence for one precision-first decode."""

    grid_size: int
    tile_count: int
    hard_edges_per_axis: int
    minimum_edge_confidence: float
    minimum_real_row_margin: float
    minimum_real_column_margin: float
    minimum_dustbin_margin: float
    maximum_component_size: int
    right_eligible_edges: int
    down_eligible_edges: int
    attempted_constraints: int
    added_constraints: int
    consistent_redundant_constraints: int
    contradiction_rejections: int
    collision_rejections: int
    span_rejections: int
    size_cap_rejections: int
    component_count: int
    largest_component: int
    component_sizes: tuple[int, ...]
    rigid_tiles_packed: int
    objective: float
    strict_permutation: bool
    runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrecisionFirstDecodeResult:
    """Strict layout, retained components, and audit diagnostics."""

    layout: np.ndarray
    components: tuple[dict[int, tuple[int, int]], ...]
    selected_edges: tuple[PrecisionEdgeEvidence, ...]
    diagnostics: PrecisionFirstDecoderDiagnostics

    def report(self, *, include_layout: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decoder": "socket-precision-first-components-v1",
            "diagnostics": self.diagnostics.as_dict(),
        }
        if include_layout:
            payload["tile_at_position"] = self.layout.tolist()
        return payload


def _assignment_array(value: Any, *, grid: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    expected = (grid * grid + 1, grid * grid + 1)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
    usable = result.copy()
    usable[-1, -1] = 0.0
    if not np.isfinite(usable).all():
        raise ValueError(f"{name} contains non-finite usable entries")
    return np.ascontiguousarray(result)


def _maximum_real_alternative(
    matrix: np.ndarray,
    *,
    source: int,
    target: int,
    outgoing: bool,
) -> float:
    count = matrix.shape[0] - 1
    candidates = (
        matrix[source, :count].copy()
        if outgoing
        else matrix[:count, target].copy()
    )
    candidates[target if outgoing else source] = -np.inf
    # Self-pairs are forbidden by the SocketMatcher and hard OT projection.
    candidates[source if outgoing else target] = -np.inf
    maximum = float(np.max(candidates))
    if not np.isfinite(maximum):
        raise RuntimeError("socket edge has no finite real alternative")
    return maximum


def precision_edge_evidence(
    matrix: Any,
    edge: SocketEdge,
    *,
    grid: int,
    config: PrecisionFirstDecoderConfig,
) -> PrecisionEdgeEvidence:
    """Compute the frozen dirty-only admission features for one edge."""

    value = _assignment_array(matrix, grid=grid, name=f"{edge.axis}_log_assignment")
    count = grid * grid
    selected = float(value[edge.source, edge.target])
    row_margin = selected - _maximum_real_alternative(
        value,
        source=edge.source,
        target=edge.target,
        outgoing=True,
    )
    column_margin = selected - _maximum_real_alternative(
        value,
        source=edge.source,
        target=edge.target,
        outgoing=False,
    )
    dustbin_margin = min(
        selected - float(value[edge.source, count]),
        selected - float(value[count, edge.target]),
    )
    eligible = bool(
        edge.confidence >= config.minimum_edge_confidence
        and row_margin >= config.minimum_real_row_margin
        and column_margin >= config.minimum_real_column_margin
        and dustbin_margin >= config.minimum_dustbin_margin
    )
    return PrecisionEdgeEvidence(
        edge=edge,
        real_row_margin=row_margin,
        real_column_margin=column_margin,
        dustbin_margin=dustbin_margin,
        eligible=eligible,
    )


def select_precision_edges(
    matrix: Any,
    *,
    grid: int,
    axis: str,
    config: PrecisionFirstDecoderConfig,
) -> tuple[PrecisionEdgeEvidence, ...]:
    """Select an adaptive number of high-margin hard-matching edges."""

    value = _assignment_array(matrix, grid=grid, name=f"{axis}_log_assignment")
    matching = hard_partial_axis_matching(value, grid=grid, axis=axis)
    evidence = tuple(
        precision_edge_evidence(value, edge, grid=grid, config=config)
        for edge in matching.edges
    )
    return tuple(item for item in evidence if item.eligible)


class _CappedTranslationComponents:
    """Decoder-compatible relative-coordinate graph with a hard size cap."""

    def __init__(self, *, count: int, grid: int, maximum_size: int) -> None:
        self.count = count
        self.grid = grid
        self.maximum_size = maximum_size
        self.tile_component = np.full(count, -1, dtype=np.int32)
        self.components: list[dict[int, tuple[int, int]]] = []

    def _span_ok(self, component: dict[int, tuple[int, int]]) -> bool:
        coordinates = np.asarray(tuple(component.values()), dtype=np.int32)
        span = coordinates.max(axis=0) - coordinates.min(axis=0)
        return bool(np.all(span < self.grid))

    def _new(self, edge: SocketEdge) -> str:
        component_id = len(self.components)
        delta = (edge.delta_row, edge.delta_column)
        self.components.append({edge.source: (0, 0), edge.target: delta})
        self.tile_component[edge.source] = component_id
        self.tile_component[edge.target] = component_id
        return "added"

    def add(self, edge: SocketEdge) -> str:
        source_component = int(self.tile_component[edge.source])
        target_component = int(self.tile_component[edge.target])
        delta = (edge.delta_row, edge.delta_column)
        if source_component < 0 and target_component < 0:
            return self._new(edge)

        if source_component >= 0 and target_component < 0:
            component = self.components[source_component]
            if len(component) >= self.maximum_size:
                return "size_cap"
            row, column = component[edge.source]
            coordinate = (row + delta[0], column + delta[1])
            if coordinate in component.values():
                return "collision"
            component[edge.target] = coordinate
            if not self._span_ok(component):
                del component[edge.target]
                return "span"
            self.tile_component[edge.target] = source_component
            return "added"

        if source_component < 0 and target_component >= 0:
            component = self.components[target_component]
            if len(component) >= self.maximum_size:
                return "size_cap"
            row, column = component[edge.target]
            coordinate = (row - delta[0], column - delta[1])
            if coordinate in component.values():
                return "collision"
            component[edge.source] = coordinate
            if not self._span_ok(component):
                del component[edge.source]
                return "span"
            self.tile_component[edge.source] = target_component
            return "added"

        if source_component == target_component:
            component = self.components[source_component]
            source_position = component[edge.source]
            target_position = component[edge.target]
            observed = (
                target_position[0] - source_position[0],
                target_position[1] - source_position[1],
            )
            return "consistent" if observed == delta else "contradiction"

        left = self.components[source_component]
        right = self.components[target_component]
        if len(left) + len(right) > self.maximum_size:
            return "size_cap"
        source_position = left[edge.source]
        target_position = right[edge.target]
        shift = (
            source_position[0] + delta[0] - target_position[0],
            source_position[1] + delta[1] - target_position[1],
        )
        moved = {
            tile: (row + shift[0], column + shift[1])
            for tile, (row, column) in right.items()
        }
        if set(left.values()) & set(moved.values()):
            return "collision"
        merged = {**left, **moved}
        if not self._span_ok(merged):
            return "span"
        left.update(moved)
        self.components[target_component] = {}
        for tile in moved:
            self.tile_component[tile] = source_component
        return "added"

    def complete_components(self) -> tuple[dict[int, tuple[int, int]], ...]:
        result = [component for component in self.components if component]
        result.extend(
            {tile: (0, 0)}
            for tile in range(self.count)
            if self.tile_component[tile] < 0
        )
        return tuple(result)


def _strict_layout(value: Any, *, count: int) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (count,) or not np.array_equal(np.sort(layout), np.arange(count)):
        raise RuntimeError("precision-first decoder did not return a strict permutation")
    return np.ascontiguousarray(layout)


def decode_precision_first(
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int = 24,
    config: PrecisionFirstDecoderConfig | None = None,
) -> PrecisionFirstDecodeResult:
    """Decode only high-margin OT edges while capping component percolation."""

    started = perf_counter()
    count = grid * grid
    config = PrecisionFirstDecoderConfig() if config is None else config
    config.validate(tile_count=count)
    right = _assignment_array(right_log_assignment, grid=grid, name="right_log_assignment")
    down = _assignment_array(down_log_assignment, grid=grid, name="down_log_assignment")
    right_edges = select_precision_edges(right, grid=grid, axis="right", config=config)
    down_edges = select_precision_edges(down, grid=grid, axis="down", config=config)
    selected = tuple(
        sorted(
            right_edges + down_edges,
            key=lambda item: (
                -item.edge.confidence,
                item.edge.axis,
                item.edge.source,
                item.edge.target,
            ),
        )
    )
    builder = _CappedTranslationComponents(
        count=count,
        grid=grid,
        maximum_size=config.maximum_component_size,
    )
    statuses = {
        "added": 0,
        "consistent": 0,
        "contradiction": 0,
        "collision": 0,
        "span": 0,
        "size_cap": 0,
    }
    for item in selected:
        statuses[builder.add(item.edge)] += 1
    components = builder.complete_components()
    component_sizes = tuple(sorted((len(component) for component in components), reverse=True))

    border_unary = socket_border_unary(right, down, grid=grid)
    layout, rigid_tiles_packed = socket_decoder._pack_rigid_components(
        list(components),
        right[:count, :count],
        down[:count, :count],
        border_unary,
        grid=grid,
        border_weight=config.border_weight,
    )
    layout = _strict_layout(layout, count=count)
    objective = socket_layout_objective(
        layout,
        right[:count, :count],
        down[:count, :count],
        border_unary,
        grid=grid,
        border_weight=config.border_weight,
    )
    diagnostics = PrecisionFirstDecoderDiagnostics(
        grid_size=grid,
        tile_count=count,
        hard_edges_per_axis=count - grid,
        minimum_edge_confidence=float(config.minimum_edge_confidence),
        minimum_real_row_margin=float(config.minimum_real_row_margin),
        minimum_real_column_margin=float(config.minimum_real_column_margin),
        minimum_dustbin_margin=float(config.minimum_dustbin_margin),
        maximum_component_size=config.maximum_component_size,
        right_eligible_edges=len(right_edges),
        down_eligible_edges=len(down_edges),
        attempted_constraints=len(selected),
        added_constraints=statuses["added"],
        consistent_redundant_constraints=statuses["consistent"],
        contradiction_rejections=statuses["contradiction"],
        collision_rejections=statuses["collision"],
        span_rejections=statuses["span"],
        size_cap_rejections=statuses["size_cap"],
        component_count=len(components),
        largest_component=component_sizes[0],
        component_sizes=component_sizes,
        rigid_tiles_packed=rigid_tiles_packed,
        objective=float(objective),
        strict_permutation=True,
        runtime_seconds=perf_counter() - started,
    )
    return PrecisionFirstDecodeResult(layout, components, selected, diagnostics)


__all__ = [
    "PrecisionEdgeEvidence",
    "PrecisionFirstDecodeResult",
    "PrecisionFirstDecoderConfig",
    "PrecisionFirstDecoderDiagnostics",
    "decode_precision_first",
    "precision_edge_evidence",
    "select_precision_edges",
]
