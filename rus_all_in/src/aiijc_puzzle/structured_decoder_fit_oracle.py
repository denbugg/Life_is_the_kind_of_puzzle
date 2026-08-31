"""FIT-only capacity accounting for a compatibility-aware decoder.

This module deliberately contains no image, matcher, model, source-name or
checkpoint access.  It consumes a frozen strict control layout and a frozen
fixed-coverage reciprocal edge head, then uses an exact organizer-train
reference only for an explicitly target-assisted capacity ceiling.

The bounded first action family is an edge-implied rigid component merge/edit.
Oracle-correct relations already realised by the current state define rigid
components.  For a proposed ``source -> target`` relation, either endpoint
component may be translated to satisfy that socket; directly displaced tiles
bijectively fill the vacated cells.  ``STOP`` is always available.  An edit is
accepted only when it strictly increases the number of realised supplied true
relations and does not reduce the satisfied-pair count.  Thus every returned
layout is a strict permutation and the ceiling cannot hide pair losses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, order=True)
class DirectedEdge:
    """One frozen directed right/down relation over tile-bag identities."""

    axis: int
    source: int
    target: int
    confidence: float

    def __post_init__(self) -> None:
        if self.axis not in (0, 1):
            raise ValueError("axis must be 0 (right) or 1 (down)")
        if isinstance(self.source, bool) or not isinstance(self.source, int):
            raise ValueError("source must be an integer")
        if isinstance(self.target, bool) or not isinstance(self.target, int):
            raise ValueError("target must be an integer")
        if self.source == self.target:
            raise ValueError("self edges are forbidden")
        if not np.isfinite(self.confidence):
            raise ValueError("confidence must be finite")


@dataclass(frozen=True)
class LayoutMetrics:
    """Absolute and relative layout diagnostics against one FIT reference."""

    satisfied_pairs: int
    exact_tiles: int
    mean_absolute_manhattan: float
    radius2_recall: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class OracleAction:
    """One accepted pair-safe edge-guided edit."""

    step: int
    action_type: str
    axis: int
    source: int
    target: int
    confidence: float
    moved_component_size: int
    row_shift: int
    column_shift: int
    pair_delta: int
    supplied_true_edge_delta: int
    exact_delta: int
    manhattan_delta: float
    radius2_delta: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredDecoderOracleResult:
    """Deterministic target-assisted ceiling for one frozen FIT board."""

    selected_edge_count: int
    selected_true_edge_count: int
    initially_realised_selected_true_edges: int
    missing_selected_true_edge_count: int
    compatible_missing_true_edge_headroom: int
    initial_pair_safe_action_count: int
    accepted_action_count: int
    realised_supplied_true_edge_gain: int
    control: LayoutMetrics
    ceiling: LayoutMetrics
    actions: tuple[OracleAction, ...]
    ceiling_layout: np.ndarray

    def __post_init__(self) -> None:
        layout = np.ascontiguousarray(self.ceiling_layout, dtype=np.int32)
        layout.setflags(write=False)
        object.__setattr__(self, "ceiling_layout", layout)

    @property
    def pair_delta(self) -> int:
        return self.ceiling.satisfied_pairs - self.control.satisfied_pairs

    @property
    def exact_delta(self) -> int:
        return self.ceiling.exact_tiles - self.control.exact_tiles

    @property
    def manhattan_delta(self) -> float:
        return (
            self.ceiling.mean_absolute_manhattan
            - self.control.mean_absolute_manhattan
        )

    @property
    def radius2_delta(self) -> float:
        return self.ceiling.radius2_recall - self.control.radius2_recall

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_edge_count": self.selected_edge_count,
            "selected_true_edge_count": self.selected_true_edge_count,
            "initially_realised_selected_true_edges": (
                self.initially_realised_selected_true_edges
            ),
            "missing_selected_true_edge_count": self.missing_selected_true_edge_count,
            "compatible_missing_true_edge_headroom": (
                self.compatible_missing_true_edge_headroom
            ),
            "initial_pair_safe_action_count": self.initial_pair_safe_action_count,
            "accepted_action_count": self.accepted_action_count,
            "realised_supplied_true_edge_gain": self.realised_supplied_true_edge_gain,
            "control": self.control.as_dict(),
            "ceiling": self.ceiling.as_dict(),
            "delta": {
                "satisfied_pairs": self.pair_delta,
                "exact_tiles": self.exact_delta,
                "mean_absolute_manhattan": self.manhattan_delta,
                "radius2_recall": self.radius2_delta,
            },
            "actions": [action.as_dict() for action in self.actions],
        }


def strict_layout(value: Any, *, grid: int, name: str = "layout") -> np.ndarray:
    """Return one contiguous strict permutation of ``grid**2`` tile ids."""

    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    count = grid * grid
    layout = np.asarray(value)
    if layout.shape != (count,) or not np.issubdtype(layout.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array with shape {(count,)}")
    result = np.ascontiguousarray(layout, dtype=np.int32)
    if not np.array_equal(np.sort(result), np.arange(count, dtype=np.int32)):
        raise ValueError(f"{name} must contain every tile identity exactly once")
    return result


def validate_fixed_reciprocal_head(
    edges: tuple[DirectedEdge, ...] | list[DirectedEdge],
    *,
    grid: int,
    requested_per_axis: int,
) -> tuple[DirectedEdge, ...]:
    """Fail closed on coverage, identity range and reciprocal uniqueness."""

    count = grid * grid
    if (
        isinstance(requested_per_axis, bool)
        or not isinstance(requested_per_axis, int)
        or requested_per_axis <= 0
    ):
        raise ValueError("requested_per_axis must be a positive integer")
    frozen = tuple(edges)
    if len(frozen) != 2 * requested_per_axis:
        raise ValueError("fixed reciprocal head has the wrong total selected count")
    if len({(edge.axis, edge.source, edge.target) for edge in frozen}) != len(frozen):
        raise ValueError("fixed reciprocal head contains duplicate directed edges")
    for axis in (0, 1):
        directional = [edge for edge in frozen if edge.axis == axis]
        if len(directional) != requested_per_axis:
            raise ValueError("fixed reciprocal head has the wrong directional count")
        sources = [edge.source for edge in directional]
        targets = [edge.target for edge in directional]
        if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
            raise ValueError("fixed reciprocal head violates source/target uniqueness")
        if any(
            edge.source < 0
            or edge.source >= count
            or edge.target < 0
            or edge.target >= count
            for edge in directional
        ):
            raise ValueError("fixed reciprocal head contains an out-of-range tile id")
    return tuple(sorted(frozen, key=lambda edge: (edge.axis, edge.source, edge.target)))


def _positions(layout: np.ndarray) -> np.ndarray:
    result = np.empty(len(layout), dtype=np.int32)
    result[layout] = np.arange(len(layout), dtype=np.int32)
    return result


def true_edge_set(reference: Any, *, grid: int) -> frozenset[tuple[int, int, int]]:
    """Return all exact directed right/down relations of a strict reference."""

    layout = strict_layout(reference, grid=grid, name="reference")
    board = layout.reshape(grid, grid)
    values = {
        (0, int(board[row, column]), int(board[row, column + 1]))
        for row in range(grid)
        for column in range(grid - 1)
    }
    values.update(
        (1, int(board[row, column]), int(board[row + 1, column]))
        for row in range(grid - 1)
        for column in range(grid)
    )
    return frozenset(values)


def realised_edge_set(layout: Any, *, grid: int) -> frozenset[tuple[int, int, int]]:
    """Return every directed geometric contact realised by one strict layout."""

    return true_edge_set(layout, grid=grid)


def layout_metrics(layout: Any, reference: Any, *, grid: int) -> LayoutMetrics:
    """Measure pairs, exact, absolute Manhattan and radius-2 recall."""

    candidate = strict_layout(layout, grid=grid, name="layout")
    target = strict_layout(reference, grid=grid, name="reference")
    candidate_positions = _positions(candidate)
    target_positions = _positions(target)
    delta = np.abs(candidate_positions // grid - target_positions // grid)
    delta += np.abs(candidate_positions % grid - target_positions % grid)
    pairs = len(realised_edge_set(candidate, grid=grid) & true_edge_set(target, grid=grid))
    return LayoutMetrics(
        satisfied_pairs=int(pairs),
        exact_tiles=int(np.count_nonzero(candidate == target)),
        mean_absolute_manhattan=float(np.mean(delta)),
        radius2_recall=float(np.mean(delta <= 2)),
    )


def _edge_is_realised(
    edge: DirectedEdge,
    positions: np.ndarray,
    *,
    grid: int,
) -> bool:
    source_position = int(positions[edge.source])
    target_position = int(positions[edge.target])
    if edge.axis == 0:
        return (
            source_position % grid < grid - 1
            and target_position == source_position + 1
        )
    return source_position // grid < grid - 1 and target_position == source_position + grid


def _translate_component_with_local_fill(
    layout: np.ndarray,
    component: tuple[int, ...],
    row_shift: int,
    column_shift: int,
    *,
    grid: int,
) -> np.ndarray | None:
    """Rigidly translate one component and bijectively fill vacated cells."""

    if not component or (row_shift == 0 and column_shift == 0):
        return None
    positions = _positions(layout)
    tiles = np.asarray(component, dtype=np.int32)
    old_positions = positions[tiles]
    old_rows, old_columns = divmod(old_positions, grid)
    new_rows = old_rows + row_shift
    new_columns = old_columns + column_shift
    if (
        np.any(new_rows < 0)
        or np.any(new_rows >= grid)
        or np.any(new_columns < 0)
        or np.any(new_columns >= grid)
    ):
        return None
    new_positions = new_rows * grid + new_columns
    old_set = set(int(value) for value in old_positions)
    new_set = set(int(value) for value in new_positions)
    if len(new_set) != len(component):
        raise RuntimeError("rigid component translation produced an internal collision")
    vacated = sorted(old_set - new_set)
    entered = sorted(new_set - old_set)
    displaced = [int(layout[position]) for position in entered]
    result = layout.copy()
    for tile, destination in zip(tiles, new_positions, strict=True):
        result[int(destination)] = int(tile)
    for destination, tile in zip(vacated, displaced, strict=True):
        result[destination] = tile
    return strict_layout(result, grid=grid, name="rigid_component_edit")


def _true_realised_components(
    layout: np.ndarray,
    truth: frozenset[tuple[int, int, int]],
    *,
    grid: int,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Partition tiles by exact relations already realised in the state."""

    count = len(layout)
    parent = np.arange(count, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            previous = int(parent[value])
            parent[value] = root
            value = previous
        return root

    def union(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a != b:
            if a > b:
                a, b = b, a
            parent[b] = a

    for axis, source, target in realised_edge_set(layout, grid=grid) & truth:
        del axis
        union(source, target)
    groups: dict[int, list[int]] = {}
    for tile in range(count):
        groups.setdefault(find(tile), []).append(tile)
    components = tuple(
        tuple(group) for group in sorted(groups.values(), key=lambda value: value[0])
    )
    component_of = np.empty(count, dtype=np.int32)
    for index, component in enumerate(components):
        component_of[np.asarray(component, dtype=np.int32)] = index
    return components, component_of


@dataclass(frozen=True)
class _CandidateAction:
    action_type: str
    edge: DirectedEdge
    component_size: int
    row_shift: int
    column_shift: int
    layout: np.ndarray
    metrics: LayoutMetrics
    supplied_true_gain: int


def _edge_component_edits(
    layout: np.ndarray,
    edge: DirectedEdge,
    components: tuple[tuple[int, ...], ...],
    component_of: np.ndarray,
    *,
    grid: int,
) -> list[tuple[str, tuple[int, ...], int, int, np.ndarray]]:
    """Enumerate the two edge-implied rigid component translations."""

    source_component = int(component_of[edge.source])
    target_component = int(component_of[edge.target])
    if source_component == target_component:
        return []
    positions = _positions(layout)
    source_position = int(positions[edge.source])
    target_position = int(positions[edge.target])
    source_row, source_column = divmod(source_position, grid)
    target_row, target_column = divmod(target_position, grid)
    delta_row, delta_column = ((0, 1) if edge.axis == 0 else (1, 0))
    proposals = []
    # Hold source component fixed; move target component to the requested socket.
    target_shift = (
        source_row + delta_row - target_row,
        source_column + delta_column - target_column,
    )
    moved_target = _translate_component_with_local_fill(
        layout,
        components[target_component],
        target_shift[0],
        target_shift[1],
        grid=grid,
    )
    if moved_target is not None:
        proposals.append(
            (
                "translate-target-component",
                components[target_component],
                target_shift[0],
                target_shift[1],
                moved_target,
            )
        )
    # Hold target component fixed; move source component to its predecessor socket.
    source_shift = (
        target_row - delta_row - source_row,
        target_column - delta_column - source_column,
    )
    moved_source = _translate_component_with_local_fill(
        layout,
        components[source_component],
        source_shift[0],
        source_shift[1],
        grid=grid,
    )
    if moved_source is not None:
        proposals.append(
            (
                "translate-source-component",
                components[source_component],
                source_shift[0],
                source_shift[1],
                moved_source,
            )
        )
    # Exact duplicate layouts can arise from two singleton components.
    unique: dict[bytes, tuple[str, tuple[int, ...], int, int, np.ndarray]] = {}
    for proposal in proposals:
        key = proposal[-1].astype("<i4", copy=False).tobytes()
        unique.setdefault(key, proposal)
    return list(unique.values())


def _realised_supplied_true_count(
    positions: np.ndarray,
    supplied_true: tuple[DirectedEdge, ...],
    *,
    grid: int,
) -> int:
    return sum(
        int(_edge_is_realised(edge, positions, grid=grid)) for edge in supplied_true
    )


def _pair_safe_candidates(
    layout: np.ndarray,
    reference: np.ndarray,
    supplied_true: tuple[DirectedEdge, ...],
    *,
    grid: int,
) -> list[_CandidateAction]:
    truth = true_edge_set(reference, grid=grid)
    current_positions = _positions(layout)
    current_metrics = layout_metrics(layout, reference, grid=grid)
    current_realised = _realised_supplied_true_count(
        current_positions, supplied_true, grid=grid
    )
    components, component_of = _true_realised_components(
        layout, truth, grid=grid
    )
    candidates: list[_CandidateAction] = []
    for edge in supplied_true:
        if _edge_is_realised(edge, current_positions, grid=grid):
            continue
        for action_type, component, row_shift, column_shift, edited in (
            _edge_component_edits(
                layout, edge, components, component_of, grid=grid
            )
        ):
            metrics = layout_metrics(edited, reference, grid=grid)
            realised = _realised_supplied_true_count(
                _positions(edited), supplied_true, grid=grid
            )
            true_gain = realised - current_realised
            if (
                metrics.satisfied_pairs >= current_metrics.satisfied_pairs
                and true_gain > 0
            ):
                candidates.append(
                    _CandidateAction(
                        action_type=action_type,
                        edge=edge,
                        component_size=len(component),
                        row_shift=row_shift,
                        column_shift=column_shift,
                        layout=edited,
                        metrics=metrics,
                        supplied_true_gain=true_gain,
                    )
                )
    return candidates


def evaluate_pair_safe_oracle(
    control_layout: Any,
    exact_reference: Any,
    selected_edges: tuple[DirectedEdge, ...] | list[DirectedEdge],
    *,
    grid: int = 24,
    requested_per_axis: int = 29,
) -> StructuredDecoderOracleResult:
    """Evaluate one fixed-head, pair-safe, target-assisted FIT ceiling."""

    control = strict_layout(control_layout, grid=grid, name="control_layout")
    reference = strict_layout(exact_reference, grid=grid, name="exact_reference")
    head = validate_fixed_reciprocal_head(
        selected_edges, grid=grid, requested_per_axis=requested_per_axis
    )
    truth = true_edge_set(reference, grid=grid)
    supplied_true = tuple(
        edge for edge in head if (edge.axis, edge.source, edge.target) in truth
    )
    control_positions = _positions(control)
    initially_realised = _realised_supplied_true_count(
        control_positions, supplied_true, grid=grid
    )
    missing_count = len(supplied_true) - initially_realised
    initial_components, initial_component_of = _true_realised_components(
        control, truth, grid=grid
    )
    compatible_headroom = 0
    for edge in supplied_true:
        if _edge_is_realised(edge, control_positions, grid=grid):
            continue
        proposals = _edge_component_edits(
            control,
            edge,
            initial_components,
            initial_component_of,
            grid=grid,
        )
        if any(
            _realised_supplied_true_count(
                _positions(proposal[-1]), supplied_true, grid=grid
            )
            > initially_realised
            for proposal in proposals
        ):
            compatible_headroom += 1
    control_metrics = layout_metrics(control, reference, grid=grid)
    initial_candidates = _pair_safe_candidates(
        control, reference, supplied_true, grid=grid
    )

    current = control.copy()
    current_metrics = control_metrics
    actions: list[OracleAction] = []
    while True:
        candidates = _pair_safe_candidates(
            current, reference, supplied_true, grid=grid
        )
        if not candidates:
            break
        current_positions = _positions(current)
        selected = max(
            candidates,
            key=lambda item: (
                item.supplied_true_gain,
                item.metrics.satisfied_pairs - current_metrics.satisfied_pairs,
                item.metrics.exact_tiles - current_metrics.exact_tiles,
                current_metrics.mean_absolute_manhattan
                - item.metrics.mean_absolute_manhattan,
                item.metrics.radius2_recall - current_metrics.radius2_recall,
                item.edge.confidence,
                -item.component_size,
                int(item.action_type == "translate-target-component"),
                -item.edge.axis,
                -int(current_positions[item.edge.source]),
                -int(current_positions[item.edge.target]),
                -item.row_shift,
                -item.column_shift,
            ),
        )
        edge = selected.edge
        edited = selected.layout
        metrics = selected.metrics
        true_gain = selected.supplied_true_gain
        actions.append(
            OracleAction(
                step=len(actions) + 1,
                action_type=selected.action_type,
                axis=edge.axis,
                source=edge.source,
                target=edge.target,
                confidence=edge.confidence,
                moved_component_size=selected.component_size,
                row_shift=selected.row_shift,
                column_shift=selected.column_shift,
                pair_delta=metrics.satisfied_pairs - current_metrics.satisfied_pairs,
                supplied_true_edge_delta=true_gain,
                exact_delta=metrics.exact_tiles - current_metrics.exact_tiles,
                manhattan_delta=(
                    metrics.mean_absolute_manhattan
                    - current_metrics.mean_absolute_manhattan
                ),
                radius2_delta=metrics.radius2_recall - current_metrics.radius2_recall,
            )
        )
        current = edited
        current_metrics = metrics
    final_positions = _positions(current)
    final_realised = _realised_supplied_true_count(
        final_positions, supplied_true, grid=grid
    )
    if current_metrics.satisfied_pairs < control_metrics.satisfied_pairs:
        raise RuntimeError("pair-safe oracle returned a pair-negative layout")
    strict_layout(current, grid=grid, name="ceiling_layout")
    return StructuredDecoderOracleResult(
        selected_edge_count=len(head),
        selected_true_edge_count=len(supplied_true),
        initially_realised_selected_true_edges=initially_realised,
        missing_selected_true_edge_count=missing_count,
        compatible_missing_true_edge_headroom=compatible_headroom,
        initial_pair_safe_action_count=len(initial_candidates),
        accepted_action_count=len(actions),
        realised_supplied_true_edge_gain=final_realised - initially_realised,
        control=control_metrics,
        ceiling=current_metrics,
        actions=tuple(actions),
        ceiling_layout=current,
    )


__all__ = [
    "DirectedEdge",
    "LayoutMetrics",
    "OracleAction",
    "StructuredDecoderOracleResult",
    "evaluate_pair_safe_oracle",
    "layout_metrics",
    "realised_edge_set",
    "strict_layout",
    "true_edge_set",
    "validate_fixed_reciprocal_head",
]
