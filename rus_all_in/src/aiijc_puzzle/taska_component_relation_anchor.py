"""One conservative post-tail component anchor from cross-component votes.

The confirmed TASKA six-arm layout already contains sizeable rigid islands,
but its absolute gauge is usually wrong.  This module does not rebuild those
islands and does not apply a whole-board roll.  Instead it:

1. forms components from focal-kept candidate edges already realised by the
   supplied strict layout;
2. lets every remaining selected-supply edge vote for the one rigid shift
   that would attach either of its endpoint components to the other endpoint;
3. evaluates only those relation-implied shifts, moving exactly one component
   and locally relocating the directly displaced tiles; and
4. accepts the highest-vote hypothesis only when the original all-1104 TASKA
   seam cost strictly improves, otherwise returning the input bit-for-bit.

The API has no target, clean image, filename, source coordinate, or tile-ID
feature.  Tile identities are opaque permutation indices, and every output is
an upright one-to-one permutation of the original 576 fragments.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost


@dataclass(frozen=True)
class ComponentRelationAnchorDiagnostics:
    """Auditable target-blind decision evidence."""

    component_count: int
    nontrivial_component_count: int
    realised_focal_kept_edge_count: int
    relation_hypothesis_count: int
    cost_improving_hypothesis_count: int
    changed: bool
    selected_component_index: int | None
    selected_component_size: int
    selected_row_shift: int
    selected_column_shift: int
    selected_vote_support: int
    selected_vote_weight: float
    baseline_total_cost: float
    selected_total_cost: float


@dataclass(frozen=True)
class ComponentRelationAnchorResult:
    """One strict layout and its target-blind anchor diagnostics."""

    layout: np.ndarray
    diagnostics: ComponentRelationAnchorDiagnostics


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if int(self.size[a]) < int(self.size[b]):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _grid_count(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _strict_layout(value: Any, *, grid: int) -> np.ndarray:
    count = _grid_count(grid)
    raw = np.asarray(value)
    if raw.shape != (count,) or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("layout must be a one-dimensional integer grid permutation")
    layout = np.ascontiguousarray(raw, dtype=np.int32)
    if not np.array_equal(np.sort(layout), np.arange(count, dtype=np.int32)):
        raise ValueError("layout must use every original tile exactly once")
    return layout


def _edges_and_logits(
    edges: Sequence[RawTailEdge],
    logits: Any,
    *,
    count: int,
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    result = tuple(edges)
    if len(set(result)) != len(result):
        raise ValueError("candidate_edges contain duplicates")
    for edge in result:
        if not isinstance(edge, RawTailEdge):
            raise TypeError("candidate_edges must contain RawTailEdge values")
        if not 0 <= edge.source < count or not 0 <= edge.target < count:
            raise ValueError("candidate edge tile id is outside the input bag")
    values = np.ascontiguousarray(logits, dtype=np.float64)
    if values.shape != (len(result),) or not np.isfinite(values).all():
        raise ValueError("focal_logits must contain one finite value per edge")
    return result, values


def _realised(layout: np.ndarray, edge: RawTailEdge, *, grid: int) -> bool:
    position = np.empty(len(layout), dtype=np.int32)
    position[layout] = np.arange(len(layout), dtype=np.int32)
    source_position = int(position[edge.source])
    target_position = int(position[edge.target])
    source_row, source_column = divmod(source_position, grid)
    target_row, target_column = divmod(target_position, grid)
    if edge.axis == "right":
        return target_row == source_row and target_column == source_column + 1
    return target_column == source_column and target_row == source_row + 1


def build_realised_focal_components(
    layout: Any,
    candidate_edges: Sequence[RawTailEdge],
    focal_logits: Any,
    *,
    grid: int = 24,
    focal_threshold: float = 0.0,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Partition tiles using realised edges at or above the frozen threshold."""

    strict = _strict_layout(layout, grid=grid)
    count = len(strict)
    edges, logits = _edges_and_logits(candidate_edges, focal_logits, count=count)
    if not np.isfinite(focal_threshold):
        raise ValueError("focal_threshold must be finite")
    graph = _DisjointSet(count)
    realised = 0
    for edge, logit in zip(edges, logits, strict=True):
        if float(logit) >= focal_threshold and _realised(strict, edge, grid=grid):
            graph.union(edge.source, edge.target)
            realised += 1
    groups: dict[int, list[int]] = defaultdict(list)
    for tile in range(count):
        groups[graph.find(tile)].append(tile)
    components = tuple(
        tuple(group)
        for group in sorted(groups.values(), key=lambda group: (-len(group), group[0]))
    )
    return components, realised


def translate_component_with_local_fill(
    layout: Any,
    component: Sequence[int],
    row_shift: int,
    column_shift: int,
    *,
    grid: int = 24,
) -> np.ndarray:
    """Move one rigid component and relocate only the directly displaced tiles."""

    strict = _strict_layout(layout, grid=grid)
    count = len(strict)
    tiles = np.asarray(tuple(component))
    if tiles.ndim != 1 or len(tiles) < 2 or tiles.dtype.kind not in {"i", "u"}:
        raise ValueError("component must contain at least two integer tile ids")
    tiles = np.ascontiguousarray(tiles, dtype=np.int32)
    if (
        len(np.unique(tiles)) != len(tiles)
        or np.any(tiles < 0)
        or np.any(tiles >= count)
    ):
        raise ValueError("component tile ids must be unique and inside the input bag")
    if isinstance(row_shift, bool) or not isinstance(row_shift, int):
        raise ValueError("row_shift must be an integer")
    if isinstance(column_shift, bool) or not isinstance(column_shift, int):
        raise ValueError("column_shift must be an integer")
    position = np.empty(count, dtype=np.int32)
    position[strict] = np.arange(count, dtype=np.int32)
    old_positions = position[tiles]
    old_rows, old_columns = divmod(old_positions, grid)
    new_rows = old_rows + row_shift
    new_columns = old_columns + column_shift
    if (
        np.any(new_rows < 0)
        or np.any(new_rows >= grid)
        or np.any(new_columns < 0)
        or np.any(new_columns >= grid)
    ):
        raise ValueError("component shift is outside the board")
    new_positions = new_rows * grid + new_columns
    old_set = set(int(value) for value in old_positions)
    new_set = set(int(value) for value in new_positions)
    if len(new_set) != len(tiles):
        raise RuntimeError("rigid translation produced an internal collision")
    vacated = sorted(old_set - new_set)
    entered = sorted(new_set - old_set)
    displaced = [int(strict[position]) for position in entered]
    result = strict.copy()
    for tile, destination in zip(tiles, new_positions, strict=True):
        result[int(destination)] = int(tile)
    for destination, tile in zip(vacated, displaced, strict=True):
        result[destination] = tile
    return _strict_layout(result, grid=grid)


def _component_bounds(
    position: np.ndarray,
    component: Sequence[int],
    *,
    grid: int,
) -> tuple[int, int, int, int]:
    rows, columns = divmod(position[np.asarray(component, dtype=np.int32)], grid)
    return int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max())


def _feasible(
    shift: tuple[int, int],
    bounds: tuple[int, int, int, int],
    *,
    grid: int,
) -> bool:
    row_shift, column_shift = shift
    minimum_row, maximum_row, minimum_column, maximum_column = bounds
    return (
        -minimum_row <= row_shift < grid - maximum_row
        and -minimum_column <= column_shift < grid - maximum_column
    )


def _softplus_sum(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.logaddexp(0.0, array).sum(dtype=np.float64))


def anchor_one_component_from_relation_votes(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    focal_logits: Any,
    *,
    grid: int = 24,
    focal_threshold: float = 0.0,
    minimum_cost_gain: float = 0.0,
) -> ComponentRelationAnchorResult:
    """Apply the single frozen relation-vote + all-bond-guard anchor rule."""

    strict = _strict_layout(layout, grid=grid)
    count = len(strict)
    edges, logits = _edges_and_logits(candidate_edges, focal_logits, count=count)
    if not np.isfinite(minimum_cost_gain) or minimum_cost_gain < 0:
        raise ValueError("minimum_cost_gain must be finite and non-negative")
    components, realised_count = build_realised_focal_components(
        strict,
        edges,
        logits,
        grid=grid,
        focal_threshold=focal_threshold,
    )
    component_of = np.empty(count, dtype=np.int32)
    for component_index, component in enumerate(components):
        component_of[np.asarray(component, dtype=np.int32)] = component_index
    position = np.empty(count, dtype=np.int32)
    position[strict] = np.arange(count, dtype=np.int32)
    rows, columns = divmod(position, grid)
    votes: list[dict[tuple[int, int], list[float]]] = [
        defaultdict(list) for _ in components
    ]
    for edge, logit in zip(edges, logits, strict=True):
        source_component = int(component_of[edge.source])
        target_component = int(component_of[edge.target])
        if source_component == target_component:
            continue
        delta_row, delta_column = (0, 1) if edge.axis == "right" else (1, 0)
        source_shift = (
            int(rows[edge.target] - delta_row - rows[edge.source]),
            int(columns[edge.target] - delta_column - columns[edge.source]),
        )
        target_shift = (
            int(rows[edge.source] + delta_row - rows[edge.target]),
            int(columns[edge.source] + delta_column - columns[edge.target]),
        )
        votes[source_component][source_shift].append(float(logit))
        votes[target_component][target_shift].append(float(logit))

    baseline_cost = total_taska_adjacent_seam_cost(
        strict,
        cost_right,
        cost_down,
        grid=grid,
    )
    hypothesis_count = 0
    improving_count = 0
    best_key: tuple[float, int, float, int, int, int] | None = None
    best_layout = strict
    best_cost = baseline_cost
    best_component: int | None = None
    best_shift = (0, 0)
    best_support = 0
    best_weight = 0.0
    for component_index, component in enumerate(components):
        if len(component) < 2:
            continue
        bounds = _component_bounds(position, component, grid=grid)
        for shift, evidence in votes[component_index].items():
            if shift == (0, 0) or not _feasible(shift, bounds, grid=grid):
                continue
            hypothesis_count += 1
            candidate = translate_component_with_local_fill(
                strict,
                component,
                shift[0],
                shift[1],
                grid=grid,
            )
            candidate_cost = total_taska_adjacent_seam_cost(
                candidate,
                cost_right,
                cost_down,
                grid=grid,
            )
            if not candidate_cost < baseline_cost - minimum_cost_gain:
                continue
            improving_count += 1
            support = len(evidence)
            weight = _softplus_sum(evidence)
            # The first two terms are the fixed relation vote.  Lower all-bond
            # cost, lower component index and a stable row/column order only
            # break exact floating-point ties.
            key = (
                weight,
                support,
                -candidate_cost,
                -component_index,
                -shift[0],
                -shift[1],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_layout = candidate
                best_cost = candidate_cost
                best_component = component_index
                best_shift = shift
                best_support = support
                best_weight = weight

    frozen = np.array(best_layout, dtype=np.int32, copy=True)
    frozen.setflags(write=False)
    diagnostics = ComponentRelationAnchorDiagnostics(
        component_count=len(components),
        nontrivial_component_count=sum(len(component) >= 2 for component in components),
        realised_focal_kept_edge_count=realised_count,
        relation_hypothesis_count=hypothesis_count,
        cost_improving_hypothesis_count=improving_count,
        changed=best_component is not None,
        selected_component_index=best_component,
        selected_component_size=(
            0 if best_component is None else len(components[best_component])
        ),
        selected_row_shift=best_shift[0],
        selected_column_shift=best_shift[1],
        selected_vote_support=best_support,
        selected_vote_weight=best_weight,
        baseline_total_cost=baseline_cost,
        selected_total_cost=best_cost,
    )
    return ComponentRelationAnchorResult(layout=frozen, diagnostics=diagnostics)


__all__ = [
    "ComponentRelationAnchorDiagnostics",
    "ComponentRelationAnchorResult",
    "anchor_one_component_from_relation_votes",
    "build_realised_focal_components",
    "translate_component_with_local_fill",
]
