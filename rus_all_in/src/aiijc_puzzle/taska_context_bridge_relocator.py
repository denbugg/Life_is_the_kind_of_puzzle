# ruff: noqa: E501
"""Relocate one weak bridge subtree while preserving an OOF high-confidence core.

The classifier is not an edge-orderer here.  It is used only to identify an
already realised focal-positive supply edge whose removal disconnects a wholly
low-confidence subtree from the realised graph.  At most that one subtree is
translated rigidly on the final frozen six-arm layout; positions occupied by
the p>=0.5 core are forbidden and the original dense seam objective chooses
among every remaining board-feasible translation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_component_relation_anchor import translate_component_with_local_fill
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_selective_fullres_fusion import strict_layout

GRID = 24
COUNT = GRID * GRID
PROBABILITY_THRESHOLD = 0.5
MINIMUM_COST_GAIN = 1e-9


@dataclass(frozen=True)
class ContextBridgeRelocationDiagnostics:
    candidate_edge_count: int
    high_confidence_core_tile_count: int
    weak_bridge_count: int
    eligible_weak_subtree_count: int
    feasible_translation_count: int
    changed: bool
    selected_edge_index: int | None
    selected_subtree_tile_count: int
    selected_row_shift: int
    selected_column_shift: int
    selected_bridge_probability: float | None
    baseline_total_cost: float
    selected_total_cost: float


@dataclass(frozen=True)
class ContextBridgeRelocationResult:
    layout: np.ndarray
    diagnostics: ContextBridgeRelocationDiagnostics


def _validated_probabilities(value: Any, count: int) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError("probabilities must align with the realised edge corpus")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    return result


def _validated_edges(value: Sequence[RawTailEdge]) -> tuple[RawTailEdge, ...]:
    edges = tuple(value)
    if len(set(edges)) != len(edges):
        raise ValueError("bridge corpus contains duplicate edges")
    for edge in edges:
        if not isinstance(edge, RawTailEdge) or edge.axis not in {"right", "down"}:
            raise TypeError("bridge corpus must contain directed right/down RawTailEdge values")
        if (
            not 0 <= edge.source < COUNT
            or not 0 <= edge.target < COUNT
            or edge.source == edge.target
        ):
            raise ValueError("bridge corpus edge is invalid")
    return edges


def _reachable_without_edge(
    adjacency: Sequence[Sequence[tuple[int, int]]], source: int, blocked_index: int
) -> set[int]:
    reached = {source}
    pending: deque[int] = deque((source,))
    while pending:
        current = pending.popleft()
        for other, index in adjacency[current]:
            if index != blocked_index and other not in reached:
                reached.add(other)
                pending.append(other)
    return reached


def relocate_one_weak_bridge_subtree(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    realised_edges: Sequence[RawTailEdge],
    probabilities: Any,
    *,
    grid: int = GRID,
) -> ContextBridgeRelocationResult:
    """Apply the fixed p<0.5 bridge cut and best immutable-core-safe relocation."""

    if grid != GRID:
        raise ValueError("this frozen intervention is registered for 24x24 only")
    current = strict_layout(layout, grid=grid)
    edges = _validated_edges(realised_edges)
    values = _validated_probabilities(probabilities, len(edges))
    right = np.ascontiguousarray(cost_right, dtype=np.float64)
    down = np.ascontiguousarray(cost_down, dtype=np.float64)
    if right.shape != (COUNT, COUNT) or down.shape != (COUNT, COUNT):
        raise ValueError("raw dense costs must be 576x576")
    if not np.isfinite(right).all() or not np.isfinite(down).all():
        raise ValueError("raw dense costs must be finite")

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(COUNT)]
    for index, edge in enumerate(edges):
        adjacency[edge.source].append((edge.target, index))
        adjacency[edge.target].append((edge.source, index))
    core_tiles: set[int] = set()
    for edge, probability in zip(edges, values, strict=True):
        if probability >= PROBABILITY_THRESHOLD:
            core_tiles.add(edge.source)
            core_tiles.add(edge.target)
    position = np.empty(COUNT, dtype=np.int32)
    position[current] = np.arange(COUNT, dtype=np.int32)
    core_positions = set(int(value) for value in position[np.fromiter(core_tiles, dtype=np.int32)])
    baseline = total_taska_adjacent_seam_cost(current, right, down, grid=grid)

    weak_bridges = 0
    eligible_subtrees = 0
    feasible = 0
    # Key: lower raw cost, lower bridge probability, larger subtree, earlier
    # supplied edge, then row-major translation.  Comparison uses a tuple with
    # all signs reversed so max() is deterministic and explicit.
    best: (
        tuple[tuple[float, float, int, int, int, int], np.ndarray, int, int, int, float] | None
    ) = None
    for edge_index, (edge, probability) in enumerate(zip(edges, values, strict=True)):
        if probability >= PROBABILITY_THRESHOLD:
            continue
        source_side = _reachable_without_edge(adjacency, edge.source, edge_index)
        if edge.target in source_side:
            continue
        weak_bridges += 1
        target_side = _reachable_without_edge(adjacency, edge.target, edge_index)
        for subtree in (source_side, target_side):
            if len(subtree) < 2 or core_tiles.intersection(subtree):
                continue
            eligible_subtrees += 1
            tiles = tuple(sorted(subtree))
            old_positions = position[np.asarray(tiles, dtype=np.int32)]
            rows, columns = divmod(old_positions, grid)
            for row_shift in range(-int(rows.min()), grid - int(rows.max())):
                for column_shift in range(-int(columns.min()), grid - int(columns.max())):
                    if row_shift == 0 and column_shift == 0:
                        continue
                    destination = set(
                        int(value) for value in ((rows + row_shift) * grid + columns + column_shift)
                    )
                    if destination.intersection(core_positions):
                        continue
                    candidate = translate_component_with_local_fill(
                        current, tiles, row_shift, column_shift, grid=grid
                    )
                    if core_positions and not np.array_equal(
                        candidate[list(core_positions)], current[list(core_positions)]
                    ):
                        raise RuntimeError("immutable high-confidence core was moved")
                    feasible += 1
                    cost = total_taska_adjacent_seam_cost(candidate, right, down, grid=grid)
                    if not cost < baseline - MINIMUM_COST_GAIN:
                        continue
                    key = (
                        -cost,
                        -float(probability),
                        len(tiles),
                        -edge_index,
                        -row_shift,
                        -column_shift,
                    )
                    if best is None or key > best[0]:
                        best = (
                            key,
                            candidate,
                            edge_index,
                            len(tiles),
                            row_shift,
                            column_shift,
                            float(cost),
                        )
    if best is None:
        selected = current.copy()
        selected.setflags(write=False)
        return ContextBridgeRelocationResult(
            selected,
            ContextBridgeRelocationDiagnostics(
                len(edges),
                len(core_tiles),
                weak_bridges,
                eligible_subtrees,
                feasible,
                False,
                None,
                0,
                0,
                0,
                None,
                baseline,
                baseline,
            ),
        )
    _, selected, edge_index, size, row_shift, column_shift, cost = best
    selected = strict_layout(selected, grid=grid)
    selected.setflags(write=False)
    return ContextBridgeRelocationResult(
        selected,
        ContextBridgeRelocationDiagnostics(
            len(edges),
            len(core_tiles),
            weak_bridges,
            eligible_subtrees,
            feasible,
            True,
            edge_index,
            size,
            row_shift,
            column_shift,
            float(values[edge_index]),
            baseline,
            cost,
        ),
    )


__all__ = [
    "ContextBridgeRelocationDiagnostics",
    "ContextBridgeRelocationResult",
    "MINIMUM_COST_GAIN",
    "PROBABILITY_THRESHOLD",
    "relocate_one_weak_bridge_subtree",
]
