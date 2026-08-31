"""Fixed dense-contact translation consensus diagnostic for TASKA components.

The emitter is deliberately target blind.  It reconstructs the same per-tile
top-k contacts used by :mod:`taska_joint_component_pose`, canonicalises every
physical right/down edge, and retains only component-pair translations that
are supported by at least two reciprocal contacts spanning both axes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.taska_joint_component_pose import _external_topk


@dataclass(frozen=True)
class DenseConsensusBoard:
    """Target-free consensus contacts for one strict puzzle layout."""

    edge_source: np.ndarray
    edge_target: np.ndarray
    edge_axis: np.ndarray
    edge_group: np.ndarray
    group_component_low: np.ndarray
    group_component_high: np.ndarray
    group_relative_translation: np.ndarray
    group_support: np.ndarray
    group_right_support: np.ndarray
    group_down_support: np.ndarray
    outgoing_physical_contact_count: int
    incoming_physical_contact_count: int
    reciprocal_missing_contact_count: int


def _strict_layout(value: Any, *, grid: int) -> np.ndarray:
    count = grid * grid
    result = np.ascontiguousarray(value, dtype=np.int32)
    if result.shape != (count,) or not np.array_equal(
        np.sort(result), np.arange(count, dtype=np.int32)
    ):
        raise ValueError("layout must be a strict grid-sized permutation")
    return result


def _normalise_components(value: Any, *, count: int) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.int32)
    if result.shape != (count,) or np.any(result < 0):
        raise ValueError("component_of_tile is malformed")
    labels = np.unique(result)
    if not np.array_equal(labels, np.arange(len(labels), dtype=np.int32)):
        raise ValueError("component labels must be contiguous")
    return result


def _normalise_relative(value: Any, *, count: int) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.int16)
    if result.shape != (count, 2):
        raise ValueError("component-relative coordinates are malformed")
    return result


def _normalise_cost(value: Any, *, count: int) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != (count, count) or not np.isfinite(result).all():
        raise ValueError("directional cost matrix is malformed")
    return result


def _is_realised(
    source: int,
    target: int,
    axis: int,
    *,
    tile_rows: np.ndarray,
    tile_columns: np.ndarray,
) -> bool:
    if axis == 0:
        return bool(
            tile_rows[target] == tile_rows[source]
            and tile_columns[target] == tile_columns[source] + 1
        )
    return bool(
        tile_rows[target] == tile_rows[source] + 1
        and tile_columns[target] == tile_columns[source]
    )


def _canonical_group(
    source: int,
    target: int,
    axis: int,
    *,
    component_of: np.ndarray,
    relative: np.ndarray,
) -> tuple[int, int, int, int]:
    source_component = int(component_of[source])
    target_component = int(component_of[target])
    delta_row, delta_column = ((0, 1) if axis == 0 else (1, 0))
    # The edge requires O_target - O_source = rel_source + delta - rel_target.
    translation_row = int(relative[source, 0]) + delta_row - int(relative[target, 0])
    translation_column = (
        int(relative[source, 1]) + delta_column - int(relative[target, 1])
    )
    if source_component < target_component:
        return (
            source_component,
            target_component,
            translation_row,
            translation_column,
        )
    return (
        target_component,
        source_component,
        -translation_row,
        -translation_column,
    )


def build_dense_contact_consensus(
    *,
    layout: Any,
    component_of_tile: Any,
    component_relative_coordinates: Any,
    cost_right: Any,
    cost_down: Any,
    grid: int = 24,
    dense_topk: int = 8,
    minimum_support: int = 2,
) -> DenseConsensusBoard:
    """Emit the single fixed reciprocal, two-axis component consensus rule.

    A counted physical edge must be present in both the source outgoing and
    target incoming top-k retrievals after applying the same board-feasibility
    filter as the frozen joint-pose cache.  Already realised control-layout
    contacts are removed before grouping.
    """

    strict = _strict_layout(layout, grid=grid)
    count = grid * grid
    component_of = _normalise_components(component_of_tile, count=count)
    relative = _normalise_relative(component_relative_coordinates, count=count)
    right = _normalise_cost(cost_right, count=count)
    down = _normalise_cost(cost_down, count=count)
    if dense_topk != 8:
        raise ValueError("this fixed diagnostic requires dense_topk=8")
    if minimum_support != 2:
        raise ValueError("this fixed diagnostic requires minimum_support=2")

    position = np.empty(count, dtype=np.int32)
    position[strict] = np.arange(count, dtype=np.int32)
    tile_rows, tile_columns = divmod(position, grid)
    component_tiles = tuple(
        np.flatnonzero(component_of == index)
        for index in range(int(component_of.max()) + 1)
    )

    # Bit 1 means source-outgoing retrieval; bit 2 means target-incoming.
    physical_flags: dict[tuple[int, int, int], int] = defaultdict(int)
    for tile in range(count):
        component_index = int(component_of[tile])
        specifications = (
            (right[tile], 0, 1, 0, False),
            (right[:, tile], 0, 1, 0, True),
            (down[tile], 1, 0, 1, False),
            (down[:, tile], 1, 0, 1, True),
        )
        for costs, delta_row, delta_column, axis, reverse in specifications:
            targets, _, _ = _external_topk(
                costs,
                component_of,
                component_index,
                tile,
                topk=dense_topk,
            )
            for other_value in targets:
                other = int(other_value)
                if reverse:
                    source, target = other, tile
                    shift_row = int(tile_rows[other] + delta_row - tile_rows[tile])
                    shift_column = int(
                        tile_columns[other] + delta_column - tile_columns[tile]
                    )
                    flag = 2
                else:
                    source, target = tile, other
                    shift_row = int(tile_rows[other] - delta_row - tile_rows[tile])
                    shift_column = int(
                        tile_columns[other] - delta_column - tile_columns[tile]
                    )
                    flag = 1
                moving_tiles = component_tiles[component_index]
                destination_rows = tile_rows[moving_tiles] + shift_row
                destination_columns = tile_columns[moving_tiles] + shift_column
                if (
                    np.any(destination_rows < 0)
                    or np.any(destination_rows >= grid)
                    or np.any(destination_columns < 0)
                    or np.any(destination_columns >= grid)
                ):
                    continue
                physical_flags[(source, target, axis)] |= flag

    outgoing_count = sum(bool(flags & 1) for flags in physical_flags.values())
    incoming_count = sum(bool(flags & 2) for flags in physical_flags.values())
    reciprocal_missing = {
        edge
        for edge, flags in physical_flags.items()
        if flags == 3
        and not _is_realised(
            *edge,
            tile_rows=tile_rows,
            tile_columns=tile_columns,
        )
    }
    grouped: dict[tuple[int, int, int, int], list[tuple[int, int, int]]] = (
        defaultdict(list)
    )
    for edge in sorted(reciprocal_missing):
        grouped[
            _canonical_group(
                *edge,
                component_of=component_of,
                relative=relative,
            )
        ].append(edge)

    retained = [
        (key, tuple(edges))
        for key, edges in sorted(grouped.items())
        if len(edges) >= minimum_support and {edge[2] for edge in edges} == {0, 1}
    ]
    sources: list[int] = []
    targets: list[int] = []
    axes: list[int] = []
    edge_groups: list[int] = []
    component_low: list[int] = []
    component_high: list[int] = []
    translations: list[tuple[int, int]] = []
    supports: list[int] = []
    right_supports: list[int] = []
    down_supports: list[int] = []
    for group_index, (key, edges) in enumerate(retained):
        low, high, row, column = key
        component_low.append(low)
        component_high.append(high)
        translations.append((row, column))
        supports.append(len(edges))
        right_supports.append(sum(edge[2] == 0 for edge in edges))
        down_supports.append(sum(edge[2] == 1 for edge in edges))
        for source, target, axis in edges:
            sources.append(source)
            targets.append(target)
            axes.append(axis)
            edge_groups.append(group_index)

    return DenseConsensusBoard(
        edge_source=np.ascontiguousarray(sources, dtype=np.int32),
        edge_target=np.ascontiguousarray(targets, dtype=np.int32),
        edge_axis=np.ascontiguousarray(axes, dtype=np.uint8),
        edge_group=np.ascontiguousarray(edge_groups, dtype=np.int32),
        group_component_low=np.ascontiguousarray(component_low, dtype=np.int32),
        group_component_high=np.ascontiguousarray(component_high, dtype=np.int32),
        group_relative_translation=np.ascontiguousarray(
            translations, dtype=np.int16
        ).reshape(-1, 2),
        group_support=np.ascontiguousarray(supports, dtype=np.int16),
        group_right_support=np.ascontiguousarray(right_supports, dtype=np.int16),
        group_down_support=np.ascontiguousarray(down_supports, dtype=np.int16),
        outgoing_physical_contact_count=int(outgoing_count),
        incoming_physical_contact_count=int(incoming_count),
        reciprocal_missing_contact_count=int(len(reciprocal_missing)),
    )


__all__ = ["DenseConsensusBoard", "build_dense_contact_consensus"]
