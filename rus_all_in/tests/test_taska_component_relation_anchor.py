from __future__ import annotations

import inspect

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_component_relation_anchor import (
    anchor_one_component_from_relation_votes,
    build_realised_focal_components,
    translate_component_with_local_fill,
)


def _candidate_costs(layout: np.ndarray, *, grid: int) -> tuple[np.ndarray, np.ndarray]:
    count = grid * grid
    right = np.full((count, count), 10.0, dtype=np.float64)
    down = np.full((count, count), 10.0, dtype=np.float64)
    board = layout.reshape(grid, grid)
    right[board[:, :-1], board[:, 1:]] = 0.0
    down[board[:-1], board[1:]] = 0.0
    return right, down


def test_local_fill_is_strict_and_moves_only_component_and_displaced_tiles() -> None:
    layout = np.arange(16, dtype=np.int32)
    moved = translate_component_with_local_fill(layout, (0, 1), 1, 0, grid=4)
    assert moved.tolist() == [4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert np.array_equal(np.sort(moved), np.arange(16))


def test_relation_vote_moves_one_realised_component_under_strict_cost_guard() -> None:
    layout = np.arange(16, dtype=np.int32)
    expected = translate_component_with_local_fill(layout, (0, 1), 1, 0, grid=4)
    right, down = _candidate_costs(expected, grid=4)
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(0, 8, "down"),
    )
    logits = np.asarray([4.0, 3.0], dtype=np.float64)
    components, realised = build_realised_focal_components(
        layout,
        edges,
        logits,
        grid=4,
    )
    assert realised == 1
    assert components[0] == (0, 1)

    result = anchor_one_component_from_relation_votes(
        layout,
        right,
        down,
        edges,
        logits,
        grid=4,
    )
    assert np.array_equal(result.layout, expected)
    assert result.diagnostics.changed
    assert result.diagnostics.selected_component_size == 2
    assert (result.diagnostics.selected_row_shift, result.diagnostics.selected_column_shift) == (
        1,
        0,
    )
    assert result.diagnostics.selected_vote_support == 1
    assert result.diagnostics.selected_total_cost < result.diagnostics.baseline_total_cost


def test_cost_guard_falls_back_bit_for_bit() -> None:
    layout = np.arange(16, dtype=np.int32)
    zero = np.zeros((16, 16), dtype=np.float64)
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(0, 8, "down"),
    )
    result = anchor_one_component_from_relation_votes(
        layout,
        zero,
        zero,
        edges,
        np.asarray([4.0, 3.0]),
        grid=4,
    )
    assert np.array_equal(result.layout, layout)
    assert not result.diagnostics.changed
    assert result.diagnostics.selected_component_index is None
    assert result.diagnostics.selected_total_cost == result.diagnostics.baseline_total_cost


def test_public_anchor_api_has_no_reference_or_identity_inputs() -> None:
    parameters = set(inspect.signature(anchor_one_component_from_relation_votes).parameters)
    forbidden = {"target", "reference", "filename", "source_filename", "tile_id"}
    assert not parameters & forbidden

