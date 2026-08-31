from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.coordinate_cyclic_placer import (
    CoordinateCyclicConfig,
    coordinate_cyclic_score_profiles,
    select_coordinate_cyclic_translation,
)


def _perfect_coordinate_logits(layout: np.ndarray, *, grid: int) -> tuple[np.ndarray, np.ndarray]:
    count = grid * grid
    position = np.empty(count, dtype=np.int32)
    position[layout] = np.arange(count, dtype=np.int32)
    rows, columns = divmod(position, grid)
    row = np.full((count, grid), -20.0, dtype=np.float64)
    column = np.full((count, grid), -20.0, dtype=np.float64)
    row[np.arange(count), rows] = 20.0
    column[np.arange(count), columns] = 20.0
    return row, column


def _perfect_assignment(layout: np.ndarray, *, grid: int, axis: str) -> np.ndarray:
    count = grid * grid
    board = np.asarray(layout).reshape(grid, grid)
    value = np.full((count + 1, count + 1), -30.0, dtype=np.float64)
    value[count, count] = -1e4
    if axis == "right":
        for row in range(grid):
            value[count, board[row, 0]] = 0.0
            value[board[row, -1], count] = 0.0
            for column in range(grid - 1):
                value[board[row, column], board[row, column + 1]] = 0.0
    elif axis == "down":
        for column in range(grid):
            value[count, board[0, column]] = 0.0
            value[board[-1, column], count] = 0.0
            for row in range(grid - 1):
                value[board[row, column], board[row + 1, column]] = 0.0
    else:
        raise ValueError(axis)
    return value


def test_coordinate_profiles_and_selection_use_tile_at_position_mapping() -> None:
    grid = 4
    reference = np.random.default_rng(11).permutation(grid * grid)
    row, column = _perfect_coordinate_logits(reference, grid=grid)
    initial = np.roll(reference.reshape(grid, grid), shift=(1, 3), axis=(0, 1)).reshape(-1)
    row_profile, column_profile = coordinate_cyclic_score_profiles(
        initial,
        row,
        column,
        grid=grid,
    )
    result = select_coordinate_cyclic_translation(initial, row, column, grid=grid)
    assert int(np.argmax(row_profile)) == 3
    assert int(np.argmax(column_profile)) == 1
    assert (result.diagnostics.selected_row_roll, result.diagnostics.selected_column_roll) == (
        3,
        1,
    )
    assert np.array_equal(result.layout, reference)
    assert result.diagnostics.strict_permutation


def test_per_tile_offsets_and_global_positive_scale_preserve_coordinate_choice() -> None:
    grid = 4
    count = grid * grid
    generator = np.random.default_rng(17)
    layout = generator.permutation(count)
    row = generator.normal(size=(count, grid))
    column = generator.normal(size=(count, grid))
    baseline = select_coordinate_cyclic_translation(layout, row, column, grid=grid)
    row_offset = generator.normal(size=(count, 1))
    column_offset = generator.normal(size=(count, 1))
    transformed = select_coordinate_cyclic_translation(
        layout,
        7.0 * row + row_offset,
        7.0 * column + column_offset,
        grid=grid,
    )
    assert np.array_equal(transformed.layout, baseline.layout)


def test_tile_relabelling_equivariance() -> None:
    grid = 4
    count = grid * grid
    generator = np.random.default_rng(23)
    layout = generator.permutation(count)
    row = generator.normal(size=(count, grid))
    column = generator.normal(size=(count, grid))
    baseline = select_coordinate_cyclic_translation(layout, row, column, grid=grid)
    old_to_new = generator.permutation(count)
    relabelled_layout = old_to_new[layout]
    relabelled_row = np.empty_like(row)
    relabelled_column = np.empty_like(column)
    relabelled_row[old_to_new] = row
    relabelled_column[old_to_new] = column
    relabelled = select_coordinate_cyclic_translation(
        relabelled_layout,
        relabelled_row,
        relabelled_column,
        grid=grid,
    )
    assert np.array_equal(relabelled.layout, old_to_new[baseline.layout])


def test_row_coordinate_column_socket_hybrid_recovers_both_axes() -> None:
    grid = 4
    count = grid * grid
    reference = np.random.default_rng(29).permutation(count)
    initial = np.roll(reference.reshape(grid, grid), shift=(2, 1), axis=(0, 1)).reshape(-1)
    row, _ = _perfect_coordinate_logits(reference, grid=grid)
    flat_column = np.zeros((count, grid), dtype=np.float64)
    result = select_coordinate_cyclic_translation(
        initial,
        row,
        flat_column,
        right_log_assignment=_perfect_assignment(reference, grid=grid, axis="right"),
        down_log_assignment=_perfect_assignment(reference, grid=grid, axis="down"),
        grid=grid,
        config=CoordinateCyclicConfig(
            row_coordinate_weight=1.0,
            row_socket_weight=0.0,
            column_coordinate_weight=0.0,
            column_socket_weight=1.0,
        ),
    )
    assert np.array_equal(result.layout, reference)


def test_flat_profiles_keep_origin_and_input_validation_is_fail_closed() -> None:
    grid = 3
    count = grid * grid
    layout = np.random.default_rng(31).permutation(count)
    flat = np.zeros((count, grid), dtype=np.float64)
    result = select_coordinate_cyclic_translation(layout, flat, flat, grid=grid)
    assert np.array_equal(result.layout, layout)
    assert not result.diagnostics.changed
    assert result.diagnostics.combined_score_gain == pytest.approx(0.0)
    with pytest.raises(ValueError, match="strict tile permutation"):
        select_coordinate_cyclic_translation(np.zeros(count), flat, flat, grid=grid)
    with pytest.raises(ValueError, match="row_logits"):
        select_coordinate_cyclic_translation(layout, np.zeros((count, 2)), flat, grid=grid)
    with pytest.raises(ValueError, match="require both socket"):
        select_coordinate_cyclic_translation(
            layout,
            flat,
            flat,
            grid=grid,
            config=CoordinateCyclicConfig(row_socket_weight=1.0),
        )
    with pytest.raises(ValueError, match="row cyclic profile"):
        CoordinateCyclicConfig(
            row_coordinate_weight=0.0,
            column_coordinate_weight=1.0,
        ).validate()
