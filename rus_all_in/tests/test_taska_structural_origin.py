from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.taska_structural_origin import (
    select_structural_border_cyclic_origin,
)


def _permutation(grid: int) -> np.ndarray:
    return np.arange(grid * grid, dtype=np.int32)


def test_recovers_known_roll_and_preserves_strict_permutation() -> None:
    grid = 4
    layout = _permutation(grid)
    wanted = np.roll(layout.reshape(grid, grid), (2, 3), (0, 1))
    unary = np.zeros((grid * grid, grid, grid), dtype=np.float64)
    for row in range(grid):
        for column in range(grid):
            if row in (0, grid - 1) or column in (0, grid - 1):
                unary[wanted[row, column], row, column] = 1.0

    result = select_structural_border_cyclic_origin(layout, unary, grid=grid)

    assert (result.selected_row_roll, result.selected_column_roll) == (2, 3)
    assert np.array_equal(result.layout.reshape(grid, grid), wanted)
    assert np.array_equal(np.sort(result.layout), _permutation(grid))
    assert result.changed is True


def test_exact_ties_keep_first_row_major_roll() -> None:
    grid = 3
    result = select_structural_border_cyclic_origin(
        _permutation(grid),
        np.zeros((grid * grid, grid, grid)),
        grid=grid,
    )

    assert (result.selected_row_roll, result.selected_column_roll) == (0, 0)
    assert result.selected_score == result.unchanged_score == 0.0
    assert result.changed is False


def test_interior_unary_is_ignored() -> None:
    grid = 4
    unary = np.zeros((grid * grid, grid, grid), dtype=np.float64)
    unary[5, 1, 1] = 1.0e9

    result = select_structural_border_cyclic_origin(_permutation(grid), unary, grid=grid)

    assert (result.selected_row_roll, result.selected_column_roll) == (0, 0)


def test_tile_relabelling_is_equivariant() -> None:
    grid = 4
    layout = _permutation(grid)
    unary = np.random.default_rng(5).normal(size=(grid * grid, grid, grid))
    relabel = np.random.default_rng(7).permutation(grid * grid).astype(np.int32)

    original = select_structural_border_cyclic_origin(layout, unary, grid=grid)
    transformed = select_structural_border_cyclic_origin(
        relabel[layout],
        unary[np.argsort(relabel)],
        grid=grid,
    )

    assert (
        transformed.selected_row_roll,
        transformed.selected_column_roll,
    ) == (original.selected_row_roll, original.selected_column_roll)
    assert np.array_equal(transformed.layout, relabel[original.layout])


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        select_structural_border_cyclic_origin(np.arange(8), np.zeros((9, 3, 3)), grid=3)
    with pytest.raises(ValueError, match="every original tile"):
        select_structural_border_cyclic_origin(
            np.zeros(9, dtype=np.int32),
            np.zeros((9, 3, 3)),
            grid=3,
        )
    with pytest.raises(TypeError, match="integer"):
        select_structural_border_cyclic_origin(np.arange(9.0), np.zeros((9, 3, 3)), grid=3)
    damaged = np.zeros((9, 3, 3))
    damaged[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_structural_border_cyclic_origin(_permutation(3), damaged, grid=3)
