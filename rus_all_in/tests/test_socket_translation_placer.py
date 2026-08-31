from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)


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


def test_perfect_socket_evidence_recovers_global_cyclic_origin() -> None:
    grid = 4
    reference = np.random.default_rng(17).permutation(grid * grid)
    initial = np.roll(reference.reshape(grid, grid), shift=(2, 1), axis=(0, 1)).reshape(-1)
    result = select_global_cyclic_translation(
        initial,
        _perfect_assignment(reference, grid=grid, axis="right"),
        _perfect_assignment(reference, grid=grid, axis="down"),
        grid=grid,
    )
    assert np.array_equal(result.layout, reference)
    assert result.diagnostics.selected_row_roll == 2
    assert result.diagnostics.selected_column_roll == 3
    assert result.diagnostics.objective_gain > 0
    assert result.diagnostics.strict_permutation


def test_flat_evidence_keeps_original_layout_on_tie() -> None:
    grid = 3
    count = grid * grid
    layout = np.random.default_rng(23).permutation(count)
    assignment = np.zeros((count + 1, count + 1), dtype=np.float64)
    assignment[count, count] = -1e4
    result = select_global_cyclic_translation(layout, assignment, assignment, grid=grid)
    assert np.array_equal(result.layout, layout)
    assert not result.diagnostics.changed
    assert result.diagnostics.objective_gain == pytest.approx(0.0)
    assert result.diagnostics.candidates_evaluated == count


def test_noisy_translation_is_strict_and_never_lowers_declared_objective() -> None:
    grid = 4
    count = grid * grid
    generator = np.random.default_rng(29)
    layout = generator.permutation(count)
    right = generator.normal(size=(count + 1, count + 1))
    down = generator.normal(size=(count + 1, count + 1))
    right[count, count] = down[count, count] = -1e4
    result = select_global_cyclic_translation(
        layout,
        right,
        down,
        grid=grid,
        config=CyclicTranslationConfig(border_weight=0.2),
    )
    assert np.array_equal(np.sort(result.layout), np.arange(count))
    assert result.diagnostics.final_objective >= result.diagnostics.initial_objective - 1e-7


def test_translation_placer_validates_inputs() -> None:
    assignment = np.zeros((5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        select_global_cyclic_translation(np.arange(16), assignment, assignment, grid=4)
    correct = np.zeros((17, 17), dtype=np.float64)
    with pytest.raises(ValueError, match="strict tile permutation"):
        select_global_cyclic_translation(np.zeros(16), correct, correct, grid=4)
    with pytest.raises(ValueError, match="border_weight"):
        select_global_cyclic_translation(
            np.arange(16),
            correct,
            correct,
            grid=4,
            config=CyclicTranslationConfig(border_weight=-1.0),
        )
