from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_row_phase_dp import solve_taska_row_phase_dp


def _phase_layout(layout: np.ndarray, phases: tuple[int, ...], grid: int) -> np.ndarray:
    board = layout.reshape(grid, grid)
    return np.concatenate(
        [np.roll(board[row], phase) for row, phase in enumerate(phases)]
    )


def test_row_phase_dp_matches_exhaustive_global_minimum() -> None:
    grid = 3
    rng = np.random.default_rng(20260831)
    layout = rng.permutation(grid * grid)
    right = rng.normal(size=(grid * grid, grid * grid))
    down = rng.normal(size=(grid * grid, grid * grid))

    result = solve_taska_row_phase_dp(layout, right, down, grid=grid)
    brute_costs = [
        total_taska_adjacent_seam_cost(
            _phase_layout(layout, phases, grid), right, down, grid=grid
        )
        for phases in product(range(grid), repeat=grid)
    ]

    assert result.diagnostics.after_total_cost == pytest.approx(min(brute_costs))
    np.testing.assert_array_equal(np.sort(result.layout), np.arange(grid * grid))
    np.testing.assert_array_equal(
        result.layout,
        _phase_layout(layout, result.diagnostics.phases, grid),
    )
    assert not result.layout.flags.writeable
    assert result.diagnostics.objective_monotone


def test_row_phase_dp_keeps_unique_zero_phase_optimum() -> None:
    grid = 3
    layout = np.arange(grid * grid)
    right = np.full((grid * grid, grid * grid), 10.0)
    down = np.full((grid * grid, grid * grid), 10.0)
    board = layout.reshape(grid, grid)
    right[board[:, :-1], board[:, 1:]] = 0.0
    down[board[:-1], board[1:]] = 0.0

    result = solve_taska_row_phase_dp(layout, right, down, grid=grid)

    np.testing.assert_array_equal(result.layout, layout)
    assert result.diagnostics.phases == (0, 0, 0)
    assert result.diagnostics.changed_row_count == 0
    assert result.diagnostics.total_cost_improvement == pytest.approx(0.0)


def test_row_phase_dp_rejects_non_permutation() -> None:
    cost = np.zeros((9, 9))
    with pytest.raises(ValueError, match="strict original-tile permutation"):
        solve_taska_row_phase_dp(np.zeros(9), cost, cost, grid=3)
