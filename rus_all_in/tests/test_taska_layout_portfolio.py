from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.taska_layout_portfolio import (
    select_lower_taska_seam_cost_layout,
    select_lowest_taska_seam_cost_layout,
    total_taska_adjacent_seam_cost,
)


def _matrices(grid: int) -> tuple[np.ndarray, np.ndarray]:
    count = grid * grid
    right = np.full((count, count), 9.0)
    down = np.full((count, count), 9.0)
    return right, down


def test_total_cost_counts_every_directed_board_bond() -> None:
    grid = 2
    right, down = _matrices(grid)
    layout = np.arange(grid * grid, dtype=np.int32)
    right[0, 1] = 1.0
    right[2, 3] = 2.0
    down[0, 2] = 3.0
    down[1, 3] = 4.0

    assert total_taska_adjacent_seam_cost(layout, right, down, grid=grid) == 10.0


def test_selector_chooses_lower_cost_layout_and_returns_strict_read_only_copy() -> None:
    grid = 2
    right, down = _matrices(grid)
    raw = np.asarray([0, 1, 2, 3], dtype=np.int32)
    calibrated = np.asarray([0, 2, 1, 3], dtype=np.int32)
    right[0, 2] = right[1, 3] = 0.1
    down[0, 1] = down[2, 3] = 0.1

    result = select_lower_taska_seam_cost_layout(
        raw,
        calibrated,
        right,
        down,
        grid=grid,
    )

    assert result.choice == "calibrated"
    assert result.calibrated_total_cost < result.raw_total_cost
    np.testing.assert_array_equal(result.layout, calibrated)
    assert not result.layout.flags.writeable


def test_exact_cost_tie_keeps_raw_layout() -> None:
    grid = 2
    right = np.ones((4, 4))
    down = np.ones((4, 4))
    raw = np.asarray([3, 2, 1, 0], dtype=np.int32)
    calibrated = np.arange(4, dtype=np.int32)

    result = select_lower_taska_seam_cost_layout(
        raw,
        calibrated,
        right,
        down,
        grid=grid,
    )

    assert result.choice == "raw"
    np.testing.assert_array_equal(result.layout, raw)


def test_selector_fails_closed_on_non_permutation_or_nonfinite_cost() -> None:
    right, down = _matrices(2)
    layout = np.arange(4, dtype=np.int32)
    with pytest.raises(ValueError, match="every tile exactly once"):
        select_lower_taska_seam_cost_layout(
            [0, 0, 2, 3],
            layout,
            right,
            down,
            grid=2,
        )
    right[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_lower_taska_seam_cost_layout(
            layout,
            layout,
            right,
            down,
            grid=2,
        )


def test_named_selector_is_stable_and_returns_all_costs() -> None:
    raw = np.asarray([0, 1, 2, 3], dtype=np.int32)
    focal = np.asarray([3, 2, 1, 0], dtype=np.int32)
    costs = np.zeros((4, 4), dtype=np.float64)
    result = select_lowest_taska_seam_cost_layout(
        {"raw": raw, "focal": focal},
        costs,
        costs,
        grid=2,
    )
    assert result.choice == "raw"
    assert result.total_costs == (("raw", 0.0), ("focal", 0.0))
    assert np.array_equal(result.layout, raw)
    assert not result.layout.flags.writeable


def test_named_selector_rejects_empty_roster() -> None:
    costs = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="non-empty"):
        select_lowest_taska_seam_cost_layout({}, costs, costs, grid=2)
