from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_adjacent_tail import (
    _globally_best_swap,
    _swap_delta_matrix,
    exact_taska_swap_delta,
    polish_unprotected_taska_tail_with_adjacent_swaps,
)
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost


def _brute_swap_delta(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    first: int,
    second: int,
    *,
    grid: int,
) -> float:
    before = total_taska_adjacent_seam_cost(layout, right, down, grid=grid)
    swapped = layout.copy()
    swapped[first], swapped[second] = swapped[second], swapped[first]
    after = total_taska_adjacent_seam_cost(swapped, right, down, grid=grid)
    return after - before


def test_exact_delta_matches_brute_force_for_every_adjacent_orientation() -> None:
    generator = np.random.default_rng(6821)
    grid = 4
    count = grid * grid
    layout = generator.permutation(count).astype(np.int32)
    right = generator.normal(size=(count, count))
    down = generator.normal(size=(count, count))

    for first in range(count):
        first_row, first_column = divmod(first, grid)
        for second in range(count):
            second_row, second_column = divmod(second, grid)
            if abs(first_row - second_row) + abs(first_column - second_column) != 1:
                continue
            exact = exact_taska_swap_delta(
                layout,
                right,
                down,
                first,
                second,
                grid=grid,
            )
            assert exact == pytest.approx(
                _brute_swap_delta(
                    layout,
                    right,
                    down,
                    first,
                    second,
                    grid=grid,
                ),
                abs=1e-12,
            )


def test_mixed_delta_matrix_matches_brute_force_for_all_position_pairs() -> None:
    generator = np.random.default_rng(6833)
    grid = 4
    count = grid * grid
    layout = generator.permutation(count).astype(np.int32)
    right = generator.normal(size=(count, count))
    down = generator.normal(size=(count, count))
    free_positions = np.arange(count, dtype=np.int64)
    rows, columns = divmod(free_positions, grid)
    adjacent = (
        np.abs(rows[:, None] - rows[None, :])
        + np.abs(columns[:, None] - columns[None, :])
    ) == 1
    adjacent_pairs = np.argwhere(np.triu(adjacent, k=1))

    delta = _swap_delta_matrix(
        layout,
        right,
        down,
        free_positions,
        adjacent_pairs,
        grid=grid,
    )
    for first in range(count):
        assert np.isinf(delta[first, : first + 1]).all()
        for second in range(first + 1, count):
            assert delta[first, second] == pytest.approx(
                _brute_swap_delta(
                    layout,
                    right,
                    down,
                    first,
                    second,
                    grid=grid,
                ),
                abs=1e-12,
            )


def test_adjacent_tail_is_strict_monotone_and_preserves_protected_relations() -> None:
    generator = np.random.default_rng(6847)
    grid = 4
    count = grid * grid
    layout = np.arange(count, dtype=np.int32)
    right = generator.normal(size=(count, count))
    down = generator.normal(size=(count, count))
    protected_edge = RawTailEdge(0, 1, "right")

    result = polish_unprotected_taska_tail_with_adjacent_swaps(
        layout,
        right,
        down,
        (protected_edge,),
        grid=grid,
        max_swaps=12,
    )

    assert np.array_equal(np.sort(result.layout), np.arange(count))
    assert not result.layout.flags.writeable
    assert result.layout[0] == 0
    assert result.layout[1] == 1
    assert result.diagnostics.protected_tile_count == 2
    assert result.diagnostics.final_realised_edge_count >= 1
    assert result.diagnostics.accepted_swap_count <= 12
    assert (
        result.diagnostics.accepted_adjacent_swap_count
        + result.diagnostics.accepted_nonadjacent_swap_count
        == result.diagnostics.accepted_swap_count
    )
    assert result.diagnostics.final_total_cost <= result.diagnostics.initial_total_cost
    assert result.diagnostics.final_total_cost == pytest.approx(
        total_taska_adjacent_seam_cost(result.layout, right, down, grid=grid)
    )


def test_global_best_rule_uses_stable_row_major_tie() -> None:
    delta = np.asarray(
        [
            [np.inf, -3.0, -3.0],
            [np.inf, np.inf, -2.0],
            [np.inf, np.inf, np.inf],
        ]
    )
    assert _globally_best_swap(delta) == (0, 1, -3.0)


def test_zero_gain_tie_keeps_layout_unchanged() -> None:
    layout = np.asarray([3, 1, 0, 2], dtype=np.int32)
    costs = np.zeros((4, 4), dtype=np.float64)
    result = polish_unprotected_taska_tail_with_adjacent_swaps(
        layout,
        costs,
        costs,
        (),
        grid=2,
        max_swaps=96,
        minimum_gain=1e-9,
    )
    assert np.array_equal(result.layout, layout)
    assert result.diagnostics.accepted_swap_count == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"grid": 1}, "grid"),
        ({"max_swaps": -1}, "max_swaps"),
        ({"minimum_gain": -0.1}, "minimum_gain"),
    ],
)
def test_invalid_controls_are_rejected(kwargs: dict[str, object], match: str) -> None:
    layout = np.arange(4)
    costs = np.zeros((4, 4))
    controls: dict[str, object] = {"grid": 2}
    controls.update(kwargs)
    with pytest.raises(ValueError, match=match):
        polish_unprotected_taska_tail_with_adjacent_swaps(
            layout,
            costs,
            costs,
            (),
            **controls,
        )
