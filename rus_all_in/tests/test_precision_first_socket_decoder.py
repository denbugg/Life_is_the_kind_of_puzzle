from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.precision_first_socket_decoder import (
    PrecisionFirstDecoderConfig,
    decode_precision_first,
    select_precision_edges,
)


def _perfect_assignment(layout: np.ndarray, *, grid: int, axis: str) -> np.ndarray:
    count = grid * grid
    board = np.asarray(layout).reshape(grid, grid)
    value = np.full((count + 1, count + 1), -20.0, dtype=np.float64)
    value[count, count] = -1e4
    if axis == "right":
        for row in range(grid):
            value[count, board[row, 0]] = 0.0
            value[board[row, -1], count] = 0.0
            for column in range(grid - 1):
                value[board[row, column], board[row, column + 1]] = 0.0
    else:
        for column in range(grid):
            value[count, board[0, column]] = 0.0
            value[board[-1, column], count] = 0.0
            for row in range(grid - 1):
                value[board[row, column], board[row + 1, column]] = 0.0
    return value


def test_oracle_precision_first_recovers_exact_grid_without_cap() -> None:
    grid = 4
    reference = np.random.default_rng(211).permutation(grid * grid)
    result = decode_precision_first(
        _perfect_assignment(reference, grid=grid, axis="right"),
        _perfect_assignment(reference, grid=grid, axis="down"),
        grid=grid,
        config=PrecisionFirstDecoderConfig(maximum_component_size=grid * grid),
    )
    assert np.array_equal(result.layout, reference)
    assert result.diagnostics.largest_component == grid * grid
    assert result.diagnostics.strict_permutation


def test_component_cap_prevents_oracle_graph_from_percolating() -> None:
    grid = 4
    reference = np.random.default_rng(223).permutation(grid * grid)
    result = decode_precision_first(
        _perfect_assignment(reference, grid=grid, axis="right"),
        _perfect_assignment(reference, grid=grid, axis="down"),
        grid=grid,
        config=PrecisionFirstDecoderConfig(maximum_component_size=3),
    )
    assert result.diagnostics.largest_component <= 3
    assert result.diagnostics.size_cap_rejections > 0
    assert np.array_equal(np.sort(result.layout), np.arange(grid * grid))


def test_admission_requires_margin_over_real_and_dustbin_alternatives() -> None:
    grid = 3
    reference = np.arange(grid * grid)
    right = _perfect_assignment(reference, grid=grid, axis="right")
    permissive = PrecisionFirstDecoderConfig(maximum_component_size=9)
    assert len(select_precision_edges(right, grid=grid, axis="right", config=permissive)) == 6

    strict = PrecisionFirstDecoderConfig(
        minimum_edge_confidence=100.0,
        maximum_component_size=9,
    )
    assert not select_precision_edges(right, grid=grid, axis="right", config=strict)


def test_config_validation_is_fail_closed() -> None:
    assignment = np.zeros((10, 10), dtype=np.float64)
    with pytest.raises(ValueError, match="maximum_component_size"):
        decode_precision_first(
            assignment,
            assignment,
            grid=3,
            config=PrecisionFirstDecoderConfig(maximum_component_size=1),
        )
