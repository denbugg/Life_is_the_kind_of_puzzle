from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_monotone_hungarian_tail import (
    repack_unprotected_taska_tail,
)


def test_repack_is_strict_monotone_and_preserves_realised_edge() -> None:
    layout = np.arange(9, dtype=np.int32)
    generator = np.random.default_rng(0)
    right = generator.normal(size=(9, 9))
    down = generator.normal(size=(9, 9))
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    result = repack_unprotected_taska_tail(
        layout,
        right,
        down,
        [RawTailEdge(0, 1, "right")],
        grid=3,
        max_rounds=6,
    )
    assert np.array_equal(np.sort(result.layout), np.arange(9))
    position = np.empty(9, dtype=np.int32)
    position[result.layout] = np.arange(9)
    assert position[1] == position[0] + 1
    assert result.diagnostics.accepted_round_count == 2
    assert result.diagnostics.final_total_cost < result.diagnostics.initial_total_cost
    assert result.diagnostics.initial_realised_edge_count == 1
    assert result.diagnostics.final_realised_edge_count == 1
    assert not result.layout.flags.writeable


def test_zero_rounds_is_a_read_only_noop() -> None:
    layout = np.arange(4, dtype=np.int32)
    costs = np.ones((4, 4), dtype=np.float64)
    result = repack_unprotected_taska_tail(
        layout,
        costs,
        costs,
        [],
        grid=2,
        max_rounds=0,
    )
    assert np.array_equal(result.layout, layout)
    assert result.diagnostics.accepted_round_count == 0
    assert not result.layout.flags.writeable


@pytest.mark.parametrize("max_rounds", [-1, True])
def test_invalid_round_budget_fails(max_rounds: object) -> None:
    layout = np.arange(4, dtype=np.int32)
    costs = np.ones((4, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        repack_unprotected_taska_tail(
            layout,
            costs,
            costs,
            [],
            grid=2,
            max_rounds=max_rounds,  # type: ignore[arg-type]
        )
