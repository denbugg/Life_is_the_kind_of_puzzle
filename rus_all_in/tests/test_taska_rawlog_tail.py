from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_rawlog_tail import select_taska_rawlog_tail


def test_fixed_alternate_objective_and_original_cost_selection() -> None:
    generator = np.random.default_rng(1301)
    layout = generator.permutation(16).astype(np.int32)
    original_right = generator.uniform(0.0, 3.0, size=(16, 16))
    original_down = generator.uniform(0.0, 3.0, size=(16, 16))
    right_log = generator.normal(size=(16, 16))
    down_log = generator.normal(size=(16, 16))

    result = select_taska_rawlog_tail(
        layout,
        original_right,
        original_down,
        right_log,
        down_log,
        (),
        grid=4,
        max_swaps=6,
    )

    for candidate in (result.control.layout, result.rawlog_tail.layout, result.selection.layout):
        assert np.array_equal(np.sort(candidate), np.arange(16))
        assert not candidate.flags.writeable
    assert result.rawlog_tail.diagnostics.final_total_cost <= (
        result.rawlog_tail.diagnostics.initial_total_cost
    )
    assert result.rawlog_tail.diagnostics.final_total_cost == pytest.approx(
        total_taska_adjacent_seam_cost(
            result.rawlog_tail.layout,
            -right_log,
            -down_log,
            grid=4,
        )
    )
    original_costs = dict(result.selection.total_costs)
    assert tuple(original_costs) == ("control", "rawlog_tail")
    assert result.selection.choice == min(original_costs, key=original_costs.__getitem__)
    assert original_costs[result.selection.choice] == pytest.approx(
        total_taska_adjacent_seam_cost(
            result.selection.layout,
            original_right,
            original_down,
            grid=4,
        )
    )


def test_both_trajectories_freeze_the_same_realised_edge_tiles() -> None:
    generator = np.random.default_rng(1307)
    layout = np.arange(16, dtype=np.int32)
    original_right = generator.normal(size=(16, 16))
    original_down = generator.normal(size=(16, 16))
    right_log = generator.normal(size=(16, 16))
    down_log = generator.normal(size=(16, 16))
    edge = RawTailEdge(0, 1, "right")

    result = select_taska_rawlog_tail(
        layout,
        original_right,
        original_down,
        right_log,
        down_log,
        (edge,),
        grid=4,
        max_swaps=12,
    )

    assert result.control.layout[0] == result.rawlog_tail.layout[0] == 0
    assert result.control.layout[1] == result.rawlog_tail.layout[1] == 1
    assert result.control.diagnostics.protected_tile_count == 2
    assert result.rawlog_tail.diagnostics.protected_tile_count == 2
    assert result.control.diagnostics.initial_realised_edge_count == 1
    assert result.rawlog_tail.diagnostics.initial_realised_edge_count == 1


def test_exact_original_cost_tie_retains_control() -> None:
    layout = np.asarray([3, 1, 0, 2], dtype=np.int32)
    zeros = np.zeros((4, 4), dtype=np.float64)
    result = select_taska_rawlog_tail(
        layout,
        zeros,
        zeros,
        zeros,
        zeros,
        (),
        grid=2,
    )
    assert result.selection.choice == "control"
    assert np.array_equal(result.selection.layout, layout)


def test_nonfinite_matcher_log_is_rejected() -> None:
    layout = np.arange(4, dtype=np.int32)
    zeros = np.zeros((4, 4), dtype=np.float64)
    invalid = zeros.copy()
    invalid[0, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        select_taska_rawlog_tail(
            layout,
            zeros,
            zeros,
            invalid,
            zeros,
            (),
            grid=2,
        )
