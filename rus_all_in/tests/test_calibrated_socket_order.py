from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    hard_partial_axis_matching,
    prioritise_component_edges,
)


def _assignment(grid: int, seed: int) -> np.ndarray:
    count = grid * grid
    value = np.random.default_rng(seed).normal(size=(count + 1, count + 1))
    np.fill_diagonal(value[:count, :count], -1e4)
    value[count, count] = -1e4
    return value


def test_calibrated_priority_changes_membership_and_greedy_order() -> None:
    grid = 3
    count = grid * grid
    right = hard_partial_axis_matching(_assignment(grid, 1), grid=grid, axis="right")
    down = hard_partial_axis_matching(_assignment(grid, 2), grid=grid, axis="down")
    matrices = {
        "right": np.zeros((count, count), dtype=np.float64),
        "down": np.zeros((count, count), dtype=np.float64),
    }
    preferred_right = right.edges[-1]
    preferred_down = down.edges[-1]
    matrices["right"][preferred_right.source, preferred_right.target] = 20.0
    matrices["down"][preferred_down.source, preferred_down.target] = 10.0
    ordered = prioritise_component_edges(
        right,
        down,
        edge_budget_per_axis=2,
        tile_count=count,
        component_edge_priority=matrices,
    )
    assert ordered[0] == preferred_right
    assert preferred_down in ordered
    assert sum(edge.axis == "right" for edge in ordered) == 2
    assert sum(edge.axis == "down" for edge in ordered) == 2


def test_decoder_priority_is_opt_in_and_default_is_unchanged() -> None:
    grid = 3
    count = grid * grid
    right = _assignment(grid, 11)
    down = _assignment(grid, 12)
    config = SocketDecoderConfig(
        component_edge_budget_per_axis=3,
        swap_edge_budget_per_axis=3,
        max_swap_steps=2,
    )
    control = decode_socket_assignments(right, down, grid=grid, config=config)
    explicit_none = decode_socket_assignments(
        right,
        down,
        grid=grid,
        config=config,
        component_edge_priority=None,
    )
    np.testing.assert_array_equal(control.layout, explicit_none.layout)
    assert not control.diagnostics.component_edge_priority_used

    priorities = {
        "right": np.random.default_rng(13).normal(size=(count, count)),
        "down": np.random.default_rng(14).normal(size=(count, count)),
    }
    candidate = decode_socket_assignments(
        right,
        down,
        grid=grid,
        config=config,
        component_edge_priority=priorities,
    )
    assert np.array_equal(np.sort(candidate.layout), np.arange(count))
    assert candidate.diagnostics.component_edge_priority_used
    assert candidate.diagnostics.swap_edge_budget_per_axis == 3


def test_component_priority_contract_fails_closed() -> None:
    grid = 3
    count = grid * grid
    right = hard_partial_axis_matching(_assignment(grid, 21), grid=grid, axis="right")
    down = hard_partial_axis_matching(_assignment(grid, 22), grid=grid, axis="down")
    with pytest.raises(ValueError, match="exactly"):
        prioritise_component_edges(
            right,
            down,
            edge_budget_per_axis=2,
            tile_count=count,
            component_edge_priority={"right": np.zeros((count, count))},
        )
    with pytest.raises(ValueError, match="shape"):
        prioritise_component_edges(
            right,
            down,
            edge_budget_per_axis=2,
            tile_count=count,
            component_edge_priority={
                "right": np.zeros((count - 1, count - 1)),
                "down": np.zeros((count, count)),
            },
        )
