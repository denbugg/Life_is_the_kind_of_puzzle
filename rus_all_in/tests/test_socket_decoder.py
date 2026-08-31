from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    hard_partial_axis_matching,
    texture_centrality_unary,
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
    elif axis == "down":
        for column in range(grid):
            value[count, board[0, column]] = 0.0
            value[board[-1, column], count] = 0.0
            for row in range(grid - 1):
                value[board[row, column], board[row + 1, column]] = 0.0
    else:
        raise ValueError(axis)
    return value


def test_hard_partial_projection_enforces_exact_cardinality_and_one_to_one() -> None:
    grid = 4
    layout = np.random.default_rng(7).permutation(grid * grid)
    matching = hard_partial_axis_matching(
        _perfect_assignment(layout, grid=grid, axis="right"),
        grid=grid,
        axis="right",
    )
    assert len(matching.edges) == grid * (grid - 1)
    assert len(matching.outgoing_unmatched) == grid
    assert len(matching.incoming_unmatched) == grid
    assert len({edge.source for edge in matching.edges}) == len(matching.edges)
    assert len({edge.target for edge in matching.edges}) == len(matching.edges)


def test_global_projection_can_reject_a_mutual_top1_distractor() -> None:
    # Two real links and two unmatched sockets are required on a 2x2 axis.
    # 0->3 is the mutual row/column top-1 distractor, but selecting it would
    # force the very bad 2->1 edge.  Exact-cardinality matching instead keeps
    # the globally consistent pair 0->1 and 2->3.
    grid = 2
    count = grid * grid
    value = np.full((count + 1, count + 1), -30.0, dtype=np.float64)
    value[count, count] = -1e4
    value[0, 1] = 10.0
    value[2, 3] = 10.0
    value[0, 3] = 11.0
    value[2, 1] = -20.0
    value[1, count] = value[3, count] = 8.0
    value[count, 0] = value[count, 2] = 8.0

    matching = hard_partial_axis_matching(value, grid=grid, axis="right")
    assert {(edge.source, edge.target) for edge in matching.edges} == {(0, 1), (2, 3)}


def test_oracle_socket_constraints_recover_complete_grid_exactly() -> None:
    grid = 4
    reference = np.random.default_rng(19).permutation(grid * grid)
    result = decode_socket_assignments(
        _perfect_assignment(reference, grid=grid, axis="right"),
        _perfect_assignment(reference, grid=grid, axis="down"),
        grid=grid,
    )
    assert np.array_equal(result.layout, reference)
    assert result.diagnostics.component_count == 1
    assert result.diagnostics.largest_component == grid * grid
    assert result.diagnostics.added_constraints == grid * grid - 1
    assert result.diagnostics.contradiction_rejections == 0
    assert result.diagnostics.strict_permutation
    report = result.report(include_layout=True)
    assert report["decoder"] == "socket-translation-components-qap-v1"
    assert report["tile_at_position"] == reference.tolist()
    assert len(report["layout_sha256"]) == 64


def test_noisy_decoder_is_strict_and_bounded_polish_never_lowers_energy() -> None:
    grid = 4
    count = grid * grid
    generator = np.random.default_rng(23)
    right = generator.normal(size=(count + 1, count + 1))
    down = generator.normal(size=(count + 1, count + 1))
    right[count, count] = down[count, count] = -1e4
    result = decode_socket_assignments(
        right,
        down,
        grid=grid,
        config=SocketDecoderConfig(max_swap_steps=6),
    )
    assert np.array_equal(np.sort(result.layout), np.arange(count))
    assert 0 <= result.diagnostics.accepted_swaps <= 6
    assert result.diagnostics.final_objective >= result.diagnostics.initial_objective - 1e-7
    assert result.diagnostics.objective_gain >= -1e-7


def test_decoder_validates_assignment_shape_and_budget() -> None:
    with pytest.raises(ValueError, match="shape"):
        decode_socket_assignments(np.zeros((5, 5)), np.zeros((5, 5)), grid=4)
    assignment = np.zeros((17, 17), dtype=np.float64)
    with pytest.raises(ValueError, match="component_edge_budget"):
        decode_socket_assignments(
            assignment,
            assignment,
            grid=4,
            config=SocketDecoderConfig(component_edge_budget_per_axis=13),
        )


def test_optional_component_shift_unary_is_strictly_off_by_default() -> None:
    grid = 4
    count = grid * grid
    generator = np.random.default_rng(31)
    right = generator.normal(size=(count + 1, count + 1))
    down = generator.normal(size=(count + 1, count + 1))
    right[count, count] = down[count, count] = -1e4
    semantic_anchor = generator.normal(size=(count, count)) * 1000.0
    control = decode_socket_assignments(
        right,
        down,
        grid=grid,
        config=SocketDecoderConfig(max_swap_steps=0),
    )
    declared_but_disabled = decode_socket_assignments(
        right,
        down,
        grid=grid,
        config=SocketDecoderConfig(max_swap_steps=0),
        component_shift_unary=semantic_anchor,
    )
    assert np.array_equal(control.layout, declared_but_disabled.layout)
    assert not declared_but_disabled.diagnostics.component_shift_unary_used

    enabled = decode_socket_assignments(
        right,
        down,
        grid=grid,
        config=SocketDecoderConfig(
            max_swap_steps=0,
            component_shift_unary_weight=0.1,
        ),
        component_shift_unary=semantic_anchor,
    )
    assert np.array_equal(np.sort(enabled.layout), np.arange(count))
    assert enabled.diagnostics.component_shift_unary_used


def test_texture_centrality_is_soft_and_leaves_monochrome_tiles_unanchored() -> None:
    grid = 4
    count = grid * grid
    tiles = np.full((count, 12, 12, 3), 96, dtype=np.uint8)
    checkerboard = (np.indices((12, 12)).sum(axis=0) % 2) * 255
    tiles[-1] = checkerboard[..., None]

    unary = texture_centrality_unary(tiles, grid=grid)
    assert unary.shape == (count, count)
    assert unary.dtype == np.float32
    assert np.allclose(unary[:-1], 0.0)
    assert np.allclose(unary.mean(axis=1), 0.0, atol=1e-6)

    centre_slot = grid + 1
    corner_slot = 0
    assert unary[-1, centre_slot] > unary[-1, corner_slot]
    assert np.unique(unary[-1]).size > 2  # radial field, not a hard centre mask


def test_texture_centrality_is_scale_and_tile_order_equivariant() -> None:
    grid = 3
    generator = np.random.default_rng(101)
    tiles = generator.integers(0, 256, size=(grid * grid, 8, 8, 3), dtype=np.uint8)
    expected = texture_centrality_unary(tiles, grid=grid)
    floating = texture_centrality_unary(tiles.astype(np.float32) / 255.0, grid=grid)
    assert np.allclose(expected, floating, atol=1e-6)

    permutation = generator.permutation(grid * grid)
    permuted = texture_centrality_unary(tiles[permutation], grid=grid)
    assert np.allclose(permuted, expected[permutation], atol=1e-6)


def test_texture_centrality_validates_input_contract() -> None:
    with pytest.raises(ValueError, match="shape"):
        texture_centrality_unary(np.zeros((4, 8, 8)), grid=2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        texture_centrality_unary(np.full((4, 8, 8, 3), 2.0), grid=2)
