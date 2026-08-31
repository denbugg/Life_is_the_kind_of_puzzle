from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail


def test_polish_is_strict_monotone_and_read_only() -> None:
    generator = np.random.default_rng(19)
    layout = generator.permutation(16).astype(np.int32)
    right = generator.normal(size=(16, 16))
    down = generator.normal(size=(16, 16))

    result = polish_unprotected_taska_tail(
        layout,
        right,
        down,
        (),
        grid=4,
        max_swaps=5,
    )

    assert np.array_equal(np.sort(result.layout), np.arange(16))
    assert not result.layout.flags.writeable
    assert result.diagnostics.accepted_swap_count <= 5
    assert result.diagnostics.final_total_cost <= result.diagnostics.initial_total_cost
    assert result.diagnostics.final_total_cost == pytest.approx(
        total_taska_adjacent_seam_cost(result.layout, right, down, grid=4)
    )


def test_initially_realised_edge_tiles_never_move() -> None:
    generator = np.random.default_rng(23)
    layout = np.arange(16, dtype=np.int32)
    right = generator.normal(size=(16, 16))
    down = generator.normal(size=(16, 16))
    edge = RawTailEdge(0, 1, "right")

    result = polish_unprotected_taska_tail(
        layout,
        right,
        down,
        (edge,),
        grid=4,
        max_swaps=12,
    )

    assert result.layout[0] == 0
    assert result.layout[1] == 1
    assert result.diagnostics.protected_tile_count == 2
    assert result.diagnostics.initial_realised_edge_count == 1
    assert result.diagnostics.final_realised_edge_count >= 1


def test_zero_cost_tie_keeps_layout_unchanged() -> None:
    layout = np.asarray([3, 1, 0, 2], dtype=np.int32)
    costs = np.zeros((4, 4), dtype=np.float64)
    result = polish_unprotected_taska_tail(
        layout,
        costs,
        costs,
        (),
        grid=2,
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
        polish_unprotected_taska_tail(layout, costs, costs, (), **controls)


def test_invalid_candidate_edge_is_rejected() -> None:
    layout = np.arange(4)
    costs = np.zeros((4, 4))
    with pytest.raises(ValueError, match="out-of-range"):
        polish_unprotected_taska_tail(
            layout,
            costs,
            costs,
            (RawTailEdge(0, 9, "right"),),
            grid=2,
        )
