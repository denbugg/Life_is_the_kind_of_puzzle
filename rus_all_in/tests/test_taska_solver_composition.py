from __future__ import annotations

import numpy as np

from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_solver_composition import (
    select_then_polish_taska_layout,
    select_then_polish_taska_layouts,
)


def test_composition_selects_then_monotonically_polishes() -> None:
    generator = np.random.default_rng(41)
    raw = generator.permutation(16).astype(np.int32)
    calibrated = generator.permutation(16).astype(np.int32)
    right = generator.normal(size=(16, 16))
    down = generator.normal(size=(16, 16))

    result = select_then_polish_taska_layout(
        raw,
        calibrated,
        right,
        down,
        (),
        grid=4,
        max_swaps=5,
    )

    selected_cost = total_taska_adjacent_seam_cost(
        result.selection.layout,
        right,
        down,
        grid=4,
    )
    final_cost = total_taska_adjacent_seam_cost(
        result.polish.layout,
        right,
        down,
        grid=4,
    )
    assert final_cost <= selected_cost
    assert np.array_equal(np.sort(result.polish.layout), np.arange(16))
    assert not result.polish.layout.flags.writeable


def test_named_composition_keeps_choice_provenance() -> None:
    raw = np.arange(9, dtype=np.int32)
    focal = raw[::-1].copy()
    costs = np.zeros((9, 9), dtype=np.float64)
    result = select_then_polish_taska_layouts(
        {"raw": raw, "focal": focal},
        costs,
        costs,
        (),
        grid=3,
    )
    assert result.selection.choice == "raw"
    assert np.array_equal(result.polish.layout, raw)
