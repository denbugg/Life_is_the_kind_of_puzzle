from __future__ import annotations

import numpy as np

from aiijc_puzzle.taska_focal_gated_protected_tail import (
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_focal_gated_tail_capacity import (
    FOCAL_GATED_CANDIDATE_MAX_SWAPS,
    FOCAL_GATED_CONTROL_MAX_SWAPS,
    compare_focal_gated_tail96_to_tail192,
)


def test_capacity_control_is_exact_existing_focal_gated_tail96() -> None:
    generator = np.random.default_rng(17)
    grid = 4
    count = grid * grid
    layout = generator.permutation(count).astype(np.int32)
    right = generator.uniform(size=(count, count))
    down = generator.uniform(size=(count, count))
    edges = ()
    logits = np.empty(0, dtype=np.float64)
    expected = polish_taska_tail_with_focal_gate(
        layout, right, down, edges, logits, grid=grid
    )
    result = compare_focal_gated_tail96_to_tail192(
        layout, right, down, edges, logits, grid=grid
    )
    np.testing.assert_array_equal(result.control.layout, expected.layout)
    assert result.control.diagnostics == expected.diagnostics
    assert result.diagnostics.control_max_swaps == FOCAL_GATED_CONTROL_MAX_SWAPS
    assert result.diagnostics.candidate_max_swaps == FOCAL_GATED_CANDIDATE_MAX_SWAPS
    assert result.candidate.diagnostics.final_total_cost <= (
        result.control.diagnostics.tail.final_total_cost + 1e-12
    )
    for candidate in (result.control.layout, result.candidate.layout):
        np.testing.assert_array_equal(np.sort(candidate), np.arange(count))


def test_capacity_has_no_runtime_budget_argument() -> None:
    assert FOCAL_GATED_CONTROL_MAX_SWAPS == 96
    assert FOCAL_GATED_CANDIDATE_MAX_SWAPS == 192
