from __future__ import annotations

import numpy as np

from puzzle_assembly.axis_refine import (
    alternating_axis_refine,
    band_cost_matrix,
    solve_band_path,
)
from puzzle_assembly.compatibility import CompatibilityMatrices


def _perfect_scores() -> CompatibilityMatrices:
    count = 24 * 24
    right = np.full((count, count), 10.0, dtype=np.float32)
    down = np.full((count, count), 10.0, dtype=np.float32)
    grid = np.arange(count).reshape(24, 24)
    right[grid[:, :-1], grid[:, 1:]] = 0.0
    down[grid[:-1, :], grid[1:, :]] = 0.0
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices("perfect", right, down)


def test_row_cost_and_path_recover_permuted_rows() -> None:
    scores = _perfect_scores()
    permutation = np.asarray(
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
    )
    layout = np.arange(24 * 24).reshape(24, 24)[permutation].ravel()
    costs = band_cost_matrix(
        layout, scores, axis="row", aggregation="mean", rank_normalize=False
    )
    order, value = solve_band_path(
        costs,
        np.zeros(24),
        np.zeros(24),
        boundary_weight=0.0,
        random_restarts=0,
        local_passes=4,
    )
    recovered = permutation[order]
    assert recovered.tolist() == list(range(24))
    assert value == 0.0


def test_alternating_refine_is_deterministic_and_valid() -> None:
    scores = _perfect_scores()
    row_order = np.roll(np.arange(24), 5)
    column_order = np.roll(np.arange(24), 9)
    layout = np.arange(24 * 24).reshape(24, 24)[row_order][:, column_order].ravel()
    first = alternating_axis_refine(
        layout,
        scores,
        cycles=2,
        aggregation="mean",
        rank_normalize=False,
        boundary_weight=0.0,
        random_restarts=0,
        local_passes=4,
        seam_guard_ratio=1.0,
        seed=123,
    )
    second = alternating_axis_refine(
        layout,
        scores,
        cycles=2,
        aggregation="mean",
        rank_normalize=False,
        boundary_weight=0.0,
        random_restarts=0,
        local_passes=4,
        seam_guard_ratio=1.0,
        seed=123,
    )
    assert np.array_equal(first.position_to_slot, second.position_to_slot)
    assert set(first.position_to_slot.tolist()) == set(range(24 * 24))
    assert first.objective_after <= first.objective_before
