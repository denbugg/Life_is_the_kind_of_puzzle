from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.tile_position_distance import (
    evaluate_best_cyclic_aligned_tile_position_distance,
    evaluate_tile_position_distance,
)


def test_exact_layout_has_zero_distance_and_unit_radius_recalls() -> None:
    reference = np.arange(16, dtype=np.int32)
    metrics = evaluate_tile_position_distance(reference, reference, grid=4)

    assert metrics.exact_tile_count == 16
    assert metrics.mean_manhattan_cells == 0.0
    assert metrics.median_manhattan_cells == 0.0
    assert metrics.p90_manhattan_cells == 0.0
    assert metrics.normalized_mean_l1 == 0.0
    assert metrics.mean_euclidean_cells == 0.0
    assert metrics.within_radius_0_recall == 1.0
    assert metrics.within_radius_1_recall == 1.0
    assert metrics.within_radius_2_recall == 1.0


def test_radius_zero_recall_is_exact_tile_fraction() -> None:
    reference = np.arange(9, dtype=np.int32)
    candidate = reference.copy()
    candidate[[0, 1]] = candidate[[1, 0]]
    metrics = evaluate_tile_position_distance(candidate, reference, grid=3)

    assert metrics.exact_tile_count == 7
    assert metrics.within_radius_0_recall == pytest.approx(7 / 9)
    assert metrics.within_radius_1_recall == 1.0
    assert metrics.mean_manhattan_cells == pytest.approx(2 / 9)
    assert metrics.normalized_mean_l1 == pytest.approx((2 / 9) / 4)


def test_global_roll_is_visible_absolute_but_perfect_after_alignment() -> None:
    grid = 4
    reference = np.arange(grid * grid, dtype=np.int32)
    candidate = np.roll(
        reference.reshape(grid, grid), shift=(1, -1), axis=(0, 1)
    ).reshape(-1)

    absolute = evaluate_tile_position_distance(candidate, reference, grid=grid)
    aligned = evaluate_best_cyclic_aligned_tile_position_distance(
        candidate, reference, grid=grid
    )

    assert absolute.exact_tile_count == 0
    assert absolute.mean_manhattan_cells > 0
    assert aligned.selected_row_roll == 3
    assert aligned.selected_column_roll == 1
    assert aligned.changed
    assert aligned.candidates_evaluated == 16
    assert aligned.metrics.exact_tile_count == 16
    assert aligned.metrics.mean_manhattan_cells == 0.0


def test_non_permutation_fails_closed() -> None:
    reference = np.arange(9, dtype=np.int32)
    candidate = reference.copy()
    candidate[-1] = candidate[0]
    with pytest.raises(ValueError, match="every tile identity"):
        evaluate_tile_position_distance(candidate, reference, grid=3)
