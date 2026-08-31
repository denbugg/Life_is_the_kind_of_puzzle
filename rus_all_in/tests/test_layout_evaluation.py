from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.layout_evaluation import (
    RECOVERED_REFERENCE_CAVEAT,
    evaluate_layout,
)


def test_exact_layout_reports_counts_rates_and_exact_reference() -> None:
    truth = np.arange(16)
    result = evaluate_layout(truth, truth, reference_is_exact=True)

    assert result.tile_count == 16
    assert result.grid_size == 4
    assert result.correct_tile_count == 16
    assert result.direct_placement == 1.0
    assert result.correct_row_count == 16
    assert result.row_accuracy == 1.0
    assert result.correct_column_count == 16
    assert result.column_accuracy == 1.0
    assert result.translation_aligned_count == 16
    assert result.translation_aligned_placement == 1.0
    assert result.right_adjacency_correct == 12
    assert result.down_adjacency_correct == 12
    assert result.adjacency_correct == 24
    assert result.adjacency_total == 24
    assert result.adjacency == 1.0
    assert result.reference_is_exact is True
    assert result.reliability_caveat is None


def test_translated_layout_keeps_relative_geometry_but_not_absolute_columns() -> None:
    truth = np.arange(16)
    translated = np.roll(truth.reshape(4, 4), 1, axis=1).reshape(-1)
    result = evaluate_layout(translated, truth)

    assert result.correct_tile_count == 0
    assert result.direct_placement == 0.0
    assert result.correct_row_count == 16
    assert result.row_accuracy == 1.0
    assert result.correct_column_count == 0
    assert result.column_accuracy == 0.0
    assert result.translation_aligned_count == 12
    assert result.translation_aligned_placement == 0.75
    assert result.right_adjacency_correct == 8
    assert result.down_adjacency_correct == 12
    assert result.adjacency_correct == 20
    assert result.adjacency == 20 / 24
    assert result.reference_is_exact is False
    assert result.reliability_caveat == RECOVERED_REFERENCE_CAVEAT
    assert result.as_dict()["correct_tile_count"] == 0


@pytest.mark.parametrize(
    ("predicted", "reference", "message"),
    [
        ([], [], "non-zero perfect square"),
        ([0, 1, 2], [0, 1, 2], "perfect square"),
        ([0, 0, 2, 3], [0, 1, 2, 3], "permutation"),
        ([0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8], "same size"),
    ],
)
def test_layout_validation_fails_closed(
    predicted: list[int], reference: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_layout(predicted, reference)
