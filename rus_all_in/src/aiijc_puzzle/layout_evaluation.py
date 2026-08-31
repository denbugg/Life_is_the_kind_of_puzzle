"""Reusable geometry metrics for strict tile-at-position permutations.

The organizer data contains clean target images but no authoritative permutation
labels.  Metrics computed against a target-assisted recovered permutation are
therefore diagnostics, not proof of exact placement against hidden labels.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

RECOVERED_REFERENCE_CAVEAT = (
    "Organizer permutation labels are unavailable. The reference permutation was "
    "recovered from dirty/clean pixels and may mislabel ambiguous or heavily corrupted tiles."
)


@dataclass(frozen=True)
class LayoutEvaluation:
    """Counts and rates for one strict predicted/reference layout pair."""

    tile_count: int
    grid_size: int
    correct_tile_count: int
    direct_placement: float
    correct_row_count: int
    row_accuracy: float
    correct_column_count: int
    column_accuracy: float
    translation_aligned_count: int
    translation_aligned_placement: float
    right_adjacency_correct: int
    right_adjacency_total: int
    right_adjacency: float
    down_adjacency_correct: int
    down_adjacency_total: int
    down_adjacency: float
    adjacency_correct: int
    adjacency_total: int
    adjacency: float
    reference_is_exact: bool
    reliability_caveat: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def _strict_square_permutation(value: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {result.shape}")
    grid_size = int(np.sqrt(len(result)))
    if not len(result) or grid_size * grid_size != len(result):
        raise ValueError(f"{name} length must be a non-zero perfect square, got {len(result)}")
    if not np.array_equal(np.sort(result), np.arange(len(result))):
        raise ValueError(f"{name} must be a permutation of 0..{len(result) - 1}")
    return result


def evaluate_layout(
    predicted_tile_at_position: Sequence[int] | np.ndarray,
    reference_tile_at_position: Sequence[int] | np.ndarray,
    *,
    reference_is_exact: bool = False,
) -> LayoutEvaluation:
    """Evaluate a strict layout against an exact or target-assisted reference.

    Both arrays use the canonical ``tile_at_position`` convention: element ``p``
    is the shuffled-input tile index placed at row-major grid position ``p``.
    Translation alignment is diagnostic only; direct placement remains the
    literal absolute-position metric.
    """

    predicted = _strict_square_permutation(
        predicted_tile_at_position, name="predicted_tile_at_position"
    )
    reference = _strict_square_permutation(
        reference_tile_at_position, name="reference_tile_at_position"
    )
    if predicted.shape != reference.shape:
        raise ValueError(
            "predicted and reference layouts must have the same size, got "
            f"{predicted.shape} and {reference.shape}"
        )

    tile_count = len(predicted)
    grid_size = int(np.sqrt(tile_count))
    reference_position = np.empty(tile_count, dtype=np.int64)
    predicted_position = np.empty(tile_count, dtype=np.int64)
    reference_position[reference] = np.arange(tile_count)
    predicted_position[predicted] = np.arange(tile_count)

    reference_rows, reference_columns = divmod(reference_position, grid_size)
    predicted_rows, predicted_columns = divmod(predicted_position, grid_size)
    correct_tile_count = int(np.count_nonzero(predicted == reference))
    correct_row_count = int(np.count_nonzero(predicted_rows == reference_rows))
    correct_column_count = int(np.count_nonzero(predicted_columns == reference_columns))

    shifts = Counter(
        zip(
            (reference_rows - predicted_rows).tolist(),
            (reference_columns - predicted_columns).tolist(),
            strict=True,
        )
    )
    translation_aligned_count = max(shifts.values())

    board = predicted.reshape(grid_size, grid_size)
    left = reference_position[board[:, :-1]]
    right = reference_position[board[:, 1:]]
    top = reference_position[board[:-1]]
    bottom = reference_position[board[1:]]
    right_correct = int(
        np.count_nonzero((right - left == 1) & (right // grid_size == left // grid_size))
    )
    down_correct = int(np.count_nonzero(bottom - top == grid_size))
    directional_total = grid_size * (grid_size - 1)
    adjacency_correct = right_correct + down_correct
    adjacency_total = 2 * directional_total

    return LayoutEvaluation(
        tile_count=tile_count,
        grid_size=grid_size,
        correct_tile_count=correct_tile_count,
        direct_placement=correct_tile_count / tile_count,
        correct_row_count=correct_row_count,
        row_accuracy=correct_row_count / tile_count,
        correct_column_count=correct_column_count,
        column_accuracy=correct_column_count / tile_count,
        translation_aligned_count=translation_aligned_count,
        translation_aligned_placement=translation_aligned_count / tile_count,
        right_adjacency_correct=right_correct,
        right_adjacency_total=directional_total,
        right_adjacency=right_correct / directional_total,
        down_adjacency_correct=down_correct,
        down_adjacency_total=directional_total,
        down_adjacency=down_correct / directional_total,
        adjacency_correct=adjacency_correct,
        adjacency_total=adjacency_total,
        adjacency=adjacency_correct / adjacency_total,
        reference_is_exact=reference_is_exact,
        reliability_caveat=None if reference_is_exact else RECOVERED_REFERENCE_CAVEAT,
    )
