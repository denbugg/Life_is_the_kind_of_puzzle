from __future__ import annotations

import numpy as np
import pytest

from scripts.run_direct_residual_union_priority_opened64 import (
    COUNT,
    DECODER_EDGE_BUDGET,
    GRID,
    _edge_is_correct,
    _fixed_top144_correct,
    _strict_layout,
)


def _true_edges(axis: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(COUNT, dtype=np.int32)
    if axis == 0:
        valid = positions % GRID != GRID - 1
        return positions[valid], positions[valid] + 1
    valid = positions < COUNT - GRID
    return positions[valid], positions[valid] + GRID


def test_edge_truth_uses_noncyclic_grid_neighbours() -> None:
    reference = np.arange(COUNT, dtype=np.int32)
    horizontal = _edge_is_correct(
        np.asarray([0, GRID - 2, GRID - 1], dtype=np.int32),
        np.asarray([1, GRID - 1, GRID], dtype=np.int32),
        axis=0,
        reference=reference,
    )
    vertical = _edge_is_correct(
        np.asarray([0, COUNT - GRID - 1, COUNT - GRID], dtype=np.int32),
        np.asarray([GRID, COUNT - 1, 0], dtype=np.int32),
        axis=1,
        reference=reference,
    )
    assert horizontal.tolist() == [True, True, False]
    assert vertical.tolist() == [True, True, False]


def test_fixed_top144_scores_frozen_priorities_without_redecoding() -> None:
    prefix = "case_0000"
    archive: dict[str, np.ndarray] = {}
    for axis in (0, 1):
        source, target = _true_edges(axis)
        priority = np.linspace(1.0, 0.0, len(source), dtype=np.float64)
        archive[f"{prefix}__axis_{axis}_source"] = source
        archive[f"{prefix}__axis_{axis}_target"] = target
        archive[f"{prefix}__axis_{axis}_baseline_priority"] = priority
        archive[f"{prefix}__axis_{axis}_treatment_priority"] = priority
    assert _fixed_top144_correct(
        archive,
        prefix,
        np.arange(COUNT, dtype=np.int32),
        arm="baseline",
    ) == 2 * DECODER_EDGE_BUDGET


def test_strict_layout_rejects_duplicate_tile_identity() -> None:
    assert np.array_equal(_strict_layout(np.arange(COUNT)), np.arange(COUNT))
    duplicate = np.arange(COUNT, dtype=np.int32)
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="strict"):
        _strict_layout(duplicate)
