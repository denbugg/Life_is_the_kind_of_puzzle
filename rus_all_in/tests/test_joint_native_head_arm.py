from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from aiijc_puzzle.joint_native_head_arm import (
    FROZEN_SOLVER_CONFIG,
    dense_raw_side_costs,
    frozen_head_edges,
    reference_from_target_slots,
    solve_joint_native_head_arm,
)
from aiijc_puzzle.taska_pair_pipeline import SOLVER_CONFIG

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_joint_native_head_arm_fit as runner  # noqa: E402


def test_dense_raw_side_costs_matches_direct_mean_square() -> None:
    values = np.arange(4 * 4 * 2 * 1, dtype=np.float64).reshape(4, 4, 2, 1)
    right, down = dense_raw_side_costs(values, grid=2)
    expected_right = np.mean(np.square(values[0, :, None] - values[1, None, :]), axis=(2, 3))
    expected_down = np.mean(np.square(values[2, :, None] - values[3, None, :]), axis=(2, 3))
    assert np.allclose(right, expected_right)
    assert np.allclose(down, expected_down)


def test_joint_native_arm_is_deterministic_strict_and_uses_frozen_config() -> None:
    assert asdict(FROZEN_SOLVER_CONFIG) == asdict(SOLVER_CONFIG)
    rng = np.random.default_rng(17)
    raw_sides = rng.normal(size=(4, 9, 3, 2)).astype(np.float32)
    sources = [np.asarray([0], dtype=np.int32), np.asarray([3], dtype=np.int32)]
    targets = [np.asarray([1], dtype=np.int32), np.asarray([6], dtype=np.int32)]
    confidences = [np.asarray([0.9]), np.asarray([0.8])]
    first = solve_joint_native_head_arm(
        raw_sides,
        sources,
        targets,
        confidences,
        grid=3,
        requested_per_axis=1,
    )
    second = solve_joint_native_head_arm(
        raw_sides,
        sources,
        targets,
        confidences,
        grid=3,
        requested_per_axis=1,
    )
    assert np.array_equal(first.layout, second.layout)
    assert np.array_equal(np.sort(first.layout), np.arange(9))
    assert first.layout_sha256 == second.layout_sha256
    assert first.diagnostics["head_edge_count"] == 2


def test_fixed_head_rejects_directional_source_collision() -> None:
    with pytest.raises(ValueError, match="source/target uniqueness"):
        frozen_head_edges(
            [np.asarray([0, 0]), np.asarray([1, 2])],
            [np.asarray([1, 2]), np.asarray([4, 5])],
            [np.asarray([0.9, 0.8]), np.asarray([0.7, 0.6])],
            grid=3,
            requested_per_axis=2,
        )


def test_reference_round_trip_from_target_slots() -> None:
    grid = 3
    count = grid * grid
    reference = np.asarray([4, 1, 8, 0, 3, 2, 7, 5, 6], dtype=np.int32)
    board = reference.reshape(grid, grid)
    candidates = np.zeros((2, count, 2), dtype=np.int32)
    slots = np.full((2, count), -1, dtype=np.int16)
    for row in range(grid):
        for column in range(grid):
            tile = int(board[row, column])
            if column + 1 < grid:
                candidates[0, tile, 0] = int(board[row, column + 1])
                slots[0, tile] = 0
            if row + 1 < grid:
                candidates[1, tile, 0] = int(board[row + 1, column])
                slots[1, tile] = 0
    observed = reference_from_target_slots(candidates, slots, grid=grid)
    assert np.array_equal(observed, reference)


def test_target_free_cache_loader_does_not_materialise_unrequested_label(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.npz"
    np.savez(
        path,
        raw_sides=np.zeros((4, 4, 2, 1), dtype=np.float32),
        target_slots=np.asarray([{"poison": True}], dtype=object),
    )
    (raw_sides,) = runner._cache_arrays(path, ("raw_sides",))
    assert raw_sides.shape == (4, 4, 2, 1)
    with pytest.raises(ValueError, match="Object arrays"):
        runner._cache_arrays(path, ("target_slots",))
