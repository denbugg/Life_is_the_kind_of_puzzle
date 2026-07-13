from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.contextual_confidence import (
    CONFIDENCE_MAP_NAMES,
    apply_confidence_to_fixed_candidate,
    solver_layout_confidence,
)
from puzzle_assembly.geometry import GRID, TILE_COUNT


def _synthetic_grid_compatibility() -> CompatibilityMatrices:
    right = np.full((TILE_COUNT, TILE_COUNT), 1.0, dtype=np.float32)
    down = np.full_like(right, 1.0)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    grid = np.arange(TILE_COUNT, dtype=np.int32).reshape(GRID, GRID)
    for row in range(GRID):
        for column in range(GRID):
            tile = int(grid[row, column])
            if column + 1 < GRID:
                right[tile, int(grid[row, column + 1])] = 0.0
            if row + 1 < GRID:
                down[tile, int(grid[row + 1, column])] = 0.0
    # Deterministic non-ties for true outer sides without changing internal
    # reciprocal top-1 joins.
    jitter = np.arange(TILE_COUNT, dtype=np.float32) * 1e-6
    right += jitter[None, :]
    down += jitter[None, ::-1]
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices("synthetic_grid", right, down)


def test_solver_layout_confidence_contract_and_signal() -> None:
    compatibility = _synthetic_grid_compatibility()
    exact = np.arange(TILE_COUNT, dtype=np.int32)
    result = solver_layout_confidence(compatibility, exact)
    assert tuple(result.maps) == CONFIDENCE_MAP_NAMES
    for values in result.maps.values():
        assert values.shape == (GRID, GRID)
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
        assert float(values.min()) >= 0.0
        assert float(values.max()) <= 1.0
    assert result.diagnostics["placed_edge_count"] == 2 * GRID * (GRID - 1)
    assert result.diagnostics["mutual_top1_placed_edge_count"] >= 2 * GRID * (GRID - 1) - 2 * GRID
    assert float(result.maps["rank_gap_cycle"].mean()) > 0.8

    random_layout = np.random.default_rng(20260712).permutation(TILE_COUNT)
    random_result = solver_layout_confidence(compatibility, random_layout)
    assert float(result.maps["rank_gap_pair"].mean()) > 10.0 * max(
        float(random_result.maps["rank_gap_pair"].mean()), 1e-12
    )


def test_confidence_gate_identity_and_full_candidate() -> None:
    rng = np.random.default_rng(17)
    base = rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    candidate = np.clip(base.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    zeros = np.zeros((GRID, GRID), dtype=np.float32)
    ones = np.ones_like(zeros)
    assert np.array_equal(
        apply_confidence_to_fixed_candidate(
            base, candidate, zeros, threshold=0.0, strength=1.0
        ),
        base,
    )
    assert np.array_equal(
        apply_confidence_to_fixed_candidate(
            base, candidate, ones, threshold=0.0, strength=1.0
        ),
        candidate,
    )
    assert np.array_equal(
        apply_confidence_to_fixed_candidate(
            base, candidate, 0.9 * ones, threshold=0.95, strength=1.0
        ),
        base,
    )


def test_confidence_public_api_is_target_blind() -> None:
    forbidden = {"target", "truth", "slot_to_target", "source", "filename", "clean"}
    for function in (solver_layout_confidence, apply_confidence_to_fixed_candidate):
        assert not (set(inspect.signature(function).parameters) & forbidden)


def test_confidence_rejects_nonfinite_off_diagonal() -> None:
    compatibility = _synthetic_grid_compatibility()
    right = compatibility.right.copy()
    right[0, 1] = np.inf
    broken = CompatibilityMatrices("broken", right, compatibility.down)
    with pytest.raises(ValueError, match="off-diagonal"):
        solver_layout_confidence(broken, np.arange(TILE_COUNT, dtype=np.int32))


def test_v2_runner_help_imports_in_project_environment() -> None:
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/evaluate_contextual_confidence_v2.py"),
            "--help",
        ],
        cwd=repo,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "freeze-development" in completed.stdout
