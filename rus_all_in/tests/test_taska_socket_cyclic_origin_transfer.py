from __future__ import annotations

import numpy as np

from aiijc_puzzle.taska_socket_cyclic_origin_transfer import (
    transfer_socket_cyclic_origin,
)


def _flat_assignments(grid: int) -> tuple[np.ndarray, np.ndarray]:
    count = grid * grid
    right = np.full((count + 1, count + 1), -10.0, dtype=np.float64)
    down = right.copy()
    right[count, count] = 0.0
    down[count, count] = 0.0
    return right, down


def test_transfer_preserves_strict_upright_permutation() -> None:
    grid = 3
    count = grid * grid
    control = np.arange(count, dtype=np.int32)
    right, down = _flat_assignments(grid)
    result = transfer_socket_cyclic_origin(control, right, down, grid=grid)

    assert np.array_equal(np.sort(result.layout), control)
    assert result.diagnostics.candidates_evaluated == count
    assert result.diagnostics.border_weight == 5.0


def test_transfer_keeps_zero_roll_on_exactly_flat_evidence() -> None:
    grid = 3
    count = grid * grid
    control = np.arange(count, dtype=np.int32)
    right = np.zeros((count + 1, count + 1), dtype=np.float64)
    down = np.zeros_like(right)
    result = transfer_socket_cyclic_origin(control, right, down, grid=grid)

    assert np.array_equal(result.layout, control)
    assert result.diagnostics.selected_row_roll == 0
    assert result.diagnostics.selected_column_roll == 0
    assert not result.diagnostics.changed
