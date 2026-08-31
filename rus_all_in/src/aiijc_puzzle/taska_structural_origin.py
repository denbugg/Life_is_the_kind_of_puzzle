"""Choose a global cyclic origin with a frozen structural border unary.

The selector is deliberately narrow: it keeps every relative tile relation in
one supplied layout, enumerates the ``grid * grid`` whole-board cyclic rolls,
and maximises only the unary evidence assigned to physical border slots.  It
does not consume seam costs, targets, filenames, or tile identities other than
the opaque indices needed to address the supplied unary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StructuralOriginResult:
    """One strict globally rolled layout and target-free diagnostics."""

    layout: np.ndarray
    selected_row_roll: int
    selected_column_roll: int
    selected_score: float
    unchanged_score: float

    @property
    def changed(self) -> bool:
        return bool(self.selected_row_roll or self.selected_column_roll)


def _validated_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid


def _strict_layout(value: Any, *, grid: int) -> np.ndarray:
    raw = np.asarray(value)
    count = grid * grid
    if raw.shape != (count,):
        raise ValueError(f"layout must have shape {(count,)}")
    if raw.dtype.kind not in "iu" or raw.dtype == np.dtype(bool):
        raise TypeError("layout must have an integer dtype")
    layout = np.ascontiguousarray(raw, dtype=np.int32)
    if not np.array_equal(np.sort(layout), np.arange(count, dtype=np.int32)):
        raise ValueError("layout must use every original tile exactly once")
    return layout


def _validated_unary(value: Any, *, grid: int) -> np.ndarray:
    raw = np.asarray(value)
    expected = (grid * grid, grid, grid)
    if raw.shape != expected:
        raise ValueError(f"border_unary must have shape {expected}")
    if raw.dtype.kind not in "iuf" or raw.dtype == np.dtype(bool):
        raise TypeError("border_unary must have a real numeric dtype")
    unary = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(unary).all():
        raise ValueError("border_unary must contain only finite values")
    return unary


def select_structural_border_cyclic_origin(
    layout: Any,
    border_unary: Any,
    *,
    grid: int = 24,
) -> StructuralOriginResult:
    """Select the highest-scoring whole-board roll, ties in row-major order.

    Only physical border positions are scored, even if a caller supplies
    non-zero interior unary entries.  ``(0, 0)`` is considered first and an
    exact tie never replaces the earlier row-major candidate.
    """

    size = _validated_grid(grid)
    strict = _strict_layout(layout, grid=size)
    unary = _validated_unary(border_unary, grid=size)
    board = strict.reshape(size, size)
    rows, columns = np.indices((size, size))
    border = (rows == 0) | (rows == size - 1) | (columns == 0) | (columns == size - 1)
    border_rows = rows[border]
    border_columns = columns[border]

    best_score = -np.inf
    best_roll = (0, 0)
    best_layout: np.ndarray | None = None
    unchanged_score: float | None = None
    for row_roll in range(size):
        for column_roll in range(size):
            rolled = np.roll(
                board,
                shift=(row_roll, column_roll),
                axis=(0, 1),
            )
            border_tiles = rolled[border]
            score = float(unary[border_tiles, border_rows, border_columns].sum())
            if row_roll == 0 and column_roll == 0:
                unchanged_score = score
            if score > best_score:
                best_score = score
                best_roll = (row_roll, column_roll)
                best_layout = np.ascontiguousarray(rolled.reshape(-1), dtype=np.int32)

    if best_layout is None or unchanged_score is None or not np.isfinite(best_score):
        raise RuntimeError("structural origin selection failed")
    if not np.array_equal(np.sort(best_layout), np.arange(size * size, dtype=np.int32)):
        raise RuntimeError("structural origin selection broke the strict permutation")
    best_layout.setflags(write=False)
    return StructuralOriginResult(
        layout=best_layout,
        selected_row_roll=best_roll[0],
        selected_column_roll=best_roll[1],
        selected_score=best_score,
        unchanged_score=unchanged_score,
    )


__all__ = ["StructuralOriginResult", "select_structural_border_cyclic_origin"]
