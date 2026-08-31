"""Dirty-only absolute anchoring for an already assembled socket board.

Local puzzle solvers can recover useful relative fragments while leaving the
whole board at an arbitrary cyclic origin.  This module evaluates all
``grid**2`` cyclic row/column translations of a strict layout and selects the
one best supported by the same two inference-visible signals produced by
``SocketMatcher``:

* right/down real-socket compatibility determines where the two global cuts
  should fall;
* four dustbin probabilities determine which tiles plausibly touch the frame.

No pixels are replaced, no target/reference layout is accepted, and the tile
permutation is preserved exactly.  The operation is deliberately bounded: it
does not change relative coordinates except at the selected cyclic cuts.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.socket_decoder import socket_border_unary, socket_layout_objective


@dataclass(frozen=True)
class CyclicTranslationConfig:
    """Frozen weights for whole-board cyclic translation selection."""

    # The base decoder's 0.20 weight was too weak to move 31/32 development
    # layouts.  A single fixed weight 5.0 was selected there, then confirmed
    # without a sweep on 48 fresh exact-synthetic boards.
    border_weight: float = 5.0
    minimum_gain: float = 1e-9

    def validate(self) -> None:
        if not np.isfinite(self.border_weight) or self.border_weight < 0:
            raise ValueError("border_weight must be finite and non-negative")
        if not np.isfinite(self.minimum_gain) or self.minimum_gain < 0:
            raise ValueError("minimum_gain must be finite and non-negative")


@dataclass(frozen=True)
class CyclicTranslationDiagnostics:
    """Auditable evidence for one target-blind translation decision."""

    grid_size: int
    tile_count: int
    candidates_evaluated: int
    selected_row_roll: int
    selected_column_roll: int
    changed: bool
    border_weight: float
    initial_objective: float
    final_objective: float
    objective_gain: float
    strict_permutation: bool
    runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CyclicTranslationResult:
    """A strict tile-at-position layout and its translation diagnostics."""

    layout: np.ndarray
    diagnostics: CyclicTranslationDiagnostics

    def report(self, *, include_layout: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "placer": "socket-global-cyclic-translation-v1",
            "layout_sha256": hashlib.sha256(
                np.asarray(self.layout, dtype="<i4").tobytes()
            ).hexdigest(),
            "diagnostics": self.diagnostics.as_dict(),
        }
        if include_layout:
            payload["tile_at_position"] = self.layout.tolist()
        return payload


def _as_square_assignment(value: Any, *, count: int, name: str) -> np.ndarray:
    matrix = value
    if hasattr(matrix, "detach"):
        matrix = matrix.detach()
    if hasattr(matrix, "cpu"):
        matrix = matrix.cpu()
    if hasattr(matrix, "numpy"):
        matrix = matrix.numpy()
    result = np.asarray(matrix, dtype=np.float64)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    expected = (count + 1, count + 1)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
    usable = result.copy()
    usable[count, count] = 0.0
    if not np.isfinite(usable).all():
        raise ValueError(f"{name} contains non-finite usable entries")
    return np.ascontiguousarray(result)


def _strict_layout(value: Any, *, count: int) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (count,):
        raise ValueError(f"layout must have shape {(count,)}, got {layout.shape}")
    if not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError("layout must be a strict tile permutation")
    return np.ascontiguousarray(layout)


def select_global_cyclic_translation(
    layout: Any,
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int = 24,
    config: CyclicTranslationConfig | None = None,
) -> CyclicTranslationResult:
    """Choose a global cyclic origin using socket cut and border evidence.

    ``layout`` uses the canonical tile-at-position convention.  Positive rolls
    follow :func:`numpy.roll`: a row roll of one moves the previous last row to
    the top.  Shift ``(0, 0)`` is always a candidate and wins exact/near ties,
    so enabling the placer cannot reduce its declared dirty-only objective.
    """

    started = perf_counter()
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least 2")
    config = CyclicTranslationConfig() if config is None else config
    config.validate()
    count = grid * grid
    initial = _strict_layout(layout, count=count)
    right = _as_square_assignment(
        right_log_assignment, count=count, name="right_log_assignment"
    )
    down = _as_square_assignment(
        down_log_assignment, count=count, name="down_log_assignment"
    )
    right_real = right[:count, :count]
    down_real = down[:count, :count]
    border = socket_border_unary(right, down, grid=grid)

    initial_objective = socket_layout_objective(
        initial,
        right_real,
        down_real,
        border,
        grid=grid,
        border_weight=config.border_weight,
    )
    board = initial.reshape(grid, grid)
    best_layout = initial
    best_objective = initial_objective
    best_roll = (0, 0)
    # Start at (0, 0) and require a real gain.  This makes ties stable and
    # prevents a flat/noisy border head from introducing an arbitrary origin.
    for row_roll in range(grid):
        for column_roll in range(grid):
            if row_roll == 0 and column_roll == 0:
                continue
            candidate = np.roll(board, shift=(row_roll, column_roll), axis=(0, 1)).reshape(-1)
            objective = socket_layout_objective(
                candidate,
                right_real,
                down_real,
                border,
                grid=grid,
                border_weight=config.border_weight,
            )
            if objective > best_objective + config.minimum_gain:
                best_layout = np.ascontiguousarray(candidate, dtype=np.int32)
                best_objective = objective
                best_roll = (row_roll, column_roll)

    best_layout = _strict_layout(best_layout, count=count)
    if best_objective + 1e-7 < initial_objective:
        raise RuntimeError("cyclic translation decreased its declared objective")
    diagnostics = CyclicTranslationDiagnostics(
        grid_size=grid,
        tile_count=count,
        candidates_evaluated=count,
        selected_row_roll=best_roll[0],
        selected_column_roll=best_roll[1],
        changed=best_roll != (0, 0),
        border_weight=float(config.border_weight),
        initial_objective=float(initial_objective),
        final_objective=float(best_objective),
        objective_gain=float(best_objective - initial_objective),
        strict_permutation=True,
        runtime_seconds=perf_counter() - started,
    )
    return CyclicTranslationResult(best_layout, diagnostics)
