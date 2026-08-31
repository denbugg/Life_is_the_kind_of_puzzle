"""Absolute-coordinate anchoring for an already assembled cyclic puzzle.

The Socket decoder often recovers useful relative geometry but leaves the
whole board at the wrong toroidal origin.  This module evaluates only the
``grid**2`` global cyclic translations of a strict tile permutation.  It uses
the absolute-coordinate head's row/column logits and, optionally, the frozen
Socket border/cut objective.  No component is deformed and no tile is added,
removed, replaced, or warped.

The implementation is deliberately model-agnostic: callers may pass logits
from the ordinary coordinate view or any state-dict-neutral fused view (for
example transpose-averaged logits) as long as they are expressed in the
original board frame.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.socket_decoder import socket_border_unary, socket_layout_objective


@dataclass(frozen=True)
class CoordinateCyclicConfig:
    """Weights for two independent row/column cyclic-origin profiles.

    Every non-constant profile is centred and divided by one positive
    board-global standard deviation before blending.  This preserves the
    argmax of an individual score family while giving the coordinate and
    Socket profiles comparable units.  It never rescales tiles independently.
    """

    row_coordinate_weight: float = 1.0
    row_socket_weight: float = 0.0
    column_coordinate_weight: float = 1.0
    column_socket_weight: float = 0.0
    socket_border_weight: float = 5.0

    def validate(self) -> None:
        weights = (
            self.row_coordinate_weight,
            self.row_socket_weight,
            self.column_coordinate_weight,
            self.column_socket_weight,
            self.socket_border_weight,
        )
        if any(not np.isfinite(value) or value < 0 for value in weights):
            raise ValueError("all cyclic profile weights must be finite and non-negative")
        if self.row_coordinate_weight + self.row_socket_weight <= 0:
            raise ValueError("the row cyclic profile must have a positive weight")
        if self.column_coordinate_weight + self.column_socket_weight <= 0:
            raise ValueError("the column cyclic profile must have a positive weight")


@dataclass(frozen=True)
class CoordinateCyclicDiagnostics:
    """Auditable dirty-only diagnostics for one cyclic-origin decision."""

    grid_size: int
    tile_count: int
    candidates_evaluated: int
    selected_row_roll: int
    selected_column_roll: int
    changed: bool
    row_coordinate_weight: float
    row_socket_weight: float
    column_coordinate_weight: float
    column_socket_weight: float
    socket_border_weight: float
    initial_combined_score: float
    final_combined_score: float
    combined_score_gain: float
    initial_coordinate_mean_log_probability: float
    final_coordinate_mean_log_probability: float
    coordinate_mean_log_probability_gain: float
    strict_permutation: bool
    runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoordinateCyclicResult:
    """A strict tile-at-position layout and its origin diagnostics."""

    layout: np.ndarray
    diagnostics: CoordinateCyclicDiagnostics

    def report(self, *, include_layout: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "placer": "absolute-coordinate-global-cyclic-translation-v1",
            "layout_sha256": hashlib.sha256(
                np.asarray(self.layout, dtype="<i4").tobytes()
            ).hexdigest(),
            "diagnostics": self.diagnostics.as_dict(),
        }
        if include_layout:
            payload["tile_at_position"] = self.layout.tolist()
        return payload


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    result = value
    if hasattr(result, "detach"):
        result = result.detach()
    if hasattr(result, "cpu"):
        result = result.cpu()
    if hasattr(result, "numpy"):
        result = result.numpy()
    array = np.asarray(result, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _strict_layout(value: Any, *, count: int) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (count,):
        raise ValueError(f"layout must have shape {(count,)}, got {layout.shape}")
    if not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError("layout must be a strict tile permutation")
    return np.ascontiguousarray(layout)


def _axis_log_probabilities(value: Any, *, count: int, grid: int, name: str) -> np.ndarray:
    logits = _as_numpy(value, name=name)
    if logits.shape != (count, grid):
        raise ValueError(f"{name} must have shape {(count, grid)}, got {logits.shape}")
    maximum = logits.max(axis=1, keepdims=True)
    log_normaliser = maximum + np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
    return np.ascontiguousarray(logits - log_normaliser)


def _standardise_profile(value: np.ndarray) -> np.ndarray:
    centred = np.asarray(value, dtype=np.float64) - float(np.mean(value))
    scale = float(np.std(centred))
    if scale <= 1e-12:
        return np.zeros_like(centred)
    return centred / scale


def coordinate_cyclic_score_profiles(
    layout: Any,
    row_logits: Any,
    column_logits: Any,
    *,
    grid: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean log-probability for every row and column cyclic roll.

    ``layout`` follows the canonical tile-at-position convention.  A positive
    roll follows :func:`numpy.roll`.  Using a mean instead of a sum only
    divides every candidate by the same tile count, so the selected origin is
    identical.  Per-tile log-softmax constants also cancel across candidates;
    the resulting score is therefore consistent with the coordinate CE
    training target and robust to arbitrary per-tile logit offsets.
    """

    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least 2")
    count = grid * grid
    initial = _strict_layout(layout, count=count)
    row_log_probability = _axis_log_probabilities(
        row_logits,
        count=count,
        grid=grid,
        name="row_logits",
    )
    column_log_probability = _axis_log_probabilities(
        column_logits,
        count=count,
        grid=grid,
        name="column_logits",
    )
    board = initial.reshape(grid, grid)
    row_profile = np.empty(grid, dtype=np.float64)
    column_profile = np.empty(grid, dtype=np.float64)
    for roll in range(grid):
        row_board = np.roll(board, shift=roll, axis=0)
        row_profile[roll] = float(
            row_log_probability[
                row_board,
                np.arange(grid, dtype=np.int32)[:, None],
            ].mean(dtype=np.float64)
        )
        column_board = np.roll(board, shift=roll, axis=1)
        column_profile[roll] = float(
            column_log_probability[
                column_board,
                np.arange(grid, dtype=np.int32)[None, :],
            ].mean(dtype=np.float64)
        )
    return row_profile, column_profile


def socket_cyclic_score_profiles(
    layout: Any,
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int = 24,
    border_weight: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column profiles of the frozen Socket cut/border objective."""

    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least 2")
    if not np.isfinite(border_weight) or border_weight < 0:
        raise ValueError("border_weight must be finite and non-negative")
    count = grid * grid
    initial = _strict_layout(layout, count=count)
    right = _as_numpy(right_log_assignment, name="right_log_assignment")
    down = _as_numpy(down_log_assignment, name="down_log_assignment")
    expected = (count + 1, count + 1)
    if right.shape != expected or down.shape != expected:
        raise ValueError(f"socket assignments must both have shape {expected}")
    right_real = right[:count, :count]
    down_real = down[:count, :count]
    border = socket_border_unary(right, down, grid=grid)
    board = initial.reshape(grid, grid)
    scores = np.empty((grid, grid), dtype=np.float64)
    for row_roll in range(grid):
        for column_roll in range(grid):
            candidate = np.roll(
                board,
                shift=(row_roll, column_roll),
                axis=(0, 1),
            ).reshape(-1)
            scores[row_roll, column_roll] = socket_layout_objective(
                candidate,
                right_real,
                down_real,
                border,
                grid=grid,
                border_weight=border_weight,
            )
    # This objective is separable for a global cyclic roll: horizontal cuts
    # and left/right evidence depend only on column roll, while vertical cuts
    # and top/bottom evidence depend only on row roll.  Marginal means recover
    # the two profiles up to irrelevant constants and are numerically robust.
    return scores.mean(axis=1), scores.mean(axis=0)


def select_coordinate_cyclic_translation(
    layout: Any,
    row_logits: Any,
    column_logits: Any,
    *,
    right_log_assignment: Any | None = None,
    down_log_assignment: Any | None = None,
    grid: int = 24,
    config: CoordinateCyclicConfig | None = None,
) -> CoordinateCyclicResult:
    """Choose one global cyclic origin from coordinate and optional Socket cues."""

    started = perf_counter()
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least 2")
    config = CoordinateCyclicConfig() if config is None else config
    config.validate()
    count = grid * grid
    initial = _strict_layout(layout, count=count)
    coordinate_row, coordinate_column = coordinate_cyclic_score_profiles(
        initial,
        row_logits,
        column_logits,
        grid=grid,
    )
    needs_socket = config.row_socket_weight > 0 or config.column_socket_weight > 0
    if needs_socket:
        if right_log_assignment is None or down_log_assignment is None:
            raise ValueError("positive socket profile weights require both socket assignments")
        socket_row, socket_column = socket_cyclic_score_profiles(
            initial,
            right_log_assignment,
            down_log_assignment,
            grid=grid,
            border_weight=config.socket_border_weight,
        )
    else:
        socket_row = np.zeros(grid, dtype=np.float64)
        socket_column = np.zeros(grid, dtype=np.float64)

    combined_row = (
        config.row_coordinate_weight * _standardise_profile(coordinate_row)
        + config.row_socket_weight * _standardise_profile(socket_row)
    )
    combined_column = (
        config.column_coordinate_weight * _standardise_profile(coordinate_column)
        + config.column_socket_weight * _standardise_profile(socket_column)
    )
    # np.argmax is stable and returns zero for a complete tie, retaining the
    # input origin rather than inventing an arbitrary roll.
    row_roll = int(np.argmax(combined_row))
    column_roll = int(np.argmax(combined_column))
    best = np.roll(
        initial.reshape(grid, grid),
        shift=(row_roll, column_roll),
        axis=(0, 1),
    ).reshape(-1)
    best = _strict_layout(best, count=count)
    initial_combined = float(combined_row[0] + combined_column[0])
    final_combined = float(combined_row[row_roll] + combined_column[column_roll])
    coordinate_initial = float(coordinate_row[0] + coordinate_column[0])
    coordinate_final = float(
        coordinate_row[row_roll] + coordinate_column[column_roll]
    )
    diagnostics = CoordinateCyclicDiagnostics(
        grid_size=grid,
        tile_count=count,
        candidates_evaluated=count,
        selected_row_roll=row_roll,
        selected_column_roll=column_roll,
        changed=(row_roll, column_roll) != (0, 0),
        row_coordinate_weight=float(config.row_coordinate_weight),
        row_socket_weight=float(config.row_socket_weight),
        column_coordinate_weight=float(config.column_coordinate_weight),
        column_socket_weight=float(config.column_socket_weight),
        socket_border_weight=float(config.socket_border_weight),
        initial_combined_score=initial_combined,
        final_combined_score=final_combined,
        combined_score_gain=final_combined - initial_combined,
        initial_coordinate_mean_log_probability=coordinate_initial,
        final_coordinate_mean_log_probability=coordinate_final,
        coordinate_mean_log_probability_gain=coordinate_final - coordinate_initial,
        strict_permutation=True,
        runtime_seconds=perf_counter() - started,
    )
    return CoordinateCyclicResult(best, diagnostics)
