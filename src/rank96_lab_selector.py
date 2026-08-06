"""Fixed E11 selector between upright Rank96 and Rank512 boards.

The selector is deliberately small and label-free.  Both boards must come from
the same raw ranker score matrices and differ only in the buddies edge budget.
For each board, E11 measures one-pixel-inset CIE-Lab continuity across every
horizontal and vertical tile boundary.  The board with the smaller mean
squared Lab discontinuity is selected; an exact tie keeps Rank96.

Tiles are never rotated, reflected, recoloured, or otherwise transformed.
Restoration is intentionally outside this module so production code can run
the fixed NLM tail exactly once after selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from skimage.color import rgb2lab


GRID = 24
TILE_SIZE = 20
NUM_TILES = GRID * GRID
DEPTH = 1
INNER_LOW = DEPTH
INNER_HIGH = TILE_SIZE - 1 - DEPTH
RANK96_MAX_EDGES = 96
RANK512_MAX_EDGES = 512
MIN_MARGIN = 0.0
REPAIR_PASSES = 0
RANK96_ARM = "rank96"
RANK512_ARM = "rank512"
LAB_SCALE = np.asarray((100.0, 128.0, 128.0), dtype=np.float32)


class LabSelectorError(ValueError):
    """An input violates the frozen E11 geometry or numerical contract."""


@dataclass(frozen=True)
class LabDepth1Selection:
    """The two solver candidates and E11's deterministic decision."""

    rank96_board: np.ndarray
    rank512_board: np.ndarray
    rank96_objective: float
    rank512_objective: float
    rank96_lab_score: float
    rank512_lab_score: float
    selected_arm: str

    @property
    def selected_board(self) -> np.ndarray:
        return self.rank96_board if self.selected_arm == RANK96_ARM else self.rank512_board

    @property
    def lab_margin_rank96_minus_rank512(self) -> float:
        return float(self.rank96_lab_score - self.rank512_lab_score)


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (NUM_TILES, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise LabSelectorError(f"tiles must be upright uint8 RGB {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def _validate_board(board: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(board)
    if value.shape != (NUM_TILES,) or not np.issubdtype(value.dtype, np.integer):
        raise LabSelectorError(f"{label} must be an integer vector of length {NUM_TILES}")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(NUM_TILES, dtype=np.int64)):
        raise LabSelectorError(f"{label} is not a permutation over 0..{NUM_TILES - 1}")
    return np.ascontiguousarray(value)


def _validate_dense(value: np.ndarray, *, label: str) -> np.ndarray:
    # Own an immutable snapshot so neither solver arm nor the caller can alter
    # the shared dense_rd bytes between the fixed 96/512 solves.
    matrix = np.array(value, dtype=np.float32, order="C", copy=True)
    expected = (NUM_TILES, NUM_TILES)
    if matrix.shape != expected:
        raise LabSelectorError(f"{label} must have shape {expected}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise LabSelectorError(f"{label} must be finite")
    if np.any(matrix < 0.0):
        raise LabSelectorError(f"{label} must be nonnegative")
    matrix.setflags(write=False)
    return matrix


def scaled_lab_tiles(tiles: np.ndarray) -> np.ndarray:
    """Return the exact E11 Lab representation for upright uint8 tiles."""

    value = _validate_tiles(tiles).astype(np.float32) / 255.0
    lab = rgb2lab(value, channel_axis=-1).astype(np.float32)
    return np.ascontiguousarray(lab / LAB_SCALE)


def _lab_depth1_score_from_scaled(lab: np.ndarray, board: np.ndarray) -> float:
    order = _validate_board(board, label="board").reshape(GRID, GRID)
    left = order[:, :-1]
    right = order[:, 1:]
    upper = order[:-1, :]
    lower = order[1:, :]

    horizontal_delta = lab[left, :, INNER_HIGH, :] - lab[right, :, INNER_LOW, :]
    vertical_delta = lab[upper, INNER_HIGH, :, :] - lab[lower, INNER_LOW, :, :]
    horizontal_mse = np.square(horizontal_delta).mean(dtype=np.float64)
    vertical_mse = np.square(vertical_delta).mean(dtype=np.float64)
    score = -0.5 * (float(horizontal_mse) + float(vertical_mse))
    if not np.isfinite(score):
        raise LabSelectorError("CIE-Lab seam score is non-finite")
    return score


def lab_depth1_board_score(tiles: np.ndarray, board: np.ndarray) -> float:
    """Score all 1,104 inset seams; larger (less negative) is better."""

    return _lab_depth1_score_from_scaled(scaled_lab_tiles(tiles), board)


def select_lab_depth1_board(
    tiles: np.ndarray,
    rank96_board: np.ndarray,
    rank512_board: np.ndarray,
    *,
    rank96_objective: float = 0.0,
    rank512_objective: float = 0.0,
) -> LabDepth1Selection:
    """Apply E11 to two precomputed boards, resolving a tie toward Rank96."""

    tile_array = _validate_tiles(tiles)
    board96 = _validate_board(rank96_board, label="rank96 board").copy()
    board512 = _validate_board(rank512_board, label="rank512 board").copy()
    objective96 = float(rank96_objective)
    objective512 = float(rank512_objective)
    if not np.isfinite(objective96) or not np.isfinite(objective512):
        raise LabSelectorError("solver objectives must be finite")
    lab = scaled_lab_tiles(tile_array)
    score96 = _lab_depth1_score_from_scaled(lab, board96)
    score512 = _lab_depth1_score_from_scaled(lab, board512)
    selected = RANK96_ARM if score96 >= score512 else RANK512_ARM
    return LabDepth1Selection(
        rank96_board=board96,
        rank512_board=board512,
        rank96_objective=objective96,
        rank512_objective=objective512,
        rank96_lab_score=score96,
        rank512_lab_score=score512,
        selected_arm=selected,
    )


def solve_and_select_lab_depth1(
    tiles: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    *,
    solver: Callable[..., tuple[np.ndarray, float]] | None = None,
) -> LabDepth1Selection:
    """Solve the only two E11 candidates from shared dense matrices and select."""

    tile_array = _validate_tiles(tiles)
    right_array = _validate_dense(right, label="right")
    down_array = _validate_dense(down, label="down")
    if solver is None:
        from solve_buddies import solve_buddies_from_scores

        solver = solve_buddies_from_scores

    outputs: list[tuple[np.ndarray, float]] = []
    for arm, max_edges in ((RANK96_ARM, RANK96_MAX_EDGES), (RANK512_ARM, RANK512_MAX_EDGES)):
        board, objective = solver(
            right_array,
            down_array,
            max_edges=max_edges,
            min_margin=MIN_MARGIN,
            repair_passes=REPAIR_PASSES,
        )
        board_array = _validate_board(board, label=f"{arm} board")
        objective_value = float(objective)
        if not np.isfinite(objective_value):
            raise LabSelectorError(f"{arm} solver objective is non-finite")
        outputs.append((board_array, objective_value))

    (board96, objective96), (board512, objective512) = outputs
    return select_lab_depth1_board(
        tile_array,
        board96,
        board512,
        rank96_objective=objective96,
        rank512_objective=objective512,
    )
