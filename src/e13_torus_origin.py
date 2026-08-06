"""Pure E13 global-origin selector for one upright 24x24 tile board.

E13 treats the completed board as a torus only while choosing its outer row
and column boundary.  It computes the exact E11 scaled CIE-Lab depth-1 seam
MSE for all 24 horizontal (row) cuts and all 24 vertical (column) cuts.  The
maximum-energy seam in each axis becomes the excluded outer boundary.

Cut ``0`` is the board's existing outer boundary and therefore means no roll.
``numpy.argmax`` supplies the frozen first-maximum tie rule.  The selected
board is one global row/column roll; tiles are never rotated, reflected,
recoloured, or changed internally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2lab


GRID = 24
TILE_SIZE = 20
NUM_TILES = GRID * GRID
DEPTH = 1
INNER_LOW = DEPTH
INNER_HIGH = TILE_SIZE - 1 - DEPTH
LAB_SCALE = np.asarray((100.0, 128.0, 128.0), dtype=np.float32)


class TorusOriginError(ValueError):
    """An input violates the fixed E13 geometry or numerical contract."""


@dataclass(frozen=True)
class TorusOriginSelection:
    """The deterministic cut scores and resulting global board roll."""

    original_board: np.ndarray
    rolled_board: np.ndarray
    horizontal_cut_energies: np.ndarray
    vertical_cut_energies: np.ndarray
    row_cut: int
    column_cut: int

    @property
    def row_roll(self) -> int:
        """The signed ``numpy.roll`` row shift applied to the board grid."""

        return -self.row_cut

    @property
    def column_roll(self) -> int:
        """The signed ``numpy.roll`` column shift applied to the board grid."""

        return -self.column_cut

    @property
    def excluded_horizontal_energy(self) -> float:
        return float(self.horizontal_cut_energies[self.row_cut])

    @property
    def excluded_vertical_energy(self) -> float:
        return float(self.vertical_cut_energies[self.column_cut])

    @property
    def retained_internal_horizontal_mse(self) -> float:
        retained = np.delete(self.horizontal_cut_energies, self.row_cut)
        return float(retained.mean(dtype=np.float64))

    @property
    def retained_internal_vertical_mse(self) -> float:
        retained = np.delete(self.vertical_cut_energies, self.column_cut)
        return float(retained.mean(dtype=np.float64))

    @property
    def retained_internal_lab_score(self) -> float:
        """The exact E11 score of the rolled board's 1,104 internal seams."""

        return -0.5 * (
            self.retained_internal_horizontal_mse
            + self.retained_internal_vertical_mse
        )


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (NUM_TILES, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise TorusOriginError(
            f"tiles must be upright uint8 RGB {expected}, got {value.dtype} {value.shape}"
        )
    return np.ascontiguousarray(value)


def _validate_board(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board)
    if value.shape != (NUM_TILES,) or not np.issubdtype(value.dtype, np.integer):
        raise TorusOriginError(f"board must be an integer vector of length {NUM_TILES}")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(NUM_TILES, dtype=np.int64)):
        raise TorusOriginError(f"board is not a permutation over 0..{NUM_TILES - 1}")
    return np.ascontiguousarray(value)


def scaled_cie_lab_tiles(tiles: np.ndarray) -> np.ndarray:
    """Return the exact scaled float32 Lab representation used by E11."""

    value = _validate_tiles(tiles).astype(np.float32) / 255.0
    lab = rgb2lab(value, channel_axis=-1).astype(np.float32)
    scaled = np.ascontiguousarray(lab / LAB_SCALE)
    if not np.isfinite(scaled).all():
        raise TorusOriginError("scaled CIE-Lab tiles contain a non-finite value")
    return scaled


def toroidal_cut_energies(
    tiles: np.ndarray, board: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal/vertical cut MSE arrays in cut-index order.

    Horizontal cut ``k`` lies between tile rows ``k-1`` and ``k`` modulo 24.
    Vertical cut ``k`` lies between tile columns ``k-1`` and ``k`` modulo 24.
    Consequently, cut zero is the current outer boundary on either axis.
    """

    lab = scaled_cie_lab_tiles(tiles)
    grid = _validate_board(board).reshape(GRID, GRID)
    horizontal = np.empty(GRID, dtype=np.float64)
    vertical = np.empty(GRID, dtype=np.float64)

    for cut in range(GRID):
        upper = grid[(cut - 1) % GRID, :]
        lower = grid[cut, :]
        horizontal_delta = (
            lab[upper, INNER_HIGH, :, :] - lab[lower, INNER_LOW, :, :]
        )
        horizontal[cut] = np.square(horizontal_delta).mean(dtype=np.float64)

        left = grid[:, (cut - 1) % GRID]
        right = grid[:, cut]
        vertical_delta = lab[left, :, INNER_HIGH, :] - lab[right, :, INNER_LOW, :]
        vertical[cut] = np.square(vertical_delta).mean(dtype=np.float64)

    if not np.isfinite(horizontal).all() or not np.isfinite(vertical).all():
        raise TorusOriginError("toroidal CIE-Lab cut energy is non-finite")
    horizontal.setflags(write=False)
    vertical.setflags(write=False)
    return horizontal, vertical


def select_torus_origin(tiles: np.ndarray, board: np.ndarray) -> TorusOriginSelection:
    """Choose the two maximum-energy cuts and apply exactly one global roll."""

    original = _validate_board(board).copy()
    horizontal, vertical = toroidal_cut_energies(tiles, original)
    # np.argmax returns the first maximum.  Thus an exact all-cut tie resolves
    # to cut 0, which is explicitly the no-roll choice.
    row_cut = int(np.argmax(horizontal))
    column_cut = int(np.argmax(vertical))
    rolled_grid = np.roll(
        original.reshape(GRID, GRID),
        shift=(-row_cut, -column_cut),
        axis=(0, 1),
    )
    rolled = _validate_board(rolled_grid.reshape(-1)).copy()
    original.setflags(write=False)
    rolled.setflags(write=False)
    return TorusOriginSelection(
        original_board=original,
        rolled_board=rolled,
        horizontal_cut_energies=horizontal,
        vertical_cut_energies=vertical,
        row_cut=row_cut,
        column_cut=column_cut,
    )
