"""Fixed photometric-invariant local-rank matcher view.

Every upright dirty tile and RGB channel is transformed independently.  The
output is matcher-only evidence and is never rendered in place of an original
tile.
"""

from __future__ import annotations

import numpy as np

from aiijc_puzzle.candidate_supply import classical_costs
from aiijc_puzzle.legacy_upgrade import cost_to_logp
from aiijc_puzzle.wiener_matcher_view import fixed_top32

TILE_COUNT = 576
TILE_SIZE = 20
WINDOW = 3


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.dtype != np.uint8 or value.shape != expected:
        raise ValueError(f"tiles must be uint8 with shape {expected}")
    return np.ascontiguousarray(value)


def local_midrank_tiles(tiles: np.ndarray) -> np.ndarray:
    """Map each pixel to its mid-rank among its eight 3x3 neighbours.

    Strictly lower neighbours contribute one and equal neighbours contribute
    one half.  The centre is excluded.  Thus positive affine photometry and,
    more generally, strictly monotone intensity transforms preserve the view.
    Reflection is confined within each tile.
    """

    source = _validate_tiles(tiles)
    padded = np.pad(source, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="reflect")
    centre = source[:, :, :, :]
    rank = np.zeros(source.shape, dtype=np.float32)
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            neighbour = padded[
                :,
                row_offset : row_offset + TILE_SIZE,
                column_offset : column_offset + TILE_SIZE,
                :,
            ]
            rank += neighbour < centre
            rank += np.float32(0.5) * (neighbour == centre)
    rank *= np.float32(255.0 / 8.0)
    if rank.shape != source.shape or not np.isfinite(rank).all():
        raise RuntimeError("local rank transform produced malformed pixels")
    return np.ascontiguousarray(rank, dtype=np.float32)


def fixed_local_rank_top32(tiles: np.ndarray) -> np.ndarray:
    """Return stable right/down top-32 identities for the fixed rank view."""

    view = local_midrank_tiles(tiles)
    right_cost, down_cost = classical_costs(view)
    scores = (cost_to_logp(right_cost), cost_to_logp(down_cost))
    return fixed_top32(scores)


__all__ = ["TILE_COUNT", "TILE_SIZE", "WINDOW", "fixed_local_rank_top32", "local_midrank_tiles"]
