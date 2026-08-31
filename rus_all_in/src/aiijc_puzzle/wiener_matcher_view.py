"""Fixed local-Wiener matcher view for legal puzzle candidate supply.

The filter is applied independently to every upright 20x20 dirty tile and to
every colour channel.  It is an inference-visible matcher view only: filtered
pixels are never rendered and never replace an original tile.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from aiijc_puzzle.candidate_supply import classical_costs
from aiijc_puzzle.legacy_upgrade import cost_to_logp

TILE_COUNT = 576
TILE_SIZE = 20
WINDOW = 3


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.dtype != np.uint8 or value.shape != expected:
        raise ValueError(f"tiles must be uint8 with shape {expected}")
    return np.ascontiguousarray(value)


def local_wiener_tiles(tiles: np.ndarray) -> np.ndarray:
    """Return the single fixed 3x3 local-Wiener matcher view.

    The classical Wiener shrinkage coefficient is estimated without a noise
    parameter.  Local mean and variance use a reflected 3x3 window; the noise
    variance is the mean local variance of the same tile/channel.  Filter size
    one on the tile and channel axes prevents cross-tile or cross-channel data
    flow.
    """

    source = _validate_tiles(tiles).astype(np.float32)
    size = (1, WINDOW, WINDOW, 1)
    mean = uniform_filter(source, size=size, mode="reflect", output=np.float32)
    mean_square = uniform_filter(
        source * source,
        size=size,
        mode="reflect",
        output=np.float32,
    )
    variance = np.maximum(mean_square - mean * mean, 0.0)
    noise_variance = variance.mean(axis=(1, 2), keepdims=True, dtype=np.float32)
    gain = np.maximum(variance - noise_variance, 0.0) / np.maximum(variance, 1e-6)
    filtered = mean + gain * (source - mean)
    if filtered.shape != source.shape or not np.isfinite(filtered).all():
        raise RuntimeError("local Wiener filter produced malformed pixels")
    return np.ascontiguousarray(np.clip(filtered, 0.0, 255.0), dtype=np.float32)


def fixed_wiener_directional_scores(
    tiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed right/down row log-probabilities for the Wiener view."""

    filtered = local_wiener_tiles(tiles)
    right_cost, down_cost = classical_costs(filtered)
    scores = (cost_to_logp(right_cost), cost_to_logp(down_cost))
    if any(value.shape != (TILE_COUNT, TILE_COUNT) for value in scores):
        raise RuntimeError("local Wiener score shape changed")
    if any(not np.isfinite(value).all() for value in scores):
        raise RuntimeError("local Wiener scores are non-finite")
    return tuple(np.ascontiguousarray(value, dtype=np.float32) for value in scores)


def fixed_top32(scores: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Return stable target-free top-32 identities for both directions."""

    axes = []
    for matrix in scores:
        value = np.asarray(matrix, dtype=np.float32).copy()
        if value.shape != (TILE_COUNT, TILE_COUNT) or not np.isfinite(value).all():
            raise ValueError("score matrices must be finite 576 x 576 arrays")
        np.fill_diagonal(value, -np.inf)
        axes.append(np.argsort(-value, axis=1, kind="stable")[:, :32])
    return np.ascontiguousarray(np.stack(axes), dtype=np.int32)


__all__ = [
    "TILE_COUNT",
    "TILE_SIZE",
    "WINDOW",
    "fixed_top32",
    "fixed_wiener_directional_scores",
    "local_wiener_tiles",
]
