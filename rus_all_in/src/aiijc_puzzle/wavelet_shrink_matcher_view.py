"""Fixed Haar-BayesShrink matcher view for legal puzzle candidate supply.

Every upright dirty tile and RGB channel is transformed independently.  The
output is matcher-only evidence: it is never rendered and never replaces an
original tile.
"""

from __future__ import annotations

import numpy as np

from aiijc_puzzle.candidate_supply import classical_costs
from aiijc_puzzle.legacy_upgrade import cost_to_logp
from aiijc_puzzle.wiener_matcher_view import fixed_top32

TILE_COUNT = 576
TILE_SIZE = 20
MAD_NORMAL = np.float32(0.67448975)


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.dtype != np.uint8 or value.shape != expected:
        raise ValueError(f"tiles must be uint8 with shape {expected}")
    return np.ascontiguousarray(value)


def _soft_threshold(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    magnitude = np.maximum(np.abs(values) - threshold, 0.0)
    return np.sign(values) * magnitude


def haar_bayesshrink_tiles(tiles: np.ndarray) -> np.ndarray:
    """Return one fixed one-level orthonormal Haar-BayesShrink view.

    A 2-D orthonormal Haar transform is applied to non-overlapping ``2x2``
    blocks, which exactly divide a ``20x20`` tile.  Noise scale is estimated
    independently for every tile/channel from the diagonal-detail MAD.  Each
    of the three detail bands then receives its standard parameter-free
    BayesShrink soft threshold ``sigma_noise**2 / sigma_signal``.  The low-pass
    band is kept exactly.  There is no learned or externally selected strength.
    """

    source = _validate_tiles(tiles).astype(np.float32)
    x00 = source[:, 0::2, 0::2]
    x01 = source[:, 0::2, 1::2]
    x10 = source[:, 1::2, 0::2]
    x11 = source[:, 1::2, 1::2]

    low = np.float32(0.5) * (x00 + x01 + x10 + x11)
    horizontal = np.float32(0.5) * (x00 - x01 + x10 - x11)
    vertical = np.float32(0.5) * (x00 + x01 - x10 - x11)
    diagonal = np.float32(0.5) * (x00 - x01 - x10 + x11)

    noise_sigma = np.median(np.abs(diagonal), axis=(1, 2), keepdims=True)
    noise_sigma = noise_sigma.astype(np.float32) / MAD_NORMAL
    noise_variance = noise_sigma * noise_sigma

    shrunk: list[np.ndarray] = []
    for detail in (horizontal, vertical, diagonal):
        observed_variance = np.mean(
            detail * detail,
            axis=(1, 2),
            keepdims=True,
            dtype=np.float32,
        )
        signal_sigma = np.sqrt(
            np.maximum(observed_variance - noise_variance, 0.0),
            dtype=np.float32,
        )
        threshold = np.divide(
            noise_variance,
            signal_sigma,
            out=np.full_like(noise_variance, np.inf),
            where=signal_sigma > np.float32(1e-6),
        )
        shrunk.append(_soft_threshold(detail, threshold))
    horizontal, vertical, diagonal = shrunk

    reconstructed = np.empty_like(source)
    reconstructed[:, 0::2, 0::2] = np.float32(0.5) * (
        low + horizontal + vertical + diagonal
    )
    reconstructed[:, 0::2, 1::2] = np.float32(0.5) * (
        low - horizontal + vertical - diagonal
    )
    reconstructed[:, 1::2, 0::2] = np.float32(0.5) * (
        low + horizontal - vertical - diagonal
    )
    reconstructed[:, 1::2, 1::2] = np.float32(0.5) * (
        low - horizontal - vertical + diagonal
    )
    if reconstructed.shape != source.shape or not np.isfinite(reconstructed).all():
        raise RuntimeError("Haar-BayesShrink produced malformed pixels")
    return np.ascontiguousarray(np.clip(reconstructed, 0.0, 255.0), dtype=np.float32)


def fixed_wavelet_top32(tiles: np.ndarray) -> np.ndarray:
    """Return stable right/down top-32 identities for the fixed wavelet view."""

    view = haar_bayesshrink_tiles(tiles)
    right_cost, down_cost = classical_costs(view)
    scores = (cost_to_logp(right_cost), cost_to_logp(down_cost))
    return fixed_top32(scores)


__all__ = [
    "MAD_NORMAL",
    "TILE_COUNT",
    "TILE_SIZE",
    "fixed_wavelet_top32",
    "haar_bayesshrink_tiles",
]
