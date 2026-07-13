"""Target-blind D4 test-time consensus for directional compatibility costs.

The frozen tile restorer is not exactly equivariant to reflections or a
180-degree rotation.  This module exposes the four involutive views used by
the bounded D4 diagnostic and combines *row ranks* of independently computed
compatibility matrices.  A median rewards view-stable candidates while a
median absolute deviation penalizes orientation-sensitive candidates.

No target, source identity, permutation, or layout label is accepted here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .compatibility import CompatibilityMatrices, rank_normalize
from .geometry import TILE, TILE_COUNT


D4_VIEWS = ("identity", "hflip", "vflip", "rot180")


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    values = np.asarray(tiles)
    if values.shape != (TILE_COUNT, TILE, TILE, 3):
        raise ValueError(
            f"tiles must have shape {(TILE_COUNT, TILE, TILE, 3)}, got {values.shape}"
        )
    if values.dtype != np.uint8:
        raise TypeError(f"tiles must be uint8, got {values.dtype}")
    return values


def transform_tiles(tiles: np.ndarray, view: str) -> np.ndarray:
    """Apply one frozen D4 view to every tile and return contiguous uint8."""

    values = _validate_tiles(tiles)
    if view == "identity":
        transformed = values
    elif view == "hflip":
        transformed = values[:, :, ::-1, :]
    elif view == "vflip":
        transformed = values[:, ::-1, :, :]
    elif view == "rot180":
        transformed = values[:, ::-1, ::-1, :]
    else:
        raise ValueError(f"unknown D4 view {view!r}; expected one of {D4_VIEWS}")
    return np.ascontiguousarray(transformed)


def inverse_transform_tiles(tiles: np.ndarray, view: str) -> np.ndarray:
    """Undo a frozen D4 view.

    All four selected transforms are involutions, so the inverse is the same
    operation.  Keeping a named inverse makes the inference contract explicit.
    """

    return transform_tiles(tiles, view)


def _rank_stack(
    scores: Mapping[str, CompatibilityMatrices], side: str
) -> np.ndarray:
    missing = [view for view in D4_VIEWS if view not in scores]
    extra = sorted(set(scores) - set(D4_VIEWS))
    if missing or extra:
        raise ValueError(f"D4 score views mismatch; missing={missing}, extra={extra}")
    ranked = []
    for view in D4_VIEWS:
        matrix = np.asarray(getattr(scores[view], side))
        if matrix.shape != (TILE_COUNT, TILE_COUNT):
            raise ValueError(f"{view}.{side} has invalid shape {matrix.shape}")
        values = rank_normalize(matrix)
        diagonal = np.diag_indices(TILE_COUNT)
        values[diagonal] = 0.0
        ranked.append(values)
    return np.stack(ranked, axis=0).astype(np.float32, copy=False)


def d4_rank_consensus(
    scores: Mapping[str, CompatibilityMatrices],
    *,
    identity_weight: float = 0.50,
    median_weight: float = 0.40,
    mad_weight: float = 0.10,
    name: str = "d4_rank_consensus_i50_m40_mad10",
) -> CompatibilityMatrices:
    """Combine four view-specific compatibility matrices without labels.

    Lower remains better.  The median absolute deviation is therefore added
    as an instability penalty.  The frozen weights must be finite,
    non-negative, and sum to one so the resulting scale remains comparable to
    the existing rank-fused compatibility matrices.
    """

    weights = np.asarray(
        [identity_weight, median_weight, mad_weight], dtype=np.float64
    )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("consensus weights must be finite and non-negative")
    if not np.isclose(float(weights.sum()), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("consensus weights must sum to one")

    outputs: dict[str, np.ndarray] = {}
    for side in ("right", "down"):
        stack = _rank_stack(scores, side)
        median = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - median[None, :, :]), axis=0)
        combined = (
            float(identity_weight) * stack[0]
            + float(median_weight) * median
            + float(mad_weight) * mad
        ).astype(np.float32)
        np.fill_diagonal(combined, np.inf)
        if not np.isfinite(combined[~np.eye(TILE_COUNT, dtype=bool)]).all():
            raise RuntimeError(f"non-finite off-diagonal D4 {side} costs")
        outputs[side] = combined
    return CompatibilityMatrices(name, outputs["right"], outputs["down"])


__all__ = [
    "D4_VIEWS",
    "d4_rank_consensus",
    "inverse_transform_tiles",
    "transform_tiles",
]
