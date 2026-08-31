"""Fixed legal guided-filter view for classical puzzle seam matching.

The transform is deliberately narrow: every upright 20x20 dirty tile is
filtered independently with the same luminance-guided local linear model.
The resulting pixels are matcher-only evidence.  They are never rendered or
substituted for an original tile in a solved layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from aiijc_puzzle.candidate_supply import classical_costs
from aiijc_puzzle.legacy_upgrade import cost_to_logp

TILE_COUNT = 576
TILE_SIZE = 20
LUMA_WEIGHTS = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass(frozen=True)
class GuidedMatcherViewConfig:
    """The single preregistered guided-filter recipe."""

    radius: int = 2
    epsilon: float = 1600.0
    guided_weight: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.radius, bool) or not isinstance(self.radius, int) or self.radius < 1:
            raise ValueError("radius must be a positive integer")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not np.isfinite(self.guided_weight) or not 0 < self.guided_weight < 1:
            raise ValueError("guided_weight must be strictly between zero and one")


FIXED_CONFIG = GuidedMatcherViewConfig()


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    source = np.asarray(tiles)
    if source.shape != (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3) or source.dtype != np.uint8:
        raise ValueError(
            "tiles must be uint8 with shape "
            f"{(TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)}, got {source.dtype} {source.shape}"
        )
    return np.ascontiguousarray(source)


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    width = 2 * radius + 1
    return uniform_filter(
        values,
        size=(1, width, width, 1),
        mode="reflect",
        output=np.float32,
    )


def guided_luminance_tiles(
    tiles: np.ndarray,
    *,
    config: GuidedMatcherViewConfig = FIXED_CONFIG,
) -> np.ndarray:
    """Return one permutation-equivariant matcher-only guided view.

    A scalar luminance guide controls a local linear reconstruction of each RGB
    channel.  The leading tile axis and trailing colour axis have filter size
    one, so information cannot cross tile identities or colour channels.
    """

    source_u8 = _validate_tiles(tiles)
    source = source_u8.astype(np.float32)
    guide = np.sum(source * LUMA_WEIGHTS, axis=-1, keepdims=True, dtype=np.float32)
    mean_guide = _box_mean(guide, config.radius)
    mean_guide_squared = _box_mean(guide * guide, config.radius)
    variance = np.maximum(mean_guide_squared - mean_guide * mean_guide, 0.0)

    mean_source = _box_mean(source, config.radius)
    mean_cross = _box_mean(guide * source, config.radius)
    covariance = mean_cross - mean_guide * mean_source
    slope = covariance / (variance + np.float32(config.epsilon))
    intercept = mean_source - slope * mean_guide
    mean_slope = _box_mean(slope, config.radius)
    mean_intercept = _box_mean(intercept, config.radius)
    filtered = mean_slope * guide + mean_intercept
    if filtered.shape != source.shape or not np.isfinite(filtered).all():
        raise RuntimeError("guided filter produced malformed or non-finite pixels")
    return np.ascontiguousarray(np.clip(filtered, 0.0, 255.0), dtype=np.float32)


def guided_fused_directional_scores(
    tiles: np.ndarray,
    bilateral_scores: tuple[np.ndarray, np.ndarray],
    *,
    config: GuidedMatcherViewConfig = FIXED_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse the fixed guided view with frozen bilateral row log-probabilities."""

    source = _validate_tiles(tiles)
    bilateral_right, bilateral_down = (
        np.asarray(matrix, dtype=np.float32) for matrix in bilateral_scores
    )
    expected = (TILE_COUNT, TILE_COUNT)
    if bilateral_right.shape != expected or bilateral_down.shape != expected:
        raise ValueError(f"bilateral score matrices must both have shape {expected}")
    if not np.isfinite(bilateral_right).all() or not np.isfinite(bilateral_down).all():
        raise ValueError("bilateral score matrices must be finite")

    guided = guided_luminance_tiles(source, config=config)
    right_cost, down_cost = classical_costs(guided)
    guided_scores = (cost_to_logp(right_cost), cost_to_logp(down_cost))
    raw_weight = np.float32(1.0 - config.guided_weight)
    guided_weight = np.float32(config.guided_weight)
    return tuple(
        np.ascontiguousarray(raw_weight * control + guided_weight * candidate, dtype=np.float32)
        for control, candidate in zip(
            (bilateral_right, bilateral_down), guided_scores, strict=True
        )
    )


__all__ = [
    "FIXED_CONFIG",
    "GuidedMatcherViewConfig",
    "guided_fused_directional_scores",
    "guided_luminance_tiles",
]
