"""Robust straight-line continuation compatibility across tile seams.

This scorer is intentionally complementary to pixel-border, Sobel-strip, and
binary-edge costs.  It projects strong contour samples from several *interior*
rows/columns to the proposed seam, then compares compact histograms of their
endpoint, angle, polarity, and colour contrast.  Interior sampling and a small
endpoint tolerance make the descriptor less sensitive to JPEG ringing and to a
single corrupted border pixel.

The public functions still return the project's standard 576x576 directional
``CompatibilityMatrices``.  Descriptor extraction itself accepts any batch
size so ``synthetic_line_seam_test`` can exercise the ranking logic cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .compatibility import CompatibilityMatrices, rank_normalize
from .geometry import TILE, TILE_COUNT


_Side = Literal["left", "right", "top", "bottom"]


@dataclass(frozen=True)
class _GradientFields:
    rgb_x: np.ndarray
    rgb_y: np.ndarray
    gray_x: np.ndarray
    gray_y: np.ndarray
    salience: np.ndarray


def _validate_tiles(tiles: np.ndarray, *, exact_count: bool = True) -> np.ndarray:
    values = np.asarray(tiles)
    expected_count = TILE_COUNT if exact_count else (values.shape[0] if values.ndim else -1)
    if values.ndim != 4 or values.shape != (expected_count, TILE, TILE, 3):
        count = TILE_COUNT if exact_count else "N"
        raise ValueError(f"expected {(count, TILE, TILE, 3)}, got {values.shape}")
    if values.dtype != np.uint8:
        raise TypeError(f"tiles must be uint8, got {values.dtype}")
    return values


def _binomial_blur(values: np.ndarray) -> np.ndarray:
    """A dependency-free 3x3 [1, 2, 1] blur for JPEG/noise suppression."""
    padded = np.pad(values, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="reflect")
    return (
        padded[:, :-2, :-2]
        + 2.0 * padded[:, :-2, 1:-1]
        + padded[:, :-2, 2:]
        + 2.0 * padded[:, 1:-1, :-2]
        + 4.0 * padded[:, 1:-1, 1:-1]
        + 2.0 * padded[:, 1:-1, 2:]
        + padded[:, 2:, :-2]
        + 2.0 * padded[:, 2:, 1:-1]
        + padded[:, 2:, 2:]
    ) * (1.0 / 16.0)


def _gradient_fields(tiles: np.ndarray) -> _GradientFields:
    values = _binomial_blur(tiles.astype(np.float32) / 255.0)
    padded_x = np.pad(values, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
    padded_y = np.pad(values, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
    rgb_x = 0.5 * (padded_x[:, :, 2:] - padded_x[:, :, :-2])
    rgb_y = 0.5 * (padded_y[:, 2:] - padded_y[:, :-2])

    # Luminance gives a stable geometric orientation; RGB derivatives below
    # retain colour-only evidence once that geometry has selected a contour.
    luma = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    gray_x = np.einsum("...c,c->...", rgb_x, luma, optimize=True)
    gray_y = np.einsum("...c,c->...", rgb_y, luma, optimize=True)
    gray_magnitude = np.hypot(gray_x, gray_y)
    colour_magnitude = np.sqrt(np.mean(rgb_x * rgb_x + rgb_y * rgb_y, axis=3))
    magnitude = gray_magnitude + 0.25 * colour_magnitude

    # Per-tile soft thresholding is much more stable than a global absolute
    # edge threshold under the task's tile-wise contrast/noise corruption.
    low = np.percentile(magnitude, 55.0, axis=(1, 2), keepdims=True)
    high = np.percentile(magnitude, 90.0, axis=(1, 2), keepdims=True)
    salience = np.clip((magnitude - low) / np.maximum(high - low, 1e-4), 0.0, 1.0)
    return _GradientFields(
        rgb_x=rgb_x.astype(np.float32),
        rgb_y=rgb_y.astype(np.float32),
        gray_x=gray_x.astype(np.float32),
        gray_y=gray_y.astype(np.float32),
        salience=salience.astype(np.float32),
    )


def _smooth_tangent(histogram: np.ndarray) -> np.ndarray:
    padded = np.pad(histogram, ((0, 0), (1, 1), (0, 0)), mode="constant")
    return 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]


def _accumulate_bilinear(
    histogram: np.ndarray,
    tangent: np.ndarray,
    slope: np.ndarray,
    polarity: np.ndarray,
    weight: np.ndarray,
) -> None:
    count, sample_count = tangent.shape
    slope_bins = histogram.shape[2]
    tangent = np.clip(tangent, 0.0, float(TILE - 1))
    slope = np.clip(slope, 0.0, float(slope_bins - 1))
    tangent_low = np.floor(tangent).astype(np.int32)
    slope_low = np.floor(slope).astype(np.int32)
    tangent_high = np.minimum(tangent_low + 1, TILE - 1)
    slope_high = np.minimum(slope_low + 1, slope_bins - 1)
    tangent_fraction = tangent - tangent_low
    slope_fraction = slope - slope_low
    tile_indices = np.repeat(np.arange(count, dtype=np.int32), sample_count)
    polarity = polarity.reshape(-1).astype(np.int32)

    for tangent_index, tangent_factor in (
        (tangent_low, 1.0 - tangent_fraction),
        (tangent_high, tangent_fraction),
    ):
        for slope_index, slope_factor in (
            (slope_low, 1.0 - slope_fraction),
            (slope_high, slope_fraction),
        ):
            contribution = (weight * tangent_factor * slope_factor).reshape(-1)
            np.add.at(
                histogram,
                (
                    tile_indices,
                    tangent_index.reshape(-1),
                    slope_index.reshape(-1),
                    polarity,
                ),
                contribution,
            )


def _accumulate_colour(
    histogram: np.ndarray,
    tangent: np.ndarray,
    signed_colour: np.ndarray,
    weight: np.ndarray,
) -> None:
    count, sample_count = tangent.shape
    tangent = np.clip(tangent, 0.0, float(TILE - 1))
    tangent_low = np.floor(tangent).astype(np.int32)
    tangent_high = np.minimum(tangent_low + 1, TILE - 1)
    fraction = tangent - tangent_low
    tile_indices = np.repeat(np.arange(count, dtype=np.int32), sample_count)

    # Splitting signed RGB contrast into positive/negative bins preserves edge
    # polarity while keeping the final descriptor non-negative and cosine-safe.
    colour = np.concatenate(
        [np.maximum(signed_colour, 0.0), np.maximum(-signed_colour, 0.0)], axis=2
    )
    scale = np.percentile(np.abs(signed_colour), 90.0, axis=1, keepdims=True)
    colour_scale = np.maximum(np.concatenate([scale, scale], axis=2), 1e-4)
    colour = np.clip(colour / colour_scale, 0.0, 2.0)
    for tangent_index, tangent_factor in (
        (tangent_low, 1.0 - fraction),
        (tangent_high, fraction),
    ):
        for channel in range(6):
            contribution = (weight * tangent_factor * colour[:, :, channel]).reshape(-1)
            np.add.at(
                histogram[:, :, channel],
                (tile_indices, tangent_index.reshape(-1)),
                contribution,
            )


def _side_descriptor(
    fields: _GradientFields,
    side: _Side,
    *,
    strip: int,
    slope_bins: int,
    max_slope: float,
    structure_weight: float,
) -> np.ndarray:
    """Return an ``N x tangent-position x feature`` seam descriptor."""
    if strip < 1 or strip > TILE - 2:
        raise ValueError(f"strip must be in [1, {TILE - 2}]")
    if slope_bins < 3 or slope_bins % 2 == 0:
        raise ValueError("slope_bins must be an odd integer >= 3")
    if not np.isfinite(max_slope) or max_slope <= 0:
        raise ValueError("max_slope must be positive and finite")
    if not 0.0 <= structure_weight <= 1.0:
        raise ValueError("structure_weight must be in [0, 1]")

    count = len(fields.gray_x)
    depths = np.arange(1, strip + 1, dtype=np.int32)
    tangent_grid = np.arange(TILE, dtype=np.float32)[None, None, :]
    depth_weight = (1.0 / np.sqrt(depths.astype(np.float32)))[None, :, None]

    if side in {"left", "right"}:
        normal_indices = depths if side == "left" else TILE - 1 - depths
        gray_normal = fields.gray_x[:, :, normal_indices].transpose(0, 2, 1)
        gray_cross = fields.gray_y[:, :, normal_indices].transpose(0, 2, 1)
        colour_cross = fields.rgb_y[:, :, normal_indices, :].transpose(0, 2, 1, 3)
        salience = fields.salience[:, :, normal_indices].transpose(0, 2, 1)
        seam = -0.5 if side == "left" else TILE - 0.5
    else:
        normal_indices = depths if side == "top" else TILE - 1 - depths
        gray_normal = fields.gray_y[:, normal_indices, :]
        gray_cross = fields.gray_x[:, normal_indices, :]
        colour_cross = fields.rgb_x[:, normal_indices, :, :]
        salience = fields.salience[:, normal_indices, :]
        seam = -0.5 if side == "top" else TILE - 0.5

    normal_grid = normal_indices.astype(np.float32)[None, :, None]
    denominator = np.where(
        np.abs(gray_cross) >= 1e-4,
        gray_cross,
        np.where(gray_cross < 0.0, -1e-4, 1e-4),
    )
    # For a level-set contour, tangent=(-gy, gx).  The two formulas below are
    # respectively dy/dx and dx/dy, represented uniformly as -normal/cross.
    slope = np.clip(-gray_normal / denominator, -max_slope, max_slope)
    endpoint = tangent_grid + slope * (seam - normal_grid)

    gradient_magnitude = np.hypot(gray_normal, gray_cross)
    crossingness = np.abs(gray_cross) / np.maximum(gradient_magnitude, 1e-4)
    inside = (endpoint >= -1.0) & (endpoint <= TILE)
    weight = salience * crossingness * depth_weight * inside.astype(np.float32)
    polarity = gray_cross < 0.0
    slope_coordinate = (slope + max_slope) * ((slope_bins - 1) / (2.0 * max_slope))

    sample_count = strip * TILE
    structure = np.zeros((count, TILE, slope_bins, 2), dtype=np.float32)
    _accumulate_bilinear(
        structure,
        endpoint.reshape(count, sample_count),
        slope_coordinate.reshape(count, sample_count),
        polarity.reshape(count, sample_count),
        weight.reshape(count, sample_count),
    )
    # A short slope-axis blur tolerates angle quantisation without discarding
    # the useful distinction between horizontal and diagonal continuation.
    slope_padded = np.pad(structure, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="constant")
    structure = (
        0.2 * slope_padded[:, :, :-2]
        + 0.6 * slope_padded[:, :, 1:-1]
        + 0.2 * slope_padded[:, :, 2:]
    )
    structure = _smooth_tangent(structure.reshape(count, TILE, -1))

    colour = np.zeros((count, TILE, 6), dtype=np.float32)
    _accumulate_colour(
        colour,
        endpoint.reshape(count, sample_count),
        colour_cross.reshape(count, sample_count, 3),
        weight.reshape(count, sample_count),
    )
    colour = _smooth_tangent(colour)

    structure /= np.maximum(
        np.linalg.norm(structure.reshape(count, -1), axis=1)[:, None, None], 1e-6
    )
    colour /= np.maximum(
        np.linalg.norm(colour.reshape(count, -1), axis=1)[:, None, None], 1e-6
    )
    return np.concatenate(
        [
            np.sqrt(structure_weight) * structure,
            np.sqrt(1.0 - structure_weight) * colour,
        ],
        axis=2,
    ).astype(np.float32)


def _shift_tangent(descriptor: np.ndarray, offset: int) -> np.ndarray:
    if offset == 0:
        return descriptor
    shifted = np.zeros_like(descriptor)
    if offset > 0:
        shifted[:, offset:] = descriptor[:, :-offset]
    else:
        shifted[:, :offset] = descriptor[:, -offset:]
    return shifted


def _pairwise_descriptor_cost(
    query: np.ndarray,
    key: np.ndarray,
    *,
    max_offset: int,
    offset_penalty: float,
) -> np.ndarray:
    if query.shape != key.shape or query.ndim != 3:
        raise ValueError("query and key descriptors must have matching NxTxF shapes")
    if max_offset < 0 or max_offset >= TILE:
        raise ValueError(f"max_offset must be in [0, {TILE - 1}]")
    if not np.isfinite(offset_penalty) or offset_penalty < 0:
        raise ValueError("offset_penalty must be finite and non-negative")

    query_flat = query.reshape(len(query), -1)
    query_norm = np.linalg.norm(query_flat, axis=1)
    best = np.full((len(query), len(key)), -np.inf, dtype=np.float32)
    for offset in range(-max_offset, max_offset + 1):
        key_flat = _shift_tangent(key, offset).reshape(len(key), -1)
        denominator = np.maximum(
            query_norm[:, None] * np.linalg.norm(key_flat, axis=1)[None, :], 1e-6
        )
        similarity = (query_flat @ key_flat.T) / denominator
        similarity -= offset_penalty * abs(offset)
        best = np.maximum(best, similarity.astype(np.float32))
    return np.clip(1.0 - best, 0.0, 2.0).astype(np.float32)


def _single_view_line_seam(
    tiles: np.ndarray,
    *,
    name: str,
    strip: int,
    slope_bins: int,
    max_slope: float,
    max_offset: int,
    offset_penalty: float,
    structure_weight: float,
) -> CompatibilityMatrices:
    fields = _gradient_fields(tiles)
    descriptors = {
        side: _side_descriptor(
            fields,
            side,
            strip=strip,
            slope_bins=slope_bins,
            max_slope=max_slope,
            structure_weight=structure_weight,
        )
        for side in ("left", "right", "top", "bottom")
    }
    right = _pairwise_descriptor_cost(
        descriptors["right"],
        descriptors["left"],
        max_offset=max_offset,
        offset_penalty=offset_penalty,
    )
    down = _pairwise_descriptor_cost(
        descriptors["bottom"],
        descriptors["top"],
        max_offset=max_offset,
        offset_penalty=offset_penalty,
    )
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices(name, right, down)


def line_seam_compatibility(
    tiles: np.ndarray,
    *,
    prefix: str = "denoised",
    auxiliary_tiles: np.ndarray | None = None,
    auxiliary_prefix: str = "raw",
    auxiliary_weight: float = 0.35,
    strip: int = 5,
    slope_bins: int = 5,
    max_slope: float = 1.5,
    max_offset: int = 1,
    offset_penalty: float = 0.025,
    structure_weight: float = 0.8,
) -> CompatibilityMatrices:
    """Build a robust structural line-continuation seam cost.

    When ``auxiliary_tiles`` is supplied, the two views are independently
    rank-normalised before fusion.  This is suitable for a denoised primary
    view plus raw tiles: raw contours can rescue lines erased by denoising,
    while incomparable physical cost scales cannot dominate the blend.
    """
    tiles = _validate_tiles(tiles)
    if not 0.0 <= auxiliary_weight <= 1.0:
        raise ValueError("auxiliary_weight must be in [0, 1]")
    primary = _single_view_line_seam(
        tiles,
        name=f"{prefix}_line_seam",
        strip=strip,
        slope_bins=slope_bins,
        max_slope=max_slope,
        max_offset=max_offset,
        offset_penalty=offset_penalty,
        structure_weight=structure_weight,
    )
    if auxiliary_tiles is None:
        return primary

    auxiliary_tiles = _validate_tiles(auxiliary_tiles)
    auxiliary = _single_view_line_seam(
        auxiliary_tiles,
        name=f"{auxiliary_prefix}_line_seam",
        strip=strip,
        slope_bins=slope_bins,
        max_slope=max_slope,
        max_offset=max_offset,
        offset_penalty=offset_penalty,
        structure_weight=structure_weight,
    )
    primary_weight = 1.0 - auxiliary_weight
    right = (
        primary_weight * rank_normalize(primary.right)
        + auxiliary_weight * rank_normalize(auxiliary.right)
    ).astype(np.float32)
    down = (
        primary_weight * rank_normalize(primary.down)
        + auxiliary_weight * rank_normalize(auxiliary.down)
    ).astype(np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    name = f"{prefix}_{auxiliary_prefix}_line_seam_fused"
    return CompatibilityMatrices(name, right, down)


def synthetic_line_seam_test() -> dict[str, float | int]:
    """Assert that a split noisy diagonal line retrieves its true neighbour.

    This uses only eight tiles and the internal descriptor matcher, so it is a
    fast deterministic unit/smoke test rather than a dataset experiment.
    """
    rng = np.random.default_rng(1731)
    tile_count = 8
    tiles = np.empty((tile_count, TILE, TILE, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:TILE, :TILE]

    def render(
        global_x: np.ndarray, *, intercept: float, slope: float, colour: np.ndarray
    ) -> np.ndarray:
        centre = intercept + slope * global_x
        line = np.exp(-0.5 * ((yy - centre) / 0.75) ** 2)[..., None]
        background = np.asarray([196.0, 181.0, 165.0], dtype=np.float32)
        image = background[None, None, :] * (1.0 - line) + colour[None, None, :] * line
        # Independent tone/noise corruption on each tile mirrors the real task
        # without making the expected continuation pixel-identical.
        image += rng.normal(0.0, 5.0, image.shape)
        image += rng.uniform(-10.0, 10.0)
        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    line_colour = np.asarray([35.0, 68.0, 115.0])
    tiles[0] = render(xx, intercept=5.2, slope=0.24, colour=line_colour)
    tiles[1] = render(xx + TILE, intercept=5.2, slope=0.24, colour=line_colour)
    distractors = [
        (2.0, -0.35),
        (15.0, -0.10),
        (3.5, 0.05),
        (17.0, 0.30),
        (10.0, -0.55),
        (1.0, 0.60),
    ]
    for index, (intercept, slope) in enumerate(distractors, start=2):
        tiles[index] = render(
            xx + TILE,
            intercept=intercept,
            slope=slope,
            colour=np.asarray([110.0, 48.0 + 7.0 * index, 45.0]),
        )

    fields = _gradient_fields(_validate_tiles(tiles, exact_count=False))
    query = _side_descriptor(
        fields, "right", strip=5, slope_bins=5, max_slope=1.5, structure_weight=0.8
    )
    key = _side_descriptor(
        fields, "left", strip=5, slope_bins=5, max_slope=1.5, structure_weight=0.8
    )
    costs = _pairwise_descriptor_cost(query, key, max_offset=1, offset_penalty=0.025)
    candidates = np.arange(1, tile_count)
    ranking = candidates[np.argsort(costs[0, candidates], kind="stable")]
    if int(ranking[0]) != 1:
        raise AssertionError(
            f"true neighbour ranked {int(np.flatnonzero(ranking == 1)[0]) + 1}: {ranking.tolist()}"
        )
    return {
        "true_rank": 1,
        "true_cost": float(costs[0, 1]),
        "next_best_cost": float(np.min(costs[0, 2:])),
    }


__all__ = ["line_seam_compatibility", "synthetic_line_seam_test"]


if __name__ == "__main__":
    print(synthetic_line_seam_test())
