"""Target-free parametric predictors tailored to the contest SSIM metric.

The puzzle permutation destroys spatial correspondence but preserves a large
part of the image's pixel population.  This module deliberately ignores layout
and extracts only permutation-invariant statistics from the corrupted input.
Training code may regress those statistics to the RGB constant that maximises
the exact contest SSIM on paired *training* targets.

No function in this module accepts both an inference image and a target.  The
only target-facing primitive, :func:`ssim_optimal_constant_rgb`, produces a
training label and cannot construct an inference prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize_scalar

from aiijc_puzzle.protocol import GRID_SIZE, IMAGE_SIZE, TILE_SIZE

_C1 = (0.01 * 255.0) ** 2
_C2 = (0.03 * 255.0) ** 2
_WINDOW_SIZE = 7
_WINDOW_PAD = _WINDOW_SIZE // 2
_SAMPLE_COVARIANCE_FACTOR = (_WINDOW_SIZE**2) / (_WINDOW_SIZE**2 - 1)

PIXEL_QUANTILES = np.asarray(
    (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25, 0.35, 0.50,
     0.65, 0.75, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999),
    dtype=np.float64,
)
TILE_QUANTILES = np.asarray((0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99))
DIFFERENCE_QUANTILES = np.asarray((0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99))
HISTOGRAM_EDGES = np.linspace(-0.5, 255.5, 33, dtype=np.float64)
SATURATION_THRESHOLDS = (0, 1, 2, 4, 8, 16, 32)


def _validate_rgb(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    value = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image {expected}, got {value.dtype} {value.shape}")
    return value


def input_median_rgb(image: NDArray[np.generic]) -> NDArray[np.float64]:
    """Return the strong target-free baseline colour."""

    value = _validate_rgb(image)
    return np.median(value.reshape(-1, 3), axis=0).astype(np.float64)


def render_constant_rgb(
    color: NDArray[np.generic], *, shape: tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3)
) -> NDArray[np.uint8]:
    """Quantise and render one RGB colour as a writable constant frame."""

    rgb = np.asarray(color, dtype=np.float64)
    if rgb.shape != (3,) or not np.all(np.isfinite(rgb)):
        raise ValueError(f"expected three finite RGB values, got {rgb}")
    if shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"contest prediction shape must be 480x480x3, got {shape}")
    quantised = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return np.broadcast_to(quantised, shape).copy()


def feature_names() -> tuple[str, ...]:
    """Return the stable schema for :func:`extract_invariant_features`."""

    names: list[str] = []
    channels = ("r", "g", "b")
    for channel in channels:
        names.extend(
            (
                f"pixel_mean_{channel}",
                f"pixel_std_{channel}",
                f"pixel_mad_{channel}",
                f"pixel_min_{channel}",
                f"pixel_max_{channel}",
            )
        )
        names.extend(f"pixel_q{quantile:g}_{channel}" for quantile in PIXEL_QUANTILES)
        names.extend(f"hist32_{index}_{channel}" for index in range(32))
        names.extend(f"low_le_{threshold}_{channel}" for threshold in SATURATION_THRESHOLDS)
        names.extend(f"high_ge_{255 - threshold}_{channel}" for threshold in SATURATION_THRESHOLDS)
        names.extend(
            (
                f"tile_mean_mean_{channel}",
                f"tile_mean_std_{channel}",
                f"tile_std_mean_{channel}",
                f"tile_std_std_{channel}",
            )
        )
        names.extend(f"tile_mean_q{quantile:g}_{channel}" for quantile in TILE_QUANTILES)
        names.extend(f"tile_std_q{quantile:g}_{channel}" for quantile in TILE_QUANTILES)
    for difference in ("r-g", "r-b", "g-b"):
        names.extend(
            (
                f"difference_mean_{difference}",
                f"difference_std_{difference}",
            )
        )
        names.extend(
            f"difference_q{quantile:g}_{difference}" for quantile in DIFFERENCE_QUANTILES
        )
    names.extend(("cov_rr", "cov_rg", "cov_rb", "cov_gg", "cov_gb", "cov_bb"))
    names.extend(("argmin_r", "argmin_g", "argmin_b", "argmax_r", "argmax_g", "argmax_b"))
    return tuple(names)


def extract_invariant_features(image: NDArray[np.generic]) -> NDArray[np.float64]:
    """Extract histogram and tile-population features without using tile positions.

    Every feature is invariant to a permutation of the 576 complete input
    tiles.  Tile-level moments are retained because the corruption is sampled
    independently per tile and their population exposes its severity.
    """

    value = _validate_rgb(image)
    flat = value.reshape(-1, 3).astype(np.float64)
    grid = value.reshape(GRID_SIZE, TILE_SIZE, GRID_SIZE, TILE_SIZE, 3)
    tiles = grid.transpose(0, 2, 1, 3, 4).reshape(-1, TILE_SIZE * TILE_SIZE, 3)
    tile_means = tiles.mean(axis=1)
    tile_stds = tiles.std(axis=1)
    pixel_quantiles = np.quantile(flat, PIXEL_QUANTILES, axis=0)
    tile_mean_quantiles = np.quantile(tile_means, TILE_QUANTILES, axis=0)
    tile_std_quantiles = np.quantile(tile_stds, TILE_QUANTILES, axis=0)

    features: list[float] = []
    for channel in range(3):
        values = flat[:, channel]
        median = pixel_quantiles[np.flatnonzero(PIXEL_QUANTILES == 0.5)[0], channel]
        features.extend(
            (
                float(values.mean()),
                float(values.std()),
                float(np.median(np.abs(values - median))),
                float(values.min()),
                float(values.max()),
            )
        )
        features.extend(pixel_quantiles[:, channel].tolist())
        histogram, _ = np.histogram(values, bins=HISTOGRAM_EDGES)
        features.extend((histogram / len(values)).tolist())
        features.extend(float(np.mean(values <= threshold)) for threshold in SATURATION_THRESHOLDS)
        features.extend(
            float(np.mean(values >= 255 - threshold)) for threshold in SATURATION_THRESHOLDS
        )
        features.extend(
            (
                float(tile_means[:, channel].mean()),
                float(tile_means[:, channel].std()),
                float(tile_stds[:, channel].mean()),
                float(tile_stds[:, channel].std()),
            )
        )
        features.extend(tile_mean_quantiles[:, channel].tolist())
        features.extend(tile_std_quantiles[:, channel].tolist())

    for left, right in ((0, 1), (0, 2), (1, 2)):
        difference = flat[:, left] - flat[:, right]
        features.extend((float(difference.mean()), float(difference.std())))
        features.extend(np.quantile(difference, DIFFERENCE_QUANTILES).tolist())

    covariance = np.cov(flat, rowvar=False, ddof=0)
    features.extend(
        float(covariance[row, column])
        for row, column in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    )
    minimum_channel = np.argmin(flat, axis=1)
    maximum_channel = np.argmax(flat, axis=1)
    features.extend(float(np.mean(minimum_channel == channel)) for channel in range(3))
    features.extend(float(np.mean(maximum_channel == channel)) for channel in range(3))
    result = np.asarray(features, dtype=np.float64)
    if result.shape != (len(feature_names()),) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"invalid invariant feature vector {result.shape}")
    return result


def _constant_channel_components(
    channel: NDArray[np.generic],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    value = np.asarray(channel, dtype=np.float64)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected one 480x480 channel, got {value.shape}")
    local_mean = uniform_filter(value, _WINDOW_SIZE, mode="reflect")
    local_second = uniform_filter(value * value, _WINDOW_SIZE, mode="reflect")
    local_variance = np.maximum(
        0.0,
        (local_second - local_mean * local_mean) * _SAMPLE_COVARIANCE_FACTOR,
    )
    crop = np.s_[_WINDOW_PAD:-_WINDOW_PAD, _WINDOW_PAD:-_WINDOW_PAD]
    return local_mean[crop], (_C2 / (local_variance + _C2))[crop]


def constant_channel_ssim(channel: NDArray[np.generic], color: float) -> float:
    """Evaluate exact scikit-image SSIM against a constant single channel."""

    if not np.isfinite(color):
        raise ValueError("constant colour must be finite")
    local_mean, contrast_weight = _constant_channel_components(channel)
    luminance = (2.0 * local_mean * color + _C1) / (
        local_mean * local_mean + color * color + _C1
    )
    return float(np.mean(contrast_weight * luminance))


def ssim_optimal_constant_rgb(target: NDArray[np.generic]) -> NDArray[np.float64]:
    """Find the continuous RGB constant maximising exact contest SSIM.

    This is a training-label/oracle diagnostic.  The objective decomposes over
    channels because the organizer uses ``channel_axis=2``.  For a constant
    prediction its local variance and covariance are zero, yielding a cheap
    bounded one-dimensional optimisation per channel.
    """

    value = _validate_rgb(target)
    colors = []
    for channel in range(3):
        local_mean, contrast_weight = _constant_channel_components(value[:, :, channel])

        def objective(
            color: float,
            mean: NDArray[np.float64] = local_mean,
            weight: NDArray[np.float64] = contrast_weight,
        ) -> float:
            luminance = (2.0 * mean * color + _C1) / (
                mean * mean + color * color + _C1
            )
            return -float(np.mean(weight * luminance))

        result = minimize_scalar(
            objective,
            bounds=(0.0, 255.0),
            method="bounded",
            options={"xatol": 1e-7},
        )
        if not result.success:
            raise RuntimeError(f"constant SSIM optimisation failed: {result.message}")
        colors.append(float(result.x))
    return np.asarray(colors, dtype=np.float64)


@dataclass(frozen=True)
class BootstrapInterval:
    """Deterministic paired-bootstrap summary."""

    mean: float
    lower_95: float
    upper_95: float
    wins: int
    count: int


def paired_bootstrap_interval(
    differences: NDArray[np.generic], *, replicates: int = 20_000, seed: int = 20260829
) -> BootstrapInterval:
    """Return a deterministic percentile CI for paired score differences."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a finite vector with at least two elements")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise ValueError("replicates must be an integer of at least 100")
    rng = np.random.default_rng(seed)
    # Generate in moderate chunks so the helper is also safe for large panels.
    means: list[NDArray[np.float64]] = []
    remaining = replicates
    while remaining:
        count = min(remaining, 4_096)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    bootstrap_means = np.concatenate(means)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return BootstrapInterval(
        mean=float(values.mean()),
        lower_95=float(lower),
        upper_95=float(upper),
        wins=int(np.count_nonzero(values > 0.0)),
        count=len(values),
    )
