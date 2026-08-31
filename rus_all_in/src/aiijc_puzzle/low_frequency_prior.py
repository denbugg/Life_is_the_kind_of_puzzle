"""Research-only low-frequency priors for metric analysis.

The puzzle permutation preserves the unordered population of pixels and tiles.
This module deliberately predicts only smooth spatial fields from permutation-
invariant board statistics.  Fitting may use paired records from the manifest's
``train`` split; :meth:`FrozenLowFrequencyPrior.predict_all` accepts only one
dirty RGB image and cannot inspect a reference target.

These rendered fields are not eligible competition outputs: they do not place
all 576 fragments.  Only ``generic_tile_template`` may be reused, strictly as a
position unary inside :mod:`aiijc_puzzle.compliant_atlas_decoder`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from aiijc_puzzle.legacy_upgrade import constant_prediction
from aiijc_puzzle.novel_analog_layout import (
    board_signature,
    consensus_layout,
    render_layout,
    tile_semantic_features,
)
from aiijc_puzzle.protocol import IMAGE_SIZE, split_tiles

GRID_SIZES = (4, 8, 12)
BLEND_STRENGTHS = (0.5, 1.0)
HUNGARIAN_BLUR_SIGMAS = (20.0, 40.0, 80.0, 120.0)
RIDGE_ALPHA = 100.0
CLUSTER_COUNT = 8
MODEL_SCHEMA_VERSION = 1


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image {expected}, got {value.dtype} {value.shape}")
    return value


def dirty_board_features(image: np.ndarray) -> np.ndarray:
    """Return a permutation-invariant descriptor of one dirty board.

    The distributional tile signature is augmented with exact whole-image
    channel moments.  Neither part depends on the order of the 576 tiles.
    """

    value = _validate_rgb(image)
    pixels = value.reshape(-1, 3).astype(np.float32) / 255.0
    global_features = np.concatenate(
        (
            pixels.mean(axis=0),
            pixels.std(axis=0),
            np.median(pixels, axis=0),
            np.quantile(pixels, (0.1, 0.25, 0.75, 0.9), axis=0).reshape(-1),
        )
    )
    semantic = tile_semantic_features(split_tiles(value))
    return np.concatenate((board_signature(semantic), global_features)).astype(np.float32)


def target_grid(image: np.ndarray, size: int) -> np.ndarray:
    """Area-average an RGB target to a small floating-point spatial grid."""

    value = _validate_rgb(image)
    if size not in GRID_SIZES:
        raise ValueError(f"unsupported grid size {size}; expected one of {GRID_SIZES}")
    return cv2.resize(
        value.astype(np.float32) / 255.0,
        (size, size),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


@dataclass(frozen=True)
class LinearHead:
    """A standardized multi-output ridge head stored without sklearn state."""

    coefficient: np.ndarray
    intercept: np.ndarray
    alpha: float

    def predict(self, standardized_features: np.ndarray) -> np.ndarray:
        values = np.asarray(standardized_features, dtype=np.float32)
        if values.ndim == 1:
            values = values[None]
        return values @ self.coefficient.T + self.intercept


def _fit_head(features_z: np.ndarray, targets: np.ndarray, *, alpha: float) -> LinearHead:
    x = np.asarray(features_z, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError(f"incompatible ridge arrays: {x.shape}, {y.shape}")
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x, y)
    return LinearHead(
        coefficient=np.asarray(model.coef_, dtype=np.float32),
        intercept=np.asarray(model.intercept_, dtype=np.float32),
        alpha=float(alpha),
    )


def _smooth_render(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected square RGB grid, got {values.shape}")
    rendered = cv2.resize(values, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    sigma = IMAGE_SIZE / (4.0 * values.shape[0])
    return cv2.GaussianBlur(rendered, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _rgb_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)


def _blend(base: np.ndarray, candidate: np.ndarray, strength: float) -> np.ndarray:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("blend strength must be in [0, 1]")
    mixed = (1.0 - strength) * base.astype(np.float32) + strength * candidate.astype(np.float32)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _adapt_grid_color(grid: np.ndarray, color: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float32)
    desired = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return values + desired - values.mean(axis=(0, 1), keepdims=True)


@dataclass(frozen=True)
class FrozenLowFrequencyPrior:
    """Train-fitted state needed by the target-free low-frequency roster."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    color_head: LinearHead
    grid_heads: dict[int, LinearHead]
    population_mean_12: np.ndarray
    population_median_12: np.ndarray
    cluster_centers: np.ndarray
    cluster_atlas_12: np.ndarray
    generic_tile_template: np.ndarray
    metadata: dict[str, object]

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        return (values - self.feature_mean) / self.feature_scale

    def predict_all(self, dirty_image: np.ndarray) -> dict[str, np.ndarray]:
        """Build every preregistered arm without access to a clean target."""

        image = _validate_rgb(dirty_image)
        baseline = constant_prediction(image, statistic="median", per_channel=True)
        baseline_float = baseline.astype(np.float32) / 255.0
        feature = dirty_board_features(image)
        feature_z = self._standardize(feature)
        predicted_color = np.clip(self.color_head.predict(feature_z)[0], 0.0, 1.0)

        result: dict[str, np.ndarray] = {"constant_input_channel_median": baseline}
        templates = {
            "population_mean": self.population_mean_12,
            "population_median": self.population_median_12,
        }
        for name, template in templates.items():
            candidate = np.clip(_smooth_render(_adapt_grid_color(template, predicted_color)), 0, 1)
            candidate_u8 = _rgb_u8(candidate)
            for strength in BLEND_STRENGTHS:
                result[f"{name}_s{int(strength * 100):03d}"] = _blend(
                    baseline, candidate_u8, strength
                )

        for size in GRID_SIZES:
            predicted = self.grid_heads[size].predict(feature_z)[0].reshape(size, size, 3)
            candidate = _rgb_u8(np.clip(_smooth_render(predicted), 0, 1))
            for strength in BLEND_STRENGTHS:
                result[f"ridge_grid{size}_s{int(strength * 100):03d}"] = _blend(
                    baseline, candidate, strength
                )

        distances = np.square(self.cluster_centers - feature_z[None]).mean(axis=1)
        cluster = int(np.argmin(distances))
        cluster_grid = _adapt_grid_color(self.cluster_atlas_12[cluster], predicted_color)
        cluster_candidate = _rgb_u8(np.clip(_smooth_render(cluster_grid), 0, 1))
        for strength in BLEND_STRENGTHS:
            result[f"cluster8_atlas_s{int(strength * 100):03d}"] = _blend(
                baseline, cluster_candidate, strength
            )

        query_features = tile_semantic_features(split_tiles(image))
        mapping, _ = consensus_layout(
            query_features,
            self.generic_tile_template[None],
            np.zeros(1, dtype=np.float32),
        )
        layout = render_layout(image, mapping)
        desired_color = np.median(image.reshape(-1, 3), axis=0).astype(np.float32)
        for sigma in HUNGARIAN_BLUR_SIGMAS:
            blurred = cv2.GaussianBlur(layout, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(
                np.float32
            )
            blurred += desired_color - blurred.mean(axis=(0, 1), keepdims=True)
            result[f"hungarian_blur_sigma{int(sigma)}"] = np.clip(
                np.rint(blurred), 0, 255
            ).astype(np.uint8)

        expected_arms = 1 + 2 * 2 + len(GRID_SIZES) * 2 + 2 + len(HUNGARIAN_BLUR_SIGMAS)
        if len(result) != expected_arms:
            raise RuntimeError(f"internal roster mismatch: {len(result)} != {expected_arms}")
        if not np.array_equal(baseline_float, baseline.astype(np.float32) / 255.0):
            raise RuntimeError("unexpected baseline mutation")
        return result

    def save(self, path: Path) -> None:
        """Persist the frozen model as a non-pickle compressed NumPy archive."""

        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "color_coefficient": self.color_head.coefficient,
            "color_intercept": self.color_head.intercept,
            "population_mean_12": self.population_mean_12,
            "population_median_12": self.population_median_12,
            "cluster_centers": self.cluster_centers,
            "cluster_atlas_12": self.cluster_atlas_12,
            "generic_tile_template": self.generic_tile_template,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for size, head in self.grid_heads.items():
            arrays[f"grid{size}_coefficient"] = head.coefficient
            arrays[f"grid{size}_intercept"] = head.intercept
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> FrozenLowFrequencyPrior:
        """Load a model written by :meth:`save` without enabling pickle."""

        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("schema_version") != MODEL_SCHEMA_VERSION:
                raise ValueError(f"unsupported low-frequency model schema in {path}")
            color_head = LinearHead(
                archive["color_coefficient"].copy(),
                archive["color_intercept"].copy(),
                float(metadata["ridge_alpha"]),
            )
            grid_heads = {
                size: LinearHead(
                    archive[f"grid{size}_coefficient"].copy(),
                    archive[f"grid{size}_intercept"].copy(),
                    float(metadata["ridge_alpha"]),
                )
                for size in GRID_SIZES
            }
            return cls(
                feature_mean=archive["feature_mean"].copy(),
                feature_scale=archive["feature_scale"].copy(),
                color_head=color_head,
                grid_heads=grid_heads,
                population_mean_12=archive["population_mean_12"].copy(),
                population_median_12=archive["population_median_12"].copy(),
                cluster_centers=archive["cluster_centers"].copy(),
                cluster_atlas_12=archive["cluster_atlas_12"].copy(),
                generic_tile_template=archive["generic_tile_template"].copy(),
                metadata=metadata,
            )


def fit_low_frequency_prior(
    dirty_features: np.ndarray,
    target_grids: dict[int, np.ndarray],
    generic_tile_template: np.ndarray,
    *,
    ridge_alpha: float = RIDGE_ALPHA,
    cluster_count: int = CLUSTER_COUNT,
    seed: int = 20260829,
    metadata: dict[str, object] | None = None,
) -> FrozenLowFrequencyPrior:
    """Fit the frozen roster from arrays extracted from train records only."""

    x = np.asarray(dirty_features, dtype=np.float32)
    if x.ndim != 2 or len(x) < cluster_count:
        raise ValueError(f"expected at least {cluster_count} feature rows, got {x.shape}")
    if set(target_grids) != set(GRID_SIZES):
        raise ValueError(f"target grids must have sizes {GRID_SIZES}")
    normalized_targets: dict[int, np.ndarray] = {}
    for size in GRID_SIZES:
        values = np.asarray(target_grids[size], dtype=np.float32)
        if values.shape != (len(x), size, size, 3):
            raise ValueError(f"unexpected grid {size} shape {values.shape}")
        normalized_targets[size] = values
    feature_mean = x.mean(axis=0)
    feature_scale = np.maximum(x.std(axis=0), 1e-5)
    x_z = (x - feature_mean) / feature_scale
    heads = {
        size: _fit_head(x_z, values.reshape(len(x), -1), alpha=ridge_alpha)
        for size, values in normalized_targets.items()
    }
    color_targets = normalized_targets[12].mean(axis=(1, 2))
    color_head = _fit_head(x_z, color_targets, alpha=ridge_alpha)

    clustering = KMeans(
        n_clusters=cluster_count,
        random_state=seed,
        n_init=10,
        max_iter=300,
    ).fit(x_z)
    cluster_atlas = np.stack(
        [
            normalized_targets[12][clustering.labels_ == label].mean(axis=0)
            for label in range(cluster_count)
        ]
    ).astype(np.float32)
    model_metadata: dict[str, object] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "ridge_alpha": float(ridge_alpha),
        "cluster_count": int(cluster_count),
        "seed": int(seed),
        "train_records": int(len(x)),
    }
    if metadata:
        model_metadata.update(metadata)
    return FrozenLowFrequencyPrior(
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
        color_head=color_head,
        grid_heads=heads,
        population_mean_12=normalized_targets[12].mean(axis=0).astype(np.float32),
        population_median_12=np.median(normalized_targets[12], axis=0).astype(np.float32),
        cluster_centers=np.asarray(clustering.cluster_centers_, dtype=np.float32),
        cluster_atlas_12=cluster_atlas,
        generic_tile_template=np.asarray(generic_tile_template, dtype=np.float32),
        metadata=model_metadata,
    )
