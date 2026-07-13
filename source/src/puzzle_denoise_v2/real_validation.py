"""Leakage-safe metrics for fixed panels of matched real tile pairs.

The functions in this module operate only on already paired 20x20 tiles.  They
do not infer mappings or assemble images.  Metrics are reported both over all
pairs (micro) and as an equal-weight average over source images (macro), so a
source with many confident matches cannot dominate model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import tile_metrics


DEFAULT_BOOTSTRAP_SEED = 20260710


@dataclass(frozen=True)
class RealPairEvaluation:
    """Metrics for one real-pair panel or confidence stratum."""

    pair_count: int
    source_count: int
    source_ids: tuple[int, ...]
    micro_metrics: dict[str, float]
    macro_metrics: dict[str, float]
    per_source_metrics: dict[int, dict[str, float]]


@dataclass(frozen=True)
class PairedBootstrapCI:
    """Percentile interval for a source-macro candidate-minus-baseline delta."""

    metric: str
    candidate_minus_baseline: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    seed: int
    source_count: int


@dataclass(frozen=True)
class ConfidenceStratum:
    """A confidence interval with an inclusive lower and exclusive upper bound."""

    name: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("confidence stratum name must be a non-empty string")
        for label, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"confidence stratum {label} must be finite or None")
        if self.minimum is not None and self.maximum is not None and self.minimum >= self.maximum:
            raise ValueError("confidence stratum minimum must be smaller than maximum")

    def mask(self, confidence: np.ndarray) -> np.ndarray:
        selected = np.ones(len(confidence), dtype=bool)
        if self.minimum is not None:
            selected &= confidence >= self.minimum
        if self.maximum is not None:
            selected &= confidence < self.maximum
        return selected


def _validate_tiles(name: str, tiles: np.ndarray) -> np.ndarray:
    array = np.asarray(tiles)
    if array.ndim != 4 or array.shape[1:] not in ((20, 20, 3), (3, 20, 20)):
        raise ValueError(f"{name} must contain NHWC or NCHW 20x20 RGB tiles, got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {array.dtype}")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _validate_source_indices(
    source_indices: np.ndarray,
    pair_count: int,
    source_count: int | None,
) -> np.ndarray:
    indices = np.asarray(source_indices)
    if indices.ndim != 1 or len(indices) != pair_count:
        raise ValueError(f"source_indices must have shape ({pair_count},), got {indices.shape}")
    if indices.dtype == np.bool_ or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("source_indices must have an integer dtype")
    if np.any(indices < 0):
        raise ValueError("source_indices must be non-negative")
    if source_count is not None:
        if isinstance(source_count, (bool, np.bool_)) or not isinstance(source_count, (int, np.integer)):
            raise TypeError("source_count must be an integer or None")
        if source_count <= 0:
            raise ValueError("source_count must be positive")
        if np.any(indices >= source_count):
            raise ValueError("source_indices contains an index outside source_count")
    return indices.astype(np.int64, copy=False)


def _validate_boundary_band(boundary_band: int) -> int:
    if isinstance(boundary_band, (bool, np.bool_)) or not isinstance(boundary_band, (int, np.integer)):
        raise TypeError("boundary_band must be an integer")
    if not 1 <= boundary_band < 10:
        raise ValueError("boundary_band must leave non-empty boundary and interior regions")
    return int(boundary_band)


def _validated_panel(
    prediction: np.ndarray,
    target: np.ndarray,
    source_indices: np.ndarray,
    source_count: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_array = _validate_tiles("prediction", prediction)
    target_array = _validate_tiles("target", target)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target shapes differ")
    indices = _validate_source_indices(source_indices, len(target_array), source_count)
    return prediction_array, target_array, indices


def evaluate_real_pairs(
    prediction: np.ndarray,
    target: np.ndarray,
    source_indices: np.ndarray,
    *,
    source_count: int | None = None,
    boundary_band: int = 3,
) -> RealPairEvaluation:
    """Evaluate uint8 paired tiles with equal-weight source-macro metrics.

    ``source_indices`` may be non-contiguous, which is useful after confidence
    filtering.  Pass the panel's declared ``source_count`` when an upper-bound
    integrity check against its metadata is available.
    """

    prediction_array, target_array, indices = _validated_panel(
        prediction,
        target,
        source_indices,
        source_count,
    )
    band = _validate_boundary_band(boundary_band)
    source_ids_array = np.unique(indices)
    per_source: dict[int, dict[str, float]] = {}
    for source_id in source_ids_array:
        selected = indices == source_id
        per_source[int(source_id)] = tile_metrics(
            prediction_array[selected],
            target_array[selected],
            boundary_band=band,
        )

    metric_names = tuple(next(iter(per_source.values())))
    macro = {
        metric: float(np.mean([values[metric] for values in per_source.values()]))
        for metric in metric_names
    }
    return RealPairEvaluation(
        pair_count=len(target_array),
        source_count=len(source_ids_array),
        source_ids=tuple(int(value) for value in source_ids_array),
        micro_metrics=tile_metrics(prediction_array, target_array, boundary_band=band),
        macro_metrics=macro,
        per_source_metrics=per_source,
    )


def paired_source_bootstrap_delta(
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    target: np.ndarray,
    source_indices: np.ndarray,
    *,
    metric: str = "tile_ssim",
    source_count: int | None = None,
    boundary_band: int = 3,
    confidence_level: float = 0.95,
    resamples: int = 2000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedBootstrapCI:
    """Bootstrap the source-macro delta while preserving candidate/baseline pairing.

    The reported delta is always ``candidate - baseline``.  Positive is better
    for SSIM/PSNR and negative is better for non-negative error metrics.  A
    signed-bias delta must be interpreted together with its absolute value.
    """

    candidate_array, target_array, indices = _validated_panel(
        candidate_prediction,
        target,
        source_indices,
        source_count,
    )
    baseline_array = _validate_tiles("baseline_prediction", baseline_prediction)
    if baseline_array.shape != target_array.shape:
        raise ValueError("baseline_prediction and target shapes differ")
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if isinstance(resamples, (bool, np.bool_)) or not isinstance(resamples, (int, np.integer)):
        raise TypeError("resamples must be an integer")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    candidate = evaluate_real_pairs(
        candidate_array,
        target_array,
        indices,
        source_count=source_count,
        boundary_band=boundary_band,
    )
    baseline = evaluate_real_pairs(
        baseline_array,
        target_array,
        indices,
        source_count=source_count,
        boundary_band=boundary_band,
    )
    if metric not in candidate.macro_metrics:
        available = ", ".join(candidate.macro_metrics)
        raise ValueError(f"unknown metric {metric!r}; available metrics: {available}")

    deltas = np.asarray(
        [
            candidate.per_source_metrics[source_id][metric]
            - baseline.per_source_metrics[source_id][metric]
            for source_id in candidate.source_ids
        ],
        dtype=np.float64,
    )
    if not np.isfinite(deltas).all():
        raise ValueError(
            f"metric {metric!r} has a non-finite per-source delta; "
            "a percentile confidence interval is undefined"
        )

    generator = np.random.default_rng(int(seed))
    sampled_sources = generator.integers(0, len(deltas), size=(int(resamples), len(deltas)))
    bootstrap_deltas = deltas[sampled_sources].mean(axis=1)
    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(bootstrap_deltas, [alpha, 1.0 - alpha])
    return PairedBootstrapCI(
        metric=metric,
        candidate_minus_baseline=float(deltas.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=float(confidence_level),
        resamples=int(resamples),
        seed=int(seed),
        source_count=len(deltas),
    )


def confidence_stratum_masks(
    confidence: np.ndarray,
    strata: Sequence[ConfidenceStratum],
) -> dict[str, np.ndarray]:
    """Return independently defined confidence masks in caller-provided order."""

    values = np.asarray(confidence)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("confidence must be a non-empty one-dimensional array")
    if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.number):
        raise TypeError("confidence must have a numeric dtype")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("confidence values must be finite and non-negative")
    definitions = tuple(strata)
    if not definitions:
        raise ValueError("at least one confidence stratum is required")
    if any(not isinstance(stratum, ConfidenceStratum) for stratum in definitions):
        raise TypeError("strata must contain ConfidenceStratum instances")
    names = [stratum.name for stratum in definitions]
    if len(set(names)) != len(names):
        raise ValueError("confidence stratum names must be unique")
    return {stratum.name: stratum.mask(values) for stratum in definitions}


def evaluate_confidence_strata(
    prediction: np.ndarray,
    target: np.ndarray,
    source_indices: np.ndarray,
    confidence: np.ndarray,
    strata: Sequence[ConfidenceStratum],
    *,
    source_count: int | None = None,
    boundary_band: int = 3,
) -> dict[str, RealPairEvaluation]:
    """Evaluate each explicit confidence stratum as its own real-pair panel.

    Strata may overlap when cumulative thresholds are desired.  An empty
    stratum raises instead of silently producing an unusable validation row.
    """

    prediction_array, target_array, indices = _validated_panel(
        prediction,
        target,
        source_indices,
        source_count,
    )
    masks = confidence_stratum_masks(confidence, strata)
    if len(next(iter(masks.values()))) != len(target_array):
        raise ValueError(f"confidence must have shape ({len(target_array)},)")

    evaluations: dict[str, RealPairEvaluation] = {}
    for name, selected in masks.items():
        if not selected.any():
            raise ValueError(f"confidence stratum {name!r} is empty")
        evaluations[name] = evaluate_real_pairs(
            prediction_array[selected],
            target_array[selected],
            indices[selected],
            source_count=source_count,
            boundary_band=boundary_band,
        )
    return evaluations
