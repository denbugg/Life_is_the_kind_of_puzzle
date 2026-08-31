"""Dirty-visible calibration features for hard Socket OT edges.

The Socket matcher returns a fractional partial assignment, while the decoder
starts from an exact-cardinality Hungarian projection.  This module describes
each projected real edge using only evidence available in that dirty board and
fits one small linear probability model.  Exact synthetic labels deliberately
live in separate helpers so feature extraction cannot accidentally consume a
reference layout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.socket_cycle_diagnostic import (
    axis_socket_rankings,
    commutative_cycle_support,
)
from aiijc_puzzle.socket_decoder import SocketEdge, hard_partial_axis_matching

FEATURE_NAMES = (
    "projected_edge_confidence",
    "ot_row_real_margin",
    "ot_column_real_margin",
    "ot_outgoing_dustbin_margin",
    "ot_incoming_dustbin_margin",
    "ot_row_log_probability",
    "ot_column_log_probability",
    "ot_row_reciprocal_rank",
    "ot_column_reciprocal_rank",
    "raw_row_real_margin",
    "raw_column_real_margin",
    "raw_row_log_probability",
    "raw_column_log_probability",
    "raw_row_reciprocal_rank",
    "raw_column_reciprocal_rank",
    "cycle_k4_supported",
    "cycle_k4_log_support",
    "cycle_k4_inverse_best_rank_sum",
    "cycle_k4_best_log_score",
    "axis_is_down",
)


@dataclass(frozen=True)
class HardEdgeFeatures:
    """Feature rows and edge identities for one dirty board."""

    values: np.ndarray
    source: np.ndarray
    target: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class FrozenLinearCalibrator:
    """Portable standardised logistic regression plus one frozen threshold."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    threshold: float
    target_fit_precision: float
    achieved_fit_precision: float

    def predict_probability(self, values: np.ndarray) -> np.ndarray:
        array = _feature_matrix(values)
        if array.shape[1] != len(self.feature_names):
            raise ValueError("feature count differs from the frozen calibrator")
        normalised = (array - self.mean) / self.scale
        logit = normalised @ self.coefficients + self.intercept
        # Numerically stable sigmoid without an optional scipy dependency here.
        probability = np.empty_like(logit, dtype=np.float64)
        positive = logit >= 0
        probability[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
        exponent = np.exp(logit[~positive])
        probability[~positive] = exponent / (1.0 + exponent)
        return probability

    def select(self, values: np.ndarray) -> np.ndarray:
        return self.predict_probability(values) >= self.threshold


def frozen_linear_calibrator_from_payload(
    payload: Mapping[str, Any],
) -> FrozenLinearCalibrator:
    """Load the hash-locked JSON representation emitted by the fit probe."""

    if payload.get("schema") != "aiijc-socket-hard-edge-linear-calibrator-v1":
        raise ValueError("unsupported frozen Socket calibrator schema")
    estimator = payload.get("estimator")
    threshold = payload.get("single_threshold")
    if not isinstance(estimator, Mapping) or not isinstance(threshold, Mapping):
        raise ValueError("frozen calibrator is missing estimator or threshold mappings")
    feature_names = estimator.get("feature_names")
    if feature_names != list(FEATURE_NAMES):
        raise ValueError("frozen calibrator feature contract differs from this implementation")

    def vector(name: str, *, positive: bool = False) -> np.ndarray:
        result = np.asarray(estimator.get(name), dtype=np.float64)
        if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
            raise ValueError(f"frozen calibrator {name} has an invalid vector")
        if positive and np.any(result <= 0):
            raise ValueError(f"frozen calibrator {name} must be strictly positive")
        return result

    mean = vector("mean")
    scale = vector("scale", positive=True)
    coefficients = vector("coefficients")
    intercept = estimator.get("intercept")
    probability_threshold = threshold.get("probability_greater_equal")
    target_precision = threshold.get("target_fit_precision")
    achieved_precision = threshold.get("achieved_fit_precision")
    scalars = (intercept, probability_threshold, target_precision, achieved_precision)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in scalars):
        raise ValueError("frozen calibrator scalar fields are invalid")
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("frozen calibrator scalar fields must be finite")
    if not 0.0 <= float(probability_threshold) <= 1.0:
        raise ValueError("frozen calibrator probability threshold must be in [0, 1]")
    if not 0.0 < float(target_precision) <= 1.0 or not 0.0 <= float(
        achieved_precision
    ) <= 1.0:
        raise ValueError("frozen calibrator precision fields are invalid")
    return FrozenLinearCalibrator(
        feature_names=FEATURE_NAMES,
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        intercept=float(intercept),
        threshold=float(probability_threshold),
        target_fit_precision=float(target_precision),
        achieved_fit_precision=float(achieved_precision),
    )


def _feature_matrix(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"features must have shape N x {len(FEATURE_NAMES)}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("features contain non-finite values")
    return result


def _assignment(value: Any, *, grid: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    expected = (grid * grid + 1, grid * grid + 1)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
    usable = result.copy()
    usable[-1, -1] = 0.0
    if not np.isfinite(usable).all():
        raise ValueError(f"{name} contains non-finite usable entries")
    return np.ascontiguousarray(result)


def _raw_scores(value: Any, *, grid: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    expected = (grid * grid, grid * grid)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _rank(values: np.ndarray, *, selected_index: int) -> int:
    selected = float(values[selected_index])
    # Stable deterministic tie-breaking matches argsort(kind="stable").
    better = int(np.count_nonzero(values > selected))
    earlier_ties = int(np.count_nonzero(values[:selected_index] == selected))
    return 1 + better + earlier_ties


def _real_stats(
    matrix: np.ndarray,
    *,
    source: int,
    target: int,
) -> tuple[float, float, float, float, float, float]:
    selected = float(matrix[source, target])
    row = matrix[source].copy()
    column = matrix[:, target].copy()
    row[target] = -np.inf
    column[source] = -np.inf
    row_margin = selected - float(np.max(row))
    column_margin = selected - float(np.max(column))
    original_row = matrix[source]
    original_column = matrix[:, target]
    row_log_probability = selected - _logsumexp(original_row)
    column_log_probability = selected - _logsumexp(original_column)
    row_rank = _rank(original_row, selected_index=target)
    column_rank = _rank(original_column, selected_index=source)
    return (
        row_margin,
        column_margin,
        row_log_probability,
        column_log_probability,
        1.0 / row_rank,
        1.0 / column_rank,
    )


def _edge_features(
    edge: SocketEdge,
    *,
    assignment: np.ndarray,
    raw: np.ndarray,
    right_rankings: Any,
    down_rankings: Any,
) -> tuple[float, ...]:
    count = raw.shape[0]
    selected = float(assignment[edge.source, edge.target])
    ot_real = assignment[:count, :count]
    ot_stats = _real_stats(ot_real, source=edge.source, target=edge.target)
    raw_stats = _real_stats(raw, source=edge.source, target=edge.target)
    ot_row_log_probability = selected - _logsumexp(assignment[edge.source, :])
    ot_column_log_probability = selected - _logsumexp(assignment[:, edge.target])
    cycle = commutative_cycle_support(
        edge,
        right=right_rankings,
        down=down_rankings,
        top_k=4,
    )
    supported = float(cycle.supported)
    return (
        float(edge.confidence),
        ot_stats[0],
        ot_stats[1],
        selected - float(assignment[edge.source, count]),
        selected - float(assignment[count, edge.target]),
        ot_row_log_probability,
        ot_column_log_probability,
        ot_stats[4],
        ot_stats[5],
        raw_stats[0],
        raw_stats[1],
        raw_stats[2],
        raw_stats[3],
        raw_stats[4],
        raw_stats[5],
        supported,
        math.log1p(cycle.support_count),
        0.0 if cycle.best_total_rank is None else 1.0 / cycle.best_total_rank,
        (
            0.0
            if cycle.best_total_conditional_log_score is None
            else cycle.best_total_conditional_log_score
        ),
        float(edge.axis == "down"),
    )


def extract_hard_edge_features(
    *,
    right_log_assignment: Any,
    down_log_assignment: Any,
    right_raw: Any,
    down_raw: Any,
    grid: int,
) -> HardEdgeFeatures:
    """Extract cheap dirty-only features for every hard-projected real edge."""

    if grid < 2:
        raise ValueError("grid must be at least 2")
    assignments = {
        "right": _assignment(right_log_assignment, grid=grid, name="right_log_assignment"),
        "down": _assignment(down_log_assignment, grid=grid, name="down_log_assignment"),
    }
    raw_scores = {
        "right": _raw_scores(right_raw, grid=grid, name="right_raw"),
        "down": _raw_scores(down_raw, grid=grid, name="down_raw"),
    }
    rankings = {
        axis: axis_socket_rankings(value, grid=grid, maximum_k=4)
        for axis, value in assignments.items()
    }
    rows: list[tuple[float, ...]] = []
    sources: list[int] = []
    targets: list[int] = []
    axes: list[int] = []
    for axis in ("right", "down"):
        matching = hard_partial_axis_matching(assignments[axis], grid=grid, axis=axis)
        for edge in matching.edges:
            rows.append(
                _edge_features(
                    edge,
                    assignment=assignments[axis],
                    raw=raw_scores[axis],
                    right_rankings=rankings["right"],
                    down_rankings=rankings["down"],
                )
            )
            sources.append(edge.source)
            targets.append(edge.target)
            axes.append(int(axis == "down"))
    expected = 2 * grid * (grid - 1)
    if len(rows) != expected:
        raise RuntimeError(f"hard edge count invariant failed: {len(rows)} != {expected}")
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (expected, len(FEATURE_NAMES)) or not np.isfinite(values).all():
        raise RuntimeError("hard edge feature matrix invariant failed")
    return HardEdgeFeatures(
        values=np.ascontiguousarray(values),
        source=np.asarray(sources, dtype=np.int32),
        target=np.asarray(targets, dtype=np.int32),
        axis=np.asarray(axes, dtype=np.int8),
    )


def exact_edge_labels(
    features: HardEdgeFeatures,
    reference_tile_at_position: Any,
    *,
    grid: int,
) -> np.ndarray:
    """Label frozen hard edges from an exact synthetic tile permutation."""

    reference = np.asarray(reference_tile_at_position, dtype=np.int64)
    count = grid * grid
    if reference.shape != (count,) or not np.array_equal(np.sort(reference), np.arange(count)):
        raise ValueError("reference_tile_at_position must be a strict grid permutation")
    row_count = len(features.values)
    if any(len(value) != row_count for value in (features.source, features.target, features.axis)):
        raise ValueError("hard edge identity arrays have inconsistent lengths")
    position = np.empty(count, dtype=np.int32)
    position[reference] = np.arange(count, dtype=np.int32)
    source_position = position[features.source]
    target_position = position[features.target]
    right = features.axis == 0
    correct = np.where(
        right,
        (target_position == source_position + 1) & (source_position % grid != grid - 1),
        target_position == source_position + grid,
    )
    return np.asarray(correct, dtype=bool)


def choose_precision_threshold(
    probability: Any,
    labels: Any,
    *,
    target_precision: float,
) -> tuple[float, float]:
    """Choose the most inclusive single threshold meeting fit precision."""

    scores = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or truth.shape != scores.shape or not len(scores):
        raise ValueError("probability and labels must be aligned non-empty vectors")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not math.isfinite(target_precision) or not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0, 1]")
    thresholds = np.unique(scores)[::-1]
    chosen: tuple[float, float, int] | None = None
    best_fallback: tuple[float, float, int] | None = None
    for threshold in thresholds:
        selected = scores >= threshold
        count = int(selected.sum())
        precision = float(truth[selected].mean())
        candidate = (float(threshold), precision, count)
        if best_fallback is None or (precision, count) > (best_fallback[1], best_fallback[2]):
            best_fallback = candidate
        if precision >= target_precision and (chosen is None or count > chosen[2]):
            chosen = candidate
    assert best_fallback is not None
    threshold, precision, _ = chosen if chosen is not None else best_fallback
    return threshold, precision


def fit_linear_calibrator(
    values: Any,
    labels: Any,
    *,
    target_precision: float = 0.80,
) -> FrozenLinearCalibrator:
    """Fit one fixed balanced logistic model and freeze its precision threshold."""

    features = _feature_matrix(values)
    truth = np.asarray(labels, dtype=bool)
    if truth.shape != (len(features),) or len(np.unique(truth)) != 2:
        raise ValueError("labels must be an aligned binary vector with both classes")
    scaler = StandardScaler().fit(features)
    normalised = scaler.transform(features)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
        solver="lbfgs",
    ).fit(normalised, truth)
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)
    intercept = float(model.intercept_[0])
    provisional = FrozenLinearCalibrator(
        feature_names=FEATURE_NAMES,
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficients=coefficients,
        intercept=intercept,
        threshold=0.5,
        target_fit_precision=target_precision,
        achieved_fit_precision=math.nan,
    )
    probability = provisional.predict_probability(features)
    threshold, precision = choose_precision_threshold(
        probability,
        truth,
        target_precision=target_precision,
    )
    return FrozenLinearCalibrator(
        feature_names=FEATURE_NAMES,
        mean=provisional.mean,
        scale=provisional.scale,
        coefficients=provisional.coefficients,
        intercept=provisional.intercept,
        threshold=threshold,
        target_fit_precision=target_precision,
        achieved_fit_precision=precision,
    )


def fixed_heuristic_selection(values: Any) -> np.ndarray:
    """Reproduce the already-tested precision-first hard threshold policy."""

    features = _feature_matrix(values)
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    return (
        (features[:, index["projected_edge_confidence"]] >= -1.0)
        & (features[:, index["ot_row_real_margin"]] >= 0.0)
        & (features[:, index["ot_column_real_margin"]] >= 0.0)
        & (features[:, index["ot_outgoing_dustbin_margin"]] >= 0.5)
        & (features[:, index["ot_incoming_dustbin_margin"]] >= 0.5)
    )


def mutual_top1_selection(values: Any, *, variant: str) -> np.ndarray:
    """Return a fixed raw- or OT-mutual-top-1 control."""

    if variant not in {"raw", "ot"}:
        raise ValueError("variant must be 'raw' or 'ot'")
    features = _feature_matrix(values)
    row = FEATURE_NAMES.index(f"{variant}_row_reciprocal_rank")
    column = FEATURE_NAMES.index(f"{variant}_column_reciprocal_rank")
    return np.isclose(features[:, row], 1.0) & np.isclose(features[:, column], 1.0)


__all__ = [
    "FEATURE_NAMES",
    "FrozenLinearCalibrator",
    "HardEdgeFeatures",
    "choose_precision_threshold",
    "exact_edge_labels",
    "extract_hard_edge_features",
    "fit_linear_calibrator",
    "fixed_heuristic_selection",
    "frozen_linear_calibrator_from_payload",
    "mutual_top1_selection",
]
