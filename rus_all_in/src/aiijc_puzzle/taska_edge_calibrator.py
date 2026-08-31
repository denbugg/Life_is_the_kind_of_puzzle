"""Portable target-free confidence calibration for harvested TASKA edges.

The feature contract is deliberately narrow: it consumes only the two frozen
cost matrices, their raw-log counterparts, and evidence already attached to
each harvested edge.  It never receives a target, exact permutation, filename,
source coordinate, or rendered image.

The production 24x24 matcher writes a zero diagonal.  The historical one-off
feature formula intentionally includes that diagonal in the row/column
partitions, means, standard deviations, and ranks.  This module preserves that
bit contract rather than silently "fixing" the diagonal at inference time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.raw_tail_global_solver import (
    RawTailEdge,
    RawTailGlobalConfig,
    RawTailGlobalDiagnostics,
    _fill_seams,
    _place_components,
    _TranslationBuilder,
)

FEATURE_NAMES = (
    "axis_is_down",
    "selected_cost",
    "selected_raw_log",
    "harvest_edge_weight",
    "harvest_vote_count",
    "row_cost_minus_partition_0",
    "row_cost_minus_partition_1",
    "row_cost_z",
    "column_cost_minus_partition_0",
    "column_cost_minus_partition_1",
    "column_cost_z",
    "row_lower_cost_fraction",
    "column_lower_cost_fraction",
    "reverse_cost",
    "reverse_raw_log",
)
FEATURE_COUNT = len(FEATURE_NAMES)
CALIBRATOR_SCHEMA = "aiijc-taska-edge-calibrator-v1"
_STANDARD_DEVIATION_EPSILON = 1e-6


def _as_numpy(value: Any) -> Any:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return current


def _validate_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _finite_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    result = np.asarray(_as_numpy(value), dtype=np.float64)
    if result.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _finite_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    result = np.asarray(_as_numpy(value), dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _validated_edges(
    candidate_edges: Sequence[RawTailEdge],
    *,
    count: int,
) -> tuple[RawTailEdge, ...]:
    edges = tuple(candidate_edges)
    seen: set[tuple[int, int, str]] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, RawTailEdge):
            raise TypeError(f"candidate_edges[{index}] must be a RawTailEdge")
        if edge.axis not in {"right", "down"}:
            raise ValueError(f"candidate_edges[{index}] has an invalid axis")
        if isinstance(edge.source, bool) or not isinstance(edge.source, int):
            raise ValueError(f"candidate_edges[{index}].source must be an integer")
        if isinstance(edge.target, bool) or not isinstance(edge.target, int):
            raise ValueError(f"candidate_edges[{index}].target must be an integer")
        if not 0 <= edge.source < count or not 0 <= edge.target < count:
            raise ValueError(f"candidate_edges[{index}] is outside the input bag")
        if edge.source == edge.target:
            raise ValueError(f"candidate_edges[{index}] is a self-edge")
        key = (edge.source, edge.target, edge.axis)
        if key in seen:
            raise ValueError(f"candidate_edges[{index}] duplicates an earlier edge")
        seen.add(key)
    return edges


def _feature_matrix(value: Any) -> np.ndarray:
    result = np.asarray(_as_numpy(value), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != FEATURE_COUNT:
        raise ValueError(f"features must have shape rows x {FEATURE_COUNT}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("features must contain only finite values")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class TaskaEdgeFeatureBatch:
    """Feature rows aligned one-to-one with one ordered harvested edge list."""

    values: np.ndarray
    edges: tuple[RawTailEdge, ...]

    def __post_init__(self) -> None:
        values = _feature_matrix(self.values).copy()
        if len(values) != len(self.edges):
            raise ValueError("feature rows and edges must have equal length")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class PrioritizedRawTailBuildDecision:
    """Auditable component-build decision under an external edge priority."""

    edge: RawTailEdge
    raw_priority: float
    ranking_priority: float
    input_rank: int
    status: str


@dataclass(frozen=True)
class PrioritizedRawTailResult:
    """Strict raw-tail result whose component order came from external priorities."""

    layout: np.ndarray
    components: tuple[dict[int, tuple[int, int]], ...]
    decisions: tuple[PrioritizedRawTailBuildDecision, ...]
    diagnostics: RawTailGlobalDiagnostics


def extract_taska_edge_features(
    cost_right: Any,
    cost_down: Any,
    right_log: Any,
    down_log: Any,
    candidate_edges: Sequence[RawTailEdge],
    edge_weights: Any,
    edge_vote_counts: Any,
    *,
    grid: int = 24,
) -> TaskaEdgeFeatureBatch:
    """Extract the fixed 15 dirty-visible features for every harvested edge.

    The row/column lower-cost fractions divide by ``grid**2 - 1``; this is
    exactly 575 on the production 24x24 board.  The selected value and diagonal
    remain in the numerator comparison, matching the frozen one-off formula.
    """

    count = _validate_grid(grid)
    right = _finite_matrix(cost_right, count=count, name="cost_right")
    down = _finite_matrix(cost_down, count=count, name="cost_down")
    right_raw_log = _finite_matrix(right_log, count=count, name="right_log")
    down_raw_log = _finite_matrix(down_log, count=count, name="down_log")
    edges = _validated_edges(candidate_edges, count=count)
    weights = _finite_vector(edge_weights, length=len(edges), name="edge_weights")
    votes = _finite_vector(
        edge_vote_counts,
        length=len(edges),
        name="edge_vote_counts",
    )
    if np.any(votes < 0):
        raise ValueError("edge_vote_counts must be non-negative")

    values = np.empty((len(edges), FEATURE_COUNT), dtype=np.float64)
    rank_denominator = float(count - 1)
    for index, edge in enumerate(edges):
        matrix = right if edge.axis == "right" else down
        raw_log = right_raw_log if edge.axis == "right" else down_raw_log
        source, target = edge.source, edge.target
        selected = float(matrix[source, target])
        row = matrix[source]
        column = matrix[:, target]
        row_partition = np.partition(row, 2)[:3]
        column_partition = np.partition(column, 2)[:3]
        values[index] = (
            float(edge.axis == "down"),
            selected,
            float(raw_log[source, target]),
            float(weights[index]),
            float(votes[index]),
            selected - float(row_partition[0]),
            selected - float(row_partition[1]),
            (selected - float(row.mean())) / (float(row.std()) + _STANDARD_DEVIATION_EPSILON),
            selected - float(column_partition[0]),
            selected - float(column_partition[1]),
            (selected - float(column.mean())) / (float(column.std()) + _STANDARD_DEVIATION_EPSILON),
            float(np.count_nonzero(row < selected)) / rank_denominator,
            float(np.count_nonzero(column < selected)) / rank_denominator,
            float(matrix[target, source]),
            float(raw_log[target, source]),
        )
    return TaskaEdgeFeatureBatch(values=values, edges=edges)


def _portable_vector(value: Any, *, name: str, positive: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite length-{FEATURE_COUNT} vector")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    result = np.ascontiguousarray(result.copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TaskaEdgeCalibrator:
    """Portable StandardScaler plus binary logistic-regression parameters."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("calibrator feature contract differs from FEATURE_NAMES")
        mean = _portable_vector(self.mean, name="mean")
        scale = _portable_vector(self.scale, name="scale", positive=True)
        coefficients = _portable_vector(self.coefficients, name="coefficients")
        if isinstance(self.intercept, bool) or not isinstance(
            self.intercept,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError("intercept must be a finite scalar")
        intercept = float(self.intercept)
        if not math.isfinite(intercept):
            raise ValueError("intercept must be a finite scalar")
        object.__setattr__(self, "feature_names", FEATURE_NAMES)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", intercept)

    def predict_logits(self, features: Any) -> np.ndarray:
        matrix = _feature_matrix(features)
        normalised = (matrix - self.mean) / self.scale
        return normalised @ self.coefficients + self.intercept

    def predict_priorities(self, features: Any) -> np.ndarray:
        """Return positive-class probabilities used as larger-is-better priorities."""

        logits = self.predict_logits(features)
        result = np.empty_like(logits, dtype=np.float64)
        positive = logits >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        result[~positive] = exponent / (1.0 + exponent)
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATOR_SCHEMA,
            "feature_names": list(self.feature_names),
            "standard_scaler": {
                "mean": self.mean.tolist(),
                "scale": self.scale.tolist(),
            },
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 1000,
                "coefficients": self.coefficients.tolist(),
                "intercept": self.intercept,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskaEdgeCalibrator:
        if payload.get("schema") != CALIBRATOR_SCHEMA:
            raise ValueError("unsupported TASKA edge calibrator schema")
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("serialized calibrator feature contract differs")
        scaler = payload.get("standard_scaler")
        logistic = payload.get("logistic_regression")
        if not isinstance(scaler, Mapping) or not isinstance(logistic, Mapping):
            raise ValueError("serialized calibrator is missing estimator parameters")
        if logistic.get("C") != 1.0 or logistic.get("max_iter") != 1000:
            raise ValueError("serialized logistic-regression contract differs")
        try:
            intercept = float(logistic.get("intercept"))
        except (TypeError, ValueError) as error:
            raise ValueError("serialized calibrator intercept is invalid") from error
        return cls(
            feature_names=FEATURE_NAMES,
            mean=np.asarray(scaler.get("mean"), dtype=np.float64),
            scale=np.asarray(scaler.get("scale"), dtype=np.float64),
            coefficients=np.asarray(logistic.get("coefficients"), dtype=np.float64),
            intercept=intercept,
        )

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> TaskaEdgeCalibrator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("serialized calibrator JSON must contain one mapping")
        return cls.from_dict(payload)

    def save_npz(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            schema=np.asarray(CALIBRATOR_SCHEMA),
            feature_names=np.asarray(FEATURE_NAMES),
            mean=self.mean,
            scale=self.scale,
            coefficients=self.coefficients,
            intercept=np.asarray(self.intercept, dtype=np.float64),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> TaskaEdgeCalibrator:
        with np.load(Path(path), allow_pickle=False) as archive:
            required = {
                "schema",
                "feature_names",
                "mean",
                "scale",
                "coefficients",
                "intercept",
            }
            if set(archive.files) != required:
                raise ValueError("serialized calibrator NPZ has an invalid key contract")
            if str(archive["schema"].item()) != CALIBRATOR_SCHEMA:
                raise ValueError("unsupported TASKA edge calibrator schema")
            names = tuple(str(value) for value in archive["feature_names"].tolist())
            if names != FEATURE_NAMES:
                raise ValueError("serialized calibrator feature contract differs")
            return cls(
                feature_names=FEATURE_NAMES,
                mean=archive["mean"],
                scale=archive["scale"],
                coefficients=archive["coefficients"],
                intercept=float(archive["intercept"].item()),
            )


def fit_taska_edge_calibrator(features: Any, labels: Any) -> TaskaEdgeCalibrator:
    """Fit the fixed deterministic StandardScaler + LogisticRegression model."""

    matrix = _feature_matrix(features)
    raw_labels = np.asarray(_as_numpy(labels))
    if raw_labels.shape != (len(matrix),):
        raise ValueError(f"labels must have shape {(len(matrix),)}, got {raw_labels.shape}")
    if not np.isin(raw_labels, (0, 1, False, True)).all():
        raise ValueError("labels must contain only binary values")
    binary_labels = raw_labels.astype(np.int8)
    if not np.array_equal(np.unique(binary_labels), np.asarray([0, 1], dtype=np.int8)):
        raise ValueError("labels must contain both binary classes")

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=0),
    )
    pipeline.fit(matrix, binary_labels)
    scaler = pipeline.named_steps["standardscaler"]
    logistic = pipeline.named_steps["logisticregression"]
    if logistic.coef_.shape != (1, FEATURE_COUNT) or logistic.intercept_.shape != (1,):
        raise RuntimeError("sklearn returned an unexpected binary estimator shape")
    return TaskaEdgeCalibrator(
        feature_names=FEATURE_NAMES,
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficients=np.asarray(logistic.coef_[0], dtype=np.float64),
        intercept=float(logistic.intercept_[0]),
    )


def predict_taska_edge_priorities(
    calibrator: TaskaEdgeCalibrator,
    cost_right: Any,
    cost_down: Any,
    right_log: Any,
    down_log: Any,
    candidate_edges: Sequence[RawTailEdge],
    edge_weights: Any,
    edge_vote_counts: Any,
    *,
    grid: int = 24,
) -> np.ndarray:
    """Extract the fixed feature rows and return aligned calibrated priorities."""

    if not isinstance(calibrator, TaskaEdgeCalibrator):
        raise TypeError("calibrator must be a TaskaEdgeCalibrator")
    batch = extract_taska_edge_features(
        cost_right,
        cost_down,
        right_log,
        down_log,
        candidate_edges,
        edge_weights,
        edge_vote_counts,
        grid=grid,
    )
    result = np.ascontiguousarray(calibrator.predict_priorities(batch.values))
    result.setflags(write=False)
    return result


def build_prioritized_raw_tail_components(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    edge_priorities: Any,
    *,
    grid: int = 24,
    component_cap: int = 0,
) -> tuple[
    tuple[dict[int, tuple[int, int]], ...],
    tuple[PrioritizedRawTailBuildDecision, ...],
]:
    """Build rigid components in stable descending external-priority order.

    The original cost matrices are retained in every decision and are not
    rewritten.  Exact priority ties preserve the caller's harvested-edge order.
    """

    count = _validate_grid(grid)
    right = _finite_matrix(cost_right, count=count, name="cost_right")
    down = _finite_matrix(cost_down, count=count, name="cost_down")
    edges = _validated_edges(candidate_edges, count=count)
    priorities = _finite_vector(
        edge_priorities,
        length=len(edges),
        name="edge_priorities",
    )
    if (
        isinstance(component_cap, bool)
        or not isinstance(component_cap, int)
        or component_cap < 0
        or component_cap == 1
        or component_cap > count
    ):
        raise ValueError("component_cap must be zero or an integer in [2, grid**2]")

    ranked = sorted(
        enumerate(edges),
        key=lambda item: (-float(priorities[item[0]]), item[0]),
    )
    builder = _TranslationBuilder(grid=grid, cap=component_cap)
    decisions: list[PrioritizedRawTailBuildDecision] = []
    for input_rank, edge in ranked:
        matrix = right if edge.axis == "right" else down
        decisions.append(
            PrioritizedRawTailBuildDecision(
                edge=edge,
                raw_priority=-float(matrix[edge.source, edge.target]),
                ranking_priority=float(priorities[input_rank]),
                input_rank=input_rank,
                status=builder.add(edge),
            )
        )
    return builder.components(), tuple(decisions)


def solve_prioritized_raw_tail_global(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    edge_priorities: Any,
    *,
    border_unary: Any | None = None,
    grid: int = 24,
    config: RawTailGlobalConfig | None = None,
) -> PrioritizedRawTailResult:
    """Run the frozen raw-tail placement/fill after a prioritized component build.

    External priorities affect only the order in which harvested relations are
    offered to the translation-consistent component builder.  Component
    placement, frame scoring, and Hungarian seam fill receive the untouched
    original ``cost_right`` and ``cost_down`` matrices.
    """

    if config is None:
        config = RawTailGlobalConfig()
    count = _validate_grid(grid)
    config.validate(grid=grid)
    right = _finite_matrix(cost_right, count=count, name="cost_right")
    down = _finite_matrix(cost_down, count=count, name="cost_down")
    edges = _validated_edges(candidate_edges, count=count)
    priorities = _finite_vector(
        edge_priorities,
        length=len(edges),
        name="edge_priorities",
    )
    unary: np.ndarray | None = None
    if border_unary is not None:
        unary = np.asarray(_as_numpy(border_unary), dtype=np.float64)
        if unary.shape != (count, grid, grid):
            raise ValueError(
                f"border_unary must have shape {(count, grid, grid)}, got {unary.shape}"
            )
        if not np.isfinite(unary).all():
            raise ValueError("border_unary must contain only finite values")
        unary = np.ascontiguousarray(unary)

    components, decisions = build_prioritized_raw_tail_components(
        right,
        down,
        edges,
        priorities,
        grid=grid,
        component_cap=config.component_cap,
    )
    board, placed_count, placed_tiles, baseline = _place_components(
        components,
        right,
        down,
        grid=grid,
        baseline_quantile=config.baseline_quantile,
        rounds=config.search_rounds,
        seed=config.random_seed,
        border_unary=unary,
        border_weight=config.border_weight,
    )
    layout = _fill_seams(
        board,
        right,
        down,
        grid=grid,
        seed=config.random_seed,
        rounds=config.fill_rounds,
    )
    strict = np.array_equal(np.sort(layout), np.arange(count))
    if not strict:
        raise RuntimeError("prioritized raw-tail solver did not return a strict permutation")
    frozen_layout = np.asarray(layout, dtype=np.int32)
    frozen_layout.setflags(write=False)

    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.status] = counts.get(decision.status, 0) + 1
    accepted = sum(value for key, value in counts.items() if key.startswith("accepted_"))
    diagnostics = RawTailGlobalDiagnostics(
        grid_size=grid,
        tile_count=count,
        candidate_edges=len(edges),
        accepted_edges=accepted,
        rejected_edges=len(edges) - accepted,
        component_count=len(components),
        component_sizes=tuple(sorted((len(component) for component in components), reverse=True)),
        placed_component_count=placed_count,
        placed_component_tiles=placed_tiles,
        baseline_cost=baseline,
        strict_permutation=strict,
        status_counts=tuple(sorted(counts.items())),
    )
    return PrioritizedRawTailResult(
        layout=frozen_layout,
        components=components,
        decisions=decisions,
        diagnostics=diagnostics,
    )


__all__ = [
    "CALIBRATOR_SCHEMA",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "PrioritizedRawTailBuildDecision",
    "PrioritizedRawTailResult",
    "TaskaEdgeCalibrator",
    "TaskaEdgeFeatureBatch",
    "build_prioritized_raw_tail_components",
    "extract_taska_edge_features",
    "fit_taska_edge_calibrator",
    "predict_taska_edge_priorities",
    "solve_prioritized_raw_tail_global",
]
