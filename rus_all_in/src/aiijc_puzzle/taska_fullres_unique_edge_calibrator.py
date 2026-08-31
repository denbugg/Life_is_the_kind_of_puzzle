"""One frozen linear filter for accepted unique-fullres TASKA edges.

The filter is trained only on the already-open local32 organizer-train panel.
At inference it consumes dirty-visible frozen evidence and never changes pixels.
Its output is only a boolean mask over the existing unique-fullres edge suffix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge

GRID = 24
COUNT = GRID * GRID
FEATURE_NAMES = (
    "recovered_focal_logit",
    "restored_support_count",
    "raw_directional_seam_cost",
    "raw_outgoing_rank_fraction",
    "raw_incoming_rank_fraction",
    "raw_outgoing_margin_from_best",
    "raw_incoming_margin_from_best",
    "axis_is_down",
)
DECISION_THRESHOLD = 0.5


def _cost_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (COUNT, COUNT) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be one finite 576x576 matrix")
    return matrix


def unique_fullres_edge_features(
    *,
    edges: Sequence[RawTailEdge],
    focal_logits: Any,
    restored_support: Any,
    cost_right: Any,
    cost_down: Any,
) -> np.ndarray:
    """Build the preregistered eight dirty-visible features in fixed order."""

    edge_values = tuple(edges)
    logits = np.asarray(focal_logits, dtype=np.float64)
    support = np.asarray(restored_support, dtype=np.float64)
    if logits.shape != (len(edge_values),) or not np.isfinite(logits).all():
        raise ValueError("focal_logits must contain one finite value per edge")
    if support.shape != (len(edge_values),) or not np.isin(support, (3.0, 4.0)).all():
        raise ValueError("restored_support must be 3 or 4 for every accepted edge")
    matrices = {
        "right": _cost_matrix(cost_right, name="cost_right"),
        "down": _cost_matrix(cost_down, name="cost_down"),
    }
    result = np.empty((len(edge_values), len(FEATURE_NAMES)), dtype=np.float64)
    denominator = float(COUNT - 1)
    for index, edge in enumerate(edge_values):
        if edge.axis not in matrices or edge.source == edge.target:
            raise ValueError("edge must be one non-self right/down RawTailEdge")
        matrix = matrices[edge.axis]
        cost = float(matrix[edge.source, edge.target])
        outgoing = np.concatenate(
            (matrix[edge.source, : edge.source], matrix[edge.source, edge.source + 1 :])
        )
        incoming = np.concatenate(
            (matrix[: edge.target, edge.target], matrix[edge.target + 1 :, edge.target])
        )
        result[index] = (
            logits[index],
            support[index],
            cost,
            np.count_nonzero(outgoing < cost) / denominator,
            np.count_nonzero(incoming < cost) / denominator,
            cost - float(np.min(outgoing)),
            cost - float(np.min(incoming)),
            float(edge.axis == "down"),
        )
    if not np.isfinite(result).all():
        raise RuntimeError("calibrator features contain non-finite values")
    return np.ascontiguousarray(result, dtype=np.float64)


@dataclass(frozen=True)
class UniqueFullresEdgeCalibrator:
    """Portable StandardScaler + one unweighted C=1 logistic regression."""

    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    decision_threshold: float = DECISION_THRESHOLD

    def __post_init__(self) -> None:
        for name in ("scaler_mean", "scaler_scale", "coefficient"):
            value = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain one finite value per feature")
            object.__setattr__(self, name, value)
        if np.any(self.scaler_scale <= 0):
            raise ValueError("scaler_scale must be strictly positive")
        if not np.isfinite(self.intercept) or self.decision_threshold != DECISION_THRESHOLD:
            raise ValueError("calibrator intercept or fixed threshold is invalid")

    def predict_probability(self, features: Any) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("features have the wrong shape")
        logits = ((values - self.scaler_mean) / self.scaler_scale) @ self.coefficient
        logits = logits + self.intercept
        probabilities = np.empty_like(logits)
        positive = logits >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponential = np.exp(logits[~positive])
        probabilities[~positive] = exponential / (1.0 + exponential)
        return probabilities

    def keep_mask(self, features: Any) -> np.ndarray:
        return self.predict_probability(features) >= self.decision_threshold

    def diagnostics(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "feature_names": list(FEATURE_NAMES),
                "scaler_mean": self.scaler_mean.tolist(),
                "scaler_scale": self.scaler_scale.tolist(),
                "coefficient": self.coefficient.tolist(),
            }
        )
        return payload


def fit_unique_fullres_edge_calibrator(
    features: Any,
    labels: Any,
) -> UniqueFullresEdgeCalibrator:
    """Fit exactly the preregistered unweighted scaler-logistic pipeline."""

    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError("features have the wrong shape")
    if target.shape != (len(values),) or not np.isin(target, (0, 1)).all():
        raise ValueError("labels must be one binary value per edge")
    if len(np.unique(target)) != 2:
        raise ValueError("fit labels must contain both classes")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(values)
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=1000,
        random_state=0,
        solver="lbfgs",
    )
    model.fit(standardized, target)
    if model.classes_.tolist() != [0, 1]:
        raise RuntimeError("unexpected logistic class order")
    return UniqueFullresEdgeCalibrator(
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        coefficient=model.coef_[0],
        intercept=float(model.intercept_[0]),
    )


__all__ = [
    "DECISION_THRESHOLD",
    "FEATURE_NAMES",
    "UniqueFullresEdgeCalibrator",
    "fit_unique_fullres_edge_calibrator",
    "unique_fullres_edge_features",
]
