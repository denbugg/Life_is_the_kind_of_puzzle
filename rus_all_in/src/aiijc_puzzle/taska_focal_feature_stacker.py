"""Portable fixed fusion of TASKA matcher evidence and recovered focal scores.

The stacker changes only the order in which already-harvested edges are
offered to the frozen translation-consistent component builder.  Its 22 input
values are all target-free at inference: the fixed 15 TASKA edge features, one
recovered focal-verifier logit, and the verifier's six handcrafted top-5
features.  Training labels are used only while fitting this offline artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.taska_edge_calibrator import FEATURE_NAMES as TASKA_EDGE_FEATURE_NAMES

FOCAL_STACKER_FEATURE_NAMES = (
    *TASKA_EDGE_FEATURE_NAMES,
    "recovered_focal_logit",
    "focal_selected_compatibility_div10",
    "focal_compatibility_minus_row_best",
    "focal_better_candidate_count",
    "focal_is_row_best",
    "focal_top5_mean_div10",
    "focal_top5_spread",
)
FOCAL_STACKER_FEATURE_COUNT = len(FOCAL_STACKER_FEATURE_NAMES)
FOCAL_STACKER_SCHEMA = "aiijc-taska-focal-feature-stacker-v1"
FOCAL_STACKER_PARAMETERS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 0,
    "class_weight": None,
}


def _as_finite_matrix(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != FOCAL_STACKER_FEATURE_COUNT:
        raise ValueError(
            f"{name} must have shape rows x {FOCAL_STACKER_FEATURE_COUNT}, "
            f"got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _as_finite_block(value: Any, *, rows: int, columns: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (rows, columns):
        raise ValueError(f"{name} must have shape {(rows, columns)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def stack_taska_focal_features(
    edge_features: Any,
    focal_logits: Any,
    focal_features: Any,
) -> np.ndarray:
    """Return the exact 15 + 1 + 6 inference feature contract."""

    edge = np.asarray(edge_features, dtype=np.float64)
    if edge.ndim != 2 or edge.shape[1] != len(TASKA_EDGE_FEATURE_NAMES):
        raise ValueError(
            f"edge_features must have shape rows x {len(TASKA_EDGE_FEATURE_NAMES)}"
        )
    if not np.isfinite(edge).all():
        raise ValueError("edge_features must contain only finite values")
    rows = len(edge)
    logits = np.asarray(focal_logits, dtype=np.float64)
    if logits.shape != (rows,) or not np.isfinite(logits).all():
        raise ValueError("focal_logits must be one finite vector aligned to edge_features")
    focal = _as_finite_block(
        focal_features,
        rows=rows,
        columns=6,
        name="focal_features",
    )
    result = np.column_stack((edge, logits, focal))
    if result.shape != (rows, FOCAL_STACKER_FEATURE_COUNT):
        raise RuntimeError("stacked focal feature contract changed")
    result = np.ascontiguousarray(result, dtype=np.float64)
    result.setflags(write=False)
    return result


def _portable_vector(value: Any, *, name: str, positive: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (FOCAL_STACKER_FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ValueError(
            f"{name} must be one finite length-{FOCAL_STACKER_FEATURE_COUNT} vector"
        )
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    result = np.ascontiguousarray(result.copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TaskaFocalFeatureStacker:
    """Portable StandardScaler plus fixed binary logistic regression."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != FOCAL_STACKER_FEATURE_NAMES:
            raise ValueError("stacker feature-name contract differs")
        mean = _portable_vector(self.mean, name="mean")
        scale = _portable_vector(self.scale, name="scale", positive=True)
        coefficients = _portable_vector(self.coefficients, name="coefficients")
        intercept = float(self.intercept)
        if not math.isfinite(intercept):
            raise ValueError("intercept must be finite")
        object.__setattr__(self, "feature_names", FOCAL_STACKER_FEATURE_NAMES)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", intercept)

    def predict_logits(self, features: Any) -> np.ndarray:
        matrix = _as_finite_matrix(features, name="features")
        result = (matrix - self.mean) / self.scale @ self.coefficients + self.intercept
        result = np.ascontiguousarray(result, dtype=np.float64)
        result.setflags(write=False)
        return result

    def predict_priorities(self, features: Any) -> np.ndarray:
        logits = self.predict_logits(features)
        result = np.empty_like(logits)
        positive = logits >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        result[~positive] = exponent / (1.0 + exponent)
        result.setflags(write=False)
        return result

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            schema=np.asarray(FOCAL_STACKER_SCHEMA),
            feature_names=np.asarray(FOCAL_STACKER_FEATURE_NAMES),
            mean=self.mean,
            scale=self.scale,
            coefficients=self.coefficients,
            intercept=np.asarray(self.intercept, dtype=np.float64),
            C=np.asarray(FOCAL_STACKER_PARAMETERS["C"], dtype=np.float64),
            max_iter=np.asarray(FOCAL_STACKER_PARAMETERS["max_iter"], dtype=np.int32),
            random_state=np.asarray(
                FOCAL_STACKER_PARAMETERS["random_state"], dtype=np.int32
            ),
            class_weight_is_none=np.asarray(True),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> TaskaFocalFeatureStacker:
        with np.load(Path(path), allow_pickle=False) as archive:
            required = {
                "schema",
                "feature_names",
                "mean",
                "scale",
                "coefficients",
                "intercept",
                "C",
                "max_iter",
                "random_state",
                "class_weight_is_none",
            }
            if set(archive.files) != required:
                raise ValueError("serialized stacker NPZ key contract differs")
            if str(archive["schema"].item()) != FOCAL_STACKER_SCHEMA:
                raise ValueError("unsupported focal feature stacker schema")
            names = tuple(str(value) for value in archive["feature_names"].tolist())
            if names != FOCAL_STACKER_FEATURE_NAMES:
                raise ValueError("serialized stacker feature contract differs")
            if (
                float(archive["C"].item()) != FOCAL_STACKER_PARAMETERS["C"]
                or int(archive["max_iter"].item())
                != FOCAL_STACKER_PARAMETERS["max_iter"]
                or int(archive["random_state"].item())
                != FOCAL_STACKER_PARAMETERS["random_state"]
                or bool(archive["class_weight_is_none"].item()) is not True
            ):
                raise ValueError("serialized stacker estimator contract differs")
            return cls(
                feature_names=names,
                mean=archive["mean"],
                scale=archive["scale"],
                coefficients=archive["coefficients"],
                intercept=float(archive["intercept"].item()),
            )


def fit_taska_focal_feature_stacker(
    features: Any,
    labels: Any,
) -> TaskaFocalFeatureStacker:
    """Fit the one fixed unweighted StandardScaler + LogisticRegression arm."""

    matrix = _as_finite_matrix(features, name="features")
    raw_labels = np.asarray(labels)
    if raw_labels.shape != (len(matrix),) or not np.isin(raw_labels, (0, 1)).all():
        raise ValueError("labels must be one aligned binary vector")
    binary = raw_labels.astype(np.int8)
    if not np.array_equal(np.unique(binary), np.asarray([0, 1], dtype=np.int8)):
        raise ValueError("labels must contain both classes")
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(**FOCAL_STACKER_PARAMETERS),
    )
    pipeline.fit(matrix, binary)
    scaler = pipeline.named_steps["standardscaler"]
    logistic = pipeline.named_steps["logisticregression"]
    if logistic.coef_.shape != (1, FOCAL_STACKER_FEATURE_COUNT):
        raise RuntimeError("sklearn returned an unexpected stacker coefficient shape")
    portable = TaskaFocalFeatureStacker(
        feature_names=FOCAL_STACKER_FEATURE_NAMES,
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficients=np.asarray(logistic.coef_[0], dtype=np.float64),
        intercept=float(logistic.intercept_[0]),
    )
    reference = logistic.predict_proba(scaler.transform(matrix[:1024]))[:, 1]
    actual = portable.predict_priorities(matrix[:1024])
    if not np.allclose(actual, reference, atol=1e-12, rtol=1e-12):
        raise RuntimeError("portable stacker differs from fitted sklearn pipeline")
    return portable


__all__ = [
    "FOCAL_STACKER_FEATURE_COUNT",
    "FOCAL_STACKER_FEATURE_NAMES",
    "FOCAL_STACKER_PARAMETERS",
    "FOCAL_STACKER_SCHEMA",
    "TaskaFocalFeatureStacker",
    "fit_taska_focal_feature_stacker",
    "stack_taska_focal_features",
]
