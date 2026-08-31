"""Portable nonlinear calibrator for the fixed TASKA edge feature contract.

Training uses one deterministic scikit-learn histogram-gradient-boosting
classifier.  Inference does not deserialize pickle: the learned numerical tree
nodes are exported to a bounded NPZ schema and evaluated by the small traversal
implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from aiijc_puzzle.taska_edge_calibrator import FEATURE_COUNT, FEATURE_NAMES

NONLINEAR_CALIBRATOR_SCHEMA = "aiijc-taska-nonlinear-edge-calibrator-v1"
NONLINEAR_CALIBRATOR_PARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "random_state": 0,
}


def _features(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"features must have shape rows x {FEATURE_COUNT}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("features must contain only finite values")
    return np.ascontiguousarray(result)


def _readonly_vector(
    value: Any,
    *,
    dtype: np.dtype[Any],
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result = np.ascontiguousarray(result.copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TaskaNonlinearCalibrator:
    """Portable binary gradient-boosted tree ensemble."""

    feature_names: tuple[str, ...]
    baseline: float
    tree_offsets: np.ndarray
    values: np.ndarray
    feature_indices: np.ndarray
    thresholds: np.ndarray
    missing_go_to_left: np.ndarray
    left_children: np.ndarray
    right_children: np.ndarray
    is_leaf: np.ndarray

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("nonlinear calibrator feature contract differs")
        baseline = float(self.baseline)
        if not np.isfinite(baseline):
            raise ValueError("baseline must be finite")
        offsets = _readonly_vector(
            self.tree_offsets,
            dtype=np.dtype(np.int32),
            name="tree_offsets",
        )
        if len(offsets) != NONLINEAR_CALIBRATOR_PARAMETERS["max_iter"] + 1:
            raise ValueError("tree_offsets must describe exactly 100 trees")
        if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError("tree_offsets must be strictly increasing from zero")
        values = _readonly_vector(self.values, dtype=np.dtype(np.float64), name="values")
        features = _readonly_vector(
            self.feature_indices,
            dtype=np.dtype(np.int16),
            name="feature_indices",
        )
        thresholds = _readonly_vector(
            self.thresholds,
            dtype=np.dtype(np.float64),
            name="thresholds",
        )
        missing_left = _readonly_vector(
            self.missing_go_to_left,
            dtype=np.dtype(np.bool_),
            name="missing_go_to_left",
        )
        left = _readonly_vector(
            self.left_children,
            dtype=np.dtype(np.int32),
            name="left_children",
        )
        right = _readonly_vector(
            self.right_children,
            dtype=np.dtype(np.int32),
            name="right_children",
        )
        leaves = _readonly_vector(
            self.is_leaf,
            dtype=np.dtype(np.bool_),
            name="is_leaf",
        )
        node_count = int(offsets[-1])
        arrays = (values, features, thresholds, missing_left, left, right, leaves)
        if any(len(array) != node_count for array in arrays):
            raise ValueError("all node arrays must match the final tree offset")
        if not np.isfinite(values).all() or not np.isfinite(thresholds).all():
            raise ValueError("tree values and thresholds must be finite")
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            size = int(stop - start)
            nodes = slice(int(start), int(stop))
            branch = ~leaves[nodes]
            if np.any((features[nodes][branch] < 0) | (features[nodes][branch] >= FEATURE_COUNT)):
                raise ValueError("branch feature index is outside the feature contract")
            if np.any((left[nodes][branch] < 0) | (left[nodes][branch] >= size)):
                raise ValueError("left child is outside its tree")
            if np.any((right[nodes][branch] < 0) | (right[nodes][branch] >= size)):
                raise ValueError("right child is outside its tree")
        object.__setattr__(self, "feature_names", FEATURE_NAMES)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "tree_offsets", offsets)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_indices", features)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "missing_go_to_left", missing_left)
        object.__setattr__(self, "left_children", left)
        object.__setattr__(self, "right_children", right)
        object.__setattr__(self, "is_leaf", leaves)

    def predict_logits(self, features: Any) -> np.ndarray:
        matrix = _features(features)
        result = np.full(len(matrix), self.baseline, dtype=np.float64)
        samples = np.arange(len(matrix), dtype=np.int64)
        for raw_start, raw_stop in zip(
            self.tree_offsets[:-1],
            self.tree_offsets[1:],
            strict=True,
        ):
            start, stop = int(raw_start), int(raw_stop)
            nodes = np.zeros(len(matrix), dtype=np.int32)
            while True:
                absolute = start + nodes
                active = ~self.is_leaf[absolute]
                if not active.any():
                    break
                active_samples = samples[active]
                active_nodes = absolute[active]
                feature = self.feature_indices[active_nodes]
                values = matrix[active_samples, feature]
                go_left = np.where(
                    np.isnan(values),
                    self.missing_go_to_left[active_nodes],
                    values <= self.thresholds[active_nodes],
                )
                nodes[active] = np.where(
                    go_left,
                    self.left_children[active_nodes],
                    self.right_children[active_nodes],
                )
                if np.any(nodes < 0) or np.any(nodes >= stop - start):
                    raise RuntimeError("tree traversal left its node range")
            result += self.values[start + nodes]
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
            schema=np.asarray(NONLINEAR_CALIBRATOR_SCHEMA),
            feature_names=np.asarray(FEATURE_NAMES),
            baseline=np.asarray(self.baseline, dtype=np.float64),
            tree_offsets=self.tree_offsets,
            values=self.values,
            feature_indices=self.feature_indices,
            thresholds=self.thresholds,
            missing_go_to_left=self.missing_go_to_left,
            left_children=self.left_children,
            right_children=self.right_children,
            is_leaf=self.is_leaf,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> TaskaNonlinearCalibrator:
        with np.load(Path(path), allow_pickle=False) as archive:
            required = {
                "schema",
                "feature_names",
                "baseline",
                "tree_offsets",
                "values",
                "feature_indices",
                "thresholds",
                "missing_go_to_left",
                "left_children",
                "right_children",
                "is_leaf",
            }
            if set(archive.files) != required:
                raise ValueError("nonlinear calibrator NPZ key contract differs")
            if str(archive["schema"].item()) != NONLINEAR_CALIBRATOR_SCHEMA:
                raise ValueError("unsupported nonlinear calibrator schema")
            names = tuple(str(value) for value in archive["feature_names"].tolist())
            return cls(
                feature_names=names,
                baseline=float(archive["baseline"].item()),
                tree_offsets=archive["tree_offsets"],
                values=archive["values"],
                feature_indices=archive["feature_indices"],
                thresholds=archive["thresholds"],
                missing_go_to_left=archive["missing_go_to_left"],
                left_children=archive["left_children"],
                right_children=archive["right_children"],
                is_leaf=archive["is_leaf"],
            )


def _from_sklearn(model: HistGradientBoostingClassifier) -> TaskaNonlinearCalibrator:
    if not isinstance(model, HistGradientBoostingClassifier):
        raise TypeError("model must be a HistGradientBoostingClassifier")
    parameters = model.get_params()
    if any(parameters[name] != value for name, value in NONLINEAR_CALIBRATOR_PARAMETERS.items()):
        raise ValueError("histogram-gradient-boosting parameter contract differs")
    predictors = getattr(model, "_predictors", None)
    if not isinstance(predictors, list) or len(predictors) != 100:
        raise ValueError("fitted model must contain exactly 100 boosting trees")
    offsets = [0]
    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "value",
            "feature_idx",
            "num_threshold",
            "missing_go_to_left",
            "left",
            "right",
            "is_leaf",
        )
    }
    for iteration in predictors:
        if len(iteration) != 1:
            raise ValueError("portable calibrator supports one binary tree per iteration")
        nodes = iteration[0].nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("portable calibrator forbids categorical tree splits")
        for name in fields:
            fields[name].append(np.asarray(nodes[name]))
        offsets.append(offsets[-1] + len(nodes))
    baseline = np.asarray(getattr(model, "_baseline_prediction", None))
    if baseline.shape != (1, 1):
        raise ValueError("binary baseline prediction contract differs")
    return TaskaNonlinearCalibrator(
        feature_names=FEATURE_NAMES,
        baseline=float(baseline.item()),
        tree_offsets=np.asarray(offsets, dtype=np.int32),
        values=np.concatenate(fields["value"]),
        feature_indices=np.concatenate(fields["feature_idx"]),
        thresholds=np.concatenate(fields["num_threshold"]),
        missing_go_to_left=np.concatenate(fields["missing_go_to_left"]),
        left_children=np.concatenate(fields["left"]),
        right_children=np.concatenate(fields["right"]),
        is_leaf=np.concatenate(fields["is_leaf"]),
    )


def fit_taska_nonlinear_calibrator(
    features: Any,
    labels: Any,
) -> TaskaNonlinearCalibrator:
    """Fit and export the one fixed nonlinear edge-calibration arm."""

    matrix = _features(features)
    raw_labels = np.asarray(labels)
    if raw_labels.shape != (len(matrix),) or not np.isin(raw_labels, (0, 1)).all():
        raise ValueError("labels must be one aligned binary vector")
    if len(np.unique(raw_labels)) != 2:
        raise ValueError("labels must contain both classes")
    estimator = HistGradientBoostingClassifier(**NONLINEAR_CALIBRATOR_PARAMETERS)
    estimator.fit(matrix, raw_labels.astype(np.int8))
    portable = _from_sklearn(estimator)
    reference = estimator.predict_proba(matrix[: min(1024, len(matrix))])[:, 1]
    actual = portable.predict_priorities(matrix[: min(1024, len(matrix))])
    if not np.allclose(actual, reference, atol=1e-12, rtol=1e-12):
        raise RuntimeError("portable tree traversal differs from sklearn")
    return portable


__all__ = [
    "NONLINEAR_CALIBRATOR_PARAMETERS",
    "NONLINEAR_CALIBRATOR_SCHEMA",
    "TaskaNonlinearCalibrator",
    "fit_taska_nonlinear_calibrator",
]
