"""Portable board/axis pairwise ranker for harvested TASKA edges.

The ranker consumes the same 22 target-free values as the fixed focal feature
stacker.  Offline fitting differs deliberately: a scaler is fitted once on
the original edge rows, then each true edge is contrasted with up to four of
the highest-scoring false focal edges from the same board and axis.  Exact
sign-reversed pairs make the training set symmetric, so inference needs only
one standardized linear score and no intercept.

Labels and board boundaries are fitting inputs only.  The portable runtime
artifact contains a scaler and one coefficient vector; it cannot access a
target, filename, board coordinate, or replacement pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.taska_focal_feature_stacker import (
    FOCAL_STACKER_FEATURE_COUNT,
    FOCAL_STACKER_FEATURE_NAMES,
)

PAIRWISE_RANKER_SCHEMA = "aiijc-taska-focal-pairwise-ranker-v1"
PAIRWISE_RANKER_PARAMETERS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 0,
    "fit_intercept": False,
    "hard_negatives_per_positive": 4,
    "hard_negative_score_feature": "recovered_focal_logit",
    "grouping": "board_then_axis",
}
AXIS_FEATURE_INDEX = FOCAL_STACKER_FEATURE_NAMES.index("axis_is_down")
FOCAL_LOGIT_FEATURE_INDEX = FOCAL_STACKER_FEATURE_NAMES.index(
    "recovered_focal_logit"
)


def _feature_matrix(value: Any, *, name: str = "features") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    expected = FOCAL_STACKER_FEATURE_COUNT
    if result.ndim != 2 or result.shape[1] != expected:
        raise ValueError(f"{name} must have shape rows x {expected}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _vector(value: Any, *, name: str, positive: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    expected = (FOCAL_STACKER_FEATURE_COUNT,)
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite length-{expected[0]} vector")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    result = np.ascontiguousarray(result.copy())
    result.setflags(write=False)
    return result


def _binary_labels(value: Any, *, rows: int) -> np.ndarray:
    labels = np.asarray(value)
    if labels.shape != (rows,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be one aligned binary vector")
    return np.ascontiguousarray(labels, dtype=np.uint8)


def _board_offsets(value: Any, *, rows: int) -> np.ndarray:
    offsets = np.asarray(value)
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("offsets must contain at least one board interval")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("offsets must be integers")
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    if offsets[0] != 0 or offsets[-1] != rows or np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets must strictly partition all feature rows")
    return offsets


def build_symmetric_pairwise_differences(
    standardized_features: Any,
    labels: Any,
    offsets: Any,
    original_features: Any,
    *,
    hard_negatives_per_positive: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build the fixed same-board/same-axis hard-negative training rows.

    Negative hardness is determined only by recovered focal logit.  Stable
    mergesort preserves original harvested-row order on exact score ties.
    Every positive-minus-negative row is followed by its exact sign reversal.
    """

    standardized = _feature_matrix(standardized_features, name="standardized_features")
    original = _feature_matrix(original_features, name="original_features")
    if standardized.shape != original.shape:
        raise ValueError("standardized and original feature rows must align")
    binary = _binary_labels(labels, rows=len(original))
    board_offsets = _board_offsets(offsets, rows=len(original))
    if (
        isinstance(hard_negatives_per_positive, bool)
        or not isinstance(hard_negatives_per_positive, int)
        or hard_negatives_per_positive < 1
    ):
        raise ValueError("hard_negatives_per_positive must be a positive integer")
    axes = original[:, AXIS_FEATURE_INDEX]
    if not np.isin(axes, (0.0, 1.0)).all():
        raise ValueError("axis_is_down must be exactly binary")

    positive_rows: list[np.ndarray] = []
    board_axis_groups = 0
    selected_positive_count = 0
    selected_negative_pair_count = 0
    for board_start, board_stop in zip(
        board_offsets[:-1], board_offsets[1:], strict=True
    ):
        board_indices = np.arange(board_start, board_stop, dtype=np.int64)
        for axis in (0.0, 1.0):
            group = board_indices[axes[board_start:board_stop] == axis]
            positives = group[binary[group] == 1]
            negatives = group[binary[group] == 0]
            if len(positives) == 0 or len(negatives) == 0:
                continue
            board_axis_groups += 1
            selected_positive_count += len(positives)
            order = np.argsort(
                -original[negatives, FOCAL_LOGIT_FEATURE_INDEX],
                kind="stable",
            )
            hardest = negatives[order[:hard_negatives_per_positive]]
            for positive in positives:
                differences = standardized[positive] - standardized[hardest]
                positive_rows.extend(differences)
                selected_negative_pair_count += len(hardest)

    if not positive_rows:
        raise ValueError("training rows contain no valid positive/negative board-axis pairs")
    positive_matrix = np.ascontiguousarray(np.vstack(positive_rows), dtype=np.float64)
    pair_features = np.ascontiguousarray(
        np.concatenate((positive_matrix, -positive_matrix), axis=0),
        dtype=np.float64,
    )
    pair_labels = np.concatenate(
        (
            np.ones(len(positive_matrix), dtype=np.uint8),
            np.zeros(len(positive_matrix), dtype=np.uint8),
        )
    )
    pair_features.setflags(write=False)
    pair_labels.setflags(write=False)
    diagnostics = {
        "board_count": len(board_offsets) - 1,
        "board_axis_group_count": board_axis_groups,
        "selected_positive_count": selected_positive_count,
        "positive_negative_pair_count": selected_negative_pair_count,
        "symmetric_training_row_count": len(pair_features),
        "hard_negatives_per_positive": hard_negatives_per_positive,
    }
    return pair_features, pair_labels, diagnostics


@dataclass(frozen=True)
class TaskaFocalPairwiseRanker:
    """Portable original-row scaler plus intercept-free pairwise linear head."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != FOCAL_STACKER_FEATURE_NAMES:
            raise ValueError("ranker feature-name contract differs")
        object.__setattr__(self, "feature_names", FOCAL_STACKER_FEATURE_NAMES)
        object.__setattr__(self, "mean", _vector(self.mean, name="mean"))
        object.__setattr__(self, "scale", _vector(self.scale, name="scale", positive=True))
        object.__setattr__(
            self,
            "coefficients",
            _vector(self.coefficients, name="coefficients"),
        )

    def predict_scores(self, features: Any) -> np.ndarray:
        matrix = _feature_matrix(features)
        result = ((matrix - self.mean) / self.scale) @ self.coefficients
        result = np.ascontiguousarray(result, dtype=np.float64)
        result.setflags(write=False)
        return result

    def predict_priorities(self, features: Any) -> np.ndarray:
        """Return ranking-equivalent raw linear scores for the component solver."""

        return self.predict_scores(features)

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            schema=np.asarray(PAIRWISE_RANKER_SCHEMA),
            feature_names=np.asarray(FOCAL_STACKER_FEATURE_NAMES),
            mean=self.mean,
            scale=self.scale,
            coefficients=self.coefficients,
            C=np.asarray(PAIRWISE_RANKER_PARAMETERS["C"], dtype=np.float64),
            max_iter=np.asarray(PAIRWISE_RANKER_PARAMETERS["max_iter"], dtype=np.int32),
            random_state=np.asarray(
                PAIRWISE_RANKER_PARAMETERS["random_state"], dtype=np.int32
            ),
            fit_intercept=np.asarray(False),
            hard_negatives_per_positive=np.asarray(
                PAIRWISE_RANKER_PARAMETERS["hard_negatives_per_positive"],
                dtype=np.int32,
            ),
            axis_feature_index=np.asarray(AXIS_FEATURE_INDEX, dtype=np.int32),
            focal_logit_feature_index=np.asarray(
                FOCAL_LOGIT_FEATURE_INDEX, dtype=np.int32
            ),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> TaskaFocalPairwiseRanker:
        with np.load(Path(path), allow_pickle=False) as archive:
            required = {
                "schema",
                "feature_names",
                "mean",
                "scale",
                "coefficients",
                "C",
                "max_iter",
                "random_state",
                "fit_intercept",
                "hard_negatives_per_positive",
                "axis_feature_index",
                "focal_logit_feature_index",
            }
            if set(archive.files) != required:
                raise ValueError("serialized pairwise ranker NPZ key contract differs")
            if str(archive["schema"].item()) != PAIRWISE_RANKER_SCHEMA:
                raise ValueError("unsupported focal pairwise ranker schema")
            names = tuple(str(value) for value in archive["feature_names"].tolist())
            contract = (
                float(archive["C"].item()) == PAIRWISE_RANKER_PARAMETERS["C"]
                and int(archive["max_iter"].item())
                == PAIRWISE_RANKER_PARAMETERS["max_iter"]
                and int(archive["random_state"].item())
                == PAIRWISE_RANKER_PARAMETERS["random_state"]
                and bool(archive["fit_intercept"].item()) is False
                and int(archive["hard_negatives_per_positive"].item())
                == PAIRWISE_RANKER_PARAMETERS["hard_negatives_per_positive"]
                and int(archive["axis_feature_index"].item()) == AXIS_FEATURE_INDEX
                and int(archive["focal_logit_feature_index"].item())
                == FOCAL_LOGIT_FEATURE_INDEX
            )
            if names != FOCAL_STACKER_FEATURE_NAMES or not contract:
                raise ValueError("serialized pairwise ranker contract differs")
            return cls(
                feature_names=names,
                mean=archive["mean"],
                scale=archive["scale"],
                coefficients=archive["coefficients"],
            )


def fit_taska_focal_pairwise_ranker(
    features: Any,
    labels: Any,
    offsets: Any,
) -> tuple[TaskaFocalPairwiseRanker, dict[str, int]]:
    """Fit the one fixed StandardScaler + symmetric pairwise logistic head."""

    matrix = _feature_matrix(features)
    binary = _binary_labels(labels, rows=len(matrix))
    board_offsets = _board_offsets(offsets, rows=len(matrix))
    scaler = StandardScaler().fit(matrix)
    standardized = np.ascontiguousarray(scaler.transform(matrix), dtype=np.float64)
    differences, pair_labels, diagnostics = build_symmetric_pairwise_differences(
        standardized,
        binary,
        board_offsets,
        matrix,
        hard_negatives_per_positive=PAIRWISE_RANKER_PARAMETERS[
            "hard_negatives_per_positive"
        ],
    )
    logistic = LogisticRegression(
        C=PAIRWISE_RANKER_PARAMETERS["C"],
        max_iter=PAIRWISE_RANKER_PARAMETERS["max_iter"],
        random_state=PAIRWISE_RANKER_PARAMETERS["random_state"],
        fit_intercept=PAIRWISE_RANKER_PARAMETERS["fit_intercept"],
    ).fit(differences, pair_labels)
    if logistic.coef_.shape != (1, FOCAL_STACKER_FEATURE_COUNT):
        raise RuntimeError("sklearn returned an unexpected pairwise coefficient shape")
    if logistic.intercept_.shape != (1,) or float(logistic.intercept_[0]) != 0.0:
        raise RuntimeError("intercept-free pairwise fit unexpectedly produced an intercept")
    ranker = TaskaFocalPairwiseRanker(
        feature_names=FOCAL_STACKER_FEATURE_NAMES,
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficients=np.asarray(logistic.coef_[0], dtype=np.float64),
    )
    reference = logistic.decision_function(differences[:1024])
    portable = differences[:1024] @ ranker.coefficients
    if not np.allclose(portable, reference, atol=1e-12, rtol=1e-12):
        raise RuntimeError("portable pairwise head differs from fitted sklearn head")
    return ranker, diagnostics


__all__ = [
    "AXIS_FEATURE_INDEX",
    "FOCAL_LOGIT_FEATURE_INDEX",
    "PAIRWISE_RANKER_PARAMETERS",
    "PAIRWISE_RANKER_SCHEMA",
    "TaskaFocalPairwiseRanker",
    "build_symmetric_pairwise_differences",
    "fit_taska_focal_pairwise_ranker",
]
