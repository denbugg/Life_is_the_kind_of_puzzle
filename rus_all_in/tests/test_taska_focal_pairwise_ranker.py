from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.taska_focal_feature_stacker import FOCAL_STACKER_FEATURE_COUNT
from aiijc_puzzle.taska_focal_pairwise_ranker import (
    AXIS_FEATURE_INDEX,
    FOCAL_LOGIT_FEATURE_INDEX,
    PAIRWISE_RANKER_PARAMETERS,
    TaskaFocalPairwiseRanker,
    build_symmetric_pairwise_differences,
    fit_taska_focal_pairwise_ranker,
)


def _training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(618)
    board_rows = 18
    board_count = 5
    features = generator.normal(
        size=(board_count * board_rows, FOCAL_STACKER_FEATURE_COUNT)
    )
    labels = np.zeros(len(features), dtype=np.uint8)
    offsets = np.arange(board_count + 1, dtype=np.int32) * board_rows
    for board in range(board_count):
        start = board * board_rows
        features[start : start + 9, AXIS_FEATURE_INDEX] = 0.0
        features[start + 9 : start + 18, AXIS_FEATURE_INDEX] = 1.0
        labels[start : start + 2] = 1
        labels[start + 9 : start + 11] = 1
    return features, labels, offsets


def test_hard_negative_pairing_is_same_axis_stable_and_symmetric() -> None:
    original = np.zeros((8, FOCAL_STACKER_FEATURE_COUNT), dtype=np.float64)
    standardized = np.arange(
        8 * FOCAL_STACKER_FEATURE_COUNT, dtype=np.float64
    ).reshape(8, FOCAL_STACKER_FEATURE_COUNT)
    original[4:, AXIS_FEATURE_INDEX] = 1.0
    original[:, FOCAL_LOGIT_FEATURE_INDEX] = [0, 10, 10, 9, 0, 1, 3, 2]
    labels = np.asarray([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.uint8)

    features, pair_labels, diagnostics = build_symmetric_pairwise_differences(
        standardized,
        labels,
        np.asarray([0, 8]),
        original,
        hard_negatives_per_positive=2,
    )

    expected_positive = np.vstack(
        (
            standardized[0] - standardized[1],
            standardized[0] - standardized[2],
            standardized[4] - standardized[6],
            standardized[4] - standardized[7],
        )
    )
    np.testing.assert_array_equal(features[:4], expected_positive)
    np.testing.assert_array_equal(features[4:], -expected_positive)
    np.testing.assert_array_equal(pair_labels, [1, 1, 1, 1, 0, 0, 0, 0])
    assert diagnostics["board_axis_group_count"] == 2
    assert diagnostics["positive_negative_pair_count"] == 4
    assert not features.flags.writeable
    assert not pair_labels.flags.writeable


def test_fixed_pairwise_fit_matches_sklearn_and_roundtrips(tmp_path) -> None:
    features, labels, offsets = _training_data()
    actual, diagnostics = fit_taska_focal_pairwise_ranker(features, labels, offsets)

    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    differences, pair_labels, _ = build_symmetric_pairwise_differences(
        standardized,
        labels,
        offsets,
        features,
        hard_negatives_per_positive=PAIRWISE_RANKER_PARAMETERS[
            "hard_negatives_per_positive"
        ],
    )
    expected = LogisticRegression(
        C=PAIRWISE_RANKER_PARAMETERS["C"],
        max_iter=PAIRWISE_RANKER_PARAMETERS["max_iter"],
        random_state=PAIRWISE_RANKER_PARAMETERS["random_state"],
        fit_intercept=False,
    ).fit(differences, pair_labels)
    np.testing.assert_allclose(
        actual.predict_scores(features),
        expected.decision_function(standardized),
        atol=1e-12,
        rtol=1e-12,
    )
    assert diagnostics["board_count"] == 5

    artifact = tmp_path / "pairwise-ranker.npz"
    actual.save_npz(artifact)
    loaded = TaskaFocalPairwiseRanker.load_npz(artifact)
    np.testing.assert_array_equal(loaded.mean, actual.mean)
    np.testing.assert_array_equal(loaded.scale, actual.scale)
    np.testing.assert_array_equal(loaded.coefficients, actual.coefficients)
    np.testing.assert_array_equal(
        loaded.predict_scores(features), actual.predict_scores(features)
    )
    assert not loaded.predict_scores(features).flags.writeable


@pytest.mark.parametrize(
    ("labels", "offsets", "message"),
    [
        (np.zeros(90, dtype=np.uint8), np.asarray([0, 90]), "no valid"),
        (np.zeros(89, dtype=np.uint8), np.asarray([0, 90]), "aligned"),
        (np.zeros(90, dtype=np.uint8), np.asarray([1, 90]), "partition"),
    ],
)
def test_pairwise_fit_rejects_invalid_training_contract(
    labels: np.ndarray,
    offsets: np.ndarray,
    message: str,
) -> None:
    features, _, _ = _training_data()
    with pytest.raises(ValueError, match=message):
        fit_taska_focal_pairwise_ranker(features, labels, offsets)
