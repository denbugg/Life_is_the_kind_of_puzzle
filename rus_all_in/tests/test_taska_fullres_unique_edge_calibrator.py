from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_fullres_unique_edge_calibrator import (
    DECISION_THRESHOLD,
    FEATURE_NAMES,
    fit_unique_fullres_edge_calibrator,
    unique_fullres_edge_features,
)


def test_unique_fullres_features_follow_fixed_nonself_rank_contract() -> None:
    right = np.add.outer(np.arange(576), np.arange(576)).astype(np.float64)
    down = right + 10.0
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    edges = (
        RawTailEdge(2, 4, "right"),
        RawTailEdge(5, 3, "down"),
    )
    features = unique_fullres_edge_features(
        edges=edges,
        focal_logits=np.asarray([0.25, 1.5]),
        restored_support=np.asarray([3, 4]),
        cost_right=right,
        cost_down=down,
    )
    assert features.shape == (2, len(FEATURE_NAMES))
    assert features[0, 0] == 0.25
    assert features[0, 1] == 3
    assert features[0, 2] == right[2, 4]
    assert features[1, -1] == 1.0
    assert np.isfinite(features).all()


def test_portable_calibrator_matches_preregistered_sklearn_pipeline() -> None:
    generator = np.random.default_rng(123)
    features = generator.normal(size=(80, len(FEATURE_NAMES)))
    labels = (features[:, 0] - 0.5 * features[:, 2] > 0).astype(np.uint8)
    portable = fit_unique_fullres_edge_calibrator(features, labels)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    sklearn_model = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=1000,
        random_state=0,
        solver="lbfgs",
    ).fit(standardized, labels)
    expected = sklearn_model.predict_proba(standardized)[:, 1]

    np.testing.assert_allclose(portable.predict_probability(features), expected, atol=1e-12)
    np.testing.assert_array_equal(portable.keep_mask(features), expected >= DECISION_THRESHOLD)
