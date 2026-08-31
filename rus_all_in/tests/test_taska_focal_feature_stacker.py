from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.taska_focal_feature_stacker import (
    FOCAL_STACKER_FEATURE_COUNT,
    FOCAL_STACKER_PARAMETERS,
    TaskaFocalFeatureStacker,
    fit_taska_focal_feature_stacker,
    stack_taska_focal_features,
)


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(19)
    features = generator.normal(size=(128, FOCAL_STACKER_FEATURE_COUNT))
    labels = (features[:, 1] - 0.4 * features[:, 16] > 0).astype(np.uint8)
    return features, labels


def test_stacked_feature_order_and_read_only() -> None:
    edge = np.arange(45, dtype=np.float64).reshape(3, 15)
    logits = np.asarray([100.0, 101.0, 102.0])
    focal = np.arange(18, dtype=np.float64).reshape(3, 6) + 200.0
    result = stack_taska_focal_features(edge, logits, focal)
    assert result.shape == (3, 22)
    np.testing.assert_array_equal(result[:, :15], edge)
    np.testing.assert_array_equal(result[:, 15], logits)
    np.testing.assert_array_equal(result[:, 16:], focal)
    assert not result.flags.writeable


def test_portable_stacker_matches_fixed_sklearn_and_roundtrips(tmp_path) -> None:
    features, labels = _training_data()
    actual = fit_taska_focal_feature_stacker(features, labels)

    expected = make_pipeline(
        StandardScaler(),
        LogisticRegression(**FOCAL_STACKER_PARAMETERS),
    ).fit(features, labels)
    np.testing.assert_allclose(
        actual.predict_priorities(features),
        expected.predict_proba(features)[:, 1],
        atol=1e-12,
        rtol=1e-12,
    )

    artifact = tmp_path / "stacker.npz"
    actual.save_npz(artifact)
    loaded = TaskaFocalFeatureStacker.load_npz(artifact)
    np.testing.assert_array_equal(loaded.mean, actual.mean)
    np.testing.assert_array_equal(loaded.scale, actual.scale)
    np.testing.assert_array_equal(loaded.coefficients, actual.coefficients)
    np.testing.assert_array_equal(
        loaded.predict_logits(features), actual.predict_logits(features)
    )


@pytest.mark.parametrize(
    ("edge_shape", "logit_shape", "focal_shape"),
    [((4, 14), (4,), (4, 6)), ((4, 15), (5,), (4, 6)), ((4, 15), (4,), (4, 5))],
)
def test_stacked_feature_contract_rejects_misalignment(
    edge_shape: tuple[int, ...],
    logit_shape: tuple[int, ...],
    focal_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        stack_taska_focal_features(
            np.zeros(edge_shape),
            np.zeros(logit_shape),
            np.zeros(focal_shape),
        )


def test_artifact_rejects_wrong_feature_count() -> None:
    features, labels = _training_data()
    with pytest.raises(ValueError, match="rows x 22"):
        fit_taska_focal_feature_stacker(features[:, :-1], labels)

