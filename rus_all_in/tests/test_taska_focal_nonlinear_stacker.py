from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.taska_focal_nonlinear_stacker import (
    FOCAL_NONLINEAR_FEATURE_COUNT,
    FOCAL_NONLINEAR_FEATURE_NAMES,
    TaskaFocalNonlinearStacker,
    fit_taska_focal_nonlinear_stacker,
    stack_taska_focal_nonlinear_features,
)


def _training() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(20260831)
    features = generator.normal(size=(800, FOCAL_NONLINEAR_FEATURE_COUNT))
    score = features[:, 0] - 0.7 * features[:, 15] + 0.4 * features[:, 20]
    labels = (score > np.median(score)).astype(np.uint8)
    return features, labels


def test_stack_contract_is_exact_and_read_only() -> None:
    edge = np.arange(45, dtype=np.float64).reshape(3, 15)
    logits = np.asarray([2.0, 3.0, 5.0])
    focal = np.arange(18, dtype=np.float64).reshape(3, 6)

    actual = stack_taska_focal_nonlinear_features(edge, logits, focal)

    assert actual.shape == (3, 22)
    assert np.array_equal(actual[:, :15], edge)
    assert np.array_equal(actual[:, 15], logits)
    assert np.array_equal(actual[:, 16:], focal)
    assert not actual.flags.writeable


def test_portable_model_roundtrip_matches_predictions(tmp_path) -> None:
    features, labels = _training()
    fitted = fit_taska_focal_nonlinear_stacker(features, labels)
    path = tmp_path / "stacker.npz"
    fitted.save_npz(path)

    loaded = TaskaFocalNonlinearStacker.load_npz(path)

    assert loaded.feature_names == FOCAL_NONLINEAR_FEATURE_NAMES
    assert len(loaded.tree_offsets) == 101
    assert np.allclose(
        loaded.predict_logits(features),
        fitted.predict_logits(features),
        atol=0.0,
        rtol=0.0,
    )
    assert not loaded.predict_priorities(features).flags.writeable


def test_validation_rejects_misaligned_or_nonfinite_inputs() -> None:
    edge = np.zeros((3, 15))
    focal = np.zeros((3, 6))
    with pytest.raises(ValueError, match="aligned"):
        stack_taska_focal_nonlinear_features(edge, np.zeros(2), focal)
    focal[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        stack_taska_focal_nonlinear_features(edge, np.zeros(3), focal)
    features, labels = _training()
    with pytest.raises(ValueError, match="aligned"):
        fit_taska_focal_nonlinear_stacker(features, labels[:-1])
