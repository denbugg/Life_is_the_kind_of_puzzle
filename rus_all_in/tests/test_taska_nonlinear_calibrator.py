from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiijc_puzzle.taska_edge_calibrator import FEATURE_COUNT
from aiijc_puzzle.taska_nonlinear_calibrator import (
    NONLINEAR_CALIBRATOR_PARAMETERS,
    TaskaNonlinearCalibrator,
    fit_taska_nonlinear_calibrator,
)


def _data(rows: int = 600) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(71)
    features = generator.normal(size=(rows, FEATURE_COUNT))
    labels = ((features[:, 0] * features[:, 1] + features[:, 2]) > 0).astype(np.uint8)
    return features, labels


def test_fit_is_deterministic_and_nonlinear() -> None:
    features, labels = _data()
    first = fit_taska_nonlinear_calibrator(features, labels)
    second = fit_taska_nonlinear_calibrator(features, labels)
    assert len(first.tree_offsets) == NONLINEAR_CALIBRATOR_PARAMETERS["max_iter"] + 1
    assert np.array_equal(first.tree_offsets, second.tree_offsets)
    assert np.array_equal(first.values, second.values)
    assert np.array_equal(
        first.predict_priorities(features),
        second.predict_priorities(features),
    )
    assert not first.predict_priorities(features).flags.writeable


def test_npz_round_trip_is_prediction_exact(tmp_path: Path) -> None:
    features, labels = _data()
    fitted = fit_taska_nonlinear_calibrator(features, labels)
    path = tmp_path / "calibrator.npz"
    fitted.save_npz(path)
    loaded = TaskaNonlinearCalibrator.load_npz(path)
    assert np.array_equal(
        loaded.predict_priorities(features),
        fitted.predict_priorities(features),
    )


def test_bad_features_and_labels_fail_closed() -> None:
    features, labels = _data()
    with pytest.raises(ValueError, match="shape"):
        fit_taska_nonlinear_calibrator(features[:, :-1], labels)
    with pytest.raises(ValueError, match="binary"):
        fit_taska_nonlinear_calibrator(features, np.full(len(features), 2))
