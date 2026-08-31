from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.socket_confidence_calibration import (
    FEATURE_NAMES,
    choose_precision_threshold,
    exact_edge_labels,
    extract_hard_edge_features,
    fit_linear_calibrator,
    fixed_heuristic_selection,
    frozen_linear_calibrator_from_payload,
    mutual_top1_selection,
)


def _random_board(grid: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(41)
    count = grid * grid
    right_raw = generator.normal(size=(count, count))
    down_raw = generator.normal(size=(count, count))
    np.fill_diagonal(right_raw, -1e4)
    np.fill_diagonal(down_raw, -1e4)
    right = generator.normal(size=(count + 1, count + 1))
    down = generator.normal(size=(count + 1, count + 1))
    np.fill_diagonal(right[:count, :count], -1e4)
    np.fill_diagonal(down[:count, :count], -1e4)
    right[-1, -1] = down[-1, -1] = -np.inf
    return right, down, right_raw, down_raw


def test_extract_hard_features_is_finite_and_exact_cardinality() -> None:
    right, down, right_raw, down_raw = _random_board()
    features = extract_hard_edge_features(
        right_log_assignment=right,
        down_log_assignment=down,
        right_raw=right_raw,
        down_raw=down_raw,
        grid=3,
    )
    assert features.values.shape == (12, len(FEATURE_NAMES))
    assert features.values.dtype == np.float32
    assert np.isfinite(features.values).all()
    assert np.count_nonzero(features.axis == 0) == 6
    assert np.count_nonzero(features.axis == 1) == 6
    assert np.all((features.source >= 0) & (features.source < 9))
    assert np.all((features.target >= 0) & (features.target < 9))


def test_exact_labels_match_reference_neighbour_geometry() -> None:
    right, down, right_raw, down_raw = _random_board()
    features = extract_hard_edge_features(
        right_log_assignment=right,
        down_log_assignment=down,
        right_raw=right_raw,
        down_raw=down_raw,
        grid=3,
    )
    reference = np.asarray([4, 1, 7, 3, 0, 8, 6, 2, 5], dtype=np.int32)
    labels = exact_edge_labels(features, reference, grid=3)
    position = np.empty(9, dtype=np.int32)
    position[reference] = np.arange(9)
    expected = []
    for source, target, axis in zip(
        features.source,
        features.target,
        features.axis,
        strict=True,
    ):
        if axis == 0:
            expected.append(
                position[target] == position[source] + 1 and position[source] % 3 != 2
            )
        else:
            expected.append(position[target] == position[source] + 3)
    np.testing.assert_array_equal(labels, expected)


def test_precision_threshold_is_most_inclusive_at_target() -> None:
    probability = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = np.asarray([1, 1, 0, 1, 0], dtype=bool)
    threshold, precision = choose_precision_threshold(
        probability,
        labels,
        target_precision=0.75,
    )
    assert threshold == pytest.approx(0.6)
    assert precision == pytest.approx(0.75)


def test_logistic_calibrator_and_fixed_controls() -> None:
    generator = np.random.default_rng(7)
    values = generator.normal(size=(200, len(FEATURE_NAMES)))
    labels = values[:, 0] + 0.5 * values[:, 2] > 0.8
    calibrator = fit_linear_calibrator(values, labels, target_precision=0.8)
    probability = calibrator.predict_probability(values)
    selected = calibrator.select(values)
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert labels[selected].mean() >= 0.8
    assert calibrator.feature_names == FEATURE_NAMES

    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    controls = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
    controls[:, index["projected_edge_confidence"]] = (-0.9, -1.1)
    controls[:, index["ot_outgoing_dustbin_margin"]] = 0.6
    controls[:, index["ot_incoming_dustbin_margin"]] = 0.6
    controls[0, index["ot_row_reciprocal_rank"]] = 1.0
    controls[0, index["ot_column_reciprocal_rank"]] = 1.0
    np.testing.assert_array_equal(fixed_heuristic_selection(controls), [True, False])
    np.testing.assert_array_equal(
        mutual_top1_selection(controls, variant="ot"),
        [True, False],
    )


def test_contract_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="target_precision"):
        choose_precision_threshold([0.5], [True], target_precision=0.0)
    with pytest.raises(ValueError, match="variant"):
        mutual_top1_selection(np.zeros((1, len(FEATURE_NAMES))), variant="fused")
    right, down, right_raw, down_raw = _random_board()
    with pytest.raises(ValueError, match="right_raw"):
        extract_hard_edge_features(
            right_log_assignment=right,
            down_log_assignment=down,
            right_raw=right_raw[:-1],
            down_raw=down_raw,
            grid=3,
        )


def test_frozen_calibrator_json_round_trip() -> None:
    generator = np.random.default_rng(23)
    values = generator.normal(size=(100, len(FEATURE_NAMES)))
    labels = values[:, 0] > 0.5
    expected = fit_linear_calibrator(values, labels)
    payload = {
        "schema": "aiijc-socket-hard-edge-linear-calibrator-v1",
        "estimator": {
            "feature_names": list(FEATURE_NAMES),
            "mean": expected.mean.tolist(),
            "scale": expected.scale.tolist(),
            "coefficients": expected.coefficients.tolist(),
            "intercept": expected.intercept,
        },
        "single_threshold": {
            "probability_greater_equal": expected.threshold,
            "target_fit_precision": expected.target_fit_precision,
            "achieved_fit_precision": expected.achieved_fit_precision,
        },
    }
    observed = frozen_linear_calibrator_from_payload(payload)
    np.testing.assert_allclose(
        observed.predict_probability(values),
        expected.predict_probability(values),
    )
    payload["estimator"]["feature_names"] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError, match="feature contract"):
        frozen_linear_calibrator_from_payload(payload)
