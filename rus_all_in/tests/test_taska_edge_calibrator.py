from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.raw_tail_global_solver import (
    RawTailEdge,
    RawTailGlobalConfig,
    solve_raw_tail_global,
)
from aiijc_puzzle.taska_edge_calibrator import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    TaskaEdgeCalibrator,
    build_prioritized_raw_tail_components,
    extract_taska_edge_features,
    fit_taska_edge_calibrator,
    predict_taska_edge_priorities,
    solve_prioritized_raw_tail_global,
)


def _feature_inputs(
    grid: int = 3,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[RawTailEdge, ...],
    np.ndarray,
    np.ndarray,
]:
    count = grid * grid
    generator = np.random.default_rng(2034)
    right = generator.normal(size=(count, count))
    down = generator.normal(size=(count, count))
    right_log = generator.normal(size=(count, count))
    down_log = generator.normal(size=(count, count))
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    np.fill_diagonal(right_log, 0.0)
    np.fill_diagonal(down_log, 0.0)
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(3, 7, "down"),
        RawTailEdge(5, 2, "right"),
        RawTailEdge(8, 4, "down"),
    )
    return (
        right,
        down,
        right_log,
        down_log,
        edges,
        np.asarray([0.4, 0.1, 0.8, 0.3]),
        np.asarray([7, 6, 11, 4]),
    )


def test_exact_fifteen_feature_contract_includes_diagonal() -> None:
    right = np.asarray(
        [
            [0.0, 4.0, 1.0, 3.0],
            [5.0, 0.0, 2.0, 6.0],
            [7.0, 2.0, 0.0, 8.0],
            [9.0, 1.0, 4.0, 0.0],
        ]
    )
    down = right + 10.0
    right_log = right - 20.0
    down_log = down - 30.0
    edge = RawTailEdge(0, 1, "right")

    batch = extract_taska_edge_features(
        right,
        down,
        right_log,
        down_log,
        (edge,),
        [0.25],
        [9],
        grid=2,
    )

    row = right[0]
    column = right[:, 1]
    expected = np.asarray(
        [
            0.0,
            4.0,
            -16.0,
            0.25,
            9.0,
            4.0,
            3.0,
            (4.0 - row.mean()) / (row.std() + 1e-6),
            4.0,
            3.0,
            (4.0 - column.mean()) / (column.std() + 1e-6),
            1.0,
            1.0,
            5.0,
            -15.0,
        ]
    )
    assert FEATURE_COUNT == 15
    assert len(FEATURE_NAMES) == FEATURE_COUNT
    np.testing.assert_allclose(batch.values[0], expected, rtol=0.0, atol=1e-14)
    assert batch.edges == (edge,)
    assert not batch.values.flags.writeable


def test_portable_fit_exactly_matches_fixed_sklearn_pipeline() -> None:
    generator = np.random.default_rng(71)
    features = generator.normal(size=(400, FEATURE_COUNT))
    labels = (features[:, 1] - 0.4 * features[:, 7] + 0.2 * features[:, 14]) > 0.2
    expected = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=0),
    )
    expected.fit(features, labels)

    calibrator = fit_taska_edge_calibrator(features, labels)

    np.testing.assert_allclose(
        calibrator.predict_priorities(features),
        expected.predict_proba(features)[:, 1],
        rtol=1e-13,
        atol=1e-15,
    )
    repeated = fit_taska_edge_calibrator(features, labels)
    np.testing.assert_array_equal(calibrator.mean, repeated.mean)
    np.testing.assert_array_equal(calibrator.scale, repeated.scale)
    np.testing.assert_array_equal(calibrator.coefficients, repeated.coefficients)
    assert calibrator.intercept == repeated.intercept


def test_json_and_npz_round_trip_only_portable_parameters(tmp_path: Path) -> None:
    generator = np.random.default_rng(11)
    features = generator.normal(size=(200, FEATURE_COUNT))
    labels = features[:, 3] + features[:, 4] > 0.0
    calibrator = fit_taska_edge_calibrator(features, labels)
    json_path = tmp_path / "calibrator.json"
    npz_path = tmp_path / "calibrator.npz"

    calibrator.save_json(json_path)
    calibrator.save_npz(npz_path)
    json_calibrator = TaskaEdgeCalibrator.load_json(json_path)
    npz_calibrator = TaskaEdgeCalibrator.load_npz(npz_path)

    for restored in (json_calibrator, npz_calibrator):
        np.testing.assert_array_equal(restored.mean, calibrator.mean)
        np.testing.assert_array_equal(restored.scale, calibrator.scale)
        np.testing.assert_array_equal(restored.coefficients, calibrator.coefficients)
        assert restored.intercept == calibrator.intercept
        np.testing.assert_allclose(
            restored.predict_priorities(features),
            calibrator.predict_priorities(features),
        )
        assert not restored.mean.flags.writeable
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema",
        "feature_names",
        "standard_scaler",
        "logistic_regression",
    }


def test_features_and_priorities_are_equivariant_to_tile_relabeling() -> None:
    right, down, right_log, down_log, edges, weights, votes = _feature_inputs()
    original = extract_taska_edge_features(
        right,
        down,
        right_log,
        down_log,
        edges,
        weights,
        votes,
        grid=3,
    )
    relabel = np.asarray([5, 0, 8, 2, 6, 1, 7, 4, 3])
    inverse = np.argsort(relabel)
    relabeled_edges = tuple(
        RawTailEdge(int(relabel[edge.source]), int(relabel[edge.target]), edge.axis)
        for edge in edges
    )
    relabeled = extract_taska_edge_features(
        right[np.ix_(inverse, inverse)],
        down[np.ix_(inverse, inverse)],
        right_log[np.ix_(inverse, inverse)],
        down_log[np.ix_(inverse, inverse)],
        relabeled_edges,
        weights,
        votes,
        grid=3,
    )
    np.testing.assert_allclose(relabeled.values, original.values, rtol=0.0, atol=1e-14)

    generator = np.random.default_rng(9)
    fit_features = generator.normal(size=(160, FEATURE_COUNT))
    labels = fit_features[:, 2] > 0.0
    calibrator = fit_taska_edge_calibrator(fit_features, labels)
    np.testing.assert_allclose(
        calibrator.predict_priorities(relabeled.values),
        calibrator.predict_priorities(original.values),
        rtol=0.0,
        atol=1e-14,
    )
    direct = predict_taska_edge_priorities(
        calibrator,
        right,
        down,
        right_log,
        down_log,
        edges,
        weights,
        votes,
        grid=3,
    )
    np.testing.assert_allclose(direct, calibrator.predict_priorities(original.values))
    assert not direct.flags.writeable


def test_validation_fails_closed() -> None:
    inputs = _feature_inputs()
    with pytest.raises(ValueError, match="cost_right"):
        extract_taska_edge_features(
            inputs[0][:-1],
            *inputs[1:],
            grid=3,
        )
    with pytest.raises(ValueError, match="edge_weights"):
        extract_taska_edge_features(*inputs[:5], [0.2], inputs[6], grid=3)
    with pytest.raises(ValueError, match="non-negative"):
        extract_taska_edge_features(*inputs[:6], [3, -1, 2, 4], grid=3)
    with pytest.raises(ValueError, match="both binary classes"):
        fit_taska_edge_calibrator(np.zeros((5, FEATURE_COUNT)), np.zeros(5))
    with pytest.raises(ValueError, match="feature contract"):
        TaskaEdgeCalibrator(
            feature_names=tuple(reversed(FEATURE_NAMES)),
            mean=np.zeros(FEATURE_COUNT),
            scale=np.ones(FEATURE_COUNT),
            coefficients=np.zeros(FEATURE_COUNT),
            intercept=0.0,
        )


def test_raw_priorities_reproduce_frozen_solver_bitwise() -> None:
    grid = 3
    count = grid * grid
    right = np.full((count, count), 10.0)
    down = np.full((count, count), 10.0)
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(0, 2, "right"),
        RawTailEdge(3, 4, "right"),
        RawTailEdge(0, 3, "down"),
    )
    right[0, 1], right[0, 2], right[3, 4] = -3.0, -2.0, -1.0
    down[0, 3] = -4.0
    raw_priorities = np.asarray(
        [-(right if edge.axis == "right" else down)[edge.source, edge.target] for edge in edges]
    )
    config = RawTailGlobalConfig(random_seed=17, search_rounds=4, fill_rounds=2)

    baseline = solve_raw_tail_global(right, down, edges, grid=grid, config=config)
    prioritized = solve_prioritized_raw_tail_global(
        right,
        down,
        edges,
        raw_priorities,
        grid=grid,
        config=config,
    )

    np.testing.assert_array_equal(prioritized.layout, baseline.layout)
    assert prioritized.components == baseline.components
    assert prioritized.diagnostics == baseline.diagnostics
    assert [decision.edge for decision in prioritized.decisions] == [
        decision.edge for decision in baseline.decisions
    ]
    assert [decision.status for decision in prioritized.decisions] == [
        decision.status for decision in baseline.decisions
    ]
    assert all(
        decision.ranking_priority == decision.raw_priority for decision in prioritized.decisions
    )


def test_external_priorities_change_only_component_build_order() -> None:
    right = np.full((4, 4), 10.0)
    down = np.full((4, 4), 10.0)
    right[0, 2] = -2.0
    right[0, 1] = -1.0
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(0, 2, "right"),
    )

    components, decisions = build_prioritized_raw_tail_components(
        right,
        down,
        edges,
        [0.9, 0.1],
        grid=2,
    )

    assert components == ({0: (0, 0), 1: (0, 1)},)
    assert tuple(decision.edge.target for decision in decisions) == (1, 2)
    assert tuple(decision.ranking_priority for decision in decisions) == (0.9, 0.1)
    assert tuple(decision.raw_priority for decision in decisions) == (1.0, 2.0)


def test_prioritized_solver_is_equivariant_to_tile_relabeling() -> None:
    right = np.full((4, 4), 10.0)
    down = np.full((4, 4), 10.0)
    right[0, 1] = right[2, 3] = 0.0
    down[0, 2] = down[1, 3] = 0.0
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(2, 3, "right"),
        RawTailEdge(0, 2, "down"),
        RawTailEdge(1, 3, "down"),
    )
    priorities = np.asarray([0.6, 0.8, 0.9, 0.7])
    relabel = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(relabel)
    relabeled_edges = tuple(
        RawTailEdge(int(relabel[edge.source]), int(relabel[edge.target]), edge.axis)
        for edge in edges
    )

    original = solve_prioritized_raw_tail_global(
        right,
        down,
        edges,
        priorities,
        grid=2,
    )
    relabeled = solve_prioritized_raw_tail_global(
        right[np.ix_(inverse, inverse)],
        down[np.ix_(inverse, inverse)],
        relabeled_edges,
        priorities,
        grid=2,
    )

    np.testing.assert_array_equal(relabeled.layout, relabel[original.layout])
    assert original.diagnostics.strict_permutation
    assert relabeled.diagnostics.strict_permutation


def test_prioritized_solver_validates_external_priorities() -> None:
    right = np.ones((4, 4))
    down = np.ones((4, 4))
    edge = RawTailEdge(0, 1, "right")
    with pytest.raises(ValueError, match="edge_priorities"):
        solve_prioritized_raw_tail_global(right, down, (edge,), [np.nan], grid=2)
    with pytest.raises(ValueError, match="edge_priorities"):
        solve_prioritized_raw_tail_global(right, down, (edge,), [0.1, 0.2], grid=2)
