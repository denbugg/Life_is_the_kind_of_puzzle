from __future__ import annotations

import numpy as np

from aiijc_puzzle.novel_analog_layout import (
    ROLE_DIM,
    analog_position_cost,
    board_signature,
    consensus_layout,
    fit_signature_bridge,
    percentile_ranks,
    retrieve_analogs,
)


def test_board_signature_is_permutation_invariant() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(32, ROLE_DIM + 16)).astype(np.float32)
    assert np.allclose(
        board_signature(features),
        board_signature(features[rng.permutation(32)]),
        atol=1e-6,
    )


def test_percentile_ranks_are_row_permutation_equivariant() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(24, 7)).astype(np.float32)
    permutation = rng.permutation(len(features))
    expected = percentile_ranks(features)[permutation]
    assert np.array_equal(percentile_ranks(features[permutation]), expected)


def test_signature_bridge_maps_paired_linear_domain() -> None:
    rng = np.random.default_rng(6)
    dirty = rng.normal(size=(40, 12)).astype(np.float32)
    clean = 0.7 * dirty + 0.2
    bridge = fit_signature_bridge(dirty, clean, alpha=0.01)
    prediction = bridge.transform(dirty)
    assert np.mean(np.square(prediction - clean)) < 1e-4


def test_retrieve_analogs_orders_nearest_references() -> None:
    library = np.asarray([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    indices, distances = retrieve_analogs(np.asarray([0.9, 1.1]), library, k=2)
    assert indices.tolist() == [2, 1]
    assert distances[0] < distances[1]


def test_consensus_layout_recovers_identical_feature_permutation() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(16, ROLE_DIM + 16)).astype(np.float32)
    permutation = rng.permutation(len(features))
    query = features[permutation]
    slot_to_query, cost = consensus_layout(
        query,
        np.stack((features, features)),
        np.asarray([0.2, 0.4], dtype=np.float32),
    )
    expected = np.argsort(permutation)
    assert np.array_equal(slot_to_query, expected)
    assert cost.shape == (16, 16)


def test_consensus_is_invariant_to_template_order() -> None:
    rng = np.random.default_rng(8)
    query = rng.normal(size=(12, ROLE_DIM + 16)).astype(np.float32)
    templates = rng.normal(size=(3, 12, ROLE_DIM + 16)).astype(np.float32)
    distances = np.asarray([0.3, 0.7, 0.4], dtype=np.float32)
    first, _ = consensus_layout(query, templates, distances)
    second, _ = consensus_layout(query, templates[[2, 0, 1]], distances[[2, 0, 1]])
    assert np.array_equal(first, second)


def test_analog_cost_has_expected_shape() -> None:
    rng = np.random.default_rng(9)
    query = rng.normal(size=(10, ROLE_DIM + 16)).astype(np.float32)
    template = rng.normal(size=(10, ROLE_DIM + 16)).astype(np.float32)
    assert analog_position_cost(query, template).shape == (10, 10)
