from __future__ import annotations

import numpy as np
import pytest
import torch

from aiijc_puzzle.taska_incidence_gnn import (
    INCIDENCE_GNN_FEATURE_COUNT,
    INCIDENCE_GNN_TRAINING,
    TaskaIncidenceGNN,
    TaskaIncidenceGNNBundle,
    load_taska_incidence_gnn_bundle,
    save_taska_incidence_gnn_bundle,
    train_taska_incidence_gnn,
)


def _random_bundle() -> TaskaIncidenceGNNBundle:
    torch.manual_seed(31)
    model = TaskaIncidenceGNN()
    with torch.no_grad():
        model.head.weight.normal_(mean=0.0, std=0.1)
        model.head.bias.fill_(0.07)
    return TaskaIncidenceGNNBundle(
        model=model,
        mean=np.zeros(INCIDENCE_GNN_FEATURE_COUNT),
        scale=np.ones(INCIDENCE_GNN_FEATURE_COUNT),
        contract={},
    )


def test_tile_relabeling_is_equivariant() -> None:
    generator = np.random.default_rng(5)
    rows = 27
    features = generator.normal(size=(rows, INCIDENCE_GNN_FEATURE_COUNT))
    focal = generator.normal(size=rows)
    source = generator.integers(0, 18, size=rows, dtype=np.int64)
    target = generator.integers(0, 18, size=rows, dtype=np.int64)
    axis = generator.integers(0, 2, size=rows, dtype=np.int64)
    permutation = generator.permutation(576)
    bundle = _random_bundle()
    original = bundle.predict_logits(features, focal, source, target, axis)
    relabeled = bundle.predict_logits(
        features,
        focal,
        permutation[source],
        permutation[target],
        axis,
    )
    np.testing.assert_allclose(relabeled, original, rtol=0.0, atol=1e-7)


def test_weights_standardizer_and_contract_round_trip(tmp_path) -> None:
    generator = np.random.default_rng(7)
    edge_count = 12
    features = generator.normal(size=(edge_count, INCIDENCE_GNN_FEATURE_COUNT))
    focal = generator.normal(size=edge_count)
    labels = np.tile(np.asarray([0, 1, 0, 1, 1, 0], dtype=np.uint8), 2)
    offsets = np.asarray([0, 6, 12], dtype=np.int64)
    source = np.tile(np.arange(6, dtype=np.int64), 2)
    target = np.tile(np.roll(np.arange(6, dtype=np.int64), -1), 2)
    axis = np.repeat(np.asarray([0, 1], dtype=np.uint8), 6)
    bundle, history = train_taska_incidence_gnn(
        features=features,
        focal_logits=focal,
        labels=labels,
        offsets=offsets,
        source=source,
        target=target,
        axis=axis,
    )
    assert history["completed_steps"] == INCIDENCE_GNN_TRAINING["steps"]
    _, _, contract = save_taska_incidence_gnn_bundle(bundle, tmp_path)
    from aiijc_puzzle.protocol import sha256_file

    loaded = load_taska_incidence_gnn_bundle(
        contract, expected_contract_sha256=sha256_file(contract)
    )
    expected = bundle.predict_logits(features, focal, source, target, axis)
    actual = loaded.predict_logits(features, focal, source, target, axis)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="contract SHA-256"):
        load_taska_incidence_gnn_bundle(contract, expected_contract_sha256="0" * 64)


def test_residual_is_bounded() -> None:
    generator = np.random.default_rng(9)
    bundle = _random_bundle()
    rows = 14
    features = generator.normal(size=(rows, INCIDENCE_GNN_FEATURE_COUNT))
    focal = generator.normal(size=rows)
    source = np.arange(rows, dtype=np.int64)
    target = np.roll(source, -1)
    axis = np.arange(rows, dtype=np.int64) % 2
    logits = bundle.predict_logits(features, focal, source, target, axis)
    assert np.all(np.abs(logits - focal) <= 2.0 + 1e-7)
