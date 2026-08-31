from __future__ import annotations

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_context_protector import (
    FEATURE_NAMES,
    fit_context_protector,
    realised_context_features,
)
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES


def _layout() -> np.ndarray:
    return np.arange(24 * 24, dtype=np.int32)


def test_context_rows_only_include_realised_positive_supply_edges() -> None:
    layout = _layout()
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(24, 48, "down"),
        RawTailEdge(4, 7, "right"),
        RawTailEdge(7, 8, "right"),
    )
    rows = realised_context_features(
        selected_layout=layout,
        selected_arm="raw",
        selected_edges=edges,
        selected_logits=np.asarray([1.0, 0.5, 1.5, -0.25]),
        provenance={"current": edges, "selective_new": (), "unique_fullres": ()},
        pre_tail_layouts={name: layout for name in FUSION_ARM_NAMES},
        cost_right=np.arange(576 * 576, dtype=np.float64).reshape(576, 576),
        cost_down=np.arange(576 * 576, dtype=np.float64).reshape(576, 576),
    )
    assert rows.edges == edges[:2]
    assert rows.features.shape == (2, len(FEATURE_NAMES))
    assert np.all(np.isfinite(rows.features))
    assert np.all(rows.features[:, 10] == 1.0)  # raw arm one-hot
    assert np.all(rows.features[:, 16] == len(FUSION_ARM_NAMES))


def test_fixed_logistic_head_returns_natural_probability_mask() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, len(FEATURE_NAMES)))
    labels = np.asarray([0, 1] * 20, dtype=np.uint8)
    model = fit_context_protector(features, labels)
    probability = model.predict_probability(features)
    assert probability.shape == (40,)
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert np.array_equal(model.keep_mask(features), probability >= 0.5)
