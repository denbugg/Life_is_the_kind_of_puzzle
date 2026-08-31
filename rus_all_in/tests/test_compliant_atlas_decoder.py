from __future__ import annotations

import inspect

import numpy as np

from aiijc_puzzle.compliant_atlas_decoder import (
    PRODUCTION_ATLAS_WEIGHT,
    PRODUCTION_EDGE_BUDGET,
    PRODUCTION_NLM_H,
    audit_raw_permutation,
    population_position_scores,
    predict_compliant_atlas,
    solve_buddies_with_position,
)
from aiijc_puzzle.legacy_upgrade import solve_buddies
from aiijc_puzzle.novel_analog_layout import tile_semantic_features
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT, assemble_tiles, split_tiles


def synthetic_image() -> np.ndarray:
    rng = np.random.default_rng(71)
    return rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def test_population_unary_does_not_mutate_template_and_has_expected_shape() -> None:
    image = synthetic_image()
    tiles = split_tiles(image)
    template = tile_semantic_features(tiles)
    before = template.copy()
    scores = population_position_scores(tiles, template)
    assert scores.shape == (TILE_COUNT, TILE_COUNT)
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores.mean(axis=1), 0.0, atol=1e-5)
    assert np.array_equal(template, before)


def test_zero_weight_buddy_solver_reproduces_historical_layout() -> None:
    rng = np.random.default_rng(72)
    right = rng.normal(size=(TILE_COUNT, TILE_COUNT)).astype(np.float32)
    down = rng.normal(size=(TILE_COUNT, TILE_COUNT)).astype(np.float32)
    np.fill_diagonal(right, -1e4)
    np.fill_diagonal(down, -1e4)
    position = rng.normal(size=(TILE_COUNT, TILE_COUNT)).astype(np.float32)
    historical = solve_buddies(right, down, max_edges=96)
    augmented = solve_buddies_with_position(
        right,
        down,
        position,
        position_weight=0.0,
        max_edges=96,
    )
    assert np.array_equal(augmented.layout, historical.layout)


def test_permutation_audit_proves_exact_tile_preservation() -> None:
    image = synthetic_image()
    tiles = split_tiles(image)
    layout = np.random.default_rng(73).permutation(TILE_COUNT).astype(np.int32)
    raw = assemble_tiles(tiles[layout])
    audit = audit_raw_permutation(
        image,
        raw,
        layout,
        restoration_applied_after_audit=True,
    )
    assert audit.passed
    assert audit.tile_count == 576
    assert audit.unique_tile_indices == 576
    assert audit.input_output_tile_multiset_equal
    assert audit.raw_input_pixels_preserved


def test_permutation_audit_rejects_duplicate_index() -> None:
    image = synthetic_image()
    layout = np.arange(TILE_COUNT, dtype=np.int32)
    raw = assemble_tiles(split_tiles(image)[layout])
    invalid = layout.copy()
    invalid[-1] = invalid[0]
    audit = audit_raw_permutation(
        image,
        raw,
        invalid,
        restoration_applied_after_audit=True,
    )
    assert not audit.passed
    assert audit.missing_tile_indices == (TILE_COUNT - 1,)
    assert audit.duplicate_tile_indices == (0,)


def test_production_predictor_has_no_target_and_preserves_all_tiles() -> None:
    assert "target" not in inspect.signature(predict_compliant_atlas).parameters
    image = synthetic_image()
    template = tile_semantic_features(split_tiles(image))
    prediction = predict_compliant_atlas(image, template)
    assert prediction.audit.passed
    assert prediction.raw.shape == image.shape
    assert prediction.restored.shape == image.shape
    assert prediction.atlas_weight == PRODUCTION_ATLAS_WEIGHT
    assert prediction.edge_budget == PRODUCTION_EDGE_BUDGET
    assert prediction.nlm_h == PRODUCTION_NLM_H
