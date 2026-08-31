from __future__ import annotations

import numpy as np

from aiijc_puzzle.low_frequency_prior import (
    GRID_SIZES,
    FrozenLowFrequencyPrior,
    dirty_board_features,
    fit_low_frequency_prior,
)
from aiijc_puzzle.novel_analog_layout import tile_semantic_features
from aiijc_puzzle.protocol import IMAGE_SIZE, assemble_tiles, split_tiles


def test_dirty_features_are_invariant_to_tile_permutation() -> None:
    rng = np.random.default_rng(41)
    image = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    tiles = split_tiles(image)
    shuffled = assemble_tiles(tiles[rng.permutation(len(tiles))])
    assert np.allclose(dirty_board_features(image), dirty_board_features(shuffled), atol=1e-6)


def test_fit_save_load_and_target_free_prediction(tmp_path) -> None:
    rng = np.random.default_rng(42)
    rows = 16
    feature_dim = 11
    features = rng.normal(size=(rows, feature_dim)).astype(np.float32)
    target_grids = {
        size: rng.uniform(0.1, 0.9, size=(rows, size, size, 3)).astype(np.float32)
        for size in GRID_SIZES
    }
    reference = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 127, dtype=np.uint8)
    generic_template = tile_semantic_features(split_tiles(reference))
    model = fit_low_frequency_prior(
        features,
        target_grids,
        generic_template,
        cluster_count=8,
        metadata={"unit_test": True},
    )
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = FrozenLowFrequencyPrior.load(path)
    assert loaded.metadata["unit_test"] is True
    assert loaded.metadata["schema_version"] == 1
    assert set(loaded.grid_heads) == set(GRID_SIZES)
    assert np.allclose(loaded.population_mean_12, model.population_mean_12)


def test_model_rejects_malformed_rgb_input() -> None:
    rng = np.random.default_rng(43)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    try:
        dirty_board_features(image)
    except ValueError as error:
        assert "480" in str(error)
    else:
        raise AssertionError("malformed image was accepted")
