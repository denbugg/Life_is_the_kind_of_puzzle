from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.after_h20_restorer import (
    AfterH20ModelConfig,
    AfterH20TileRestorer,
    blend_around_h20,
    paired_clean_tiles,
    restore_tiles,
)
from aiijc_puzzle.protocol import assemble_tiles, split_tiles


def test_model_starts_as_exact_h20_identity() -> None:
    model = AfterH20TileRestorer(AfterH20ModelConfig(width=8, blocks=1))
    pre_h20 = torch.rand(3, 3, 20, 20)
    h20 = torch.rand(3, 3, 20, 20)

    prediction = model(pre_h20, h20)

    assert prediction.shape == h20.shape
    assert torch.equal(prediction, h20)


def test_restore_tiles_is_independent_identity_at_initialisation() -> None:
    generator = np.random.default_rng(7)
    pre_h20 = generator.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    h20 = generator.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    model = AfterH20TileRestorer(AfterH20ModelConfig(width=8, blocks=1))

    restored = restore_tiles(
        model,
        pre_h20,
        h20,
        device=torch.device("cpu"),
        batch_size=113,
    )

    assert np.array_equal(restored, h20)


def test_blend_has_exact_h20_endpoint_and_deterministic_rounding() -> None:
    h20 = np.full((480, 480, 3), 10, dtype=np.uint8)
    restored = np.full_like(h20, 13)

    assert np.array_equal(blend_around_h20(h20, restored, 0.0), h20)
    assert np.all(blend_around_h20(h20, restored, 0.5) == 12)
    assert np.array_equal(blend_around_h20(h20, restored, 1.0), restored)


def test_clean_identities_follow_predicted_dirty_layout() -> None:
    generator = np.random.default_rng(11)
    clean_tiles = generator.integers(0, 256, size=(576, 20, 20, 3), dtype=np.uint8)
    clean = assemble_tiles(clean_tiles)
    dirty_at_position = generator.permutation(576)
    dirty_tiles = np.empty_like(clean_tiles)
    dirty_tiles[dirty_at_position] = clean_tiles
    dirty = assemble_tiles(dirty_tiles)
    predicted_layout = generator.permutation(576)

    paired, margins, diagnostics = paired_clean_tiles(dirty, clean, predicted_layout)

    expected_target_position = np.empty(576, dtype=np.int64)
    expected_target_position[dirty_at_position] = np.arange(576)
    expected = clean_tiles[expected_target_position[predicted_layout]]
    assert np.array_equal(paired, expected)
    assert margins.shape == (576,)
    assert len(diagnostics["target_position_for_dirty_sha256"]) == 64


def test_split_after_restore_preserves_upright_tile_geometry() -> None:
    tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    tiles[:, 0, 0, 0] = np.arange(576, dtype=np.int64) % 256
    image = assemble_tiles(tiles)

    assert np.array_equal(split_tiles(image), tiles)
