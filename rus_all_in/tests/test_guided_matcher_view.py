from __future__ import annotations

import numpy as np
import pytest

import aiijc_puzzle.guided_matcher_view as module
from aiijc_puzzle.guided_matcher_view import (
    FIXED_CONFIG,
    GuidedMatcherViewConfig,
    guided_fused_directional_scores,
    guided_luminance_tiles,
)


def _tiles(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0,
        256,
        size=(576, 20, 20, 3),
        dtype=np.uint8,
    )


def test_fixed_recipe_is_exactly_preregisterable() -> None:
    assert GuidedMatcherViewConfig(
        radius=2,
        epsilon=1600.0,
        guided_weight=0.5,
    ) == FIXED_CONFIG
    with pytest.raises(ValueError):
        GuidedMatcherViewConfig(radius=0)
    with pytest.raises(ValueError):
        GuidedMatcherViewConfig(epsilon=0.0)
    with pytest.raises(ValueError):
        GuidedMatcherViewConfig(guided_weight=1.0)


def test_guided_view_preserves_contract_and_input() -> None:
    tiles = _tiles()
    before = tiles.copy()
    filtered = guided_luminance_tiles(tiles)
    assert filtered.shape == tiles.shape
    assert filtered.dtype == np.float32
    assert np.isfinite(filtered).all()
    assert float(filtered.min()) >= 0.0
    assert float(filtered.max()) <= 255.0
    assert np.array_equal(tiles, before)
    assert not np.array_equal(filtered, tiles.astype(np.float32))


def test_guided_view_is_tile_permutation_equivariant_and_constant_safe() -> None:
    tiles = _tiles(1)
    permutation = np.random.default_rng(2).permutation(576)
    expected = guided_luminance_tiles(tiles)[permutation]
    actual = guided_luminance_tiles(tiles[permutation])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    constant = np.full((576, 20, 20, 3), (20, 100, 240), dtype=np.uint8)
    np.testing.assert_allclose(
        guided_luminance_tiles(constant),
        constant.astype(np.float32),
        rtol=0.0,
        atol=1e-4,
    )


def test_fixed_fusion_is_arithmetic_mean_of_control_and_guided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiles = _tiles(3)
    control = (
        np.full((576, 576), -3.0, dtype=np.float32),
        np.full((576, 576), -5.0, dtype=np.float32),
    )
    right_cost = np.full((576, 576), 7.0, dtype=np.float32)
    down_cost = np.full((576, 576), 11.0, dtype=np.float32)
    monkeypatch.setattr(module, "classical_costs", lambda _: (right_cost, down_cost))
    monkeypatch.setattr(
        module,
        "cost_to_logp",
        lambda value: np.asarray(value, dtype=np.float32) * -0.25,
    )
    right, down = guided_fused_directional_scores(tiles, control)
    np.testing.assert_allclose(right, 0.5 * control[0] - 0.125 * right_cost)
    np.testing.assert_allclose(down, 0.5 * control[1] - 0.125 * down_cost)
    assert right.flags.c_contiguous and down.flags.c_contiguous


def test_rejects_malformed_tiles_and_scores() -> None:
    tiles = _tiles(4)
    with pytest.raises(ValueError):
        guided_luminance_tiles(tiles[:1])
    with pytest.raises(ValueError):
        guided_luminance_tiles(tiles.astype(np.float32))
    with pytest.raises(ValueError):
        guided_fused_directional_scores(
            tiles,
            (np.zeros((2, 2), dtype=np.float32), np.zeros((576, 576), dtype=np.float32)),
        )
