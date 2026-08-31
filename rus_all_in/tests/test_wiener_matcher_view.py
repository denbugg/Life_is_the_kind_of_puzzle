from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.wiener_matcher_view import (
    fixed_top32,
    local_wiener_tiles,
)
from scripts.run_wiener_candidate_emitter_fit_capacity import _coverage


def _tiles(seed: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, 256, size=(576, 20, 20, 3), dtype=np.uint8
    )


def test_local_wiener_is_deterministic_finite_and_bounded() -> None:
    source = _tiles()
    first = local_wiener_tiles(source)
    second = local_wiener_tiles(source.copy())
    assert first.dtype == np.float32
    assert first.shape == source.shape
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 255.0


def test_local_wiener_is_tile_permutation_equivariant() -> None:
    source = _tiles(8)
    permutation = np.random.default_rng(9).permutation(576)
    transformed = local_wiener_tiles(source)
    permuted = local_wiener_tiles(source[permutation])
    assert np.array_equal(permuted, transformed[permutation])


def test_local_wiener_does_not_mix_tiles_or_channels() -> None:
    source = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    source[17, 10, 10, 2] = 255
    transformed = local_wiener_tiles(source)
    assert not np.any(transformed[np.arange(576) != 17])
    assert not np.any(transformed[17, ..., :2])


def test_local_wiener_preserves_constant_tiles() -> None:
    source = np.full((576, 20, 20, 3), 73, dtype=np.uint8)
    assert np.array_equal(local_wiener_tiles(source), source.astype(np.float32))


def test_validation_rejects_wrong_shape_or_dtype() -> None:
    with pytest.raises(ValueError):
        local_wiener_tiles(np.zeros((10, 20, 20, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        local_wiener_tiles(np.zeros((576, 20, 20, 3), dtype=np.float32))


def test_fixed_top32_is_stable_and_excludes_self() -> None:
    row = np.arange(576, dtype=np.float32)[None].repeat(576, axis=0)
    scores = (row.copy(), row.copy())
    top = fixed_top32(scores)
    assert top.shape == (2, 576, 32)
    assert top.dtype == np.int32
    for axis in range(2):
        for source in range(576):
            assert source not in top[axis, source]
            assert len(np.unique(top[axis, source])) == 32


def test_coverage_counts_only_unique_wiener_recovery() -> None:
    topk = np.empty((4, 2, 576, 32), dtype=np.int32)
    for emitter in range(4):
        for axis in range(2):
            for source in range(576):
                candidates = [
                    value
                    for value in range(576)
                    if value != source
                ][:32]
                topk[emitter, axis, source] = candidates
    truth = np.full((2, 576), -1, dtype=np.int32)
    truth[:, :3] = np.asarray([[10, 20, 40], [11, 21, 41]])
    # First two are in the shared legacy prefix; only the third is added by Wiener.
    topk[3, 0, 2, 0] = 40
    topk[3, 1, 2, 0] = 41
    result = _coverage(topk, truth)
    assert result["right"]["wiener_unique_recovered"] == 1
    assert result["down"]["wiener_unique_recovered"] == 1
    assert result["right"]["extended_union"] == result["right"]["legacy_union"] + 1
