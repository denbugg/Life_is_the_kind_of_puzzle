from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.wavelet_shrink_matcher_view import (
    fixed_wavelet_top32,
    haar_bayesshrink_tiles,
)
from scripts.run_wavelet_candidate_fit_capacity import coverage_counts, volume_matched_null


def _tiles(seed: int = 41) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0,
        256,
        size=(576, 20, 20, 3),
        dtype=np.uint8,
    )


def test_wavelet_view_is_deterministic_finite_and_bounded() -> None:
    source = _tiles()
    first = haar_bayesshrink_tiles(source)
    second = haar_bayesshrink_tiles(source.copy())
    assert first.dtype == np.float32
    assert first.shape == source.shape
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 255.0


def test_wavelet_view_is_tile_permutation_equivariant() -> None:
    source = _tiles(42)
    permutation = np.random.default_rng(43).permutation(576)
    transformed = haar_bayesshrink_tiles(source)
    permuted = haar_bayesshrink_tiles(source[permutation])
    assert np.array_equal(permuted, transformed[permutation])


def test_wavelet_view_does_not_mix_tiles_or_channels() -> None:
    source = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    source[17, 10, 10, 2] = 255
    transformed = haar_bayesshrink_tiles(source)
    assert not np.any(transformed[np.arange(576) != 17])
    assert not np.any(transformed[17, ..., :2])


def test_wavelet_view_preserves_constant_tiles() -> None:
    source = np.full((576, 20, 20, 3), 73, dtype=np.uint8)
    assert np.array_equal(haar_bayesshrink_tiles(source), source.astype(np.float32))


def test_wavelet_view_reduces_dense_noise_energy() -> None:
    noise = np.random.default_rng(45).normal(0.0, 20.0, size=(576, 20, 20, 3))
    source = np.clip(128.0 + noise, 0.0, 255.0).astype(np.uint8)
    transformed = haar_bayesshrink_tiles(source)
    raw_energy = np.mean(np.square(source.astype(np.float32) - 128.0))
    transformed_energy = np.mean(np.square(transformed - 128.0))
    assert transformed_energy < raw_energy


def test_wavelet_validation_rejects_wrong_shape_or_dtype() -> None:
    with pytest.raises(ValueError):
        haar_bayesshrink_tiles(np.zeros((10, 20, 20, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        haar_bayesshrink_tiles(np.zeros((576, 20, 20, 3), dtype=np.float32))


def test_fixed_wavelet_top32_is_valid() -> None:
    topk = fixed_wavelet_top32(_tiles(44))
    assert topk.shape == (2, 576, 32)
    assert topk.dtype == np.int32
    for axis in range(2):
        for source in range(576):
            assert source not in topk[axis, source]
            assert len(np.unique(topk[axis, source])) == 32


def test_coverage_counts_only_unique_wavelet_recovery() -> None:
    topk = np.empty((7, 2, 576, 32), dtype=np.int32)
    for emitter in range(7):
        for axis in range(2):
            for source in range(576):
                topk[emitter, axis, source] = [
                    value for value in range(576) if value != source
                ][:32]
    truth = np.full((2, 576), -1, dtype=np.int32)
    truth[:, :3] = np.asarray([[10, 20, 40], [11, 21, 41]])
    topk[6, 0, 2, 0] = 40
    topk[6, 1, 2, 0] = 41
    result = coverage_counts(topk, truth)
    assert result["right"]["wavelet_unique_over_all6"] == 1
    assert result["down"]["wavelet_unique_over_all6"] == 1
    assert result["right"]["all7_union"] == result["right"]["all6_union"] + 1


def test_volume_matched_null_uses_only_new_identity_count() -> None:
    topk = np.empty((7, 2, 576, 32), dtype=np.int32)
    for emitter in range(7):
        for axis in range(2):
            for source in range(576):
                topk[emitter, axis, source] = [
                    value for value in range(576) if value != source
                ][:32]
    truth = np.full((2, 576), -1, dtype=np.int32)
    truth[0, 0] = 40
    # The base union has 32 unique identities; wavelet adds exactly one.
    topk[6, 0, 0, -1] = 40
    null = volume_matched_null(topk, truth)["right"]
    assert null["eligible_all6_misses"] == 1
    assert null["new_unique_proposals_on_misses"] == 1
    assert null["actual_unique_hits"] == 1
    assert null["uniform_volume_matched_expected_hits"] == pytest.approx(1 / (575 - 32))
