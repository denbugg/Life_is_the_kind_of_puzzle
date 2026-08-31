from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.local_rank_matcher_view import local_midrank_tiles
from scripts.run_local_rank_candidate_fit_capacity import coverage_counts


def _tiles(seed: int = 10, high: int = 256) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, high, size=(576, 20, 20, 3), dtype=np.uint8
    )


def test_rank_view_is_deterministic_finite_and_bounded() -> None:
    source = _tiles()
    first = local_midrank_tiles(source)
    second = local_midrank_tiles(source.copy())
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 255.0


def test_rank_view_is_tile_permutation_equivariant() -> None:
    source = _tiles(11)
    permutation = np.random.default_rng(12).permutation(576)
    expected = local_midrank_tiles(source)[permutation]
    assert np.array_equal(local_midrank_tiles(source[permutation]), expected)


def test_rank_view_is_positive_affine_photometry_invariant() -> None:
    source = _tiles(13, high=100)
    transformed = (2 * source.astype(np.uint16) + 7).astype(np.uint8)
    assert np.array_equal(local_midrank_tiles(source), local_midrank_tiles(transformed))


def test_rank_view_does_not_mix_tiles_or_channels() -> None:
    source = np.full((576, 20, 20, 3), 100, dtype=np.uint8)
    source[17, 10, 10, 2] = 200
    baseline = local_midrank_tiles(np.full_like(source, 100))
    changed = local_midrank_tiles(source)
    delta = changed != baseline
    assert not np.any(delta[np.arange(576) != 17])
    assert not np.any(delta[17, ..., :2])


def test_constant_tiles_have_midrank_four() -> None:
    source = np.full((576, 20, 20, 3), 73, dtype=np.uint8)
    assert np.all(local_midrank_tiles(source) == np.float32(127.5))


def test_rank_view_rejects_wrong_shape_or_dtype() -> None:
    with pytest.raises(ValueError):
        local_midrank_tiles(np.zeros((10, 20, 20, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        local_midrank_tiles(np.zeros((576, 20, 20, 3), dtype=np.float32))


def test_coverage_counts_rank_only_recovery() -> None:
    topk = np.empty((6, 2, 576, 32), dtype=np.int32)
    for emitter in range(6):
        for axis in range(2):
            for source in range(576):
                topk[emitter, axis, source] = [
                    value for value in range(576) if value != source
                ][:32]
    truth = np.full((2, 576), -1, dtype=np.int32)
    truth[:, 2] = 40
    topk[5, :, 2, 0] = 40
    counts = coverage_counts(topk, truth)
    assert counts["right"]["all5_union"] == 0
    assert counts["right"]["rank_unique_over_all5"] == 1
    assert counts["down"]["all6_union"] == 1
