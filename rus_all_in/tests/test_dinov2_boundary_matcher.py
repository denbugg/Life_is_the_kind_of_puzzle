from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.dinov2_boundary_matcher import (
    PATCH_GRID,
    freeze_topk,
    scores_from_patch_tokens,
)


def _tokens(count: int = 5, width: int = 8) -> np.ndarray:
    generator = np.random.default_rng(20260831)
    return generator.normal(size=(count, PATCH_GRID, PATCH_GRID, width)).astype(np.float32)


def test_scores_are_finite_directional_and_exclude_self() -> None:
    scores = scores_from_patch_tokens(_tokens())
    assert scores.right.shape == (5, 5)
    assert scores.down.shape == (5, 5)
    assert np.isfinite(scores.right).all()
    assert np.isfinite(scores.down).all()
    candidates = freeze_topk(scores.right, k=3)
    assert candidates.shape == (5, 3)
    assert all(index not in row for index, row in enumerate(candidates))


def test_tile_relabelling_equivariance() -> None:
    tokens = _tokens(count=6)
    permutation = np.asarray([3, 0, 5, 1, 4, 2])
    inverse = np.argsort(permutation)
    original = scores_from_patch_tokens(tokens)
    shuffled = scores_from_patch_tokens(tokens[permutation])
    np.testing.assert_allclose(
        shuffled.right[np.ix_(inverse, inverse)], original.right, atol=1e-6
    )
    np.testing.assert_allclose(
        shuffled.down[np.ix_(inverse, inverse)], original.down, atol=1e-6
    )


def test_invalid_patch_contract_fails_closed() -> None:
    with pytest.raises(ValueError):
        scores_from_patch_tokens(np.zeros((4, 6, 7, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        freeze_topk(np.eye(4), k=4)
