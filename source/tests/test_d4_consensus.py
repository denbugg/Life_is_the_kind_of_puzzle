from __future__ import annotations

import numpy as np
import pytest

from puzzle_assembly.compatibility import CompatibilityMatrices, rank_normalize
from puzzle_assembly.d4_consensus import (
    D4_VIEWS,
    d4_rank_consensus,
    inverse_transform_tiles,
    transform_tiles,
)
from puzzle_assembly.geometry import TILE, TILE_COUNT


def _tiles(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0, 256, size=(TILE_COUNT, TILE, TILE, 3), dtype=np.uint8
    )


def _score_bank() -> dict[str, CompatibilityMatrices]:
    rows = np.arange(TILE_COUNT, dtype=np.float32)[:, None]
    columns = np.arange(TILE_COUNT, dtype=np.float32)[None, :]
    bank = {}
    for index, view in enumerate(D4_VIEWS):
        right = np.mod(columns + index * rows, TILE_COUNT).astype(np.float32)
        down = np.mod(columns - index * rows, TILE_COUNT).astype(np.float32)
        np.fill_diagonal(right, np.inf)
        np.fill_diagonal(down, np.inf)
        bank[view] = CompatibilityMatrices(view, right, down)
    return bank


@pytest.mark.parametrize("view", D4_VIEWS)
def test_d4_inverse_is_byte_exact(view: str) -> None:
    tiles = _tiles()
    transformed = transform_tiles(tiles, view)
    restored = inverse_transform_tiles(transformed, view)
    assert transformed.flags.c_contiguous
    assert restored.flags.c_contiguous
    assert restored.dtype == np.uint8
    assert np.array_equal(restored, tiles)


def test_unknown_d4_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown D4 view"):
        transform_tiles(_tiles(), "rot90")


def test_d4_rank_consensus_matches_frozen_formula() -> None:
    bank = _score_bank()
    result = d4_rank_consensus(bank)
    for side in ("right", "down"):
        stack = np.stack(
            [rank_normalize(getattr(bank[view], side)) for view in D4_VIEWS]
        )
        stack[:, np.arange(TILE_COUNT), np.arange(TILE_COUNT)] = 0.0
        median = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - median[None]), axis=0)
        expected = 0.5 * stack[0] + 0.4 * median + 0.1 * mad
        np.fill_diagonal(expected, np.inf)
        assert np.array_equal(getattr(result, side), expected.astype(np.float32))
        assert np.isinf(np.diag(getattr(result, side))).all()


def test_d4_consensus_is_deterministic_and_target_blind_schema() -> None:
    bank = _score_bank()
    first = d4_rank_consensus(bank)
    second = d4_rank_consensus(bank)
    assert np.array_equal(first.right, second.right)
    assert np.array_equal(first.down, second.down)


def test_d4_consensus_requires_exact_views_and_unit_weight() -> None:
    bank = _score_bank()
    bank.pop("vflip")
    with pytest.raises(ValueError, match="views mismatch"):
        d4_rank_consensus(bank)
    with pytest.raises(ValueError, match="sum to one"):
        d4_rank_consensus(_score_bank(), identity_weight=0.4)
