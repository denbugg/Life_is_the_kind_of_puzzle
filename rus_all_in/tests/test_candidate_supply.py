from __future__ import annotations

import numpy as np

from aiijc_puzzle.candidate_supply import (
    classical_costs,
    merge_records,
    recover_layout,
    split_tiles,
    top_candidates,
)


def test_split_tiles_is_row_major() -> None:
    image = np.arange(4 * 4 * 3).reshape(4, 4, 3)
    tiles = split_tiles(image, grid=2)
    assert tiles.shape == (4, 2, 2, 3)
    np.testing.assert_array_equal(tiles[0], image[:2, :2])
    np.testing.assert_array_equal(tiles[3], image[2:, 2:])


def test_recover_layout_is_bijective_on_easy_tiles() -> None:
    rng = np.random.default_rng(7)
    clean = rng.integers(0, 256, size=(9, 8, 8, 3), dtype=np.uint8)
    permutation = rng.permutation(9)
    dirty = clean[permutation].astype(np.float32)
    dirty = np.clip(1.1 * dirty + 8.0, 0, 255).astype(np.uint8)
    recovered = recover_layout(dirty, clean)
    expected = np.argsort(permutation)
    np.testing.assert_array_equal(recovered.dirty_at_position, expected)
    np.testing.assert_array_equal(np.sort(recovered.dirty_at_position), np.arange(9))


def test_classical_cost_prefers_continuous_synthetic_neighbour() -> None:
    # Four tiles cut from one smooth image.  The right neighbour of tile zero is one.
    yy, xx = np.mgrid[:12, :12]
    image = np.stack((xx * 10 + yy, xx * 3 + yy * 7, xx + yy * 5), axis=-1).astype(np.uint8)
    tiles = split_tiles(image, grid=2)
    right, down = classical_costs(tiles)
    assert np.isinf(right[0, 0])
    assert np.isinf(down[0, 0])
    assert int(np.argmin(right[0])) == 1
    assert int(np.argmin(down[0])) == 2


def test_top_candidates_and_record_merge() -> None:
    cost = np.asarray([[np.inf, 2, 1], [4, np.inf, 3], [2, 1, np.inf]], dtype=float)
    np.testing.assert_array_equal(top_candidates(cost, 2), [[2, 1], [2, 0], [1, 0]])
    records = [
        {
            "emitter": "raw",
            "scope": "all",
            "direction": "right",
            "k": 1,
            "edge_count": 2,
            "candidate_count_sum": 2,
            "content_candidate_count_sum": 2,
            "exact_hits": 1,
            "best_rmse_sum": 12.0,
            "rmse_hits": {"le_10": 1},
        }
    ]
    merged = merge_records([records, records])[0]
    assert merged["exact_recall"] == 0.5
    assert merged["mean_content_candidates"] == 1.0
    assert merged["mean_best_rmse"] == 6.0
    assert merged["content_recall_rmse_le_10"] == 0.5
