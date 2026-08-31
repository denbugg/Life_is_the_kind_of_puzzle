from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.socket_pasha_matched import (
    fuse_pasha_socket_ot_rank_percentiles,
    row_rank_percentiles,
    validate_directional_scores,
)


def test_row_rank_percentiles_exclude_self_and_are_high_is_good() -> None:
    scores = np.asarray(
        [
            [100.0, 3.0, 1.0, 2.0],
            [4.0, 100.0, 2.0, 3.0],
            [1.0, 2.0, 100.0, 3.0],
            [2.0, 3.0, 1.0, 100.0],
        ],
        dtype=np.float32,
    )
    ranked = row_rank_percentiles(scores)
    assert np.array_equal(np.diag(ranked), np.full(4, -1.0, dtype=np.float32))
    assert ranked[0].tolist() == [-1.0, 1.0, 0.0, 0.5]
    assert ranked[1].tolist() == [1.0, -1.0, 0.0, 0.5]


def test_rank_fusion_is_exact_equal_weight_average() -> None:
    pasha = np.asarray(
        [[-9.0, 3.0, 2.0], [2.0, -9.0, 1.0], [3.0, 1.0, -9.0]],
        dtype=np.float32,
    )
    socket = np.asarray(
        [[-9.0, 1.0, 4.0], [1.0, -9.0, 3.0], [2.0, 4.0, -9.0]],
        dtype=np.float32,
    )
    right, down = fuse_pasha_socket_ot_rank_percentiles(
        pasha,
        pasha + 10.0,
        socket,
        socket + 20.0,
    )
    expected = 0.5 * row_rank_percentiles(pasha) + 0.5 * row_rank_percentiles(socket)
    assert np.array_equal(right, expected)
    assert np.array_equal(down, expected)
    assert np.array_equal(np.diag(right), np.full(3, -1.0, dtype=np.float32))


def test_directional_validation_rejects_nonfinite_or_wrong_shape() -> None:
    valid = np.zeros((4, 4), dtype=np.float32)
    right, down = validate_directional_scores(valid, valid, tile_count=4)
    assert right.flags.c_contiguous and down.flags.c_contiguous
    with pytest.raises(ValueError, match="shape"):
        validate_directional_scores(valid[:3], valid, tile_count=4)
    invalid = valid.copy()
    invalid[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_directional_scores(invalid, valid, tile_count=4)
