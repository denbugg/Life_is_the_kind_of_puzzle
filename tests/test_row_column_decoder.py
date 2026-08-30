import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_column_decoder import constrained_paths, solve_rows_then_columns


def perfect_scores(side):
    n = side * side
    right = np.full((n, n), -10.0)
    down = np.full((n, n), -10.0)
    for y in range(side):
        for x in range(side):
            tile = y * side + x
            if x + 1 < side:
                right[tile, tile + 1] = 10.0
            if y + 1 < side:
                down[tile, tile + side] = 10.0
    return right, down


def test_constrained_paths_is_bijective_and_bounded():
    rng = np.random.default_rng(4)
    paths = constrained_paths(rng.normal(size=(16, 16)), count=4, max_length=4)
    assert len(paths) == 4
    assert all(len(p) == 4 for p in paths)
    assert sorted(x for p in paths for x in p) == list(range(16))


def test_perfect_rows_decode_exactly():
    right, down = perfect_scores(4)
    got = solve_rows_then_columns(right, down, 4, first="rows")
    assert np.array_equal(got, np.arange(16))


def test_perfect_columns_decode_exactly():
    right, down = perfect_scores(4)
    got = solve_rows_then_columns(right, down, 4, first="columns")
    assert np.array_equal(got, np.arange(16))
