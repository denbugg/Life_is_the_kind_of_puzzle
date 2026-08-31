import numpy as np

from aiijc_puzzle.taska_two_sided_rank_selector import (
    select_two_sided_log_rank_layout,
    two_sided_log_rank_score,
)


def test_rank_score_prefers_requested_grid_contacts() -> None:
    grid = 3
    count = grid * grid
    right = np.full((count, count), 10.0)
    down = np.full((count, count), 10.0)
    np.fill_diagonal(right, 20.0)
    np.fill_diagonal(down, 20.0)
    ideal = np.arange(count, dtype=np.int32)
    board = ideal.reshape(grid, grid)
    right[board[:, :-1], board[:, 1:]] = 0.0
    down[board[:-1, :], board[1:, :]] = 0.0
    other = ideal[::-1].copy()
    assert two_sided_log_rank_score(ideal, right, down, grid=grid) < (
        two_sided_log_rank_score(other, right, down, grid=grid)
    )
    choice, layout, scores = select_two_sided_log_rank_layout(
        {"other": other, "ideal": ideal},
        ("other", "ideal"),
        right,
        down,
        grid=grid,
    )
    assert choice == "ideal"
    assert np.array_equal(layout, ideal)
    assert scores["ideal"] < scores["other"]


def test_selector_rejects_non_permutation() -> None:
    matrix = np.zeros((9, 9))
    with np.testing.assert_raises(ValueError):
        select_two_sided_log_rank_layout(
            {"bad": np.zeros(9)}, ("bad",), matrix, matrix, grid=3
        )
