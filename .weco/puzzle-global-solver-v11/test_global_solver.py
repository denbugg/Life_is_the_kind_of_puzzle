import numpy as np

from global_solver import placement_metrics, solve_complete


def perfect_case(side: int = 6) -> None:
    n = side * side
    rng = np.random.default_rng(7)
    true_positions = rng.permutation(n)
    inverse = np.empty(n, np.int32)
    inverse[true_positions] = np.arange(n)
    right = rng.normal(-2, 0.1, (n, n)).astype(np.float32)
    down = rng.normal(-2, 0.1, (n, n)).astype(np.float32)
    np.fill_diagonal(right, -100)
    np.fill_diagonal(down, -100)
    for position, tile in enumerate(inverse):
        row, col = divmod(position, side)
        if col + 1 < side:
            right[tile, inverse[position + 1]] = 8
        if row + 1 < side:
            down[tile, inverse[position + side]] = 8
    anchor = {int(tile): divmod(position, side) for position, tile in enumerate(inverse[:14])}
    result = solve_complete(right, down, side, anchor, seed=11,
                            beam_width=4, hungarian_rounds=4, swap_proposals=3000)
    metrics = placement_metrics(result.board, true_positions, side)
    assert len(np.unique(result.board)) == n
    assert metrics["coverage"] == 1.0
    assert metrics["adjacency"] > 0.95, metrics
    assert metrics["translation_aligned_placement"] > 0.95, metrics
    print({"contract": "pass", **metrics, "objective": result.objective})


if __name__ == "__main__":
    perfect_case()
