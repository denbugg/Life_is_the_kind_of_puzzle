import numpy as np

from frame_classifier import frame_labels, frame_unary
from island_field_decoder import anchor_islands, select_islands


def test_frame_labels_have_exact_side_count_per_direction():
    side = 6
    y = frame_labels(side).reshape(4, -1)
    assert np.all(y.sum(1) == side)


def test_frame_unary_prefers_predicted_left_edge():
    side, n = 3, 9
    probability = np.full((4, n), 0.05)
    probability[0, 4] = 0.95
    score = frame_unary(probability, side)
    assert score[0, 4] > score[1, 4]


def test_oracle_tail_builds_and_anchors_exact_layout():
    side, n = 4, 16
    right = np.full((n, n), -10.0)
    down = np.full((n, n), -10.0)
    for y in range(side):
        for x in range(side):
            p = y * side + x
            if x + 1 < side:
                right[p, p + 1] = 10.0
            if y + 1 < side:
                down[p, p + side] = 10.0
    components = select_islands(right, down, side, keep=2 * side * (side - 1))
    unary = np.full((n, n), -5.0)
    unary[np.arange(n), np.arange(n)] = 5.0
    layout, _ = anchor_islands(unary, components, side, beam=16, offsets=n)
    assert np.array_equal(layout, np.arange(n))


def test_anchor_is_always_bijective():
    side, n = 4, 16
    unary = np.random.default_rng(0).normal(size=(n, n))
    components = [{2: (0, 0), 7: (0, 1)}, {4: (0, 0), 11: (1, 0)}]
    layout, _ = anchor_islands(unary, components, side, beam=8, offsets=n)
    assert sorted(layout.tolist()) == list(range(n))
