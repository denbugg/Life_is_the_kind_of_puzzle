from __future__ import annotations

import numpy as np

from aiijc_puzzle.socket_cycle_diagnostic import (
    axis_socket_rankings,
    commutative_cycle_support,
)
from aiijc_puzzle.socket_decoder import SocketEdge


def _cycle_assignments() -> tuple[np.ndarray, np.ndarray]:
    grid = 3
    count = grid * grid
    right = np.full((count + 1, count + 1), -20.0)
    down = np.full_like(right, -20.0)
    right[-1, -1] = down[-1, -1] = -1e4
    # 0 -> 1 -> 4 and 0 -> 3 -> 4 form one exact commutative square.
    right[0, 1] = right[3, 4] = 4.0
    down[0, 3] = down[1, 4] = 4.0
    np.fill_diagonal(right, -1e4)
    np.fill_diagonal(down, -1e4)
    return right, down


def test_right_and_down_edges_find_the_same_commutative_square() -> None:
    right_matrix, down_matrix = _cycle_assignments()
    right = axis_socket_rankings(right_matrix, grid=3, maximum_k=2)
    down = axis_socket_rankings(down_matrix, grid=3, maximum_k=2)
    right_edge = SocketEdge(0, 1, 0, 1, 1.0, "right")
    down_edge = SocketEdge(0, 3, 1, 0, 1.0, "down")
    right_support = commutative_cycle_support(
        right_edge,
        right=right,
        down=down,
        top_k=1,
    )
    down_support = commutative_cycle_support(
        down_edge,
        right=right,
        down=down,
        top_k=1,
    )
    assert right_support.supported and down_support.supported
    assert right_support.support_count == down_support.support_count == 1
    assert right_support.best_total_rank == down_support.best_total_rank == 4


def test_candidate_outside_top_k_is_not_supported() -> None:
    right_matrix, down_matrix = _cycle_assignments()
    right_matrix[0, 2] = 5.0
    right = axis_socket_rankings(right_matrix, grid=3, maximum_k=2)
    down = axis_socket_rankings(down_matrix, grid=3, maximum_k=2)
    edge = SocketEdge(0, 1, 0, 1, 1.0, "right")
    top1 = commutative_cycle_support(edge, right=right, down=down, top_k=1)
    top2 = commutative_cycle_support(edge, right=right, down=down, top_k=2)
    assert not top1.supported
    assert top1.base_rank is None
    assert top2.supported


def test_missing_closing_edge_removes_top1_support() -> None:
    right_matrix, down_matrix = _cycle_assignments()
    right_matrix[3, 4] = -20.0
    right_matrix[3, 5] = 4.0
    right = axis_socket_rankings(right_matrix, grid=3, maximum_k=1)
    down = axis_socket_rankings(down_matrix, grid=3, maximum_k=1)
    edge = SocketEdge(0, 1, 0, 1, 1.0, "right")
    support = commutative_cycle_support(edge, right=right, down=down, top_k=1)
    assert not support.supported
