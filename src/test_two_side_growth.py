"""Focused synthetic contracts for :mod:`two_side_growth`.

Run directly so the repository does not need a pytest dependency:
    python src/test_two_side_growth.py
"""
from __future__ import annotations

import unittest

import numpy as np

from placement_metrics import neighbour_accuracy, placement_accuracy
from two_side_growth import (
    DELTAS,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    DirectionalTopK,
    PotentialDSU,
    enumerate_plaquettes,
    grow_plaquettes,
    make_synthetic_plaquette,
    pack_components,
)


def _directional_rows(
    permutation: np.ndarray,
    grid_side: int,
    *,
    true_rank: int,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    """Dense tiny candidate rows with every real edge at a requested rank."""

    count = len(permutation)
    inverse = np.empty(count, dtype=np.int64)
    inverse[permutation] = np.arange(count, dtype=np.int64)
    candidate_ids = np.empty((count, count - 1), dtype=np.int64)
    scores = np.full((count, 4, count - 1), -10.0, dtype=np.float64)
    truth: dict[tuple[int, int], int] = {}
    for anchor in range(count):
        candidates = [tile for tile in range(count) if tile != anchor]
        candidate_ids[anchor] = candidates
        cell = int(permutation[anchor])
        row, col = divmod(cell, grid_side)
        for direction, (dr, dc) in enumerate(DELTAS):
            target_row, target_col = row + dr, col + dc
            # Stable non-truth ordering for boundary rows and lower ranks.
            for slot, target in enumerate(candidates):
                scores[anchor, direction, slot] = -float(slot + 1)
            if not (0 <= target_row < grid_side and 0 <= target_col < grid_side):
                continue
            target = int(inverse[target_row * grid_side + target_col])
            truth[(anchor, direction)] = target
            target_slot = candidates.index(target)
            if true_rank == 1:
                scores[anchor, direction, target_slot] = 20.0
            elif true_rank == 2:
                scores[anchor, direction, target_slot] = 19.0
                decoy = next(tile for tile in candidates if tile != target)
                scores[anchor, direction, candidates.index(decoy)] = 20.0
            else:
                raise ValueError("test helper supports true_rank 1 or 2")
    return candidate_ids, scores, truth


def _dummy_graph(count: int) -> DirectionalTopK:
    ids = np.asarray([[(anchor + 1) % count] for anchor in range(count)], dtype=np.int64)
    scores = np.zeros((count, 4, 1), dtype=np.float64)
    return DirectionalTopK.from_candidate_rows(ids, scores, top_k=1)


class EnumerationContracts(unittest.TestCase):
    def test_perfect_square_deduplicates_four_corner_witnesses(self) -> None:
        permutation = np.arange(4, dtype=np.int64)
        candidate_ids, scores, _ = _directional_rows(permutation, 2, true_rank=1)
        graph = DirectionalTopK.from_candidate_rows(candidate_ids, scores, top_k=1)
        motifs = enumerate_plaquettes(graph, max_per_elbow=16)
        exact = [motif for motif in motifs if motif.tiles == (0, 1, 2, 3)]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].corner_mask, 0b1111)
        self.assertTrue(exact[0].tier_a)

    def test_rank_two_square_survives_when_top_one_loop_is_wrong(self) -> None:
        permutation = np.arange(4, dtype=np.int64)
        candidate_ids, scores, truth = _directional_rows(
            permutation, 2, true_rank=2
        )
        for (anchor, direction), target in truth.items():
            best_slot = int(np.argmax(scores[anchor, direction]))
            self.assertNotEqual(int(candidate_ids[anchor, best_slot]), target)
        graph = DirectionalTopK.from_candidate_rows(candidate_ids, scores, top_k=2)
        motifs = enumerate_plaquettes(graph, max_per_elbow=32)
        exact = [motif for motif in motifs if motif.tiles == (0, 1, 2, 3)]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].corner_mask, 0b1111)

    def test_missing_diagonal_closure_is_not_emitted(self) -> None:
        candidate_ids = np.asarray([[1, 2], [0, 2], [0, 1], [0, 1]], dtype=np.int64)
        scores = np.zeros((4, 4, 2), dtype=np.float64)
        graph = DirectionalTopK.from_candidate_rows(candidate_ids, scores, top_k=2)
        motifs = enumerate_plaquettes(graph, max_per_elbow=32)
        self.assertNotIn((0, 1, 2, 3), {motif.tiles for motif in motifs})


class DSUContracts(unittest.TestCase):
    def test_adjacent_two_corner_motif_cannot_create_fresh_seed(self) -> None:
        weak = make_synthetic_plaquette((0, 1, 2, 3), corner_mask=0b0011)
        self.assertFalse(weak.tier_a)
        growth = grow_plaquettes(4, 2, [weak], growth_min_corners=2)
        self.assertEqual(len(growth.seed_motifs), 0)
        self.assertEqual(len(growth.growth_motifs), 0)
        self.assertEqual(len(growth.dsu.roots()), 4)

    def test_overlapping_plaquettes_grow_exact_two_by_three(self) -> None:
        motifs = [
            make_synthetic_plaquette((0, 1, 3, 4)),
            make_synthetic_plaquette((1, 2, 4, 5)),
        ]
        growth = grow_plaquettes(9, 3, motifs)
        root, _ = growth.dsu.find(0)
        self.assertEqual(len(growth.dsu.positions[root]), 6)
        expected = {
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (1, 0),
            4: (1, 1),
            5: (1, 2),
        }
        component = growth.dsu.positions[root]
        minimum_row = min(row for row, _ in component.values())
        minimum_col = min(col for _, col in component.values())
        normalised = {
            tile: (row - minimum_row, col - minimum_col)
            for tile, (row, col) in component.items()
        }
        self.assertEqual(normalised, expected)

    def test_block_merge_requires_and_accepts_two_distinct_seams(self) -> None:
        motifs = [
            make_synthetic_plaquette((0, 1, 4, 5)),
            make_synthetic_plaquette((8, 9, 12, 13)),
            # Two vertical cross-block seams: 4-8 and 5-9.
            make_synthetic_plaquette((4, 5, 8, 9)),
        ]
        growth = grow_plaquettes(16, 4, motifs)
        root_a, _ = growth.dsu.find(0)
        root_b, _ = growth.dsu.find(13)
        self.assertEqual(root_a, root_b)
        self.assertEqual(len(growth.dsu.positions[root_a]), 8)
        self.assertEqual(len(growth.growth_motifs), 1)

    def test_one_tile_overlap_cannot_grow_a_component(self) -> None:
        seed = make_synthetic_plaquette((0, 1, 4, 5))
        one_overlap = make_synthetic_plaquette((0, 6, 7, 8))
        growth = grow_plaquettes(16, 4, [seed, one_overlap])
        root_zero, _ = growth.dsu.find(0)
        root_six, _ = growth.dsu.find(6)
        self.assertNotEqual(root_zero, root_six)
        self.assertEqual(len(growth.growth_motifs), 0)

    def test_collision_and_span_rejections_are_transactional(self) -> None:
        first = make_synthetic_plaquette((0, 1, 3, 4))
        second = make_synthetic_plaquette((1, 2, 4, 5))
        growth = grow_plaquettes(9, 3, [first, second])
        before = growth.dsu.signature()
        # The requested bottom row is already occupied by tiles 3 and 4.
        collision = make_synthetic_plaquette((0, 1, 6, 7))
        result = growth.dsu.try_growth(collision, minimum_edge=-np.inf)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "collision")
        self.assertEqual(growth.dsu.signature(), before)

        small = PotentialDSU(6, 2)
        seed_result = small.try_seed(
            make_synthetic_plaquette((0, 1, 2, 3)), minimum_edge=-np.inf
        )
        self.assertTrue(seed_result.accepted)
        before = small.signature()
        span_result = small.try_growth(
            make_synthetic_plaquette((1, 4, 3, 5)), minimum_edge=-np.inf
        )
        self.assertFalse(span_result.accepted)
        self.assertEqual(span_result.reason, "span")
        self.assertEqual(small.signature(), before)

    def test_input_order_does_not_change_growth(self) -> None:
        motifs = [
            make_synthetic_plaquette((0, 1, 3, 4)),
            make_synthetic_plaquette((1, 2, 4, 5)),
            make_synthetic_plaquette((3, 4, 6, 7)),
            make_synthetic_plaquette((4, 5, 7, 8)),
        ]
        forward = grow_plaquettes(9, 3, motifs)
        backward = grow_plaquettes(9, 3, list(reversed(motifs)))
        self.assertEqual(forward.dsu.signature(), backward.dsu.signature())


class PackingAndPermutationContracts(unittest.TestCase):
    def test_fragmented_completion_is_deterministic_strict_permutation(self) -> None:
        motif = make_synthetic_plaquette((0, 1, 3, 4))
        growth = grow_plaquettes(9, 3, [motif])
        graph = _dummy_graph(9)
        first = pack_components(growth.dsu, graph)
        second = pack_components(growth.dsu, graph)
        self.assertTrue(np.array_equal(first.placement, second.placement))
        self.assertTrue(np.array_equal(np.sort(first.placement), np.arange(9)))
        self.assertEqual(first.board.shape, (3, 3))

    def test_shuffled_tile_ids_recover_perfect_24_by_24_board(self) -> None:
        side = 24
        count = side * side
        # Multiplication by 37 permutes 0..575 (gcd(37,576)=1).  Tile ids have
        # no relation to board boundaries, guarding the fixed I11 bug.
        permutation = (np.arange(count, dtype=np.int64) * 37) % count
        truth_board = np.argsort(permutation)
        dsu = PotentialDSU(count, side)

        def motif_at(row: int, col: int):
            cell = row * side + col
            return make_synthetic_plaquette(
                (
                    int(truth_board[cell]),
                    int(truth_board[cell + 1]),
                    int(truth_board[cell + side]),
                    int(truth_board[cell + side + 1]),
                )
            )

        first = dsu.try_seed(motif_at(0, 0), minimum_edge=-np.inf)
        self.assertTrue(first.accepted)
        for row in range(side - 1):
            for col in range(side - 1):
                if row == 0 and col == 0:
                    continue
                result = dsu.try_growth(motif_at(row, col), minimum_edge=-np.inf)
                self.assertTrue(result.accepted, (row, col, result))
        root, _ = dsu.find(int(truth_board[0]))
        self.assertEqual(len(dsu.positions[root]), count)

        packed = pack_components(dsu, _dummy_graph(count))
        self.assertTrue(np.array_equal(packed.placement, truth_board))
        place, _ = placement_accuracy(packed.placement, truth_board)
        neighbour, right, down = neighbour_accuracy(packed.placement, truth_board)
        self.assertEqual((place, neighbour, right, down), (1.0, 1.0, 1.0, 1.0))

    def test_direction_constants_match_repository_contract(self) -> None:
        self.assertEqual((UP, DOWN, LEFT, RIGHT), (0, 1, 2, 3))
        self.assertEqual(
            tuple(DELTAS), ((-1, 0), (1, 0), (0, -1), (0, 1))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
