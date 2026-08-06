"""Focused contracts for the fail-closed E4 structure calibrator."""
from __future__ import annotations

import unittest

from calibrate_two_side_growth import (
    CALIBRATION_IMAGES,
    CalibrationConfig,
    aggregate_config_rows,
    build_config_grid,
    evaluate_seed_selection,
    pareto_frontier,
    select_passing_config,
    select_strict_tier_a_seeds,
)
from two_side_growth import Plaquette, true_plaquette_keys


def _motif(
    tiles: tuple[int, int, int, int],
    *,
    edge: float,
    rank_sum: int,
    mask: int = 0b1111,
) -> Plaquette:
    return Plaquette(
        tiles=tiles,
        corner_mask=mask,
        min_edge=edge,
        mean_edge=edge,
        reciprocal_rank_sum=rank_sum,
        witness_rank_sum=max(4, rank_sum // 2),
    )


class CalibrationContracts(unittest.TestCase):
    def test_calibration_split_is_fixed_and_exclusive(self) -> None:
        self.assertEqual(CALIBRATION_IMAGES, tuple(range(10, 18)))

    def test_grid_deduplicates_rank_cutoffs_capped_by_top_k(self) -> None:
        grid = build_config_grid([2], [-2.0], [1.0, 2.0, 3.0, float("inf")])
        # k=2 permits reciprocal rank sum at most 16, so mean 2/3/inf collapse.
        self.assertEqual(
            [item.maximum_reciprocal_rank_sum for item in grid], [8, 16]
        )

    def test_strict_seed_replay_applies_edge_rank_and_tier_a_cutoffs(self) -> None:
        strong_exact = _motif((0, 1, 3, 4), edge=-0.5, rank_sum=8)
        weak_wrong = _motif((2, 5, 6, 7), edge=-5.0, rank_sum=24)
        non_tier_a = _motif(
            (1, 2, 4, 5), edge=-0.1, rank_sum=8, mask=0b0011
        )
        strict = CalibrationConfig(8, -1.0, 8)
        selected = select_strict_tier_a_seeds(
            [weak_wrong, non_tier_a, strong_exact], 9, strict
        )
        self.assertEqual([motif.tiles for motif in selected], [(0, 1, 3, 4)])

    def test_metrics_show_precision_coverage_tradeoff_without_packing(self) -> None:
        truth = true_plaquette_keys(
            # identity tile->cell permutation for a 3x3 board
            __import__("numpy").arange(9),
            3,
        )
        exact = _motif((0, 1, 3, 4), edge=-0.5, rank_sum=8)
        wrong = _motif((2, 5, 6, 7), edge=-5.0, rank_sum=24)
        strict = evaluate_seed_selection(
            [exact, wrong], truth, 9, CalibrationConfig(8, -1.0, 8)
        )
        loose = evaluate_seed_selection(
            [exact, wrong], truth, 9, CalibrationConfig(8, -6.0, 32)
        )
        self.assertEqual(strict["precision"], 1.0)
        self.assertAlmostEqual(strict["seed_tile_coverage"], 4 / 9)
        self.assertEqual(loose["precision"], 0.5)
        self.assertAlmostEqual(loose["seed_tile_coverage"], 8 / 9)

    def test_fail_closed_when_no_point_passes_both_gates(self) -> None:
        config_a = CalibrationConfig(4, -2.0, 16)
        config_b = CalibrationConfig(8, -3.0, 24)
        row_a = aggregate_config_rows(
            config_a,
            [
                {
                    "accepted": 10.0,
                    "exact": 10.0,
                    "precision": 1.0,
                    "seed_tile_coverage": 0.10,
                    "accepted_true_seed_recall": 0.02,
                    "proposal_recall": 0.05,
                    "selection_seconds": 0.01,
                }
            ],
            shared_seconds=[1.0],
            minimum_precision=0.95,
            minimum_coverage=0.15,
        )
        row_b = aggregate_config_rows(
            config_b,
            [
                {
                    "accepted": 10.0,
                    "exact": 9.0,
                    "precision": 0.9,
                    "seed_tile_coverage": 0.20,
                    "accepted_true_seed_recall": 0.02,
                    "proposal_recall": 0.10,
                    "selection_seconds": 0.01,
                }
            ],
            shared_seconds=[2.0],
            minimum_precision=0.95,
            minimum_coverage=0.15,
        )
        self.assertIsNone(select_passing_config([row_a, row_b]))

    def test_passing_choice_and_pareto_are_deterministic(self) -> None:
        def row(
            key: str,
            precision: float,
            coverage: float,
            recall: float,
            seconds: float,
            passes: bool,
        ):
            return {
                "key": key,
                "config": {
                    "top_k": 8,
                    "minimum_edge": -2.0,
                    "maximum_reciprocal_rank_sum": 16,
                },
                "precision": precision,
                "mean_seed_tile_coverage": coverage,
                "mean_proposal_recall": recall,
                "estimated_mean_seconds": seconds,
                "passes_precision_coverage": passes,
            }

        high_coverage = row("a", 0.96, 0.20, 0.10, 2.0, True)
        high_precision = row("b", 0.99, 0.16, 0.10, 2.0, True)
        dominated = row("c", 0.90, 0.10, 0.05, 3.0, False)
        self.assertEqual(
            select_passing_config([high_precision, high_coverage])["key"], "a"
        )
        self.assertEqual(
            {item["key"] for item in pareto_frontier([dominated, high_coverage, high_precision])},
            {"a", "b"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
