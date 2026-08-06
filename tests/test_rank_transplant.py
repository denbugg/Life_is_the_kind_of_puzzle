from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rank_transplant import (  # noqa: E402
    INVERSE_DIRECTION,
    LEFT,
    RIGHT,
    ReciprocalPair,
    assert_disjoint_phases,
    confidence_gated_rank_transplant,
    fused_donor_scores,
    reciprocal_physical_pairs,
    row_predictions,
    select_trusted_pairs,
    transplant_raw_logits,
)


def _scores_for_targets(
    candidates: np.ndarray,
    targets: np.ndarray,
    margins: np.ndarray,
) -> np.ndarray:
    directions, count = targets.shape
    scores = np.empty((directions, count, candidates.shape[1]), dtype=np.float32)
    for direction in range(directions):
        for anchor in range(count):
            slots = np.flatnonzero(candidates[anchor] == targets[direction, anchor])
            if len(slots) != 1:
                raise AssertionError("fixture target is absent")
            top = int(slots[0])
            second = (top + 1) % candidates.shape[1]
            row = np.full(candidates.shape[1], -2.0, dtype=np.float32)
            row[second] = 0.0
            row[top] = float(margins[direction, anchor])
            scores[direction, anchor] = row
    return scores


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(
        (
            (1, 2, 3),
            (0, 2, 3),
            (0, 1, 3),
            (0, 1, 2),
        ),
        dtype=np.int64,
    )
    # Base rows prefer the tile two steps around the four-tile cycle.
    base_targets = np.asarray(
        [[(anchor + 2) % 4 for anchor in range(4)] for _ in range(4)],
        dtype=np.int64,
    )
    base = _scores_for_targets(candidates, base_targets, np.full((4, 4), 7.0))

    # Default donor rows make a directed four-cycle (no reciprocal edges).
    donor_targets = np.asarray(
        [[(anchor + 1) % 4 for anchor in range(4)] for _ in range(4)],
        dtype=np.int64,
    )
    margins = np.full((4, 4), 1.0, dtype=np.float32)
    # Two reciprocal physical pairs with different weaker-side confidence.
    donor_targets[RIGHT, 0] = 1
    donor_targets[LEFT, 1] = 0
    margins[RIGHT, 0] = 5.0
    margins[LEFT, 1] = 4.0
    donor_targets[RIGHT, 2] = 3
    donor_targets[LEFT, 3] = 2
    margins[RIGHT, 2] = 3.0
    margins[LEFT, 3] = 2.5
    donor = _scores_for_targets(candidates, donor_targets, margins)
    return candidates, base, donor


class RankTransplantContracts(unittest.TestCase):
    def test_reciprocal_pairs_are_physical_unique_and_deterministic(self) -> None:
        candidates, base, donor = _fixture()
        first = reciprocal_physical_pairs(candidates, donor, base_scores=base)
        second = reciprocal_physical_pairs(candidates, donor, base_scores=base)
        self.assertEqual(first, second)
        self.assertEqual([(pair.anchor, pair.target) for pair in first], [(0, 1), (2, 3)])
        self.assertGreater(first[0].confidence, first[1].confidence)
        for pair in first:
            self.assertEqual(pair.reverse_direction, INVERSE_DIRECTION[pair.direction])
            self.assertGreater(pair.changed_rows, 0)

    def test_top_m_rank_swap_preserves_exact_masks_and_value_multisets(self) -> None:
        candidates, base, donor = _fixture()
        frozen_base = base.copy()
        result = confidence_gated_rank_transplant(
            candidates,
            base,
            donor,
            top_m=1,
            verify=True,
        )
        self.assertTrue(np.array_equal(base, frozen_base), "input base was mutated")
        self.assertEqual(len(result.eligible_pairs), 2)
        self.assertEqual(len(result.selected_pairs), 1)
        self.assertEqual(result.changed_row_count, 2)
        self.assertTrue(np.array_equal(np.isfinite(base), np.isfinite(result.scores)))
        self.assertTrue(
            np.array_equal(np.sort(base, axis=-1), np.sort(result.scores, axis=-1))
        )
        before, _, _ = row_predictions(candidates, base)
        after, _, _ = row_predictions(candidates, result.scores)
        self.assertEqual(int(after[RIGHT, 0]), 1)
        self.assertEqual(int(after[LEFT, 1]), 0)
        unchanged = np.ones(before.shape, dtype=bool)
        unchanged[RIGHT, 0] = False
        unchanged[LEFT, 1] = False
        self.assertTrue(np.array_equal(before[unchanged], after[unchanged]))

    def test_zero_budget_is_bit_exact_identity(self) -> None:
        candidates, base, donor = _fixture()
        result = confidence_gated_rank_transplant(candidates, base, donor, top_m=0)
        self.assertEqual(result.selected_pairs, ())
        self.assertEqual(result.swapped_rows, ())
        self.assertTrue(np.array_equal(result.scores, base))

    def test_confidence_threshold_and_ties_are_deterministic(self) -> None:
        candidates, base, donor = _fixture()
        pairs = reciprocal_physical_pairs(candidates, donor, base_scores=base)
        selected = select_trusted_pairs(
            pairs,
            top_m=4,
            min_confidence=(pairs[0].confidence + pairs[1].confidence) / 2.0,
        )
        self.assertEqual(selected, (pairs[0],))
        tied = (
            ReciprocalPair(2, RIGHT, 3, LEFT, 2.0, 2.0, 2.0, 2),
            ReciprocalPair(0, RIGHT, 1, LEFT, 2.0, 2.0, 2.0, 2),
        )
        self.assertEqual(select_trusted_pairs(tied, top_m=1)[0].anchor, 0)

    def test_fused_donor_keeps_base_mask_and_has_comparable_rows(self) -> None:
        candidates, base, _ = _fixture()
        base = base.copy()
        base[0, 0, 2] = -np.inf
        spatial = np.where(np.isfinite(base), base * -0.3 + 0.7, -np.inf).astype(np.float32)
        fused_one = fused_donor_scores(base, spatial, alpha=0.75)
        fused_two = fused_donor_scores(base, spatial, alpha=0.75)
        self.assertTrue(np.array_equal(np.isfinite(fused_one), np.isfinite(base)))
        self.assertTrue(np.array_equal(fused_one, fused_two))
        for direction, anchor in np.argwhere(np.isfinite(base).any(axis=-1)):
            row = fused_one[direction, anchor]
            finite = np.isfinite(row)
            self.assertAlmostEqual(float(row[finite].mean()), 0.0, places=5)
            if finite.sum() > 1:
                self.assertAlmostEqual(float(row[finite].std()), 1.0, places=5)

    def test_invalid_reverse_direction_contract_is_rejected(self) -> None:
        candidates, base, _ = _fixture()
        invalid = ReciprocalPair(0, RIGHT, 1, RIGHT, 1.0, 1.0, 1.0, 2)
        with self.assertRaisesRegex(ValueError, "reverse direction"):
            transplant_raw_logits(candidates, base, (invalid,))

    def test_calibration_and_confirmation_must_be_disjoint(self) -> None:
        assert_disjoint_phases((10, 11), (100, 101))
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_phases((10, 11), (11, 12))


class CalibrationSelectionContracts(unittest.TestCase):
    def test_selection_uses_precision_gate_then_primary_ssim(self) -> None:
        # Importing here keeps the pure rank-transplant tests independent of
        # evaluator/model imports.
        from eval_rank_transplant import select_calibration_configuration

        summaries = {
            "high_neighbor_low_ssim": {
                "alpha": 0.5,
                "top_m": 8,
                "metrics": {"trusted_pairs_total": 8.0, "trusted_pair_precision": 0.9},
                "delta": {"solve_only_ssim": 0.001, "neighbour": 0.02, "edge_r1": 0.02},
            },
            "best_primary": {
                "alpha": 0.75,
                "top_m": 16,
                "metrics": {"trusted_pairs_total": 16.0, "trusted_pair_precision": 0.875},
                "delta": {"solve_only_ssim": 0.003, "neighbour": 0.01, "edge_r1": 0.01},
            },
            "fails_precision": {
                "alpha": 1.0,
                "top_m": 32,
                "metrics": {"trusted_pairs_total": 32.0, "trusted_pair_precision": 0.8},
                "delta": {"solve_only_ssim": 0.5, "neighbour": 0.5, "edge_r1": 0.5},
            },
        }
        selected = select_calibration_configuration(
            summaries,
            minimum_trusted_precision=0.85,
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected[0], "best_primary")


if __name__ == "__main__":
    unittest.main()
