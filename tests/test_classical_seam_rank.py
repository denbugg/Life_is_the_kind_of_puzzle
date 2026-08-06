from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classical_seam_rank import (  # noqa: E402
    DEPTHS,
    DOWN,
    LEFT,
    PREDECLARED_VARIANT_ORDER,
    RIGHT,
    UP,
    VARIANT_NAMES,
    compute_classical_candidate_scores,
    directional_border_lines,
    variant_index,
)


def _candidates(count: int) -> np.ndarray:
    return np.asarray(
        [[target for target in range(count) if target != anchor] for anchor in range(count)],
        dtype=np.int64,
    )


class DirectionAndShapeContracts(unittest.TestCase):
    def test_depth_one_indices_follow_udlr_without_rotation(self) -> None:
        values = np.zeros((1, 5, 5, 3), dtype=np.float32)
        for row in range(5):
            for column in range(5):
                values[0, row, column] = row * 10 + column

        right = directional_border_lines(values, RIGHT, 1)
        self.assertTrue(np.array_equal(right.source_boundary[0, :, 0], values[0, :, 3, 0]))
        self.assertTrue(np.array_equal(right.source_inner[0, :, 0], values[0, :, 2, 0]))
        self.assertTrue(np.array_equal(right.target_boundary[0, :, 0], values[0, :, 1, 0]))
        self.assertTrue(np.array_equal(right.target_inner[0, :, 0], values[0, :, 2, 0]))

        left = directional_border_lines(values, LEFT, 1)
        self.assertTrue(np.array_equal(left.source_boundary[0, :, 0], values[0, :, 1, 0]))
        self.assertTrue(np.array_equal(left.target_boundary[0, :, 0], values[0, :, 3, 0]))

        down = directional_border_lines(values, DOWN, 1)
        self.assertTrue(np.array_equal(down.source_boundary[0, :, 0], values[0, 3, :, 0]))
        self.assertTrue(np.array_equal(down.target_boundary[0, :, 0], values[0, 1, :, 0]))

        up = directional_border_lines(values, UP, 1)
        self.assertTrue(np.array_equal(up.source_boundary[0, :, 0], values[0, 1, :, 0]))
        self.assertTrue(np.array_equal(up.target_boundary[0, :, 0], values[0, 3, :, 0]))

    def test_all_variants_return_exact_shape_mask_and_deterministic_values(self) -> None:
        rng = np.random.default_rng(72)
        tiles = rng.integers(0, 256, size=(4, 6, 6, 3), dtype=np.uint8)
        candidates = _candidates(4)
        valid = np.ones((4, 4, 3), dtype=bool)
        valid[RIGHT, 0, 2] = False
        first = compute_classical_candidate_scores(tiles, candidates, valid)
        second = compute_classical_candidate_scores(tiles, candidates, valid)
        self.assertEqual(first.shape, (len(VARIANT_NAMES), 4, 4, 3))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.array_equal(first, second))
        expected_mask = np.broadcast_to(valid[None], first.shape)
        self.assertTrue(np.array_equal(np.isfinite(first), expected_mask))
        self.assertTrue(np.isneginf(first[:, RIGHT, 0, 2]).all())

    def test_fixed_orientation_scorer_never_rotates_target(self) -> None:
        tiles = np.zeros((3, 6, 6, 3), dtype=np.uint8)
        red = np.asarray((255, 0, 0), dtype=np.uint8)
        tiles[0, :, -1] = red
        tiles[1, :, 0] = red
        # Rotating tile 1 moves its matching left trace to a horizontal side.
        tiles[2] = np.rot90(tiles[1], 1, axes=(0, 1))
        candidates = _candidates(3)
        scores = compute_classical_candidate_scores(
            tiles,
            candidates,
            np.ones((4, 3, 2), dtype=bool),
        )
        lab_d0 = scores[variant_index("lab_ssd_d0")]
        slot_one = int(np.flatnonzero(candidates[0] == 1)[0])
        slot_rotated = int(np.flatnonzero(candidates[0] == 2)[0])
        self.assertGreater(lab_d0[RIGHT, 0, slot_one], lab_d0[RIGHT, 0, slot_rotated])
        slot_zero = int(np.flatnonzero(candidates[1] == 0)[0])
        slot_two = int(np.flatnonzero(candidates[1] == 2)[0])
        self.assertGreater(lab_d0[LEFT, 1, slot_zero], lab_d0[LEFT, 1, slot_two])

    def test_depth_zero_and_one_use_distinct_inward_traces(self) -> None:
        tiles = np.zeros((3, 6, 6, 3), dtype=np.uint8)
        red = np.asarray((255, 0, 0), dtype=np.uint8)
        green = np.asarray((0, 255, 0), dtype=np.uint8)
        blue = np.asarray((0, 0, 255), dtype=np.uint8)
        tiles[0, :, -1] = red
        tiles[0, :, -2] = green
        tiles[1, :, 0] = blue
        tiles[1, :, 1] = green
        tiles[2, :, 0] = red
        tiles[2, :, 1] = blue
        candidates = _candidates(3)
        scores = compute_classical_candidate_scores(
            tiles,
            candidates,
            np.ones((4, 3, 2), dtype=bool),
        )
        slot_one = int(np.flatnonzero(candidates[0] == 1)[0])
        slot_two = int(np.flatnonzero(candidates[0] == 2)[0])
        depth_zero = scores[variant_index("lab_ssd_d0"), RIGHT, 0]
        depth_one = scores[variant_index("lab_ssd_d1"), RIGHT, 0]
        self.assertGreater(depth_zero[slot_two], depth_zero[slot_one])
        self.assertGreater(depth_one[slot_one], depth_one[slot_two])

    def test_scorer_signature_has_no_labels_or_permutation(self) -> None:
        parameters = inspect.signature(compute_classical_candidate_scores).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("permutation", parameters)
        self.assertEqual(tuple(DEPTHS), (0, 1, 2))

    def test_invalid_floating_range_is_rejected(self) -> None:
        tiles = np.full((3, 6, 6, 3), 2.0, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            compute_classical_candidate_scores(
                tiles,
                _candidates(3),
                np.ones((4, 3, 2), dtype=bool),
            )


class CalibrationProtocolContracts(unittest.TestCase):
    @staticmethod
    def _variant(r1: float, precision: float, accepted: int = 256) -> dict[str, object]:
        return {
            "fixed_raw_plus_classical": {"all_true_edge_r1": r1},
            "trusted": {"precision": precision, "accepted_pairs": float(accepted)},
        }

    def test_variant_order_is_complete_unique_and_scientifically_predeclared(self) -> None:
        self.assertEqual(VARIANT_NAMES, PREDECLARED_VARIANT_ORDER)
        self.assertEqual(len(VARIANT_NAMES), 9)
        self.assertEqual(len(set(VARIANT_NAMES)), 9)
        for family in ("rgb_ssd", "lab_ssd", "mgc"):
            for depth in DEPTHS:
                self.assertIn(f"{family}_d{depth}", VARIANT_NAMES)

    def test_selection_takes_first_qualifier_not_best_metric(self) -> None:
        from eval_classical_seam_rank import choose_first_qualifying_variant

        variants = {
            name: self._variant(0.19, 0.69)
            for name in PREDECLARED_VARIANT_ORDER
        }
        first = PREDECLARED_VARIANT_ORDER[0]
        second = PREDECLARED_VARIANT_ORDER[1]
        last = PREDECLARED_VARIANT_ORDER[-1]
        variants[first] = self._variant(0.21, 0.69)  # precision fails
        variants[second] = self._variant(0.201, 0.701)  # first strict qualifier
        variants[last] = self._variant(0.9, 0.99)  # better, but later and ignored
        selected, audit = choose_first_qualifying_variant(
            {
                "raw": {
                    "all_true_edge_r1": 0.20,
                    "trusted": {"precision": 0.70},
                },
                "variants": variants,
            }
        )
        self.assertEqual(selected, second)
        self.assertEqual([row["variant"] for row in audit], [first, second])

    def test_selection_fails_closed_when_no_variant_improves_both(self) -> None:
        from eval_classical_seam_rank import choose_first_qualifying_variant

        selected, audit = choose_first_qualifying_variant(
            {
                "raw": {
                    "all_true_edge_r1": 0.20,
                    "trusted": {"precision": 0.70},
                },
                "variants": {
                    name: self._variant(0.20, 0.70)
                    for name in PREDECLARED_VARIANT_ORDER
                },
            }
        )
        self.assertIsNone(selected)
        self.assertEqual(len(audit), len(PREDECLARED_VARIANT_ORDER))

    def test_splits_and_replay_group_are_hard_coded_and_disjoint(self) -> None:
        from eval_classical_seam_rank import (
            CALIBRATION_IMAGE_IDS,
            CONFIRMATION_IMAGE_IDS,
            REPLAY_GROUP_COUNT,
            REPLAY_GROUP_START,
        )

        self.assertEqual(CALIBRATION_IMAGE_IDS, tuple(range(10, 18)))
        self.assertEqual(CONFIRMATION_IMAGE_IDS, tuple(range(18, 22)))
        self.assertFalse(set(CALIBRATION_IMAGE_IDS) & set(CONFIRMATION_IMAGE_IDS))
        self.assertEqual((REPLAY_GROUP_START, REPLAY_GROUP_COUNT), (10, 12))

    def test_feature_cache_replays_only_with_exact_provenance(self) -> None:
        from eval_classical_seam_rank import _load_or_compute_features

        rng = np.random.default_rng(9)
        tiles = rng.integers(0, 256, size=(3, 6, 6, 3), dtype=np.uint8)
        candidates = _candidates(3)
        valid = np.ones((4, 3, 2), dtype=bool)
        metadata = {
            "schema_version": "1",
            "image_id": "10",
            "source_cache_sha256": "cache",
            "tiles_sha256": "tiles",
            "candidates_sha256": "candidates",
            "valid_sha256": "valid",
            "mgc_ridge": "0.05",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            first = _load_or_compute_features(
                feature_path=path,
                metadata=metadata,
                tiles=tiles,
                candidates=candidates,
                valid=valid,
                force=False,
            )
            second = _load_or_compute_features(
                feature_path=path,
                metadata=metadata,
                tiles=tiles,
                candidates=candidates,
                valid=valid,
                force=False,
            )
            self.assertTrue(np.array_equal(first, second))
            changed = dict(metadata)
            changed["tiles_sha256"] = "different"
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                _load_or_compute_features(
                    feature_path=path,
                    metadata=changed,
                    tiles=tiles,
                    candidates=candidates,
                    valid=valid,
                    force=False,
                )


if __name__ == "__main__":
    unittest.main()
