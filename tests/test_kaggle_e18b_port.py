"""Regression and fail-safe tests for the reviewable E18b Kaggle port."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
E14_DIR = REPO / "autoresearch-runs" / "e14-fusion-relaxation"
E18_DIR = REPO / "autoresearch-runs" / "e18-nlm-polish"
sys.path[:0] = [str(REPO), str(E14_DIR), str(E18_DIR)]

import evaluate_e18 as reference
import kaggle_e18b_postprocess as port
from e2_raw_fusion import classical_mgc_ssd_scores, fuse_scores
from global_solver_candidate import solve_layout


def cache_path() -> Path:
    candidates = [
        Path(os.environ.get("E14_TEST_CACHE", "")),
        REPO / "outputs" / "directional_student_holdout128.npz",
        Path("/Users/fenix/Desktop/ML/ai_challenge_pazzle_predicting/outputs/"
             "directional_student_holdout128.npz"),
    ]
    return next((path for path in candidates if str(path) and path.is_file()), candidates[-1])


class E18bExactParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = cache_path()
        if not path.is_file():
            raise unittest.SkipTest("frozen holdout cache is unavailable")
        cls.data = np.load(path, mmap_mode="r")
        tiles = np.asarray(cls.data["tiles"][0])
        classical_right, classical_down = classical_mgc_ssd_scores(tiles)
        right = fuse_scores(np.asarray(cls.data["right"][0]), classical_right)
        down = fuse_scores(np.asarray(cls.data["down"][0]), classical_down)
        cls.layout = np.asarray(solve_layout(
            right, down, np.asarray(cls.data["pos"][0]), 20260818
        ), np.int32)
        cls.raw = reference.assemble(tiles, cls.layout)

    def test_pixels_are_bit_identical_to_evaluator(self) -> None:
        expected_unguarded = reference.nlm_h9(self.raw)
        expected, expected_reverted = reference.no_gray_guard(
            self.raw, expected_unguarded
        )
        actual, stats = port.no_gray_nlm_h9(self.raw)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(stats["reverted_new_gray_cells"], expected_reverted)
        self.assertEqual(stats["raw_gray_count"], reference.gray_count(self.raw))
        self.assertEqual(stats["unguarded_gray_count"], reference.gray_count(expected_unguarded))
        self.assertEqual(stats["guarded_gray_count"], reference.gray_count(expected))

    def test_layout_identity_and_no_gray_gate(self) -> None:
        before = self.layout.copy()
        output, used, reason, stats = port.polish_or_raw(self.raw)
        np.testing.assert_array_equal(self.layout, before)
        self.assertTrue(used)
        self.assertIsNone(reason)
        self.assertLessEqual(port.gray_count(output), port.gray_count(self.raw))
        self.assertLessEqual(stats["guarded_gray_count"], stats["raw_gray_count"])


class E18bFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(18)
        self.raw = rng.integers(
            0, 256, (port.IMAGE_SIZE, port.IMAGE_SIZE, 3), dtype=np.uint8
        )

    def test_cv2_error_returns_bit_identical_raw(self) -> None:
        class BrokenCv2:
            COLOR_RGB2BGR = 1

            @staticmethod
            def cvtColor(*_args, **_kwargs):
                raise RuntimeError("synthetic cv2 failure")

        output, used, reason, stats = port.polish_or_raw(
            self.raw, cv2_module=BrokenCv2()
        )
        np.testing.assert_array_equal(output, self.raw)
        self.assertFalse(used)
        self.assertEqual(reason, "RuntimeError: synthetic cv2 failure")
        self.assertEqual(stats["guarded_gray_count"], stats["raw_gray_count"])

    def test_disabled_returns_bit_identical_raw(self) -> None:
        output, used, reason, _ = port.polish_or_raw(self.raw, enabled=False)
        np.testing.assert_array_equal(output, self.raw)
        self.assertFalse(used)
        self.assertEqual(reason, "disabled")

    def test_invalid_input_can_be_made_fatal_for_debugging(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected"):
            port.polish_or_raw(self.raw[:, :-1], fallback_on_error=False)


if __name__ == "__main__":
    unittest.main()
