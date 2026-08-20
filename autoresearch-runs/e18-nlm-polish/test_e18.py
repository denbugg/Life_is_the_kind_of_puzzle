"""Small deterministic checks for the E18 no-gray guard."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_e18 import GRID, TILE, gray_count, no_gray_guard


class NoGrayGuardTest(unittest.TestCase):
    def test_reverts_only_new_gray_cell(self) -> None:
        rng = np.random.default_rng(18)
        raw = rng.integers(0, 256, (GRID * TILE, GRID * TILE, 3), dtype=np.uint8)
        filtered = raw.copy()
        filtered[:TILE, :TILE] = 127
        guarded, reverted = no_gray_guard(raw, filtered)
        self.assertEqual(reverted, 1)
        np.testing.assert_array_equal(guarded[:TILE, :TILE], raw[:TILE, :TILE])
        np.testing.assert_array_equal(guarded[TILE:, TILE:], filtered[TILE:, TILE:])
        self.assertLessEqual(gray_count(guarded), gray_count(raw))

    def test_keeps_non_gray_filtering(self) -> None:
        rng = np.random.default_rng(180)
        raw = rng.integers(0, 256, (GRID * TILE, GRID * TILE, 3), dtype=np.uint8)
        filtered = np.clip(raw.astype(np.int16) + 1, 0, 255).astype(np.uint8)
        guarded, reverted = no_gray_guard(raw, filtered)
        self.assertEqual(reverted, 0)
        np.testing.assert_array_equal(guarded, filtered)


if __name__ == "__main__":
    unittest.main()
