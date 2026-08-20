from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(HERE)]

import kaggle_e14_solver as e14
from build_restorer_sidecar import no_gray_mask
from e15_multiplex_solver import _multiplex_support


class E15Tests(unittest.TestCase):
    def test_identical_layers_reduce_to_e14_support(self):
        rng = np.random.default_rng(15)
        scores = rng.normal(size=(e14.N, e14.N)).astype(np.float32)
        np.fill_diagonal(scores, -1e4)
        graph = (*e14._topk_compatibility(scores, e14.TOP_K_EDGES),) * 2
        beliefs = rng.uniform(size=(e14.N, e14.N))
        beliefs /= beliefs.sum(axis=1, keepdims=True)
        expected = e14._directional_support(beliefs, *graph)
        actual = _multiplex_support(beliefs, graph, graph)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-16)

    def test_guard_reverts_gray_collapse_and_keeps_normal_output(self):
        raw = np.empty((2, 20, 20, 3), np.uint8)
        raw[0, ..., 0] = 210
        raw[0, ..., 1] = 40
        raw[0, ..., 2] = 20
        raw[1] = np.arange(20, dtype=np.uint8)[None, :, None] * 8
        collapsed = raw.copy()
        collapsed[0] = 128
        mask = no_gray_mask(raw, collapsed)
        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(mask[1]))


if __name__ == "__main__":
    unittest.main()
