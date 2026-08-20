from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(HERE)]

import kaggle_e14_solver as e14
from e20_common import candidate_union, topk_high
from e20_verifier import robust_row_z, verified_scores


class CountingRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.items = 0

    def forward(self, features):
        self.items += len(features)
        return features.mean((1, 2, 3))


class E20Tests(unittest.TestCase):
    def test_robust_z_is_clipped_and_median_centered(self):
        z = robust_row_z(np.asarray([-100, 0, 1, 2, 100], np.float32))
        self.assertEqual(float(z.min()), -4.0)
        self.assertEqual(float(z.max()), 4.0)
        self.assertEqual(float(np.median(z)), 0.0)

    def test_topk_excludes_self(self):
        scores = np.zeros((e14.N, e14.N), np.float32)
        np.fill_diagonal(scores, 1e6)
        ids = topk_high(scores)
        self.assertFalse(np.any(ids == np.arange(e14.N)[:, None]))
        self.assertTrue(np.all(np.asarray([len(np.unique(row)) for row in ids]) == 32))

    def test_union_is_self_excluded_and_bounded(self):
        rng = np.random.default_rng(20)
        scores = rng.normal(size=(e14.N, e14.N)).astype(np.float32)
        restored = rng.integers(0, 256, size=(e14.N, 20, 20, 3), dtype=np.uint8)
        unions, _, _ = candidate_union(scores, restored, 0)
        self.assertTrue(all(row not in ids for row, ids in enumerate(unions)))
        self.assertTrue(all(32 <= len(ids) <= 64 for ids in unions))

    def test_ranker_only_scores_union_and_bad_anchor_gets_no_bonus(self):
        rng = np.random.default_rng(21)
        scores = rng.normal(size=(e14.N, e14.N)).astype(np.float32)
        np.fill_diagonal(scores, -1e4)
        restored = rng.integers(0, 256, size=(e14.N, 20, 20, 3), dtype=np.uint8)
        good = np.ones(e14.N, np.bool_)
        good[0] = False
        model = CountingRanker()
        output, stats = verified_scores(
            model, restored, good, scores, 0, torch.device("cpu"), batch_size=2048
        )
        self.assertEqual(model.items, stats["candidate_pairs"])
        np.testing.assert_array_equal(output[0], scores[0])
        self.assertTrue(np.all(np.diag(output) == -1e4))


if __name__ == "__main__":
    unittest.main()
