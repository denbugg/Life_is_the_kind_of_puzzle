"""Regression tests for the reviewable E14 Kaggle port."""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import kaggle_e14_solver as port


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference_fusion = load_module(
    "e14_reference_fusion",
    REPO / "autoresearch-runs" / "e14-fusion-relaxation" / "e2_raw_fusion.py",
)
reference_solver = load_module(
    "e14_reference_solver", REPO / "global_solver_candidate.py"
)


def cache_path() -> Path:
    candidates = [
        Path(os.environ.get("E14_TEST_CACHE", "")),
        REPO / "outputs" / "directional_student_holdout128.npz",
        Path("/Users/fenix/Desktop/ML/ai_challenge_pazzle_predicting/outputs/"
             "directional_student_holdout128.npz"),
    ]
    return next((path for path in candidates if str(path) and path.is_file()), candidates[-1])


class E14CacheParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = cache_path()
        if not path.is_file():
            raise unittest.SkipTest("frozen holdout cache is unavailable")
        cls.data = np.load(path, mmap_mode="r")
        cls.tiles = np.asarray(cls.data["tiles"][0])
        cls.learned_right = np.asarray(cls.data["right"][0])
        cls.learned_down = np.asarray(cls.data["down"][0])
        cls.pos = np.asarray(cls.data["pos"][0])

    def test_raw_classical_scores_are_bit_identical(self):
        expected = reference_fusion.classical_mgc_ssd_scores(self.tiles)
        actual = port.classical_mgc_ssd_scores(self.tiles)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(np.all(np.diag(actual[0]) < -9999))
        self.assertTrue(np.all(np.diag(actual[1]) < -9999))

    def test_fusion_and_global_layout_are_bit_identical(self):
        classical = reference_fusion.classical_mgc_ssd_scores(self.tiles)
        expected_right = reference_fusion.fuse_scores(
            self.learned_right, classical[0]
        )
        expected_down = reference_fusion.fuse_scores(
            self.learned_down, classical[1]
        )
        actual_right, actual_down = port.fused_directional_scores(
            self.tiles, self.learned_right, self.learned_down, learned_are_logp=True
        )
        np.testing.assert_array_equal(actual_right, expected_right)
        np.testing.assert_array_equal(actual_down, expected_down)
        seed = 20260818
        expected_layout = reference_solver.solve_layout(
            expected_right, expected_down, self.pos, seed
        )
        actual_layout = port.solve_layout(
            actual_right, actual_down, self.pos, seed
        )
        np.testing.assert_array_equal(actual_layout, expected_layout)
        self.assertTrue(port.is_valid_layout(actual_layout))

    def test_alpha_is_frozen(self):
        with self.assertRaisesRegex(ValueError, "locks alpha=0.2"):
            port.fuse_scores(self.learned_right, self.learned_right, alpha=0.3)


class E14ProductionFallbackTest(unittest.TestCase):
    def test_solver_error_preserves_legacy_layout(self):
        # Importing the production module does not load checkpoints or access data.
        tqdm_package = types.ModuleType("tqdm")
        tqdm_auto = types.ModuleType("tqdm.auto")
        tqdm_auto.tqdm = lambda iterable, **_kwargs: iterable
        with mock.patch.dict(
            sys.modules, {"tqdm": tqdm_package, "tqdm.auto": tqdm_auto}
        ):
            import kaggle_solve_puzzles as production

        fallback = np.arange(port.N, dtype=np.int32)
        raw = np.zeros((port.N, port.TILE, port.TILE, 3), dtype=np.uint8)
        scores = np.zeros((port.N, port.N), dtype=np.float32)
        with mock.patch.object(
            production.kaggle_e14_solver,
            "fused_directional_scores",
            side_effect=RuntimeError("synthetic failure"),
        ):
            selected, used, reason = production.select_e14_or_fallback(
                raw, scores, scores, scores, 7, fallback
            )
        np.testing.assert_array_equal(selected, fallback)
        self.assertFalse(used)
        self.assertEqual(reason, "RuntimeError: synthetic failure")


if __name__ == "__main__":
    unittest.main()
