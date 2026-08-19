import ast
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE = Path(__file__).with_name("kaggle_solve_puzzles.py")


def load_functions(*names, globals_dict=None):
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "np": np,
        "os": os,
        "Path": Path,
        "Image": Image,
        "zipfile": zipfile,
        "GRID": 24,
        "TILE": 20,
        "N": 24 * 24,
        "POSITION_WEIGHT": 0.12,
        "SWAP_STEPS": 100,
        "GREEDY_TOPK": 4,
        "RELATION_MAX_SWAPS": 64,
        "RELATION_GUARD_WEIGHT": 0.25,
        "RELATION_MIN_GAIN": 0.75,
        "RELATION_BASE_TOL": 0.05,
        "RELATION_POSITION_TOL": 0.25,
        "linear_sum_assignment": lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("assignment fallback was not expected in this test")
        ),
    }
    namespace.update(globals_dict or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class SolverRegressionTests(unittest.TestCase):
    def test_relation_guard_accepts_strong_safe_local_repair(self):
        ns = load_functions(
            "local_value", "relation_local_value", "relation_guarded_refine"
        )
        n = ns["N"]
        layout = np.arange(n, dtype=np.int32)
        layout[1], layout[2] = layout[2], layout[1]
        zeros = np.zeros((n, n), dtype=np.float32)
        relation_right = zeros.copy()
        relation_right[0, 1] = 5.0
        result, stats = ns["relation_guarded_refine"](
            layout, zeros, zeros, zeros, relation_right, zeros
        )
        self.assertEqual(result[1], 1)
        self.assertEqual(stats["accepted"], 1)

    def test_relation_guard_rejects_weak_edge(self):
        ns = load_functions(
            "local_value", "relation_local_value", "relation_guarded_refine"
        )
        n = ns["N"]
        layout = np.arange(n, dtype=np.int32)
        layout[1], layout[2] = layout[2], layout[1]
        zeros = np.zeros((n, n), dtype=np.float32)
        relation_right = zeros.copy()
        relation_right[0, 1] = 0.5
        result, stats = ns["relation_guarded_refine"](
            layout, zeros, zeros, zeros, relation_right, zeros
        )
        self.assertTrue(np.array_equal(result, layout))
        self.assertEqual(stats["accepted"], 0)

    def test_greedy_graph_recovers_consistent_grid(self):
        ns = load_functions("greedy_graph_layout")
        n, grid = ns["N"], ns["GRID"]
        right = np.full((n, n), -10.0, dtype=np.float32)
        down = np.full((n, n), -10.0, dtype=np.float32)
        for tile in range(n):
            if tile % grid + 1 < grid:
                right[tile, tile + 1] = 10.0
            if tile + grid < n:
                down[tile, tile + grid] = 10.0
        pos = np.zeros((n, n), dtype=np.float32)
        pos[np.arange(n), np.arange(n)] = 1.0
        layout, stats = ns["greedy_graph_layout"](right, down, pos)
        self.assertTrue(np.array_equal(layout, np.arange(n)))
        self.assertEqual(stats["largest"], n)

    def test_optimizer_preserves_best_so_far(self):
        ns = load_functions("local_value", "layout_objective", "optimize_layout")
        n = ns["N"]
        layout = np.arange(n, dtype=np.int32)
        right = np.zeros((n, n), dtype=np.float32)
        down = np.zeros((n, n), dtype=np.float32)
        pos = np.zeros((n, n), dtype=np.float32)
        pos[np.arange(n), np.arange(n)] = 1.0
        result = ns["optimize_layout"](layout, right, down, pos, seed=0)
        self.assertTrue(np.array_equal(result, layout))

    def test_rl_candidate_cannot_reduce_baseline_objective(self):
        ns = load_functions("layout_objective", "select_layout_candidate")
        n, grid = ns["N"], ns["GRID"]
        baseline = np.arange(n, dtype=np.int32)
        candidate = baseline.copy()
        candidate[0], candidate[-1] = candidate[-1], candidate[0]
        right = np.full((n, n), -1.0, dtype=np.float32)
        down = np.full((n, n), -1.0, dtype=np.float32)
        for tile in range(n):
            if tile % grid + 1 < grid:
                right[tile, tile + 1] = 1.0
            if tile + grid < n:
                down[tile, tile + grid] = 1.0
        pos = np.zeros((n, n), dtype=np.float32)
        selected, baseline_value, candidate_value, accepted = ns["select_layout_candidate"](
            baseline, candidate, right, down, pos
        )
        self.assertFalse(accepted)
        self.assertGreater(baseline_value, candidate_value)
        self.assertTrue(np.array_equal(selected, baseline))

    def test_load_tiles_rejects_wrong_geometry(self):
        ns = load_functions("load_tiles")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.png"
            Image.fromarray(np.zeros((240, 960, 3), dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "Expected image shape"):
                ns["load_tiles"](path)

    def test_checkpoint_resolver_stays_inside_root(self):
        ns = load_functions("find_latest")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "selected"
            other = Path(directory) / "other"
            root.mkdir()
            other.mkdir()
            (root / "model_epoch2.pt").touch()
            (root / "model_epoch7.pt").touch()
            (other / "model_epoch99.pt").touch()
            result = ns["find_latest"](root, "model_epoch*.pt")
            self.assertEqual(result.name, "model_epoch7.pt")

    def test_model_root_requires_complete_component_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "incomplete"
            complete = root / "complete"
            incomplete.mkdir()
            complete.mkdir()
            (incomplete / "edge_matcher_epoch9.pt").touch()
            (complete / "edge_matcher_epoch2.pt").touch()
            (complete / "position_prior_epoch2.pt").touch()
            ns = load_functions(
                "resolve_model_root",
                globals_dict={"Path": Path},
            )
            # Use the complete root as the configured fallback; it must not mix with incomplete.
            result = ns["resolve_model_root"](
                complete,
                ["edge_matcher_epoch*.pt", "position_prior_epoch*.pt"],
                "assembly",
            )
            self.assertEqual(result, complete)

    def test_submission_zip_contains_only_current_files(self):
        ns = load_functions("finalize_submission_zip")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current.png"
            stale = root / "stale.png"
            current.write_bytes(b"current")
            stale.write_bytes(b"stale")
            output = root / "submission.zip"
            archived = ns["finalize_submission_zip"]([current], [current.name], output)
            self.assertEqual(archived, {"current.png"})
            with zipfile.ZipFile(output) as zf:
                self.assertEqual(zf.namelist(), ["current.png"])

    def test_rl_normalization_uses_population_std(self):
        text = SOURCE.read_text()
        self.assertIn("heuristic.std(correction=0)", text)


if __name__ == "__main__":
    unittest.main()
