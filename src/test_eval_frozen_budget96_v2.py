from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import eval_frozen_budget96_v2 as budget96_v2
import eval_frozen_end_to_end_gate as frozen


class FrozenBudget96V2Tests(unittest.TestCase):
    def test_exact_v2_root_and_only_two_fixed_arms(self) -> None:
        self.assertEqual(
            budget96_v2.EXPECTED_GATE_ROOT_SHA256,
            "7a5a5e68779a25fd8dc882062345a3e7b5e9e555da51dee97c5b5ca3e3558134",
        )
        self.assertEqual(budget96_v2.ARM_ORDER, ("budget_96", "budget_512"))
        self.assertEqual(set(budget96_v2.FIXED_ARMS), set(budget96_v2.ARM_ORDER))
        self.assertEqual(
            [budget96_v2.FIXED_ARMS[name]["max_edges"] for name in budget96_v2.ARM_ORDER],
            [96, 512],
        )
        for config in budget96_v2.FIXED_ARMS.values():
            self.assertEqual(config["min_margin"], 0.0)
            self.assertEqual(config["repair_passes"], 0)
            self.assertEqual(config["orientation"], "fixed_type1_no_rotation")
        left = dict(budget96_v2.FIXED_ARMS["budget_96"])
        right = dict(budget96_v2.FIXED_ARMS["budget_512"])
        left.pop("max_edges")
        right.pop("max_edges")
        self.assertEqual(left, right)

    def test_wrong_v1_root_is_rejected(self) -> None:
        v1_root = "ee3d74662f5326fbd1069763fd7b96dc3adb41bde0117cba1d78ff067c6bf23d"
        budget96_v2._require_expected_root(budget96_v2.EXPECTED_GATE_ROOT_SHA256)
        with self.assertRaises(frozen.IntegrityError):
            budget96_v2._require_expected_root(v1_root)

    def test_only_exact_gate_v2_and_score_cache_v2_paths_are_accepted(self) -> None:
        paths = budget96_v2._default_paths()
        budget96_v2._require_exact_paths(paths["gate"], paths["score_cache"])
        with self.assertRaises(frozen.IntegrityError):
            budget96_v2._require_exact_paths(paths["gate"].with_name("gate_v1"), paths["score_cache"])
        with self.assertRaises(frozen.CacheContractError):
            budget96_v2._require_exact_paths(
                paths["gate"], paths["score_cache"].with_name("score_cache_v1")
            )

    def test_cli_exposes_no_experiment_parameter_controls(self) -> None:
        parser = budget96_v2._build_parser()
        options = {option for action in parser._actions for option in action.option_strings}
        self.assertTrue({"--preflight", "--report"} <= options)
        for forbidden in (
            "--gate-dir",
            "--score-cache-dir",
            "--budget",
            "--max-edges",
            "--min-margin",
            "--repair-passes",
            "--device",
            "--config",
            "--sweep",
            "--rotation",
        ):
            self.assertNotIn(forbidden, options)

    def test_solver_calls_differ_only_in_max_edges(self) -> None:
        right = np.zeros((frozen.NFRAG, frozen.NFRAG), dtype=np.float32)
        down = np.zeros_like(right)
        calls: list[dict[str, object]] = []

        def fake_solver(r: np.ndarray, d: np.ndarray, **kwargs: object):
            self.assertIs(r, right)
            self.assertIs(d, down)
            calls.append(dict(kwargs))
            return np.arange(frozen.NFRAG, dtype=np.int64), 1.25

        with mock.patch("solve_buddies.solve_buddies_from_scores", side_effect=fake_solver):
            budget96_v2._solve_dense(right, down, arm="budget_96")
            budget96_v2._solve_dense(right, down, arm="budget_512")
        self.assertEqual(
            calls,
            [
                {"max_edges": 96, "min_margin": 0.0, "repair_passes": 0},
                {"max_edges": 512, "min_margin": 0.0, "repair_passes": 0},
            ],
        )
        with self.assertRaises(ValueError):
            budget96_v2._solve_dense(right, down, arm="budget_128")

    def test_loaded_wrong_root_stops_before_hash_or_cache_checks(self) -> None:
        paths = budget96_v2._default_paths()
        paths["score_cache"].mkdir(parents=True, exist_ok=True)
        with (
            mock.patch.object(
                budget96_v2.frozen,
                "load_and_verify_gate",
                return_value=({"scenes": []}, {}, "e" * 64),
            ),
            mock.patch.object(budget96_v2.frozen, "_verify_external_files") as verify_external,
            mock.patch.object(
                budget96_v2.frozen, "verify_score_cache_directory"
            ) as verify_cache,
        ):
            with self.assertRaises(frozen.IntegrityError):
                budget96_v2._load_verified_inputs(paths["gate"], paths["score_cache"])
        verify_external.assert_not_called()
        verify_cache.assert_not_called()

    def test_fixed_orientation_is_checked_before_external_files(self) -> None:
        paths = budget96_v2._default_paths()
        manifest = {
            "geometry": {"orientation": "rotation_enabled"},
            "scenes": [],
        }
        with (
            mock.patch.object(
                budget96_v2.frozen,
                "load_and_verify_gate",
                return_value=(manifest, {}, budget96_v2.EXPECTED_GATE_ROOT_SHA256),
            ),
            mock.patch.object(budget96_v2.frozen, "_verify_external_files") as verify_external,
        ):
            with self.assertRaises(frozen.IntegrityError):
                budget96_v2._load_verified_inputs(paths["gate"], paths["score_cache"])
        verify_external.assert_not_called()

    def test_report_is_create_once_or_byte_identical_with_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            digest, created = budget96_v2._write_immutable_report(path, {"answer": 96})
            self.assertTrue(created)
            original = path.read_bytes()
            original_mtime = path.stat().st_mtime_ns
            digest_again, created_again = budget96_v2._write_immutable_report(
                path, {"answer": 96}
            )
            self.assertEqual(digest_again, digest)
            self.assertFalse(created_again)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mtime_ns, original_mtime)
            self.assertTrue(path.with_suffix(".json.sha256").is_file())
            with self.assertRaises(frozen.IntegrityError):
                budget96_v2._write_immutable_report(path, {"answer": 512})

    def test_preflight_does_not_solve_or_restore(self) -> None:
        paths = budget96_v2._default_paths()
        manifest = {"scenes": [{"name": "a"}], "checkpoints": {}, "code": {}}
        loaded = (manifest, {"a": {}}, {"a": {}}, budget96_v2.EXPECTED_GATE_ROOT_SHA256,
                  {"verified": 1, "missing": []})
        with (
            mock.patch.object(budget96_v2, "_load_verified_inputs", return_value=loaded),
            mock.patch.object(budget96_v2, "_solve_dense") as solve,
            mock.patch.object(budget96_v2.frozen, "_fixed_nlm") as restore,
        ):
            result = budget96_v2.preflight_frozen_budget96_v2()
        self.assertEqual(result["status"], "preflight_ok")
        self.assertEqual(result["scene_count"], 1)
        self.assertEqual(result["cache_count"], 1)
        solve.assert_not_called()
        restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
