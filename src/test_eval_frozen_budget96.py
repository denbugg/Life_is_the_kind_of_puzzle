from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import eval_frozen_budget96 as budget96
import eval_frozen_end_to_end_gate as frozen


class FrozenBudget96Tests(unittest.TestCase):
    def test_exact_root_and_only_two_fixed_arms(self) -> None:
        self.assertEqual(
            budget96.EXPECTED_GATE_ROOT_SHA256,
            "ee3d74662f5326fbd1069763fd7b96dc3adb41bde0117cba1d78ff067c6bf23d",
        )
        self.assertEqual(budget96.ARM_ORDER, ("budget_96", "budget_512"))
        self.assertEqual(
            [budget96.FIXED_ARMS[name]["max_edges"] for name in budget96.ARM_ORDER],
            [96, 512],
        )
        for config in budget96.FIXED_ARMS.values():
            self.assertEqual(config["min_margin"], 0.0)
            self.assertEqual(config["repair_passes"], 0)
        left = dict(budget96.FIXED_ARMS["budget_96"])
        right = dict(budget96.FIXED_ARMS["budget_512"])
        left.pop("max_edges")
        right.pop("max_edges")
        self.assertEqual(left, right)

    def test_wrong_gate_root_is_rejected(self) -> None:
        budget96._require_expected_root(budget96.EXPECTED_GATE_ROOT_SHA256)
        with self.assertRaises(frozen.IntegrityError):
            budget96._require_expected_root("0" * 64)

    def test_wrong_loaded_root_stops_before_external_or_cache_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            budget96.frozen,
            "load_and_verify_gate",
            return_value=({"scenes": []}, {}, "0" * 64),
        ), mock.patch.object(
            budget96.frozen, "_verify_external_files"
        ) as verify_external, mock.patch.object(
            budget96.frozen, "verify_score_cache_directory"
        ) as verify_cache:
            with self.assertRaises(frozen.IntegrityError):
                budget96._load_verified_inputs(Path("gate"), Path(temporary))
        verify_external.assert_not_called()
        verify_cache.assert_not_called()

    def test_cli_has_paths_but_no_selection_or_device_controls(self) -> None:
        parser = budget96._build_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertTrue({"--gate-dir", "--score-cache-dir", "--report"} <= options)
        self.assertNotIn("--budget", options)
        self.assertNotIn("--max-edges", options)
        self.assertNotIn("--device", options)
        args = parser.parse_args(
            ["--gate-dir", "g", "--score-cache-dir", "c", "--report", "r.json"]
        )
        self.assertEqual((args.gate_dir, args.score_cache_dir, args.report),
                         (Path("g"), Path("c"), Path("r.json")))

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
            board_96, objective_96 = budget96._solve_dense(right, down, arm="budget_96")
            board_512, objective_512 = budget96._solve_dense(right, down, arm="budget_512")
        self.assertTrue(np.array_equal(board_96, board_512))
        self.assertEqual((objective_96, objective_512), (1.25, 1.25))
        self.assertEqual(
            calls,
            [
                {"max_edges": 96, "min_margin": 0.0, "repair_passes": 0},
                {"max_edges": 512, "min_margin": 0.0, "repair_passes": 0},
            ],
        )
        with self.assertRaises(ValueError):
            budget96._solve_dense(right, down, arm="budget_128")

    def test_input_loader_requires_complete_bound_cache_and_external_hashes(self) -> None:
        manifest = {
            "scenes": [{"name": "a"}, {"name": "b"}],
        }
        scene_arrays = {"a": {}, "b": {}}
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            with (
                mock.patch.object(
                    budget96.frozen,
                    "load_and_verify_gate",
                    return_value=(manifest, scene_arrays, budget96.EXPECTED_GATE_ROOT_SHA256),
                ) as load_gate,
                mock.patch.object(budget96.frozen, "_verify_external_files") as verify_external,
                mock.patch.object(
                    budget96.frozen,
                    "verify_score_cache_directory",
                    return_value={"verified": 2, "missing": []},
                ) as verify_cache,
                mock.patch.object(
                    budget96.frozen,
                    "_score_cache_paths",
                    side_effect=lambda root, name: (root / f"{name}.npz", root / f"{name}.sha256"),
                ),
                mock.patch.object(
                    budget96.frozen, "_score_cache_contract", side_effect=lambda *args: {"bound": args[1]["name"]}
                ) as cache_contract,
                mock.patch.object(
                    budget96.frozen, "load_score_cache", side_effect=lambda path, contract: {"path": path, **contract}
                ) as load_cache,
            ):
                loaded = budget96._load_verified_inputs(Path("gate"), cache_dir)
        self.assertIs(loaded[0], manifest)
        self.assertEqual(set(loaded[2]), {"a", "b"})
        load_gate.assert_called_once_with(Path("gate"))
        verify_external.assert_called_once_with(manifest)
        verify_cache.assert_called_once_with(
            manifest,
            budget96.EXPECTED_GATE_ROOT_SHA256,
            cache_dir.resolve(),
            require_complete=True,
        )
        self.assertEqual(cache_contract.call_count, 2)
        self.assertEqual(load_cache.call_count, 2)

    def test_input_loader_rejects_nonexact_complete_result(self) -> None:
        manifest = {"scenes": [{"name": "a"}]}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                budget96.frozen,
                "load_and_verify_gate",
                return_value=(manifest, {"a": {}}, budget96.EXPECTED_GATE_ROOT_SHA256),
            ), mock.patch.object(
                budget96.frozen, "_verify_external_files"
            ), mock.patch.object(
                budget96.frozen,
                "verify_score_cache_directory",
                return_value={"verified": 0, "missing": []},
            ):
            with self.assertRaises(frozen.CacheContractError):
                budget96._load_verified_inputs(Path("gate"), Path(temporary))

    def test_report_is_create_once_or_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            digest, created = budget96._write_immutable_report(path, {"answer": 96})
            self.assertTrue(created)
            self.assertEqual(len(digest), 64)
            original = path.read_bytes()
            original_mtime = path.stat().st_mtime_ns
            digest_again, created_again = budget96._write_immutable_report(path, {"answer": 96})
            self.assertEqual(digest_again, digest)
            self.assertFalse(created_again)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mtime_ns, original_mtime)
            with self.assertRaises(frozen.IntegrityError):
                budget96._write_immutable_report(path, {"answer": 512})

    def test_identical_report_requires_matching_digest_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            budget96._write_immutable_report(path, {"fixed": True})
            sidecar = path.with_suffix(".json.sha256")
            sidecar.write_text("wrong\n", encoding="ascii")
            with self.assertRaises(frozen.IntegrityError):
                budget96._write_immutable_report(path, {"fixed": True})


if __name__ == "__main__":
    unittest.main()
