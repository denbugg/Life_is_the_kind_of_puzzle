from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import e14_cc192_oracle as cc192  # noqa: E402
import eval_e14_cc192_discovery as evaluator  # noqa: E402
from eval_buddies_ssim_budget import RawScene  # noqa: E402


def _zeros() -> tuple[np.ndarray, np.ndarray]:
    shape = (cc192.NUM_TILES, cc192.NUM_TILES)
    return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)


class FixedCC192CoreTests(unittest.TestCase):
    def test_structure_uses_exact_candidate_prefix_and_component_builder(self) -> None:
        right, down = _zeros()
        permutation = np.arange(cc192.NUM_TILES, dtype=np.int64)
        edges = [
            (0.9, 0.5, 0, 1, 0, 1),       # true right
            (0.8, 0.4, 0, 24, 1, 0),      # true down
            (0.7, 0.3, 23, 24, 0, 1),     # false row-wrap lookalike
        ]
        components = [
            {0: (0, 0), 1: (0, 1), 24: (1, 0)},
            {100: (0, 0), 101: (0, 1)},
        ]
        with mock.patch.object(cc192, "_candidate_edges", return_value=edges) as selected:
            with mock.patch.object(
                cc192, "build_buddies_components", return_value=components
            ) as built:
                result = cc192.measure_cc192_structure(right, down, permutation)
        selected.assert_called_once()
        built.assert_called_once()
        self.assertEqual(selected.call_args.kwargs, {"max_edges": 192, "min_margin": 0.0})
        self.assertEqual(built.call_args.kwargs, {"max_edges": 192, "min_margin": 0.0})
        self.assertEqual(result.selected_edge_count, 3)
        self.assertEqual(result.true_edge_count, 2)
        self.assertEqual(result.selected_edge_precision, 2 / 3)
        self.assertEqual(result.covered_tiles, 5)
        self.assertEqual(result.component_coverage, 5 / 576)
        self.assertEqual(result.component_sizes, (3, 2))

    def test_solver_exposes_no_budget_and_calls_only_192_zero_zero(self) -> None:
        right, down = _zeros()
        captured: dict[str, object] = {}

        def solver(r: np.ndarray, d: np.ndarray, **kwargs: object):
            captured.update(kwargs)
            return np.arange(576, dtype=np.int64), 1.5

        board, objective = cc192.solve_cc192(right, down, solver=solver)
        self.assertEqual(
            captured,
            {"max_edges": 192, "min_margin": 0.0, "repair_passes": 0},
        )
        self.assertTrue(np.array_equal(board, np.arange(576)))
        self.assertEqual(objective, 1.5)
        self.assertNotIn("max_edges", cc192.solve_cc192.__annotations__)

    def test_invalid_dense_permutation_and_solver_output_fail_closed(self) -> None:
        right, down = _zeros()
        bad_dense = right.copy()
        bad_dense[0, 1] = np.nan
        with self.assertRaises(cc192.CC192ContractError):
            cc192.measure_cc192_structure(bad_dense, down, np.arange(576))
        bad_permutation = np.arange(576)
        bad_permutation[-1] = 0
        with self.assertRaises(cc192.CC192ContractError):
            cc192.measure_cc192_structure(right, down, bad_permutation)

        def bad_solver(*_args: object, **_kwargs: object):
            return np.zeros(576, dtype=np.int64), 0.0

        with self.assertRaises(cc192.CC192ContractError):
            cc192.solve_cc192(right, down, solver=bad_solver)


class FrozenE14ProtocolTests(unittest.TestCase):
    def test_protocol_and_gates_are_literal_exact(self) -> None:
        self.assertEqual(
            evaluator.STRUCTURAL_RULE,
            {
                "mean_selected_edge_precision_min": 0.95,
                "mean_component_coverage_min": 0.45,
            },
        )
        self.assertEqual(
            evaluator.END_TO_END_RULE,
            {
                "cc192_minus_rr96_mean_solve_ssim_min": 0.010,
                "cc192_minus_rr96_mean_final_ssim_min": 0.015,
                "cc192_minus_rr96_final_wins_min": 6,
                "cc192_minus_rr96_worst_final_delta_min": -0.020,
            },
        )
        protocol = evaluator.E14_PROTOCOL
        self.assertEqual(protocol["calibration_ids"], list(range(10, 18)))
        self.assertEqual(protocol["arms"]["RR96"]["max_edges"], 96)
        self.assertEqual(protocol["arms"]["CC192"]["max_edges"], 192)
        self.assertEqual(protocol["arms"]["CC192"]["repair_passes"], 0)
        self.assertFalse(protocol["geometry"]["rotation"])
        self.assertFalse(protocol["geometry"]["reflection"])
        self.assertIn("budget_sweep", protocol["excluded"])
        self.assertIn("rank_or_energy_transplant", protocol["excluded"])
        self.assertEqual(protocol["runtime_provenance"], evaluator.EXPECTED_RUNTIME_PROVENANCE)

    def test_cli_exposes_paths_only(self) -> None:
        destinations = {
            action.dest
            for action in evaluator.build_parser()._actions
            if action.dest != "help"
        }
        self.assertEqual(
            destinations,
            {"raw_cache_dir", "calibration_report", "e12_report", "report"},
        )
        self.assertTrue(
            destinations.isdisjoint(
                {"budget", "max_edges", "margin", "repair", "device", "arm", "transplant"}
            )
        )

    def test_runtime_is_pinned_fail_closed(self) -> None:
        self.assertEqual(evaluator._runtime_provenance(), evaluator.EXPECTED_RUNTIME_PROVENANCE)
        with mock.patch.object(evaluator.platform, "python_version", return_value="0.0.0"):
            with self.assertRaisesRegex(evaluator.E14ContractError, "runtime drifted"):
                evaluator._runtime_provenance()

    def test_e_drive_guard(self) -> None:
        self.assertEqual(
            evaluator._require_e_drive(Path("E:/pazzle_work/report.json"), label="test").drive.upper(),
            "E:",
        )
        with self.assertRaisesRegex(evaluator.E14ContractError, "must stay on E"):
            evaluator._require_e_drive(ROOT / "report.json", label="test")


class GateTests(unittest.TestCase):
    def test_structural_gate_is_inclusive_and_requires_both_checks(self) -> None:
        passing = {
            "mean_selected_edge_precision": 0.95,
            "mean_component_coverage": 0.45,
        }
        self.assertTrue(evaluator.structural_decision(passing)["passed"])
        failing = {**passing, "mean_component_coverage": 0.449999}
        self.assertFalse(evaluator.structural_decision(failing)["passed"])

    def test_end_to_end_gate_is_inclusive_and_requires_all_checks(self) -> None:
        passing = {
            "metrics": {
                "solve_only_ssim": {"mean_delta": 0.010},
                "final_ssim": {
                    "mean_delta": 0.015,
                    "wins": 6,
                    "worst_delta": -0.020,
                },
            }
        }
        self.assertTrue(evaluator.end_to_end_decision(passing)["passed"])
        failing = {
            "metrics": {
                **passing["metrics"],
                "final_ssim": {**passing["metrics"]["final_ssim"], "wins": 5},
            }
        }
        self.assertFalse(evaluator.end_to_end_decision(failing)["passed"])

    def test_rr_reproducibility_pins_both_means_and_all_hashes(self) -> None:
        rows = {
            image: {
                "solve_only_ssim": evaluator.EXPECTED_RR_MEAN_SOLVE_SSIM,
                "final_ssim": evaluator.EXPECTED_RR_MEAN_FINAL_SSIM,
                "board_sha256": f"{image:064x}",
            }
            for image in range(10, 18)
        }
        observed = evaluator.verify_rr_means(rows)
        self.assertEqual(len(observed["board_hashes"]), 8)
        drifted = {image: dict(row) for image, row in rows.items()}
        drifted[10]["final_ssim"] += 1e-3
        with self.assertRaisesRegex(evaluator.E14ContractError, "mean final"):
            evaluator.verify_rr_means(drifted)


class CanvasContractTests(unittest.TestCase):
    def test_cc192_assembles_corrupted_tiles_and_calls_nlm_once(self) -> None:
        corrupted = np.full((576, 20, 20, 3), 17, dtype=np.uint8)
        target = np.full((480, 480, 3), 201, dtype=np.uint8)
        scene = RawScene(
            image_id=10,
            validation_name="synthetic.png",
            cache_path=Path("E:/unused.npz"),
            cache_sha256="0" * 64,
            candidate_ids=np.empty((576, 0), dtype=np.int64),
            base_scores=np.empty((4, 576, 0), dtype=np.float32),
            permutation=np.arange(576, dtype=np.int64),
            tiles_uint8=corrupted,
            target_uint8=target,
        )
        calls = 0
        seen: dict[str, np.ndarray] = {}

        def restorer(image: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            seen["image"] = image.copy()
            return image

        with mock.patch.object(evaluator, "sk_ssim", side_effect=(0.1, 0.2)):
            result = evaluator.evaluate_cc192_board(
                scene, np.arange(576, dtype=np.int64), 2.0, restorer=restorer
            )
        self.assertEqual(calls, 1)
        self.assertTrue(np.all(seen["image"] == 17))
        self.assertFalse(np.array_equal(seen["image"], target))
        self.assertEqual(result["solve_only_ssim"], 0.1)
        self.assertEqual(result["final_ssim"], 0.2)


if __name__ == "__main__":
    unittest.main()
