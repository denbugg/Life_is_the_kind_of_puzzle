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

import e13_torus_origin as torus  # noqa: E402
import eval_e13_torus_origin as evaluator  # noqa: E402
import rank96_lab_selector as e11  # noqa: E402
from eval_buddies_ssim_budget import RawScene  # noqa: E402


def _identity_board() -> np.ndarray:
    return np.arange(torus.NUM_TILES, dtype=np.int64)


def _uniform_shifted_grid(row_cut: int, column_cut: int) -> np.ndarray:
    tiles = np.empty(
        (torus.NUM_TILES, torus.TILE_SIZE, torus.TILE_SIZE, 3), dtype=np.uint8
    )
    for row in range(torus.GRID):
        for column in range(torus.GRID):
            row_level = (row - row_cut) % torus.GRID
            column_level = (column - column_cut) % torus.GRID
            value = 4 * (row_level + column_level)
            tiles[row * torus.GRID + column] = value
    return tiles


class PureSelectorTests(unittest.TestCase):
    def test_non_toroidal_score_matches_exact_e11_formula(self) -> None:
        rng = np.random.default_rng(20260813)
        tiles = rng.integers(
            0,
            256,
            size=(torus.NUM_TILES, torus.TILE_SIZE, torus.TILE_SIZE, 3),
            dtype=np.uint8,
        )
        board = rng.permutation(torus.NUM_TILES).astype(np.int64)
        horizontal, vertical = torus.toroidal_cut_energies(tiles, board)
        reconstructed = -0.5 * (
            float(horizontal[1:].mean(dtype=np.float64))
            + float(vertical[1:].mean(dtype=np.float64))
        )
        self.assertAlmostEqual(reconstructed, e11.lab_depth1_board_score(tiles, board), places=14)

    def test_selects_known_row_and_column_cut_and_rolls_once(self) -> None:
        expected_row_cut = 7
        expected_column_cut = 13
        tiles = _uniform_shifted_grid(expected_row_cut, expected_column_cut)
        board = _identity_board()
        selected = torus.select_torus_origin(tiles, board)
        self.assertEqual(selected.row_cut, expected_row_cut)
        self.assertEqual(selected.column_cut, expected_column_cut)
        self.assertEqual(selected.row_roll, -expected_row_cut)
        self.assertEqual(selected.column_roll, -expected_column_cut)
        expected = np.roll(
            board.reshape(torus.GRID, torus.GRID),
            shift=(-expected_row_cut, -expected_column_cut),
            axis=(0, 1),
        ).reshape(-1)
        self.assertTrue(np.array_equal(selected.rolled_board, expected))
        self.assertTrue(
            np.array_equal(np.sort(selected.rolled_board), np.arange(torus.NUM_TILES))
        )
        self.assertAlmostEqual(
            selected.retained_internal_lab_score,
            e11.lab_depth1_board_score(tiles, selected.rolled_board),
            places=14,
        )

    def test_exact_all_cut_tie_is_cut_zero_and_no_roll(self) -> None:
        tiles = np.full(
            (torus.NUM_TILES, torus.TILE_SIZE, torus.TILE_SIZE, 3),
            127,
            dtype=np.uint8,
        )
        board = _identity_board()
        selected = torus.select_torus_origin(tiles, board)
        self.assertTrue(np.array_equal(selected.horizontal_cut_energies, np.zeros(24)))
        self.assertTrue(np.array_equal(selected.vertical_cut_energies, np.zeros(24)))
        self.assertEqual((selected.row_cut, selected.column_cut), (0, 0))
        self.assertTrue(np.array_equal(selected.rolled_board, board))

    def test_selection_does_not_mutate_inputs_and_results_are_read_only(self) -> None:
        tiles = _uniform_shifted_grid(3, 4)
        board = _identity_board()
        tiles_before = tiles.copy()
        board_before = board.copy()
        selected = torus.select_torus_origin(tiles, board)
        self.assertTrue(np.array_equal(tiles, tiles_before))
        self.assertTrue(np.array_equal(board, board_before))
        self.assertFalse(selected.original_board.flags.writeable)
        self.assertFalse(selected.rolled_board.flags.writeable)
        self.assertFalse(selected.horizontal_cut_energies.flags.writeable)
        self.assertFalse(selected.vertical_cut_energies.flags.writeable)

    def test_invalid_dtype_shape_and_board_fail_closed(self) -> None:
        tiles = np.zeros(
            (torus.NUM_TILES, torus.TILE_SIZE, torus.TILE_SIZE, 3), dtype=np.uint8
        )
        with self.assertRaises(torus.TorusOriginError):
            torus.select_torus_origin(tiles.astype(np.float32), _identity_board())
        with self.assertRaises(torus.TorusOriginError):
            torus.select_torus_origin(tiles[:-1], _identity_board())
        bad = _identity_board()
        bad[-1] = 0
        with self.assertRaises(torus.TorusOriginError):
            torus.select_torus_origin(tiles, bad)


class FrozenDiscoveryContractTests(unittest.TestCase):
    def test_protocol_and_thresholds_are_literal_exact(self) -> None:
        self.assertEqual(
            evaluator.RR_PROMOTION_RULE,
            {
                "mean_solve_delta_min": 0.002,
                "mean_final_delta_min": 0.003,
                "final_wins_min": 5,
                "worst_final_delta_min": -0.015,
            },
        )
        self.assertEqual(
            evaluator.CC_ORIGIN_DIAGNOSIS_RULE,
            {
                "mean_solve_delta_min": 0.0075,
                "mean_final_delta_min": 0.015,
                "final_wins_min": 6,
                "worst_final_delta_min": -0.020,
                "absolute_cc_solve_at_least_rr_baseline": True,
                "absolute_cc_final_at_least_rr_baseline": True,
            },
        )
        protocol = evaluator.E13_PROTOCOL
        self.assertEqual(protocol["calibration_ids"], list(range(10, 18)))
        self.assertEqual(protocol["selector"]["depth"], 1)
        self.assertEqual(protocol["selector"]["horizontal_cuts"], 24)
        self.assertEqual(protocol["selector"]["vertical_cuts"], 24)
        self.assertFalse(protocol["geometry"]["rotation"])
        self.assertFalse(protocol["geometry"]["reflection"])
        self.assertFalse(protocol["geometry"]["tile_changes"])
        self.assertEqual(protocol["restoration"]["calls"], "once_after_roll_per_arm_per_scene")
        self.assertEqual(protocol["arms"]["RR96"]["role"], "deployable_discovery")
        self.assertEqual(
            protocol["arms"]["CC96"]["role"], "diagnostic_only_not_deployable"
        )
        self.assertEqual(protocol["runtime_provenance"], evaluator.EXPECTED_RUNTIME_PROVENANCE)

    def test_runtime_provenance_is_pinned_fail_closed(self) -> None:
        self.assertEqual(evaluator._runtime_provenance(), evaluator.EXPECTED_RUNTIME_PROVENANCE)
        with mock.patch.object(evaluator.platform, "python_version", return_value="0.0.0"):
            with self.assertRaisesRegex(evaluator.E13ContractError, "runtime differs"):
                evaluator._runtime_provenance()

    def test_cli_exposes_only_input_and_report_paths(self) -> None:
        parser = evaluator.build_parser()
        destinations = {
            action.dest for action in parser._actions if action.dest not in {"help"}
        }
        self.assertEqual(
            destinations,
            {"raw_cache_dir", "calibration_report", "e12_report", "report"},
        )
        forbidden = {
            "threshold",
            "cut",
            "rotation",
            "reflection",
            "arm",
            "device",
            "nlm",
            "h",
            "max_edges",
        }
        self.assertTrue(destinations.isdisjoint(forbidden))

    def test_rr_rule_is_inclusive_and_requires_every_check(self) -> None:
        summary = {
            "metrics": {
                "solve_only_ssim": {"mean_delta": 0.002},
                "final_ssim": {
                    "mean_delta": 0.003,
                    "wins": 5,
                    "worst_delta": -0.015,
                },
            }
        }
        passed = evaluator.rr_promotion_decision(summary)
        self.assertTrue(passed["passed"])
        failed = {
            "metrics": {
                **summary["metrics"],
                "final_ssim": {
                    **summary["metrics"]["final_ssim"],
                    "wins": 4,
                },
            }
        }
        self.assertFalse(evaluator.rr_promotion_decision(failed)["passed"])

    def test_cc_rule_requires_absolute_rr_baseline_recovery(self) -> None:
        rr = {
            "metrics": {
                "solve_only_ssim": {"before_mean": 0.10},
                "final_ssim": {"before_mean": 0.16},
            }
        }
        cc = {
            "metrics": {
                "solve_only_ssim": {"mean_delta": 0.0075, "after_mean": 0.10},
                "final_ssim": {
                    "mean_delta": 0.015,
                    "after_mean": 0.16,
                    "wins": 6,
                    "worst_delta": -0.020,
                },
            }
        }
        self.assertTrue(evaluator.cc_origin_diagnosis_decision(cc, rr)["passed"])
        below_rr = {
            "metrics": {
                **cc["metrics"],
                "solve_only_ssim": {
                    **cc["metrics"]["solve_only_ssim"],
                    "after_mean": 0.099999,
                },
            }
        }
        decision = evaluator.cc_origin_diagnosis_decision(below_rr, rr)
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["checks"]["absolute_cc_solve_at_least_rr_baseline"])

    def test_report_and_cache_path_guard_is_e_drive_only(self) -> None:
        self.assertEqual(
            evaluator._require_e_drive(Path("E:/pazzle_work/example.json"), label="test").drive.upper(),
            "E:",
        )
        with self.assertRaisesRegex(evaluator.E13ContractError, "must stay on E"):
            evaluator._require_e_drive(ROOT / "example.json", label="test")


class RolledEvaluationTests(unittest.TestCase):
    def test_restorer_is_called_exactly_once_after_roll(self) -> None:
        tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
        target = np.zeros((480, 480, 3), dtype=np.uint8)
        scene = RawScene(
            image_id=10,
            validation_name="synthetic.png",
            cache_path=Path("E:/unused.npz"),
            cache_sha256="0" * 64,
            candidate_ids=np.empty((576, 0), dtype=np.int64),
            base_scores=np.empty((4, 576, 0), dtype=np.float32),
            permutation=np.arange(576, dtype=np.int64),
            tiles_uint8=tiles,
            target_uint8=target,
        )
        calls = 0

        def restorer(image: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            return image

        with mock.patch.object(evaluator, "sk_ssim", side_effect=(0.1, 0.2)) as metric:
            result = evaluator.evaluate_rolled_board(
                scene, np.arange(576, dtype=np.int64), 1.25, restorer=restorer
            )
        self.assertEqual(calls, 1)
        self.assertEqual(metric.call_count, 2)
        self.assertEqual(result["solve_only_ssim"], 0.1)
        self.assertEqual(result["final_ssim"], 0.2)
        self.assertEqual(result["placement"], 1.0)
        self.assertEqual(result["neighbour"], 1.0)


if __name__ == "__main__":
    unittest.main()
