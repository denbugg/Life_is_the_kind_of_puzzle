from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_e16_clean_render_oracle as evaluator  # noqa: E402
from imgio import to_frags  # noqa: E402


def _target() -> np.ndarray:
    row, col = np.indices((480, 480))
    return np.stack(
        (row % 251, col % 253, (3 * row + 5 * col) % 255), axis=2
    ).astype(np.uint8)


class FrozenContractTests(unittest.TestCase):
    def test_protocol_and_decision_are_literal_exact(self) -> None:
        self.assertEqual(
            evaluator.DECISION_RULE,
            {
                "mean_final_ssim_delta_min": 0.050,
                "strict_wins_min": 8,
                "worst_final_delta_min": 0.020,
            },
        )
        protocol = evaluator.E16_PROTOCOL
        self.assertEqual(protocol["calibration_ids"], list(range(10, 18)))
        self.assertEqual(
            protocol["candidate"]["clean_tiles"],
            "imgio.to_frags(target_uint8)[permutation]",
        )
        self.assertEqual(protocol["candidate"]["restoration"], None)
        self.assertFalse(protocol["geometry"]["rotation"])
        self.assertFalse(protocol["geometry"]["reflection"])
        self.assertIn("candidate_solver", protocol["excluded"])
        self.assertIn("candidate_NLM", protocol["excluded"])
        self.assertEqual(
            set(evaluator._source_provenance()),
            {
                "eval_e16_clean_render_oracle.py",
                "eval_clean_score_oracle.py",
                "eval_e14_cc192_discovery.py",
                "imgio.py",
            },
        )

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
                {"threshold", "restorer", "nlm", "diffusion", "board", "device"}
            )
        )

    def test_runtime_and_E_drive_are_fail_closed(self) -> None:
        self.assertEqual(
            evaluator._runtime_provenance(), evaluator.EXPECTED_RUNTIME_PROVENANCE
        )
        with mock.patch.object(
            evaluator.platform, "python_version", return_value="0.0.0"
        ):
            with self.assertRaisesRegex(evaluator.E16ContractError, "runtime drifted"):
                evaluator._runtime_provenance()
        with self.assertRaisesRegex(evaluator.E16ContractError, "must stay on E"):
            evaluator._require_e_drive(ROOT / "report.json", label="test")

    def test_clean_render_signature_has_no_restoration_controls(self) -> None:
        self.assertEqual(
            set(inspect.signature(evaluator.clean_render).parameters),
            {"target_uint8", "permutation", "board"},
        )

    def test_report_path_guards_run_before_input_loading(self) -> None:
        base = evaluator.E16Paths(
            raw_cache_dir=Path("E:/pazzle_work/e16_unit_raw"),
            calibration_report=ROOT / "calibration.json",
            e12_report=Path("E:/pazzle_work/e16_unit_e12.json"),
            report=Path("E:/pazzle_work/e16_unit_raw/report.json"),
        )
        with self.assertRaisesRegex(evaluator.E16ContractError, "inside an input cache"):
            evaluator.run_oracle(base)
        clean_cache = evaluator.E16Paths(
            **{
                **base.__dict__,
                "report": evaluator.DEFAULT_E12_REPORT.parent
                / "score_cache"
                / "report.json",
            }
        )
        with self.assertRaisesRegex(evaluator.E16ContractError, "inside an input cache"):
            evaluator.run_oracle(clean_cache)
        wrong_suffix = evaluator.E16Paths(
            **{**base.__dict__, "report": Path("E:/pazzle_work/e16_report.txt")}
        )
        with self.assertRaisesRegex(evaluator.E16ContractError, "must be a .json"):
            evaluator.run_oracle(wrong_suffix)
        overwrite = evaluator.E16Paths(
            **{**base.__dict__, "report": base.e12_report}
        )
        with self.assertRaisesRegex(evaluator.E16ContractError, "must not overwrite"):
            evaluator.run_oracle(overwrite)


class MappingAndLeakageTests(unittest.TestCase):
    def test_non_self_inverse_permutation_reconstructs_target_exactly(self) -> None:
        target = _target()
        permutation = np.roll(np.arange(576, dtype=np.int64), 7)
        self.assertFalse(np.array_equal(permutation, np.argsort(permutation)))
        truth_board = np.argsort(permutation)
        clean_tiles, rendered = evaluator.clean_render(
            target, permutation, truth_board
        )
        self.assertTrue(np.array_equal(rendered, target))
        self.assertTrue(
            np.array_equal(to_frags(rendered), clean_tiles[truth_board])
        )

    def test_each_output_block_is_an_upright_unblended_input_order_tile(self) -> None:
        target = _target()
        permutation = np.roll(np.arange(576, dtype=np.int64), 11)
        board = np.arange(576, dtype=np.int64)[::-1]
        clean_tiles, rendered = evaluator.clean_render(target, permutation, board)
        self.assertTrue(np.array_equal(to_frags(rendered), clean_tiles[board]))

    def test_target_changes_pixels_but_never_board_hash_or_calls_nlm(self) -> None:
        target = _target()
        permutation = np.roll(np.arange(576, dtype=np.int64), 5)
        board = np.arange(576, dtype=np.int64)
        board_sha = evaluator.e12.array_sha256(board)
        rr = {"board_sha256": board_sha, "final_ssim": 0.1}

        def scene(value: np.ndarray):
            return SimpleNamespace(
                image_id=10,
                validation_name="synthetic.png",
                target_uint8=value,
                permutation=permutation,
            )

        with mock.patch.object(evaluator.e12, "fixed_nlm") as nlm:
            first = evaluator.evaluate_scene(scene(target), board, rr)
            second = evaluator.evaluate_scene(scene(255 - target), board, rr)
        nlm.assert_not_called()
        self.assertEqual(first["board_sha256"], board_sha)
        self.assertEqual(second["board_sha256"], board_sha)
        self.assertNotEqual(
            first["clean_render_canvas_sha256"],
            second["clean_render_canvas_sha256"],
        )
        self.assertEqual(first["candidate_solver_calls"], 0)
        self.assertEqual(first["candidate_restorer_calls"], 0)

    def test_rr_board_hash_mismatch_fails_before_clean_render(self) -> None:
        scene = SimpleNamespace(
            image_id=10,
            validation_name="synthetic.png",
            target_uint8=_target(),
            permutation=np.arange(576, dtype=np.int64),
        )
        with mock.patch.object(evaluator, "clean_render") as render:
            with self.assertRaisesRegex(evaluator.E16ContractError, "board hash drifted"):
                evaluator.evaluate_scene(
                    scene,
                    np.arange(576, dtype=np.int64),
                    {"board_sha256": "0" * 64, "final_ssim": 0.1},
                )
        render.assert_not_called()


class DecisionTests(unittest.TestCase):
    def test_decision_is_inclusive_and_requires_all_checks(self) -> None:
        passing = {
            "mean_final_ssim_delta": 0.050,
            "strict_wins": 8,
            "worst_final_ssim_delta": 0.020,
        }
        self.assertTrue(evaluator.decision(passing)["passed"])
        failures = (
            {**passing, "mean_final_ssim_delta": 0.049999},
            {**passing, "strict_wins": 7},
            {**passing, "worst_final_ssim_delta": 0.019999},
        )
        for failing in failures:
            with self.subTest(failing=failing):
                self.assertFalse(evaluator.decision(failing)["passed"])

    def test_summary_aligns_exactly_eight_ids_and_counts_strict_wins(self) -> None:
        rows = [
            {
                "image": image,
                "rr96_final_ssim": 0.1,
                "clean_render_ssim": 0.2 if image < 17 else 0.1,
            }
            for image in range(10, 18)
        ]
        summary = evaluator.summarize(rows)
        self.assertEqual(summary["images"], 8)
        self.assertEqual(summary["strict_wins"], 7)
        self.assertEqual(summary["ties"], 1)
        with self.assertRaisesRegex(evaluator.E16ContractError, "exactly eight"):
            evaluator.summarize(rows[:-1])

    def test_complete_report_is_revalidated_from_rows(self) -> None:
        rr_rows = {
            image: {"board_sha256": f"{image:064x}"}
            for image in range(10, 18)
        }
        rows = [
            {
                "image": image,
                "board_sha256": f"{image:064x}",
                "rr96_final_ssim": 0.1,
                "clean_render_ssim": 0.2,
            }
            for image in range(10, 18)
        ]
        summary = evaluator.summarize(rows)
        result = evaluator.decision(summary)
        report = {
            "status": "complete",
            "schema": evaluator.REPORT_SCHEMA,
            "experiment": evaluator.EXPERIMENT,
            "run_contract_sha256": "a" * 64,
            "protocol": evaluator.E16_PROTOCOL,
            "rows": rows,
            "completed_images": list(range(10, 18)),
            "summary": summary,
            "decision": result,
            "stage": result["status"],
        }
        evaluator._validate_complete_report(
            report, contract_digest="a" * 64, rr_rows=rr_rows
        )
        malformed = {**report, "summary": {**summary, "strict_wins": 0}}
        with self.assertRaisesRegex(evaluator.E16ContractError, "summary"):
            evaluator._validate_complete_report(
                malformed, contract_digest="a" * 64, rr_rows=rr_rows
            )


if __name__ == "__main__":
    unittest.main()
