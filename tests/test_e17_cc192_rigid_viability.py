from __future__ import annotations

import copy
import json
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

import eval_e17_cc192_rigid_viability as evaluator  # noqa: E402


def _zeros() -> tuple[np.ndarray, np.ndarray]:
    shape = (576, 576)
    return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)


def _structure_values() -> dict[str, object]:
    pure_sizes = [210]
    return {
        "selected_claims": 192,
        "true_full_prefix_claims": 192,
        "full_prefix_precision": 1.0,
        "incremental_claims": 96,
        "true_incremental_claims": 96,
        "incremental96_precision": 1.0,
        "component_count": 1,
        "component_tiles": 210,
        "component_coverage": 210 / 576,
        "exact_pure_component_count": 1,
        "exact_pure_rigid_tiles": 210,
        "exact_pure_rigid_tile_coverage": 210 / 576,
        "largest_exact_pure_component_size": 210,
        "exact_pure_component_sizes": pure_sizes,
    }


def _row(image: int) -> dict[str, object]:
    return {
        "image": image,
        "validation_name": f"validation_{image}",
        "clean_score_cache_sha256": f"{image:064x}",
        **_structure_values(),
    }


def _complete_report() -> tuple[dict[str, object], dict[str, object], list[object]]:
    rows = [_row(image) for image in evaluator.e12.CALIBRATION_IDS]
    contract: dict[str, object] = {
        "clean_score_caches": {
            str(image): {
                "path": f"E:/cache/{image}.npz",
                "sha256": f"{image:064x}",
            }
            for image in evaluator.e12.CALIBRATION_IDS
        }
    }
    digest = evaluator.e12.canonical_digest(contract)
    summary = evaluator.summarize(rows)
    result = evaluator.decision(summary)
    report: dict[str, object] = {
        "schema_version": evaluator.SCHEMA_VERSION,
        "schema": evaluator.REPORT_SCHEMA,
        "experiment": evaluator.EXPERIMENT,
        "status": "complete",
        "stage": result["status"],
        "protocol": evaluator.E17_PROTOCOL,
        "protocol_sha256": evaluator.e12.canonical_digest(
            evaluator.E17_PROTOCOL
        ),
        "run_contract": contract,
        "run_contract_sha256": digest,
        "rows": rows,
        "completed_images": list(evaluator.e12.CALIBRATION_IDS),
        "summary": summary,
        "decision": result,
        "runtime_seconds": 1.0,
    }
    scenes = [
        SimpleNamespace(image_id=image, validation_name=f"validation_{image}")
        for image in evaluator.e12.CALIBRATION_IDS
    ]
    return report, contract, scenes


class GeometryTests(unittest.TestCase):
    def test_exact_component_purity_is_translation_invariant_and_whole(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        pure = {0: (10, 20), 1: (10, 21), 24: (11, 20)}
        self.assertTrue(evaluator.component_is_exactly_pure(pure, permutation))
        contaminated = {**pure, 25: (11, 22)}
        self.assertFalse(
            evaluator.component_is_exactly_pure(contaminated, permutation)
        )
        self.assertFalse(
            evaluator.component_is_exactly_pure({0: (0, 0)}, permutation)
        )

    def test_directional_truth_rejects_row_wrap(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        self.assertTrue(
            evaluator._edge_is_true((1.0, 0.1, 0, 1, 0, 1), permutation)
        )
        self.assertFalse(
            evaluator._edge_is_true((1.0, 0.1, 23, 24, 0, 1), permutation)
        )

    def test_incremental_prefix_is_exactly_edges_96_through_191(self) -> None:
        right, down = _zeros()
        edges = []
        for index in range(192):
            anchor = index % 552
            target = anchor + 24
            direction = (0, 1) if index < 96 else (1, 0)
            edges.append(
                (
                    float(192 - index),
                    1.0,
                    anchor,
                    target,
                    direction[0],
                    direction[1],
                )
            )
        components = [{0: (0, 0), 24: (1, 0)}]
        with mock.patch.object(evaluator, "_candidate_edges", return_value=edges):
            with mock.patch.object(
                evaluator, "build_buddies_components", return_value=components
            ):
                measured = evaluator.measure_structure(
                    right, down, np.arange(576, dtype=np.int64)
                )
        self.assertEqual(measured["selected_claims"], 192)
        self.assertEqual(measured["true_full_prefix_claims"], 96)
        self.assertEqual(measured["incremental_claims"], 96)
        self.assertEqual(measured["true_incremental_claims"], 96)

    def test_dense_and_component_integrity_are_fail_closed(self) -> None:
        right, down = _zeros()
        right[0, 1] = -0.1
        with self.assertRaisesRegex(evaluator.E17ContractError, "nonnegative"):
            evaluator.measure_structure(
                right, down, np.arange(576, dtype=np.int64)
            )
        with self.assertRaisesRegex(evaluator.E17ContractError, "overlaps"):
            evaluator._validate_component({0: (0, 0), 1: (0, 0)})


class FrozenContractTests(unittest.TestCase):
    def test_protocol_and_gate_are_literal_exact(self) -> None:
        self.assertEqual(
            evaluator.DECISION_RULE,
            {
                "selected_claims_each": 192,
                "mean_full_prefix_precision_min": 0.95,
                "mean_incremental96_precision_min": 0.90,
                "worst_incremental96_precision_min": 0.80,
                "mean_exact_pure_rigid_tile_coverage_min": 0.35,
                "worst_exact_pure_rigid_tile_coverage_min": 0.25,
                "mean_largest_exact_pure_component_size_min": 8.0,
            },
        )
        protocol = evaluator.E17_PROTOCOL
        self.assertEqual(protocol["graph"]["full_prefix"], 192)
        self.assertEqual(protocol["graph"]["incremental_indices"], [96, 191])
        self.assertFalse(protocol["purity"]["modal_trim"])
        self.assertFalse(protocol["purity"]["oracle_edge_removal"])
        self.assertFalse(protocol["geometry"]["rotation"])
        self.assertIn("candidate_board", protocol["excluded"])
        self.assertIn("NLM", protocol["excluded"])
        self.assertIn("SSIM", protocol["excluded"])

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

    def test_runtime_and_E_drive_are_fail_closed(self) -> None:
        self.assertEqual(
            evaluator._runtime_provenance(), evaluator.EXPECTED_RUNTIME_PROVENANCE
        )
        with mock.patch.object(
            evaluator.platform, "python_version", return_value="0.0.0"
        ):
            with self.assertRaisesRegex(evaluator.E17ContractError, "runtime drifted"):
                evaluator._runtime_provenance()
        with self.assertRaisesRegex(evaluator.E17ContractError, "must stay on E"):
            evaluator._require_e_drive(ROOT / "report.json", label="test")

    def test_structure_calibration_validation_does_not_require_ssim_fields(self) -> None:
        payload = json.loads(
            evaluator.DEFAULT_CALIBRATION_REPORT.read_text(encoding="utf-8")
        )
        payload.pop("selected_metrics", None)
        payload.pop("grid", None)
        payload.pop("grid_per_image", None)
        evaluator._validate_structure_calibration_payload(payload)

    def test_report_path_guards_run_before_any_input_loader(self) -> None:
        cases = (
            (
                evaluator.DEFAULT_RAW_CACHE_DIR,
                evaluator.DEFAULT_CALIBRATION_REPORT,
                evaluator.DEFAULT_E12_REPORT,
                Path("E:/pazzle_work/e17/not_json.txt"),
            ),
            (
                evaluator.DEFAULT_RAW_CACHE_DIR,
                evaluator.DEFAULT_CALIBRATION_REPORT,
                evaluator.DEFAULT_E12_REPORT,
                evaluator.DEFAULT_RAW_CACHE_DIR / "nested" / "report.json",
            ),
            (
                evaluator.DEFAULT_RAW_CACHE_DIR,
                evaluator.DEFAULT_CALIBRATION_REPORT,
                evaluator.DEFAULT_E12_REPORT,
                evaluator.DEFAULT_E12_REPORT.parent
                / "score_cache"
                / "report.json",
            ),
            (
                evaluator.DEFAULT_RAW_CACHE_DIR,
                evaluator.DEFAULT_CALIBRATION_REPORT,
                evaluator.DEFAULT_E12_REPORT,
                evaluator.DEFAULT_E12_REPORT,
            ),
            (
                evaluator.DEFAULT_RAW_CACHE_DIR,
                Path("E:/pazzle_work/e17/calibration.json"),
                evaluator.DEFAULT_E12_REPORT,
                Path("E:/pazzle_work/e17/calibration.json"),
            ),
        )
        with mock.patch.object(
            evaluator, "_load_verified_structure_inputs"
        ) as loader:
            for args in cases:
                with self.subTest(report=args[-1]):
                    with self.assertRaises(evaluator.E17ContractError):
                        evaluator.run_gate(*args)
        loader.assert_not_called()

    def test_input_validation_failure_writes_no_scientific_report(self) -> None:
        with mock.patch.object(
            evaluator,
            "_load_verified_structure_inputs",
            side_effect=evaluator.E17ContractError("pinned input drift"),
        ):
            with mock.patch.object(evaluator, "_atomic_write_json") as writer:
                with self.assertRaisesRegex(
                    evaluator.E17ContractError, "pinned input drift"
                ):
                    evaluator.run_gate(
                        evaluator.DEFAULT_RAW_CACHE_DIR,
                        evaluator.DEFAULT_CALIBRATION_REPORT,
                        evaluator.DEFAULT_E12_REPORT,
                        Path("E:/pazzle_work/e17/input_failure_test.json"),
                    )
        writer.assert_not_called()


class CompleteReportTests(unittest.TestCase):
    def test_complete_report_is_recomputed_and_fail_closed(self) -> None:
        report, contract, scenes = _complete_report()
        digest = evaluator.e12.canonical_digest(contract)
        evaluator._validate_complete_report(
            report,
            contract=contract,
            contract_digest=digest,
            scenes=scenes,
        )

        corruptions = []
        missing = copy.deepcopy(report)
        missing["rows"] = missing["rows"][:-1]
        corruptions.append(missing)
        duplicate = copy.deepcopy(report)
        duplicate["rows"][-1]["image"] = 16
        corruptions.append(duplicate)
        bad_cache = copy.deepcopy(report)
        bad_cache["rows"][0]["clean_score_cache_sha256"] = "0" * 64
        corruptions.append(bad_cache)
        bad_summary = copy.deepcopy(report)
        bad_summary["summary"]["mean_full_prefix_precision"] = 0.0
        corruptions.append(bad_summary)
        bad_decision = copy.deepcopy(report)
        bad_decision["decision"]["passed"] = False
        corruptions.append(bad_decision)
        bad_contract = copy.deepcopy(report)
        bad_contract["run_contract"] = {}
        corruptions.append(bad_contract)
        forbidden = copy.deepcopy(report)
        forbidden["rows"][0]["solve_only_ssim"] = 1.0
        corruptions.append(forbidden)

        for corrupted in corruptions:
            with self.subTest(keys=sorted(corrupted)):
                with self.assertRaises(evaluator.E17ContractError):
                    evaluator._validate_complete_report(
                        corrupted,
                        contract=contract,
                        contract_digest=digest,
                        scenes=scenes,
                    )

    def test_summary_rejects_missing_duplicate_and_wrong_ids(self) -> None:
        rows = [_row(image) for image in evaluator.e12.CALIBRATION_IDS]
        invalid = (
            rows[:-1],
            [*rows[:-1], copy.deepcopy(rows[-2])],
            [*rows[:-1], {**rows[-1], "image": 99}],
        )
        for value in invalid:
            with self.subTest(ids=[row["image"] for row in value]):
                with self.assertRaises(evaluator.E17ContractError):
                    evaluator.summarize(value)

    def test_run_gate_has_no_board_solver_restorer_or_ssim_path(self) -> None:
        scenes = [
            SimpleNamespace(
                image_id=image,
                validation_name=f"validation_{image}",
                permutation=np.arange(576, dtype=np.int64),
            )
            for image in evaluator.e12.CALIBRATION_IDS
        ]
        e12_report = {"scene_provenance_digest": "scene-digest"}
        records = {
            image: {
                "path": f"E:/cache/{image}.npz",
                "sha256": f"{image:064x}",
            }
            for image in evaluator.e12.CALIBRATION_IDS
        }

        def fake_cache(scene: object, *_args: object) -> object:
            image = int(scene.image_id)
            return SimpleNamespace(
                cc_candidates=np.zeros((576, 128), dtype=np.int64),
                cc_scores=np.zeros((4, 576, 128), dtype=np.float32),
                sha256=f"{image:064x}",
            )

        zeros = _zeros()
        with mock.patch.object(
            evaluator,
            "_load_verified_structure_inputs",
            return_value=(e12_report, {}, scenes),
        ), mock.patch.object(
            evaluator.e14, "_clean_cache_records", return_value=records
        ), mock.patch.object(
            evaluator.e14, "_load_cc_cache", side_effect=fake_cache
        ), mock.patch.object(
            evaluator.e12, "dense_from_graph", return_value=zeros
        ), mock.patch.object(
            evaluator, "measure_structure", return_value=_structure_values()
        ), mock.patch.object(
            evaluator, "_source_provenance", return_value={"source": "sha"}
        ), mock.patch.object(
            evaluator,
            "_runtime_provenance",
            return_value=evaluator.EXPECTED_RUNTIME_PROVENANCE,
        ), mock.patch.object(
            evaluator, "_atomic_write_json"
        ) as writer, mock.patch.object(
            Path, "is_file", return_value=False
        ), mock.patch.object(
            evaluator.e14,
            "load_verified_e12_inputs",
            side_effect=AssertionError("board/metric loader called"),
        ) as old_loader, mock.patch.object(
            evaluator.e14,
            "_e12_rr_rows",
            side_effect=AssertionError("RR rows read"),
        ) as rr_rows, mock.patch.object(
            evaluator.e14,
            "verify_rr_means",
            side_effect=AssertionError("SSIM means read"),
        ) as rr_means, mock.patch.object(
            evaluator.e14,
            "_replay_rr96",
            side_effect=AssertionError("board replay called"),
        ) as replay, mock.patch.object(
            evaluator.e12,
            "solve_dense",
            side_effect=AssertionError("solver called"),
        ) as solver, mock.patch.object(
            evaluator.e12,
            "fixed_nlm",
            side_effect=AssertionError("restorer called"),
        ) as restorer, mock.patch.object(
            evaluator.e12,
            "board_metrics",
            side_effect=AssertionError("SSIM/board metrics called"),
        ) as board_metrics:
            output = evaluator.run_gate(
                evaluator.DEFAULT_RAW_CACHE_DIR,
                evaluator.DEFAULT_CALIBRATION_REPORT,
                evaluator.DEFAULT_E12_REPORT,
                Path("E:/pazzle_work/e17/unit_no_write.json"),
            )

        self.assertEqual(len(output["rows"]), 8)
        self.assertNotIn("rr_reproducibility", output)
        self.assertTrue(output["decision"]["passed"])
        self.assertEqual(writer.call_count, 10)
        for forbidden_call in (
            old_loader,
            rr_rows,
            rr_means,
            replay,
            solver,
            restorer,
            board_metrics,
        ):
            forbidden_call.assert_not_called()
        for row in output["rows"]:
            self.assertFalse(
                any(
                    marker in key.lower()
                    for key in row
                    for marker in (
                        "ssim",
                        "board",
                        "canvas",
                        "nlm",
                        "placement",
                        "neighbour",
                    )
                )
            )


class DecisionTests(unittest.TestCase):
    def test_gate_is_inclusive_and_every_check_is_required(self) -> None:
        passing = {
            "selected_claims_each": [192],
            "mean_full_prefix_precision": 0.95,
            "mean_incremental96_precision": 0.90,
            "worst_incremental96_precision": 0.80,
            "mean_exact_pure_rigid_tile_coverage": 0.35,
            "worst_exact_pure_rigid_tile_coverage": 0.25,
            "mean_largest_exact_pure_component_size": 8.0,
        }
        self.assertTrue(evaluator.decision(passing)["passed"])
        failures = (
            {**passing, "selected_claims_each": [191]},
            {**passing, "mean_full_prefix_precision": 0.949999},
            {**passing, "mean_incremental96_precision": 0.899999},
            {**passing, "worst_incremental96_precision": 0.799999},
            {**passing, "mean_exact_pure_rigid_tile_coverage": 0.349999},
            {**passing, "worst_exact_pure_rigid_tile_coverage": 0.249999},
            {**passing, "mean_largest_exact_pure_component_size": 7.999999},
        )
        for failing in failures:
            with self.subTest(failing=failing):
                self.assertFalse(evaluator.decision(failing)["passed"])


if __name__ == "__main__":
    unittest.main()
