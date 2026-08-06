from __future__ import annotations

import copy
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_e18_absolute_frame_oracle as evaluator  # noqa: E402


def _diagnostics(
    *,
    rigid_coverage: float = 0.40,
    precision: float = 0.90,
    cycle_ratio: float = 0.10,
    cap_hit: bool = False,
) -> dict[str, object]:
    cycle_fraction = Fraction(cycle_ratio).limit_denominator(575)
    rigid_components = cycle_fraction.denominator + 1
    cycle_rank = cycle_fraction.numerator
    component_contacts = rigid_components - 1 + cycle_rank
    precision_fraction = Fraction(precision).limit_denominator(100)
    seam_count = precision_fraction.denominator
    true_seams = precision_fraction.numerator
    rigid_tiles = int(round(576 * rigid_coverage))
    translations = [[0, 3, 4]] + [
        [component, component // 24, component % 24]
        for component in range(1, rigid_components)
    ]
    seams = [[tile, tile + 1, 0, 1] for tile in range(seam_count)]
    return {
        "cc192_component_count": 100,
        "cc192_nontrivial_components": 50,
        "cc192_nontrivial_tiles": 272,
        "root_component_id": 0,
        "root_component_size": 12,
        "root_origins_evaluated": 400,
        "bridge_claims": 1000,
        "attachment_rounds": 20,
        "proposal_evaluations": 1000,
        "expansion_cap_hit": cap_hit,
        "absolute_layouts_retained": 8,
        "rigid_components_placed": rigid_components,
        "rigid_tiles_placed": rigid_tiles,
        "rigid_coverage": rigid_tiles / 576,
        "unplaced_nontrivial_components": 30,
        "unplaced_nontrivial_tiles": 40,
        "satisfied_bridge_claims": 30,
        "unique_component_contacts": component_contacts,
        "unique_physical_cross_seams": seam_count,
        "component_cycle_rank": cycle_rank,
        "component_cycle_rank_ratio": float(cycle_fraction),
        "accepted_cross_seams": seams,
        "true_accepted_cross_seams": true_seams,
        "accepted_cross_seam_precision": float(precision_fraction),
        "root_origin": [3, 4],
        "translations": translations,
        "wave_commits": 10,
        "wave_rounds": 2,
        "hungarian_rounds": 2,
        "terminal_neural_objective": -1.0,
        "terminal_lab_tie_score": -1.0,
    }


def _candidate_row(
    image: int,
    *,
    placement: float = 0.03,
    neighbour: float = 0.25,
    solve: float | None = None,
    precision: float = 0.90,
    cycle_ratio: float = 0.10,
    rigid_coverage: float = 0.40,
    cap_hit: bool = False,
) -> dict[str, object]:
    if solve is None:
        solve = evaluator.EXPECTED_RR_MEAN_SOLVE_SSIM + 0.006
    board = np.arange(576, dtype=np.int64)
    return {
        "image": image,
        "validation_name": f"validation_{image}",
        "arm": "E18_absolute_frame_beam",
        "placement": placement,
        "neighbour": neighbour,
        "right": neighbour,
        "down": neighbour,
        "solve_only_ssim": solve,
        "board": board.tolist(),
        "board_sha256": evaluator.e12.array_sha256(board),
        "solved_corrupted_canvas_sha256": f"{image + 100:064x}",
        "diagnostics": _diagnostics(
            rigid_coverage=rigid_coverage,
            precision=precision,
            cycle_ratio=cycle_ratio,
            cap_hit=cap_hit,
        ),
        "solver_seconds": 1.0,
        "clean_score_cache_sha256": f"{image + 200:064x}",
    }


def _rr_rows() -> dict[int, dict[str, object]]:
    return {
        image: {
            "image": image,
            "validation_name": f"validation_{image}",
            "placement": 0.0,
            "neighbour": 0.0,
            "right": 0.0,
            "down": 0.0,
            "solve_only_ssim": evaluator.EXPECTED_RR_MEAN_SOLVE_SSIM,
            "final_ssim": 0.10,
            "objective": 0.0,
            "board_sha256": f"{image:064x}",
            "solved_corrupted_canvas_sha256": f"{image + 100:064x}",
            "restored_canvas_sha256": f"{image + 300:064x}",
        }
        for image in evaluator.e12.CALIBRATION_IDS
    }


def _contract(candidate: list[dict[str, object]]) -> dict[str, object]:
    return {
        "frozen": True,
        "clean_score_caches": {
            str(row["image"]): {
                "path": f"E:/cache/{row['image']}.npz",
                "sha256": row["clean_score_cache_sha256"],
            }
            for row in candidate
        },
    }


def _validation_scenes() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            image_id=image,
            validation_name=f"validation_{image}",
            permutation=np.arange(576, dtype=np.int64),
        )
        for image in evaluator.e12.CALIBRATION_IDS
    ]


def _actual_diagnostics() -> evaluator.frame.SolveDiagnostics:
    return evaluator.frame.SolveDiagnostics(
        cc192_component_count=100,
        cc192_nontrivial_components=50,
        cc192_nontrivial_tiles=272,
        root_component_id=0,
        root_component_size=12,
        root_origins_evaluated=400,
        bridge_claims=1000,
        attachment_rounds=20,
        proposal_evaluations=1000,
        expansion_cap_hit=False,
        absolute_layouts_retained=8,
        rigid_components_placed=20,
        rigid_tiles_placed=230,
        rigid_coverage=230 / 576,
        unplaced_nontrivial_components=30,
        unplaced_nontrivial_tiles=42,
        satisfied_bridge_claims=30,
        unique_component_contacts=22,
        unique_physical_cross_seams=1,
        component_cycle_rank=3,
        component_cycle_rank_ratio=0.10,
        accepted_cross_seams=((0, 1, 0, 1),),
        root_origin=(3, 4),
        translations=((0, 3, 4),),
        wave_commits=10,
        wave_rounds=2,
        hungarian_rounds=2,
        terminal_neural_objective=-1.0,
        terminal_lab_tie_score=-1.0,
    )


class FrozenContractTests(unittest.TestCase):
    def test_protocol_rules_and_cli_are_literal(self) -> None:
        self.assertEqual(
            evaluator.DECODER_RULE,
            {
                "expansion_cap_hit_scenes_max": 0,
                "strict_bijection_scenes": 8,
                "mean_rigid_coverage_min": 0.35,
                "mean_accepted_cross_seam_precision_min": 0.85,
                "worst_accepted_cross_seam_precision_min": 0.70,
                "mean_component_cycle_rank_ratio_min": 0.05,
                "mean_placement_min": 0.02,
                "mean_neighbour_min": 0.20,
                "candidate_minus_rr96_mean_solve_ssim_min": 0.005,
            },
        )
        self.assertEqual(
            evaluator.END_TO_END_RULE,
            {
                "candidate_minus_rr96_mean_solve_ssim_min": 0.010,
                "candidate_minus_rr96_mean_final_ssim_min": 0.015,
                "candidate_minus_rr96_final_wins_min": 6,
                "candidate_minus_rr96_worst_final_delta_min": -0.020,
            },
        )
        protocol = evaluator.E18_PROTOCOL
        self.assertEqual(protocol["components"]["max_edges"], 192)
        self.assertEqual(protocol["bridges"]["top_k"], 8)
        self.assertTrue(protocol["bridges"]["single_bridge_allowed"])
        self.assertEqual(protocol["search"]["beam_width"], 256)
        self.assertEqual(protocol["search"]["absolute_layouts_global_per_scene"], 8)
        self.assertEqual(
            protocol["search"]["pre_geometry_translation_rank"],
            [
                "distinct_supporting_claim_count_desc",
                "supporting_claim_score_sum_desc",
                "maximum_supporting_claim_score_desc",
                "component_id_asc",
                "shift_row_asc",
                "shift_col_asc",
            ],
        )
        self.assertEqual(
            protocol["search"]["cap_reaching"],
            "hard_failure_not_truncated_success",
        )
        self.assertFalse(protocol["geometry"]["rotation"])
        self.assertFalse(protocol["geometry"]["reflection"])
        destinations = {
            action.dest
            for action in evaluator.build_parser()._actions
            if action.dest != "help"
        }
        self.assertEqual(
            destinations,
            {
                "raw_cache_dir",
                "calibration_report",
                "e12_report",
                "e17_report",
                "report",
            },
        )

    def test_runtime_and_live_E17_provenance_are_exact(self) -> None:
        self.assertEqual(
            evaluator._runtime_provenance(), evaluator.EXPECTED_RUNTIME_PROVENANCE
        )
        report = evaluator._verify_e17_report(evaluator.DEFAULT_E17_REPORT)
        self.assertTrue(report["decision"]["passed"])
        self.assertEqual(report["stage"], "go_E18_absolute_frame_beam")

    def test_source_provenance_is_direct_and_complete(self) -> None:
        self.assertEqual(
            set(evaluator._source_provenance()),
            {
                "e15_frame_consensus.py",
                "e18_absolute_frame_beam.py",
                "eval_clean_score_oracle.py",
                "eval_e14_cc192_discovery.py",
                "eval_e17_cc192_rigid_viability.py",
                "eval_e18_absolute_frame_oracle.py",
                "imgio.py",
                "placement_metrics.py",
                "rank96_lab_selector.py",
                "solve_buddies.py",
            },
        )

    def test_report_guards_fail_before_input_loading(self) -> None:
        cases = (
            evaluator.E18Paths(
                evaluator.DEFAULT_RAW_CACHE_DIR,
                evaluator.DEFAULT_CALIBRATION_REPORT,
                evaluator.DEFAULT_E12_REPORT,
                evaluator.DEFAULT_E17_REPORT,
                Path("E:/pazzle_work/e18/not_json.txt"),
            ),
            evaluator.E18Paths(
                evaluator.DEFAULT_RAW_CACHE_DIR,
                evaluator.DEFAULT_CALIBRATION_REPORT,
                evaluator.DEFAULT_E12_REPORT,
                evaluator.DEFAULT_E17_REPORT,
                evaluator.DEFAULT_RAW_CACHE_DIR / "report.json",
            ),
            evaluator.E18Paths(
                evaluator.DEFAULT_RAW_CACHE_DIR,
                evaluator.DEFAULT_CALIBRATION_REPORT,
                evaluator.DEFAULT_E12_REPORT,
                evaluator.DEFAULT_E17_REPORT,
                evaluator.DEFAULT_E12_REPORT.parent / "score_cache" / "report.json",
            ),
            evaluator.E18Paths(
                evaluator.DEFAULT_RAW_CACHE_DIR,
                evaluator.DEFAULT_CALIBRATION_REPORT,
                evaluator.DEFAULT_E12_REPORT,
                evaluator.DEFAULT_E17_REPORT,
                evaluator.DEFAULT_E17_REPORT,
            ),
        )
        with mock.patch.object(evaluator.e14, "load_verified_e12_inputs") as loader:
            for paths in cases:
                with self.subTest(report=paths.report):
                    with self.assertRaises(evaluator.E18ContractError):
                        evaluator.run_discovery(paths)
        loader.assert_not_called()


class TruthAndMetricTests(unittest.TestCase):
    def test_canonical_seam_truth_rejects_row_wrap(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        self.assertTrue(evaluator._seam_is_true((0, 1, 0, 1), permutation))
        self.assertFalse(evaluator._seam_is_true((23, 24, 0, 1), permutation))
        with self.assertRaisesRegex(evaluator.E18ContractError, "canonical"):
            evaluator._seam_is_true((1, 0, 0, -1), permutation)

    def test_evaluation_assembles_only_original_corrupted_upright_tiles(self) -> None:
        tiles = np.full((576, 20, 20, 3), 17, dtype=np.uint8)
        target = np.zeros((480, 480, 3), dtype=np.uint8)
        scene = SimpleNamespace(
            tiles_uint8=tiles,
            target_uint8=target,
            permutation=np.arange(576, dtype=np.int64),
        )
        result = evaluator.frame.SolveResult(
            board=np.arange(576, dtype=np.int64),
            diagnostics=_actual_diagnostics(),
        )
        with mock.patch.object(
            evaluator, "assemble", return_value=target.copy()
        ) as assembled, mock.patch.object(evaluator, "sk_ssim", return_value=1.0):
            row = evaluator.evaluate_solve_only(scene, result)
        self.assertIs(assembled.call_args.args[0], tiles)
        self.assertTrue(np.array_equal(assembled.call_args.args[1], result.board))
        self.assertEqual(row["diagnostics"]["accepted_cross_seam_precision"], 1.0)
        self.assertEqual(row["solve_only_ssim"], 1.0)

    def test_non_self_inverse_permutation_uses_position_to_tile_truth(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        permutation[:3] = [1, 2, 0]
        expected_truth = np.argsort(permutation)
        scene = SimpleNamespace(
            tiles_uint8=np.zeros((576, 20, 20, 3), dtype=np.uint8),
            target_uint8=np.zeros((480, 480, 3), dtype=np.uint8),
            permutation=permutation,
        )
        result = evaluator.frame.SolveResult(
            board=np.arange(576, dtype=np.int64),
            diagnostics=_actual_diagnostics(),
        )
        with mock.patch.object(
            evaluator, "placement_accuracy", return_value=(0.0, None)
        ) as placement, mock.patch.object(
            evaluator, "neighbour_accuracy", return_value=(0.0, 0.0, 0.0)
        ) as neighbour, mock.patch.object(
            evaluator,
            "assemble",
            return_value=np.zeros((480, 480, 3), dtype=np.uint8),
        ), mock.patch.object(evaluator, "sk_ssim", return_value=0.0):
            evaluator.evaluate_solve_only(scene, result)
        self.assertTrue(np.array_equal(placement.call_args.args[1], expected_truth))
        self.assertTrue(np.array_equal(neighbour.call_args.args[1], expected_truth))

    def test_decoder_gate_is_inclusive_and_every_check_is_required(self) -> None:
        passing = {
            "expansion_cap_hit_scenes": 0,
            "strict_bijection_scenes": 8,
            "mean_rigid_coverage": 0.35,
            "mean_accepted_cross_seam_precision": 0.85,
            "worst_accepted_cross_seam_precision": 0.70,
            "mean_component_cycle_rank_ratio": 0.05,
            "mean_placement": 0.02,
            "mean_neighbour": 0.20,
            "candidate_minus_rr96_mean_solve_ssim": 0.005,
        }
        self.assertTrue(evaluator.decoder_decision(passing)["passed"])
        failures = (
            {**passing, "expansion_cap_hit_scenes": 1},
            {**passing, "strict_bijection_scenes": 7},
            {**passing, "mean_rigid_coverage": 0.34999},
            {**passing, "mean_accepted_cross_seam_precision": 0.84999},
            {**passing, "worst_accepted_cross_seam_precision": 0.69999},
            {**passing, "mean_component_cycle_rank_ratio": 0.04999},
            {**passing, "mean_placement": 0.01999},
            {**passing, "mean_neighbour": 0.19999},
            {**passing, "candidate_minus_rr96_mean_solve_ssim": 0.00499},
        )
        for value in failures:
            with self.subTest(value=value):
                self.assertFalse(evaluator.decoder_decision(value)["passed"])

    def test_end_to_end_gate_is_inclusive_and_every_check_is_required(self) -> None:
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
        failures = []
        for metric, key, value in (
            ("solve_only_ssim", "mean_delta", 0.00999),
            ("final_ssim", "mean_delta", 0.01499),
            ("final_ssim", "wins", 5),
            ("final_ssim", "worst_delta", -0.02001),
        ):
            failed = copy.deepcopy(passing)
            failed["metrics"][metric][key] = value
            failures.append(failed)
        for value in failures:
            with self.subTest(value=value):
                self.assertFalse(evaluator.end_to_end_decision(value)["passed"])

    def test_summary_rejects_wrong_ids_and_disconnected_partial_graph(self) -> None:
        rows = [_candidate_row(image) for image in evaluator.e12.CALIBRATION_IDS]
        evaluator.summarize_decoder(rows)
        duplicate = copy.deepcopy(rows)
        duplicate[-1]["image"] = 16
        with self.assertRaises(evaluator.E18ContractError):
            evaluator.summarize_decoder(duplicate)
        disconnected = copy.deepcopy(rows)
        disconnected[0]["diagnostics"]["unique_component_contacts"] = 0
        with self.assertRaisesRegex(evaluator.E18ContractError, "not connected"):
            evaluator.summarize_decoder(disconnected)


class CompleteReportTests(unittest.TestCase):
    def test_decoder_kill_complete_report_is_recomputed_fail_closed(self) -> None:
        rr_rows = _rr_rows()
        candidate = [
            _candidate_row(image, neighbour=0.0, solve=0.0, precision=1.0)
            for image in evaluator.e12.CALIBRATION_IDS
        ]
        decoder_summary = evaluator.summarize_decoder(candidate)
        decoder_gate = evaluator.decoder_decision(decoder_summary)
        self.assertFalse(decoder_gate["passed"])
        contract = _contract(candidate)
        digest = evaluator.e12.canonical_digest(contract)
        rr_serialized = [
            {**rr_rows[image], "solver_replay_seconds": 0.0}
            for image in evaluator.e12.CALIBRATION_IDS
        ]
        report: dict[str, object] = {
            "schema_version": evaluator.SCHEMA_VERSION,
            "schema": evaluator.REPORT_SCHEMA,
            "experiment": evaluator.EXPERIMENT,
            "status": "complete",
            "stage": "kill_decoder",
            "protocol": evaluator.E18_PROTOCOL,
            "protocol_sha256": evaluator.e12.canonical_digest(
                evaluator.E18_PROTOCOL
            ),
            "run_contract": contract,
            "run_contract_sha256": digest,
            "rows": {"RR96": rr_serialized, "candidate": candidate},
            "completed_decoder_images": list(evaluator.e12.CALIBRATION_IDS),
            "completed_nlm_images": [],
            "decoder_summary": decoder_summary,
            "decisions": {
                "decoder": decoder_gate,
                "end_to_end": {"status": "not_run"},
            },
            "runtime_seconds": 1.0,
        }
        evaluator._validate_complete_report(
            report,
            contract=contract,
            contract_digest=digest,
            rr_rows=rr_rows,
            scenes=_validation_scenes(),
        )
        corruptions = []
        bad_rows = copy.deepcopy(report)
        bad_rows["rows"]["candidate"] = bad_rows["rows"]["candidate"][:-1]
        corruptions.append(bad_rows)
        bad_summary = copy.deepcopy(report)
        bad_summary["decoder_summary"]["mean_neighbour"] = 1.0
        corruptions.append(bad_summary)
        bad_decision = copy.deepcopy(report)
        bad_decision["decisions"]["decoder"]["passed"] = True
        corruptions.append(bad_decision)
        bad_nlm = copy.deepcopy(report)
        bad_nlm["completed_nlm_images"] = list(evaluator.e12.CALIBRATION_IDS)
        corruptions.append(bad_nlm)
        for corrupted in corruptions:
            with self.subTest(keys=sorted(corrupted)):
                with self.assertRaises(evaluator.E18ContractError):
                    evaluator._validate_complete_report(
                        corrupted,
                        contract=contract,
                        contract_digest=digest,
                        rr_rows=rr_rows,
                        scenes=_validation_scenes(),
                    )

    def test_keep_complete_report_is_recomputed_fail_closed(self) -> None:
        rr_rows = _rr_rows()
        candidate = [
            {
                **_candidate_row(
                    image,
                    neighbour=0.25,
                    solve=evaluator.EXPECTED_RR_MEAN_SOLVE_SSIM + 0.020,
                    precision=1.0,
                ),
                "final_ssim": 0.20,
                "restored_canvas_sha256": f"{image + 300:064x}",
            }
            for image in evaluator.e12.CALIBRATION_IDS
        ]
        decoder_summary = evaluator.summarize_decoder(candidate)
        decoder_gate = evaluator.decoder_decision(decoder_summary)
        self.assertTrue(decoder_gate["passed"])
        contract = _contract(candidate)
        digest = evaluator.e12.canonical_digest(contract)
        comparison = dict(evaluator.e14.paired_summary(candidate, rr_rows))
        comparison["candidate_arm"] = "E18_absolute_frame_beam"
        end_gate = evaluator.end_to_end_decision(comparison)
        self.assertTrue(end_gate["passed"])
        report: dict[str, object] = {
            "schema_version": evaluator.SCHEMA_VERSION,
            "schema": evaluator.REPORT_SCHEMA,
            "experiment": evaluator.EXPERIMENT,
            "status": "complete",
            "stage": end_gate["status"],
            "protocol": evaluator.E18_PROTOCOL,
            "protocol_sha256": evaluator.e12.canonical_digest(
                evaluator.E18_PROTOCOL
            ),
            "run_contract": contract,
            "run_contract_sha256": digest,
            "rows": {
                "RR96": [
                    {**rr_rows[image], "solver_replay_seconds": 0.0}
                    for image in evaluator.e12.CALIBRATION_IDS
                ],
                "candidate": candidate,
            },
            "completed_decoder_images": list(evaluator.e12.CALIBRATION_IDS),
            "completed_nlm_images": list(evaluator.e12.CALIBRATION_IDS),
            "decoder_summary": decoder_summary,
            "comparison": comparison,
            "decisions": {"decoder": decoder_gate, "end_to_end": end_gate},
            "runtime_seconds": 1.0,
        }
        evaluator._validate_complete_report(
            report,
            contract=contract,
            contract_digest=digest,
            rr_rows=rr_rows,
            scenes=_validation_scenes(),
        )
        corruptions = []
        bad_cache = copy.deepcopy(report)
        bad_cache["rows"]["candidate"][0]["clean_score_cache_sha256"] = "f" * 64
        corruptions.append(bad_cache)
        bad_rr = copy.deepcopy(report)
        bad_rr["rows"]["RR96"][0]["final_ssim"] = 0.99
        corruptions.append(bad_rr)
        bad_comparison = copy.deepcopy(report)
        bad_comparison["comparison"]["metrics"]["final_ssim"]["mean_delta"] = 0.99
        corruptions.append(bad_comparison)
        bad_end = copy.deepcopy(report)
        bad_end["decisions"]["end_to_end"]["passed"] = False
        corruptions.append(bad_end)
        bad_truth = copy.deepcopy(report)
        bad_truth["rows"]["candidate"][0]["diagnostics"][
            "accepted_cross_seams"
        ] = [[0, 2, 0, 1]]
        corruptions.append(bad_truth)
        for corrupted in corruptions:
            with self.subTest():
                with self.assertRaises(evaluator.E18ContractError):
                    evaluator._validate_complete_report(
                        corrupted,
                        contract=contract,
                        contract_digest=digest,
                        rr_rows=rr_rows,
                        scenes=_validation_scenes(),
                    )


class StagingTests(unittest.TestCase):
    def _run(self, *, decoder_pass: bool):
        scenes = [
            SimpleNamespace(
                image_id=image,
                validation_name=f"validation_{image}",
                tiles_uint8=np.zeros((576, 20, 20, 3), dtype=np.uint8),
                target_uint8=np.zeros((480, 480, 3), dtype=np.uint8),
                permutation=np.arange(576, dtype=np.int64),
            )
            for image in evaluator.e12.CALIBRATION_IDS
        ]
        rr_rows = _rr_rows()
        records = {
            image: {
                "path": f"E:/cache/{image}.npz",
                "sha256": "a" * 64,
            }
            for image in evaluator.e12.CALIBRATION_IDS
        }
        e12_report = {"scene_provenance_digest": "scene-digest"}
        e17_report = {"run_contract_sha256": "e17-contract"}

        def evaluated(scene: object, _result: object) -> dict[str, object]:
            neighbour = 0.25 if decoder_pass else 0.0
            row = _candidate_row(
                int(scene.image_id),
                neighbour=neighbour,
                solve=evaluator.EXPECTED_RR_MEAN_SOLVE_SSIM + 0.011,
            )
            row.pop("image")
            row.pop("validation_name")
            row.pop("arm")
            row.pop("solver_seconds")
            row.pop("clean_score_cache_sha256")
            row["_board"] = np.arange(576, dtype=np.int64)
            row["_solved"] = np.zeros((480, 480, 3), dtype=np.uint8)
            return row

        fake_result = SimpleNamespace()
        cache = SimpleNamespace(
            cc_candidates=np.zeros((576, 128), dtype=np.int64),
            cc_scores=np.zeros((4, 576, 128), dtype=np.float32),
            sha256="a" * 64,
        )
        paths = evaluator.E18Paths(
            evaluator.DEFAULT_RAW_CACHE_DIR,
            evaluator.DEFAULT_CALIBRATION_REPORT,
            evaluator.DEFAULT_E12_REPORT,
            evaluator.DEFAULT_E17_REPORT,
            Path("E:/pazzle_work/e18/unit_no_write.json"),
        )
        with mock.patch.object(
            evaluator.e14,
            "load_verified_e12_inputs",
            return_value=(e12_report, {}, scenes),
        ), mock.patch.object(
            evaluator, "_verify_e17_report", return_value=e17_report
        ), mock.patch.object(
            evaluator.e14, "_e12_rr_rows", return_value=rr_rows
        ), mock.patch.object(
            evaluator.e14, "verify_rr_means", return_value={"ok": True}
        ), mock.patch.object(
            evaluator.e14, "_clean_cache_records", return_value=records
        ), mock.patch.object(
            evaluator.e14, "_replay_rr96", return_value=(None, 0.0, 0.0)
        ), mock.patch.object(
            evaluator.e14, "_load_cc_cache", return_value=cache
        ), mock.patch.object(
            evaluator.e12, "dense_from_graph", return_value=(np.zeros((576, 576), dtype=np.float32),) * 2
        ), mock.patch.object(
            evaluator.frame, "solve_absolute_frame", return_value=fake_result
        ) as solver, mock.patch.object(
            evaluator, "evaluate_solve_only", side_effect=evaluated
        ), mock.patch.object(
            evaluator, "_source_provenance", return_value={"source": "sha"}
        ), mock.patch.object(
            evaluator,
            "_runtime_provenance",
            return_value=evaluator.EXPECTED_RUNTIME_PROVENANCE,
        ), mock.patch.object(
            evaluator, "_atomic_write_json"
        ), mock.patch.object(
            Path, "is_file", return_value=False
        ), mock.patch.object(
            evaluator.e12,
            "fixed_nlm",
            return_value=np.zeros((480, 480, 3), dtype=np.uint8),
        ) as nlm, mock.patch.object(
            evaluator, "sk_ssim", return_value=0.20
        ):
            result = evaluator.run_discovery(paths)
        return result, solver, nlm

    def test_decoder_reject_never_calls_nlm(self) -> None:
        result, solver, nlm = self._run(decoder_pass=False)
        self.assertEqual(result["stage"], "kill_decoder")
        self.assertEqual(solver.call_count, 8)
        nlm.assert_not_called()

    def test_decoder_pass_calls_candidate_nlm_once_per_scene(self) -> None:
        result, solver, nlm = self._run(decoder_pass=True)
        self.assertEqual(solver.call_count, 8)
        self.assertEqual(nlm.call_count, 8)
        self.assertEqual(result["stage"], "go_raw_adaptation_confirmation")
        self.assertEqual(
            result["comparison"]["candidate_arm"], "E18_absolute_frame_beam"
        )


if __name__ == "__main__":
    unittest.main()
