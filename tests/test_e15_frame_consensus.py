from __future__ import annotations

import inspect
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import e15_frame_consensus as e15  # noqa: E402
import eval_e15_frame_consensus_oracle as evaluator  # noqa: E402


def _component(component_id: int, values: dict[int, tuple[int, int]]) -> e15.Component:
    return e15.Component(
        component_id=component_id,
        entries=tuple(sorted((tile, row, col) for tile, (row, col) in values.items())),
    )


def _zero_dense() -> tuple[np.ndarray, np.ndarray]:
    shape = (e15.NUM_TILES, e15.NUM_TILES)
    return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)


class FrozenContractTests(unittest.TestCase):
    def test_constants_are_literal_exact(self) -> None:
        self.assertEqual(e15.SEED_MAX_EDGES, 96)
        self.assertEqual(e15.VOTE_MAX_EDGES, 192)
        self.assertEqual(e15.MIN_DISTINCT_SEAMS, 2)
        self.assertEqual(e15.BEAM_WIDTH, 256)
        self.assertEqual(e15.PROPOSALS_PER_STATE, 64)
        self.assertEqual(e15.RELATIVE_LAYOUTS, 8)
        self.assertEqual(e15.ABSOLUTE_LAYOUTS, 8)
        self.assertEqual(e15.EXPANSION_CAP, 500_000)
        self.assertEqual(e15.SCORE_FLOOR, 1e-8)
        self.assertEqual(e15.HUNGARIAN_ROUNDS, 2)
        self.assertEqual(e15.NULL_WEIGHT, 0.0)
        self.assertEqual(e15.REPAIR_PASSES, 0)

    def test_solver_exposes_no_orientation_budget_or_null_control(self) -> None:
        parameters = set(inspect.signature(e15.solve_frame_consensus).parameters)
        self.assertEqual(parameters, {"right", "down", "tiles"})
        self.assertTrue(
            parameters.isdisjoint(
                {"rotation", "reflection", "seed_budget", "vote_budget", "null_weight"}
            )
        )

    def test_selected_claims_allows_only_frozen_prefixes(self) -> None:
        right, down = _zero_dense()
        edge = (0.5, 0.1, 0, 1, 0, 1)
        with mock.patch.object(e15, "_candidate_edges", return_value=[edge]) as selected:
            claims = e15.selected_claims(right, down, max_edges=192)
        selected.assert_called_once()
        self.assertEqual(selected.call_args.kwargs, {"max_edges": 192, "min_margin": 0.0})
        self.assertEqual(claims[0].identity, (0, 1, 0, 1))
        with self.assertRaises(e15.FrameConsensusError):
            e15.selected_claims(right, down, max_edges=128)


class FrozenEvaluatorContractTests(unittest.TestCase):
    def test_protocol_and_all_three_gates_are_literal_exact(self) -> None:
        self.assertEqual(
            evaluator.STRUCTURAL_RULE,
            {
                "selected_cc96_claims_each": 96,
                "mean_cc96_edge_precision_min": 0.98,
                "mean_cc96_component_coverage_min": 0.25,
                "mean_two_vote_hypothesis_precision_min": 0.98,
                "worst_two_vote_hypothesis_precision_min": 0.90,
                "mean_relation_supported_tile_coverage_min": 0.15,
            },
        )
        self.assertEqual(
            evaluator.DECODER_RULE,
            {
                "expansion_cap_hit_allowed": False,
                "strict_bijection_scenes": 8,
                "mean_rigid_coverage_min": 0.20,
                "all_non_seed_attachments_two_seam": True,
                "mean_placement_min": 0.02,
                "mean_neighbour_min": 0.20,
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
        protocol = evaluator.E15_PROTOCOL
        self.assertEqual(protocol["calibration_ids"], list(range(10, 18)))
        self.assertEqual(protocol["seed"]["max_edges"], 96)
        self.assertEqual(protocol["translation_votes"]["max_edges"], 192)
        self.assertEqual(
            protocol["translation_votes"]["distinct_physical_seams_required"],
            2,
        )
        self.assertFalse(protocol["geometry"]["rotation"])
        self.assertFalse(protocol["geometry"]["reflection"])
        self.assertEqual(protocol["search"]["null_weight"], 0.0)
        self.assertEqual(protocol["search"]["absolute_layouts_per_scene"], 8)
        self.assertNotIn("absolute_layouts_per_relative", protocol["search"])
        self.assertEqual(protocol["search"]["expansion_cap_per_scene"], 500_000)
        self.assertEqual(protocol["residual"]["hungarian_rounds"], 2)
        self.assertEqual(
            protocol["restoration"]["scope"],
            "candidate_once_only_after_decoder_gate",
        )
        self.assertEqual(
            set(evaluator._source_provenance()),
            {
                "e15_frame_consensus.py",
                "eval_e15_frame_consensus_oracle.py",
                "eval_e14_cc192_discovery.py",
                "eval_clean_score_oracle.py",
                "imgio.py",
                "placement_metrics.py",
                "rank96_lab_selector.py",
                "solve_buddies.py",
            },
        )
        self.assertEqual(protocol["runtime_provenance"]["scipy"], "1.16.2")

    def test_cli_exposes_paths_only(self) -> None:
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
                "e14_report",
                "report",
            },
        )
        self.assertTrue(
            destinations.isdisjoint(
                {
                    "budget",
                    "max_edges",
                    "margin",
                    "beam",
                    "cap",
                    "rotation",
                    "reflection",
                    "null_weight",
                    "repair",
                    "device",
                }
            )
        )

    def test_runtime_is_pinned_fail_closed(self) -> None:
        self.assertEqual(
            evaluator._runtime_provenance(), evaluator.EXPECTED_RUNTIME_PROVENANCE
        )
        with mock.patch.object(
            evaluator.platform, "python_version", return_value="0.0.0"
        ):
            with self.assertRaisesRegex(evaluator.E15ContractError, "runtime drifted"):
                evaluator._runtime_provenance()

    def test_e_drive_and_input_cache_guards(self) -> None:
        self.assertEqual(
            evaluator._require_e_drive(
                Path("E:/pazzle_work/report.json"), label="test"
            ).drive.upper(),
            "E:",
        )
        with self.assertRaisesRegex(evaluator.E15ContractError, "must stay on E"):
            evaluator._require_e_drive(ROOT / "report.json", label="test")

        paths = evaluator.E15Paths(
            raw_cache_dir=Path("E:/pazzle_work/input_cache"),
            calibration_report=ROOT / "calibration.json",
            e12_report=Path("E:/pazzle_work/e12.json"),
            e14_report=Path("E:/pazzle_work/e14.json"),
            report=Path("E:/pazzle_work/input_cache/report.json"),
        )
        with self.assertRaisesRegex(
            evaluator.E15ContractError, "inside an input cache"
        ):
            evaluator.run_discovery(paths)

        clean_cache_paths = evaluator.E15Paths(
            **{
                **paths.__dict__,
                "report": evaluator.DEFAULT_E12_REPORT.parent
                / "score_cache"
                / "report.json",
            }
        )
        with self.assertRaisesRegex(
            evaluator.E15ContractError, "inside an input cache"
        ):
            evaluator.run_discovery(clean_cache_paths)

        wrong_suffix = evaluator.E15Paths(
            **{**paths.__dict__, "report": Path("E:/pazzle_work/e15_report.txt")}
        )
        with self.assertRaisesRegex(evaluator.E15ContractError, "must be a .json"):
            evaluator.run_discovery(wrong_suffix)

        for input_report in (
            paths.e12_report,
            paths.e14_report,
            paths.calibration_report,
        ):
            overwrite = evaluator.E15Paths(
                **{**paths.__dict__, "report": input_report}
            )
            with self.subTest(input_report=input_report):
                with self.assertRaises(evaluator.E15ContractError):
                    evaluator.run_discovery(overwrite)


class EvaluatorGateTests(unittest.TestCase):
    def test_structural_gate_is_inclusive_and_requires_every_check(self) -> None:
        passing = {
            "selected_cc96_claims_each": [96],
            "mean_cc96_edge_precision": 0.98,
            "mean_cc96_component_coverage": 0.25,
            "mean_two_vote_hypothesis_precision": 0.98,
            "worst_two_vote_hypothesis_precision": 0.90,
            "mean_relation_supported_tile_coverage": 0.15,
        }
        self.assertTrue(evaluator.structural_decision(passing)["passed"])
        failures = (
            {**passing, "selected_cc96_claims_each": [95]},
            {**passing, "mean_cc96_edge_precision": 0.979999},
            {**passing, "mean_cc96_component_coverage": 0.249999},
            {**passing, "mean_two_vote_hypothesis_precision": 0.979999},
            {**passing, "worst_two_vote_hypothesis_precision": 0.899999},
            {**passing, "mean_relation_supported_tile_coverage": 0.149999},
        )
        for failing in failures:
            with self.subTest(failing=failing):
                self.assertFalse(evaluator.structural_decision(failing)["passed"])

    def test_decoder_gate_is_inclusive_and_requires_every_check(self) -> None:
        passing = {
            "expansion_cap_hit_scenes": 0,
            "strict_bijection_scenes": 8,
            "mean_rigid_coverage": 0.20,
            "all_non_seed_attachments_two_seam": True,
            "mean_placement": 0.02,
            "mean_neighbour": 0.20,
        }
        self.assertTrue(evaluator.decoder_decision(passing)["passed"])
        failures = (
            {**passing, "expansion_cap_hit_scenes": 1},
            {**passing, "strict_bijection_scenes": 7},
            {**passing, "mean_rigid_coverage": 0.199999},
            {**passing, "all_non_seed_attachments_two_seam": False},
            {**passing, "mean_placement": 0.019999},
            {**passing, "mean_neighbour": 0.199999},
        )
        for failing in failures:
            with self.subTest(failing=failing):
                self.assertFalse(evaluator.decoder_decision(failing)["passed"])

    def test_end_to_end_gate_is_inclusive_and_requires_every_check(self) -> None:
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
            ("solve_only_ssim", "mean_delta", 0.009999),
            ("final_ssim", "mean_delta", 0.014999),
            ("final_ssim", "wins", 5),
            ("final_ssim", "worst_delta", -0.020001),
        ):
            metrics = {
                name: dict(values) for name, values in passing["metrics"].items()
            }
            metrics[metric][key] = value
            failures.append({"metrics": metrics})
        for failing in failures:
            with self.subTest(failing=failing):
                self.assertFalse(evaluator.end_to_end_decision(failing)["passed"])

    def test_hypothesis_truth_requires_every_distinct_claim(self) -> None:
        components = (
            _component(0, {0: (0, 0), 24: (1, 0)}),
            _component(1, {1: (0, 0), 25: (1, 0)}),
        )
        true_claim = e15.SeamClaim(0.9, 0, 1, 0, 1)
        false_claim = e15.SeamClaim(0.8, 24, 25, 1, 0)
        hypothesis = e15.TranslationHypothesis(
            hypothesis_id=0,
            left_component=0,
            right_component=1,
            offset_row=0,
            offset_col=1,
            claims=(true_claim, false_claim),
        )
        measured = evaluator.measure_structure(
            components,
            (true_claim, false_claim),
            (hypothesis,),
            np.arange(576, dtype=np.int64),
        )
        self.assertEqual(measured["true_cc96_claims"], 1)
        self.assertEqual(measured["true_two_vote_hypotheses"], 0)
        self.assertEqual(measured["two_vote_hypothesis_precision"], 0.0)
        self.assertEqual(measured["relation_supported_tiles"], 4)

    def test_empty_hypothesis_set_has_zero_precision(self) -> None:
        components = (_component(0, {0: (0, 0)}),)
        measured = evaluator.measure_structure(
            components,
            (),
            (),
            np.arange(576, dtype=np.int64),
        )
        self.assertEqual(measured["two_vote_hypotheses"], 0)
        self.assertEqual(measured["two_vote_hypothesis_precision"], 0.0)
        self.assertEqual(measured["relation_supported_tile_coverage"], 0.0)

    def test_summaries_require_exactly_eight_rows(self) -> None:
        with self.assertRaisesRegex(evaluator.E15ContractError, "exactly eight"):
            evaluator.summarize_structure([{}] * 7)
        with self.assertRaisesRegex(evaluator.E15ContractError, "exactly eight"):
            evaluator.summarize_decoder([{}] * 7)

    def test_private_arrays_are_never_serialised(self) -> None:
        board = np.arange(576, dtype=np.int64)
        solved = np.zeros((480, 480, 3), dtype=np.uint8)
        serialised = evaluator._serialisable_decoder_row(
            {"metric": 1.0, "_board": board, "_solved": solved}
        )
        self.assertEqual(serialised, {"metric": 1.0})


class StagedExecutionTests(unittest.TestCase):
    def _run_with_rows(
        self,
        structure_row: dict[str, object],
        decoder_row: dict[str, object] | None,
    ) -> tuple[dict[str, object], mock.Mock, mock.Mock]:
        scenes = [
            SimpleNamespace(
                image_id=image,
                validation_name=f"synthetic_{image}.png",
                permutation=np.arange(576, dtype=np.int64),
                tiles_uint8=np.zeros((576, 20, 20, 3), dtype=np.uint8),
                target_uint8=np.zeros((480, 480, 3), dtype=np.uint8),
            )
            for image in range(10, 18)
        ]
        rr_rows = {
            image: {
                **{metric: 0.0 for metric in evaluator.e14.BOARD_METRICS},
                "objective": 0.0,
                "board_sha256": f"{image:064x}",
                "solved_corrupted_canvas_sha256": f"{image + 20:064x}",
                "restored_canvas_sha256": f"{image + 40:064x}",
            }
            for image in range(10, 18)
        }
        clean_records = {
            image: {
                "path": f"E:/pazzle_work/score_cache/image_{image:04d}.npz",
                "sha256": f"{image + 60:064x}",
            }
            for image in range(10, 18)
        }
        clean_cache = SimpleNamespace(
            cc_candidates=np.empty((576, 0), dtype=np.int64),
            cc_scores=np.empty((4, 576, 0), dtype=np.float32),
            sha256="f" * 64,
        )
        right, down = _zero_dense()
        component = _component(0, {0: (0, 0)})
        owner = np.zeros(576, dtype=np.int64)
        paths = evaluator.E15Paths(
            raw_cache_dir=Path("E:/pazzle_work/unit_e15_raw"),
            calibration_report=ROOT / "unit_calibration.json",
            e12_report=Path("E:/pazzle_work/unit_e12.json"),
            e14_report=Path("E:/pazzle_work/unit_e14.json"),
            report=Path("E:/pazzle_work/unit_e15_stage/report.json"),
        )
        solve_mock = mock.Mock(return_value=object())
        nlm_mock = mock.Mock(
            side_effect=lambda image: np.ascontiguousarray(image, dtype=np.uint8)
        )
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    evaluator.e14,
                    "load_verified_e12_inputs",
                    return_value=(
                        {"scene_provenance_digest": "a" * 64},
                        {},
                        scenes,
                    ),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator,
                    "_verify_e14_report",
                    return_value={"run_contract_sha256": "b" * 64},
                )
            )
            stack.enter_context(
                mock.patch.object(evaluator.e14, "_e12_rr_rows", return_value=rr_rows)
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator.e14,
                    "verify_rr_means",
                    return_value={"verified": True},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator.e14,
                    "_clean_cache_records",
                    return_value=clean_records,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator.e14, "_replay_rr96", return_value=(np.arange(576), 0.0, 0.0)
                )
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator.e14, "_load_cc_cache", return_value=clean_cache
                )
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator.e12, "dense_from_graph", return_value=(right, down)
                )
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator.frame,
                    "build_seed_components",
                    return_value=((component,), owner),
                )
            )
            stack.enter_context(
                mock.patch.object(evaluator.frame, "selected_claims", return_value=())
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator.frame, "build_translation_hypotheses", return_value=()
                )
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator,
                    "measure_structure",
                    return_value=dict(structure_row),
                )
            )
            stack.enter_context(
                mock.patch.object(evaluator.frame, "solve_frame_consensus", solve_mock)
            )
            if decoder_row is not None:
                stack.enter_context(
                    mock.patch.object(
                        evaluator,
                        "evaluate_solve_only",
                        return_value=dict(decoder_row),
                    )
                )
            stack.enter_context(
                mock.patch.object(evaluator.e12, "fixed_nlm", nlm_mock)
            )
            stack.enter_context(
                mock.patch.object(evaluator, "sk_ssim", return_value=0.2)
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator,
                    "_source_provenance",
                    return_value={"unit": "c" * 64},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    evaluator,
                    "_runtime_provenance",
                    return_value=dict(evaluator.EXPECTED_RUNTIME_PROVENANCE),
                )
            )
            stack.enter_context(mock.patch.object(evaluator, "_atomic_write_json"))
            result = dict(evaluator.run_discovery(paths))
        return result, solve_mock, nlm_mock

    @staticmethod
    def _passing_structure() -> dict[str, object]:
        return {
            "selected_cc96_claims": 96,
            "true_cc96_claims": 96,
            "cc96_edge_precision": 1.0,
            "cc96_component_count": 1,
            "cc96_component_tiles": 200,
            "cc96_component_coverage": 0.30,
            "two_vote_hypotheses": 4,
            "true_two_vote_hypotheses": 4,
            "two_vote_hypothesis_precision": 1.0,
            "relation_supported_tiles": 100,
            "relation_supported_tile_coverage": 0.20,
        }

    @staticmethod
    def _decoder_row(*, neighbour: float) -> dict[str, object]:
        return {
            "placement": 0.05,
            "neighbour": neighbour,
            "right": neighbour,
            "down": neighbour,
            "solve_only_ssim": 0.2,
            "board_sha256": "d" * 64,
            "solved_corrupted_canvas_sha256": "e" * 64,
            "diagnostics": {
                "non_seed_attachment_supports": [2],
                "expansion_cap_hit": False,
                "rigid_coverage": 0.25,
                "expansions": 10,
            },
            "_board": np.arange(576, dtype=np.int64),
            "_solved": np.zeros((480, 480, 3), dtype=np.uint8),
        }

    def test_structure_reject_calls_neither_solver_nor_nlm(self) -> None:
        failing = self._passing_structure()
        failing["cc96_edge_precision"] = 0.0
        result, solver, nlm = self._run_with_rows(failing, None)
        self.assertEqual(result["stage"], "kill_structure")
        solver.assert_not_called()
        nlm.assert_not_called()

    def test_decoder_reject_never_calls_nlm(self) -> None:
        result, solver, nlm = self._run_with_rows(
            self._passing_structure(), self._decoder_row(neighbour=0.0)
        )
        self.assertEqual(result["stage"], "kill_decoder")
        self.assertEqual(solver.call_count, 8)
        nlm.assert_not_called()

    def test_passing_decoder_calls_candidate_nlm_once_per_scene_only(self) -> None:
        result, solver, nlm = self._run_with_rows(
            self._passing_structure(), self._decoder_row(neighbour=0.25)
        )
        self.assertEqual(result["stage"], "go_changed_decoder_oracle")
        self.assertEqual(solver.call_count, 8)
        self.assertEqual(nlm.call_count, 8)
        self.assertEqual(result["comparison"]["candidate_arm"], "E15_frame_consensus")


class HypothesisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
        )
        self.owner = np.asarray([0, 0, 1, 1] + [0] * 572, dtype=np.int64)

    def test_two_distinct_seams_create_one_translation(self) -> None:
        claims = (
            e15.SeamClaim(0.9, 0, 2, 0, 1),
            e15.SeamClaim(0.8, 1, 3, 0, 1),
        )
        hypotheses = e15.build_translation_hypotheses(
            self.components, self.owner, claims
        )
        self.assertEqual(len(hypotheses), 1)
        hypothesis = hypotheses[0]
        self.assertEqual(
            (hypothesis.left_component, hypothesis.right_component), (0, 1)
        )
        self.assertEqual((hypothesis.offset_row, hypothesis.offset_col), (0, 1))
        self.assertEqual(hypothesis.distinct_seams, 2)

    def test_one_seam_or_duplicate_physical_seam_is_rejected(self) -> None:
        one = (e15.SeamClaim(0.9, 0, 2, 0, 1),)
        self.assertEqual(
            e15.build_translation_hypotheses(self.components, self.owner, one), ()
        )
        duplicate = (
            e15.SeamClaim(0.9, 0, 2, 0, 1),
            e15.SeamClaim(0.8, 0, 2, 0, 1),
        )
        self.assertEqual(
            e15.build_translation_hypotheses(
                self.components, self.owner, duplicate
            ),
            (),
        )

    def test_collision_hypothesis_is_rejected(self) -> None:
        claims = (
            e15.SeamClaim(0.9, 0, 2, 0, 0),
            e15.SeamClaim(0.8, 1, 3, 0, 0),
        )
        self.assertEqual(
            e15.build_translation_hypotheses(
                self.components, self.owner, claims
            ),
            (),
        )

    def test_relative_beam_respects_hypothesis_offset(self) -> None:
        hypothesis = e15.TranslationHypothesis(
            hypothesis_id=0,
            left_component=0,
            right_component=1,
            offset_row=0,
            offset_col=1,
            claims=(
                e15.SeamClaim(0.9, 0, 2, 0, 1),
                e15.SeamClaim(0.8, 1, 3, 0, 1),
            ),
        )
        lab = np.zeros((576, 576), dtype=np.float32)
        states, expansions, cap_hit = e15.relative_translation_beam(
            self.components, (hypothesis,), lab, lab
        )
        self.assertFalse(cap_hit)
        self.assertGreaterEqual(expansions, 1)
        translations = {cid: (row, col) for cid, row, col in states[0].translations}
        self.assertEqual(
            translations[1][1] - translations[0][1], 1
        )
        self.assertEqual(states[0].seam_votes, 2)

    def test_physical_seam_evidence_is_counted_once(self) -> None:
        shared = e15.SeamClaim(0.9, 0, 2, 0, 1)
        reverse_shared = e15.SeamClaim(0.8, 2, 0, 0, -1)
        hypotheses = {
            0: e15.TranslationHypothesis(
                0,
                0,
                1,
                0,
                1,
                (shared, e15.SeamClaim(0.7, 1, 3, 0, 1)),
            ),
            1: e15.TranslationHypothesis(
                1,
                0,
                1,
                0,
                1,
                (reverse_shared, e15.SeamClaim(0.6, 24, 26, 0, 1)),
            ),
        }
        seams, score = e15._hypothesis_evidence({0, 1}, hypotheses)
        self.assertEqual(len(seams), 3)
        self.assertAlmostEqual(score, 0.9 + 0.7 + 0.6)

    def test_reaching_exact_scene_cap_is_a_cap_hit(self) -> None:
        hypothesis = e15.TranslationHypothesis(
            hypothesis_id=0,
            left_component=0,
            right_component=1,
            offset_row=0,
            offset_col=1,
            claims=(
                e15.SeamClaim(0.9, 0, 2, 0, 1),
                e15.SeamClaim(0.8, 1, 3, 0, 1),
            ),
        )
        lab = np.zeros((576, 576), dtype=np.float32)
        with mock.patch.object(e15, "EXPANSION_CAP", 1):
            states, expansions, cap_hit = e15.relative_translation_beam(
                self.components, (hypothesis,), lab, lab
            )
        self.assertEqual(expansions, 1)
        self.assertTrue(cap_hit)
        self.assertEqual(len(states[0].translations), 1)


class AbsoluteFrameAndResidualTests(unittest.TestCase):
    def test_exact_ties_keep_spatially_diverse_origins(self) -> None:
        candidates = []
        for row in range(4):
            for col in range(4):
                board = np.full((24, 24), -1, dtype=np.int64)
                board[row, col] = 0
                candidates.append(((0, 0.0), row, col, board, None))
        selected = e15._diverse_top_origins(candidates, 8)
        origins = {(row, col) for _score, row, col, _board, _payload in selected}
        self.assertEqual(len(origins), 8)
        self.assertGreater(len({row for row, _ in origins}), 1)
        self.assertGreater(len({col for _, col in origins}), 1)
        self.assertNotEqual(origins, {(0, 0)})

    def test_absolute_layout_budget_is_global_not_per_relative(self) -> None:
        components = {
            0: _component(0, {0: (0, 0)}),
            1: _component(1, {1: (0, 0)}),
        }
        growths = tuple(
            e15.RigidGrowth(
                translations=((component_id, 0, 0),),
                used_hypotheses=frozenset(),
                hypothesis_seams=0,
                hypothesis_score=0.0,
                contact_seams=0,
                contact_score=0.0,
                lab_tie_score=0.0,
                attachments=(),
                expansions=0,
                cap_hit=False,
            )
            for component_id in (0, 1)
        )
        log_right, log_down = _zero_dense()
        retained, evaluated = e15._absolute_origin_candidates(
            growths, components, log_right, log_down
        )
        self.assertEqual(evaluated, 2 * 24 * 24)
        self.assertEqual(len(retained), e15.ABSOLUTE_LAYOUTS)

    def test_relative_growth_spends_one_cumulative_scene_budget(self) -> None:
        states = tuple(
            e15.RelativeState(
                translations=((index, 0, 0),),
                used_hypotheses=frozenset(),
                seam_votes=0,
                neural_score=0.0,
                lab_tie_score=0.0,
            )
            for index in range(3)
        )
        priors: list[int] = []

        def fake_growth(*_args: object, prior_expansions: int, **_kwargs: object):
            priors.append(prior_expansions)
            updated = prior_expansions + 7
            return e15.RigidGrowth(
                translations=((0, 0, 0),),
                used_hypotheses=frozenset(),
                hypothesis_seams=0,
                hypothesis_score=0.0,
                contact_seams=0,
                contact_score=0.0,
                lab_tie_score=0.0,
                attachments=(),
                expansions=updated,
                cap_hit=updated >= 24,
            )

        right, down = _zero_dense()
        with mock.patch.object(e15, "EXPANSION_CAP", 24):
            with mock.patch.object(
                e15, "grow_rigid_multicontact", side_effect=fake_growth
            ):
                growths, expansions, cap_hit = e15._grow_relative_layouts(
                    states,
                    (),
                    (),
                    right,
                    down,
                    right,
                    down,
                    prior_expansions=10,
                    prior_cap_hit=False,
                )
        self.assertEqual(priors, [10, 17])
        self.assertEqual(len(growths), 2)
        self.assertEqual(expansions, 24)
        self.assertTrue(cap_hit)

    def test_residual_keeps_rigid_tile_and_runs_two_hungarian_rounds(self) -> None:
        right, down = _zero_dense()
        partial = np.full((24, 24), -1, dtype=np.int64)
        partial[0, 0] = 0
        board, diagnostics = e15.complete_residual(partial, right, down)
        self.assertEqual(board[0], 0)
        self.assertTrue(np.array_equal(np.sort(board), np.arange(576)))
        self.assertEqual(diagnostics.wave_commits, 0)
        self.assertEqual(diagnostics.hungarian_rounds, 2)

    def test_multicontact_wave_commits_mutual_best_tile(self) -> None:
        right, down = _zero_dense()
        partial = np.full((24, 24), -1, dtype=np.int64)
        partial[0, 0] = 0
        partial[0, 1] = 1
        partial[1, 0] = 24
        right[24, 25] = 1.0
        down[1, 25] = 1.0
        board, diagnostics = e15.complete_residual(partial, right, down)
        self.assertEqual(board.reshape(24, 24)[1, 1], 25)
        self.assertGreaterEqual(diagnostics.wave_commits, 1)
        self.assertTrue(np.array_equal(np.sort(board), np.arange(576)))

    def test_invalid_partial_duplicate_fails_closed(self) -> None:
        right, down = _zero_dense()
        partial = np.full((24, 24), -1, dtype=np.int64)
        partial[0, 0] = 5
        partial[0, 1] = 5
        with self.assertRaises(e15.FrameConsensusError):
            e15.complete_residual(partial, right, down)

    def test_terminal_objective_prefers_supported_identity_contacts(self) -> None:
        right, down = _zero_dense()
        identity = np.arange(576, dtype=np.int64)
        grid = identity.reshape(24, 24)
        right[grid[:, :-1], grid[:, 1:]] = 1.0
        down[grid[:-1, :], grid[1:, :]] = 1.0
        good = e15.terminal_neural_objective(identity, right, down)
        bad = e15.terminal_neural_objective(identity[::-1], right, down)
        self.assertGreater(good, bad)


if __name__ == "__main__":
    unittest.main()
