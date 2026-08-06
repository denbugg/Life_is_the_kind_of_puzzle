from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_e20_triangle_potential_viability as evaluator  # noqa: E402


SHA = "a" * 64


def _component(component_id: int, entries: tuple[tuple[int, int, int], ...]):
    return evaluator.e18_core.e15.Component(component_id, entries)


def _graph():
    components = [
        _component(0, ((0, 0, 0), (24, 1, 0))),
        _component(1, ((1, 0, 0), (25, 1, 0))),
    ]
    used = {0, 1, 24, 25}
    for tile in range(576):
        if tile in used:
            continue
        component_id = len(components)
        components.append(_component(component_id, ((tile, 0, 0),)))

    owner = np.full(576, -1, dtype=np.int64)
    rows = np.zeros(576, dtype=np.int64)
    cols = np.zeros(576, dtype=np.int64)
    for component in components:
        for tile, row, col in component.entries:
            owner[tile] = component.component_id
            rows[tile] = row
            cols[tile] = col
    claims = (
        evaluator.e18_core.BridgeClaim(0, 1.25, 0, 1, 0, 1, 0, 1),
        evaluator.e18_core.BridgeClaim(1, 2.50, 24, 25, 0, 1, 0, 1),
    )
    return evaluator.e18_core.GraphData(
        components=tuple(components),
        owner=owner,
        local_rows=rows,
        local_cols=cols,
        nontrivial=frozenset({0, 1}),
        claims=claims,
        claims_by_frontier={(0, 3): (claims[0],), (24, 3): (claims[1],)},
        claims_by_component={0: claims, 1: claims},
    )


def _scores() -> tuple[np.ndarray, np.ndarray]:
    right = np.zeros((576, 576), dtype=np.float32)
    down = np.zeros_like(right)
    right[0, 1] = 1.25
    right[24, 25] = 2.50
    return right, down


def _cluster() -> evaluator.pose.PoseCluster:
    return evaluator.pose.PoseCluster(
        component_ids=(0, 1),
        translations=((0, 0, 0), (1, 0, 1)),
        relative_entries=(
            (0, 0, 0),
            (1, 0, 1),
            (24, 1, 0),
            (25, 1, 1),
        ),
        bbox=(0, 1, 0, 1),
        bbox_height=2,
        bbox_width=2,
        legal_origin_bounds=(0, 22, 0, 22),
        legal_origin_count=23 * 23,
        tree_hypothesis_ids=(0,),
        cycle_hypothesis_ids=(),
        accepted_hypothesis_ids=(0,),
        accepted_relations=((0, 1, 0, 1),),
        component_contacts=((0, 1),),
        accepted_cross_seams=((0, 1, 0, 1), (24, 25, 0, 1)),
        rigid_tiles=4,
        rigid_coverage=4 / 576,
        component_cycle_rank=0,
        component_cycle_rank_ratio=0.0,
        cross_neural_sum=3.75,
        minimum_tile=0,
    )


def _result() -> evaluator.pose.TrianglePoseResult:
    cluster = _cluster()
    hypothesis = evaluator.pose.PoseHypothesis(
        hypothesis_id=0,
        u=0,
        v=1,
        dr=0,
        dc=1,
        seam_scores=(((0, 1, 0, 1), 1.25), ((24, 25, 0, 1), 2.50)),
        reciprocal_seams=(),
    )
    diagnostics = evaluator.pose.TrianglePoseDiagnostics(
        cc192_component_count=574,
        cc192_nontrivial_components=2,
        cc192_nontrivial_tiles=4,
        bridge_claims=2,
        pose_hypotheses=1,
        triangle_supported_hypotheses=0,
        eligible_hypotheses=1,
        weak_hypotheses=0,
        eligible_processed=1,
        tree_merges=1,
        cycle_acceptances=0,
        pose_conflicts=0,
        contact_rejections=0,
        collision_rejections=0,
        span_rejections=0,
        cluster_count=573,
        selected_components=2,
        selected_rigid_tiles=4,
    )
    return evaluator.pose.TrianglePoseResult(
        selected=cluster,
        clusters=(cluster,),
        hypotheses=(hypothesis,),
        diagnostics=diagnostics,
    )


def _scene(image: int = 10, permutation: np.ndarray | None = None):
    if permutation is None:
        permutation = np.arange(576, dtype=np.int64)
    return SimpleNamespace(
        image_id=image,
        validation_name=f"validation_{image}",
        permutation=np.asarray(permutation, dtype=np.int64),
        tiles_uint8=np.zeros((576, 20, 20, 3), dtype=np.uint8),
    )


def _row(image: int = 10, permutation: np.ndarray | None = None):
    return evaluator.evaluate_structure(
        _scene(image, permutation),
        _result(),
        clean_score_cache_sha256=SHA,
        graph=_graph(),
    )


def _validate_row(row: dict[str, object], image: int = 10) -> None:
    right, down = _scores()
    with mock.patch.object(
        evaluator.pose, "run_triangle_potential_dsu", return_value=_result()
    ), mock.patch.object(
        evaluator.e18_core, "build_graph_data", return_value=_graph()
    ):
        evaluator._validate_success_row(
            row,
            scene=_scene(image),
            cache_sha256=SHA,
            right=right,
            down=down,
        )


class FrozenContractTests(unittest.TestCase):
    def test_protocol_decision_cli_and_exclusions_are_literal(self) -> None:
        self.assertEqual(
            evaluator.DECISION_RULE,
            {
                "completed_scenes": 8,
                "legal_origin_scenes": 8,
                "mean_rigid_coverage_min": 0.35,
                "worst_rigid_coverage_min": 0.25,
                "mean_exact_pose_coverage_min": 0.30,
                "worst_exact_pose_coverage_min": 0.20,
                "mean_exact_relative_pose_precision_min": 0.90,
                "worst_exact_relative_pose_precision_min": 0.80,
                "mean_accepted_relation_precision_min": 0.85,
                "worst_accepted_relation_precision_min": 0.70,
                "mean_accepted_cross_seam_precision_min": 0.85,
                "worst_accepted_cross_seam_precision_min": 0.70,
                "mean_component_cycle_rank_ratio_min": 0.05,
            },
        )
        protocol = evaluator.E20_PROTOCOL
        self.assertEqual(protocol["input_graph"]["top_k"], 8)
        self.assertFalse(protocol["input_graph"]["rotation"])
        self.assertFalse(protocol["input_graph"]["reflection"])
        self.assertEqual(protocol["triangles"]["incident_hypotheses_per_component"], 8)
        self.assertEqual(protocol["triangles"]["leg_pairs_per_intermediate_max"], 28)
        self.assertEqual(protocol["selection"]["minimum_independent_paths"], 2)
        self.assertFalse(protocol["selection"]["rollback"])
        self.assertFalse(protocol["selection"]["beam"])
        self.assertTrue(protocol["output"]["sparse_only"])
        self.assertFalse(protocol["output"]["absolute_board"])
        self.assertEqual(
            protocol["measurement"]["exact_pose_bin"],
            "modal_truth_coordinate_minus_relative_coordinate",
        )
        self.assertEqual(
            protocol["authorization"],
            {
                "e19_report_sha256": evaluator.EXPECTED_E19_REPORT_SHA256,
                "e19_run_contract_sha256": evaluator.EXPECTED_E19_RUN_CONTRACT_SHA256,
                "e19_protocol_sha256": evaluator.EXPECTED_E19_PROTOCOL_SHA256,
                "required_status": "complete",
                "required_stage": "kill_relative_cap",
                "required_cap_image": 10,
                "required_proposal_evaluations": 500000,
                "required_rounds": 32,
            },
        )
        required_exclusions = {
            "absolute_board",
            "residual_completion",
            "placement",
            "neighbour",
            "SSIM",
            "NLM",
            "labels_inside_selection",
            "rotation",
            "reflection",
            "GPU",
            "diffusion",
        }
        self.assertTrue(required_exclusions.issubset(set(protocol["excluded"])))
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
                "e19_report",
                "report",
            },
        )

    def test_source_and_runtime_provenance_are_exact(self) -> None:
        self.assertEqual(
            set(evaluator._source_provenance()),
            {
                "e15_frame_consensus.py",
                "e18_absolute_frame_beam.py",
                "e20_triangle_potential_dsu.py",
                "eval_clean_score_oracle.py",
                "eval_e14_cc192_discovery.py",
                "eval_e17_cc192_rigid_viability.py",
                "eval_e18_absolute_frame_oracle.py",
                "eval_e19_relative_frame_viability.py",
                "eval_e20_triangle_potential_viability.py",
                "rank96_lab_selector.py",
                "solve_buddies.py",
            },
        )
        self.assertEqual(
            evaluator._runtime_provenance(), evaluator.EXPECTED_RUNTIME_PROVENANCE
        )

    def test_live_e19_cap_kill_is_authenticated_exactly(self) -> None:
        report = evaluator._verify_e19_cap_kill(evaluator.DEFAULT_E19_REPORT)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["stage"], "kill_relative_cap")
        self.assertEqual(report["rows"], [])
        self.assertEqual(report["completed_images"], [])
        self.assertEqual(report["cap_failure"]["image"], 10)
        self.assertEqual(report["cap_failure"]["proposal_evaluations"], 500000)
        self.assertEqual(report["cap_failure"]["rounds"], 32)

    def test_e19_authentication_rejects_critical_tamper_and_coercion(self) -> None:
        pristine = copy.deepcopy(
            evaluator._load_json(evaluator.DEFAULT_E19_REPORT, label="E19 report")
        )
        real_sha256_file = evaluator.e12.sha256_file

        def authentic_hash(path: Path) -> str:
            if Path(path).resolve() == evaluator.DEFAULT_E19_REPORT.resolve():
                return evaluator.EXPECTED_E19_REPORT_SHA256
            return real_sha256_file(path)

        def set_nested(*path_and_value):
            *path, value = path_and_value

            def mutate(payload):
                current = payload
                for key in path[:-1]:
                    current = current[key]
                current[path[-1]] = value

            return mutate

        mutations = (
            ("status", set_nested("status", "failed")),
            ("stage", set_nested("stage", "relative_structure")),
            ("protocol", set_nested("protocol", {})),
            ("protocol_sha", set_nested("protocol_sha256", "0" * 64)),
            ("contract_sha", set_nested("run_contract_sha256", "0" * 64)),
            ("rows", set_nested("rows", [{}])),
            ("completed", set_nested("completed_images", [10])),
            ("summary", set_nested("summary", {})),
            ("cap_image", set_nested("cap_failure", "image", 11)),
            (
                "cap_evaluations",
                set_nested("cap_failure", "proposal_evaluations", 499999),
            ),
            ("cap_rounds", set_nested("cap_failure", "rounds", 31)),
            ("initial_states", set_nested("cap_failure", "initial_states", 2)),
            ("cap_hit", set_nested("cap_failure", "cap_hit", False)),
            ("cap_error", set_nested("cap_failure", "error", "wrong")),
            ("decision", set_nested("decision", {"passed": True})),
            (
                "contract_protocol",
                set_nested("run_contract", "protocol_sha256", "0" * 64),
            ),
            ("runtime_bool", set_nested("runtime_seconds", True)),
            ("runtime_nan", set_nested("runtime_seconds", float("nan"))),
            ("fractional_image", set_nested("cap_failure", "image", 10.5)),
            (
                "fractional_evaluations",
                set_nested("cap_failure", "proposal_evaluations", 500000.5),
            ),
            ("fractional_rounds", set_nested("cap_failure", "rounds", 32.5)),
            ("fractional_initial", set_nested("cap_failure", "initial_states", 1.5)),
        )
        for label, mutate in mutations:
            payload = copy.deepcopy(pristine)
            mutate(payload)
            if label.startswith("fractional_") or label == "cap_error":
                payload["decision"] = evaluator.e19_eval.cap_decision(
                    payload["cap_failure"]
                )
            with self.subTest(label=label), mock.patch.object(
                evaluator.e12, "sha256_file", side_effect=authentic_hash
            ), mock.patch.object(evaluator, "_load_json", return_value=payload):
                with self.assertRaises(evaluator.E20ContractError):
                    evaluator._verify_e19_cap_kill(evaluator.DEFAULT_E19_REPORT)

    def test_e19_authentication_rejects_shared_source_drift(self) -> None:
        pristine = copy.deepcopy(
            evaluator._load_json(evaluator.DEFAULT_E19_REPORT, label="E19 report")
        )
        real_sha256_file = evaluator.e12.sha256_file
        drifted = evaluator.E19_SHARED_SOURCE_NAMES[0]

        def drifted_hash(path: Path) -> str:
            resolved = Path(path).resolve()
            if resolved == evaluator.DEFAULT_E19_REPORT.resolve():
                return evaluator.EXPECTED_E19_REPORT_SHA256
            if resolved.name == drifted:
                return "0" * 64
            return real_sha256_file(path)

        with mock.patch.object(
            evaluator.e12, "sha256_file", side_effect=drifted_hash
        ), mock.patch.object(evaluator, "_load_json", return_value=pristine):
            with self.assertRaisesRegex(
                evaluator.E20ContractError, "shared E19-to-E20 source drifted"
            ):
                evaluator._verify_e19_cap_kill(evaluator.DEFAULT_E19_REPORT)

    def test_e_drive_suffix_and_overlap_guards_precede_input_loading(self) -> None:
        valid = evaluator.E20Paths(
            evaluator.DEFAULT_RAW_CACHE_DIR,
            evaluator.DEFAULT_CALIBRATION_REPORT,
            evaluator.DEFAULT_E12_REPORT,
            evaluator.DEFAULT_E19_REPORT,
            Path("E:/pazzle_work/triangle_pose_e20/e20_unit_guard.json"),
        )
        cases = (
            evaluator.E20Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e19_report,
                Path("C:/tmp/e20.json"),
            ),
            evaluator.E20Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e19_report,
                Path("E:/pazzle_work/triangle_pose_e20/e20.txt"),
            ),
            evaluator.E20Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e19_report,
                valid.raw_cache_dir / "report.json",
            ),
            evaluator.E20Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e19_report,
                evaluator.DEFAULT_E12_REPORT.parent / "score_cache" / "report.json",
            ),
            evaluator.E20Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e19_report,
                valid.e12_report,
            ),
            evaluator.E20Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e19_report,
                valid.e19_report,
            ),
            evaluator.E20Paths(
                Path("C:/tmp/raw"),
                valid.calibration_report,
                valid.e12_report,
                valid.e19_report,
                valid.report,
            ),
            evaluator.E20Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                Path("C:/tmp/e12.json"),
                valid.e19_report,
                valid.report,
            ),
            evaluator.E20Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                Path("C:/tmp/e19.json"),
                valid.report,
            ),
        )
        for paths in cases:
            with self.subTest(paths=paths), mock.patch.object(
                evaluator, "_verify_e19_cap_kill"
            ) as verify, mock.patch.object(
                evaluator.e17, "_load_verified_structure_inputs"
            ) as loader:
                with self.assertRaises(evaluator.E20ContractError):
                    evaluator.run_gate(paths)
                verify.assert_not_called()
                loader.assert_not_called()


class TruthMeasurementTests(unittest.TestCase):
    def test_seam_truth_rejects_row_wrap_noncanonical_bool_and_float(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        self.assertTrue(evaluator._seam_is_true((0, 1, 0, 1), permutation))
        self.assertTrue(evaluator._seam_is_true((0, 24, 1, 0), permutation))
        self.assertFalse(evaluator._seam_is_true((23, 24, 0, 1), permutation))
        for seam in (
            (1, 0, 0, -1),
            (24, 0, -1, 0),
            (0, 1, 1, 1),
            (False, 1, 0, 1),
            (0.5, 1, 0, 1),
        ):
            with self.subTest(seam=seam), self.assertRaises(
                evaluator.E20ContractError
            ):
                evaluator._seam_is_true(seam, permutation)

    def test_measure_selected_rejects_bool_and_fractional_seams_before_coercion(self) -> None:
        for seam in ((False, 1, 0, 1), (0.5, 1, 0, 1)):
            cluster = replace(_cluster(), accepted_cross_seams=(seam,))
            with self.subTest(seam=seam), self.assertRaises(
                evaluator.E20ContractError
            ):
                evaluator.measure_selected(cluster, _graph(), np.arange(576))

    def test_measure_selected_strictly_types_entries_relations_and_scalar_metrics(self) -> None:
        corruptions = (
            replace(
                _cluster(),
                relative_entries=((False, 0, 0),) + _cluster().relative_entries[1:],
            ),
            replace(
                _cluster(),
                relative_entries=((0.5, 0, 0),) + _cluster().relative_entries[1:],
            ),
            replace(_cluster(), accepted_relations=((False, 1, 0, 1),)),
            replace(_cluster(), accepted_relations=((0, 1, 0.5, 1),)),
            replace(_cluster(), component_cycle_rank_ratio=True),
            replace(_cluster(), component_cycle_rank_ratio=float("nan")),
            replace(_cluster(), legal_origin_count=True),
            replace(_cluster(), legal_origin_count=529.5),
        )
        for cluster in corruptions:
            with self.subTest(cluster=cluster), self.assertRaises(
                evaluator.E20ContractError
            ):
                evaluator.measure_selected(cluster, _graph(), np.arange(576))

    def test_non_self_inverse_permutation_drives_modal_pose_by_tile_to_position(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        permutation[:3] = [1, 2, 0]
        cluster = replace(
            _cluster(),
            relative_entries=(
                (0, 0, 1),
                (1, 0, 2),
                (24, 1, 0),
                (25, 1, 1),
            ),
        )
        metrics = evaluator.measure_selected(cluster, _graph(), permutation)
        self.assertEqual(metrics["modal_truth_offset"], [0, 0])
        self.assertEqual(metrics["exact_pose_tiles"], 4)
        self.assertEqual(metrics["exact_relative_pose_precision"], 1.0)
        self.assertTrue(evaluator._seam_is_true((0, 1, 0, 1), permutation))
        self.assertFalse(evaluator._seam_is_true((1, 2, 0, 1), permutation))

    def test_modal_pose_tie_uses_lexicographically_first_signed_offset(self) -> None:
        cluster = replace(
            _cluster(),
            relative_entries=(
                (0, 1, 0),
                (1, 1, 1),
                (24, 1, 0),
                (25, 1, 1),
            ),
        )
        metrics = evaluator.measure_selected(cluster, _graph(), np.arange(576))
        self.assertEqual(metrics["modal_truth_offset"], [-1, 0])
        self.assertEqual(metrics["exact_pose_tiles"], 2)
        self.assertEqual(metrics["exact_relative_pose_precision"], 0.5)

    def test_relation_truth_requires_both_whole_components_exact(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        permutation[24], permutation[25] = 25, 24
        shifts = evaluator._component_truth_shifts(_graph(), permutation)
        self.assertIsNone(shifts[0])
        self.assertIsNone(shifts[1])
        metrics = evaluator.measure_selected(_cluster(), _graph(), permutation)
        self.assertEqual(metrics["accepted_relations"], 1)
        self.assertEqual(metrics["true_accepted_relations"], 0)
        self.assertEqual(metrics["accepted_relation_precision"], 0.0)

    def test_empty_relation_and_seam_precision_are_exactly_zero(self) -> None:
        cluster = replace(
            _cluster(), accepted_relations=(), accepted_cross_seams=()
        )
        metrics = evaluator.measure_selected(cluster, _graph(), np.arange(576))
        self.assertEqual(metrics["accepted_relations"], 0)
        self.assertEqual(metrics["accepted_relation_precision"], 0.0)
        self.assertEqual(metrics["accepted_cross_seams"], 0)
        self.assertEqual(metrics["accepted_cross_seam_precision"], 0.0)


class DecisionTests(unittest.TestCase):
    def test_decision_is_inclusive_and_every_check_is_required(self) -> None:
        passing = {
            "completed_scenes": 8,
            "legal_origin_scenes": 8,
            "mean_rigid_coverage": 0.35,
            "worst_rigid_coverage": 0.25,
            "mean_exact_pose_coverage": 0.30,
            "worst_exact_pose_coverage": 0.20,
            "mean_exact_relative_pose_precision": 0.90,
            "worst_exact_relative_pose_precision": 0.80,
            "mean_accepted_relation_precision": 0.85,
            "worst_accepted_relation_precision": 0.70,
            "mean_accepted_cross_seam_precision": 0.85,
            "worst_accepted_cross_seam_precision": 0.70,
            "mean_component_cycle_rank_ratio": 0.05,
        }
        accepted = evaluator.decision(passing)
        self.assertTrue(accepted["passed"])
        self.assertEqual(
            accepted["status"], "go_E21_one_cluster_absolute_origin_residual"
        )
        failures = []
        for key, value in (
            ("completed_scenes", 7),
            ("legal_origin_scenes", 7),
            ("mean_rigid_coverage", 0.349999),
            ("worst_rigid_coverage", 0.249999),
            ("mean_exact_pose_coverage", 0.299999),
            ("worst_exact_pose_coverage", 0.199999),
            ("mean_exact_relative_pose_precision", 0.899999),
            ("worst_exact_relative_pose_precision", 0.799999),
            ("mean_accepted_relation_precision", 0.849999),
            ("worst_accepted_relation_precision", 0.699999),
            ("mean_accepted_cross_seam_precision", 0.849999),
            ("worst_accepted_cross_seam_precision", 0.699999),
            ("mean_component_cycle_rank_ratio", 0.049999),
        ):
            failures.append({**passing, key: value})
        for summary in failures:
            with self.subTest(summary=summary):
                rejected = evaluator.decision(summary)
                self.assertFalse(rejected["passed"])
                self.assertEqual(
                    rejected["status"], "kill_top8_triangle_potential_route"
                )


class CoreReplayAndAlgebraTests(unittest.TestCase):
    def test_success_row_replays_exact_core_and_hash_and_metrics(self) -> None:
        row = _row()
        right, down = _scores()
        with mock.patch.object(
            evaluator.pose, "run_triangle_potential_dsu", return_value=_result()
        ) as replay, mock.patch.object(
            evaluator.e18_core, "build_graph_data", return_value=_graph()
        ):
            evaluator._validate_success_row(
                row,
                scene=_scene(),
                cache_sha256=SHA,
                right=right,
                down=down,
            )
        replay.assert_called_once_with(right, down)
        tampered = copy.deepcopy(row)
        tampered["core"]["diagnostics"]["tree_merges"] = 0
        with self.assertRaises(evaluator.E20ContractError):
            _validate_row(tampered)
        bad_hash = copy.deepcopy(row)
        bad_hash["core_sha256"] = "0" * 64
        with self.assertRaises(evaluator.E20ContractError):
            _validate_row(bad_hash)
        bad_metrics = copy.deepcopy(row)
        bad_metrics["metrics"]["exact_pose_tiles"] = 3
        with self.assertRaises(evaluator.E20ContractError):
            _validate_row(bad_metrics)

    def test_every_sparse_geometry_layer_rejects_tampering(self) -> None:
        graph = _graph()
        right, down = _scores()
        pristine = evaluator._core_payload(_result())
        evaluator._validate_core_geometry(
            pristine, graph=graph, right=right, down=down
        )
        corruptions: list[tuple[str, dict[str, object]]] = []

        def changed(label, key, value):
            payload = copy.deepcopy(pristine)
            payload["selected"][key] = value
            corruptions.append((label, payload))

        changed("component_ids", "component_ids", [1, 0])
        changed("translations", "translations", [[0, 0, 0], [1, 0, 2]])
        changed(
            "entries",
            "relative_entries",
            [[0, 0, 0], [1, 0, 1], [24, 1, 0], [25, 1, 2]],
        )
        changed("normalization", "translations", [[0, 1, 0], [1, 1, 1]])
        changed("bbox", "bbox", [0, 1, 0, 2])
        changed("bbox_height", "bbox_height", 3)
        changed("origin_bounds", "legal_origin_bounds", [0, 21, 0, 22])
        changed("origin_count", "legal_origin_count", 528)
        changed("rigid_tiles", "rigid_tiles", 3)
        changed("rigid_coverage", "rigid_coverage", 0.5)
        changed("tree_ids", "tree_hypothesis_ids", [])
        changed("accepted_ids", "accepted_hypothesis_ids", [])
        changed("cycle_ids", "cycle_hypothesis_ids", [0])
        changed("relations", "accepted_relations", [])
        changed("contacts", "component_contacts", [])
        changed("cycle_rank", "component_cycle_rank", 1)
        changed("cycle_ratio", "component_cycle_rank_ratio", 0.5)
        changed(
            "duplicate_seam",
            "accepted_cross_seams",
            [[0, 1, 0, 1], [0, 1, 0, 1]],
        )
        changed("float_seam", "accepted_cross_seams", [[0.5, 1, 0, 1]])
        changed("bool_seam", "accepted_cross_seams", [[False, 1, 0, 1]])
        changed("noncontact_seam", "accepted_cross_seams", [[0, 25, 0, 1]])
        changed("neural_sum", "cross_neural_sum", 3.749)
        changed("minimum_tile", "minimum_tile", 1)
        changed("relation_offset", "accepted_relations", [[0, 1, 0, 2]])

        internal = copy.deepcopy(pristine)
        internal["selected"]["accepted_cross_seams"] = [[0, 24, 1, 0]]
        internal["selected"]["cross_neural_sum"] = 0.0
        corruptions.append(("internal_not_cross_seam", internal))

        for label, payload in corruptions:
            with self.subTest(label=label), self.assertRaises(
                evaluator.E20ContractError
            ):
                evaluator._validate_core_geometry(
                    payload, graph=graph, right=right, down=down
                )


class CompleteReportTests(unittest.TestCase):
    def test_complete_report_is_recomputed_and_tamper_closed(self) -> None:
        scenes = [_scene(image) for image in evaluator.e12.CALIBRATION_IDS]
        rows = [_row(image) for image in evaluator.e12.CALIBRATION_IDS]
        summary = evaluator.summarize(rows)
        gate = evaluator.decision(summary)
        contract = {"frozen": True}
        digest = evaluator.e12.canonical_digest(contract)
        report: dict[str, object] = {
            "schema_version": evaluator.SCHEMA_VERSION,
            "schema": evaluator.REPORT_SCHEMA,
            "experiment": evaluator.EXPERIMENT,
            "status": "complete",
            "stage": gate["status"],
            "protocol": evaluator.E20_PROTOCOL,
            "protocol_sha256": evaluator.e12.canonical_digest(
                evaluator.E20_PROTOCOL
            ),
            "run_contract": contract,
            "run_contract_sha256": digest,
            "rows": rows,
            "completed_images": list(evaluator.e12.CALIBRATION_IDS),
            "summary": summary,
            "decision": gate,
            "runtime_seconds": 1.0,
        }
        records = {
            image: {"path": f"E:/cache/{image}.npz", "sha256": SHA}
            for image in evaluator.e12.CALIBRATION_IDS
        }
        cache = SimpleNamespace(
            sha256=SHA,
            cc_candidates=np.zeros((576, 1), dtype=np.int64),
            cc_scores=np.zeros((4, 576, 1), dtype=np.float32),
        )
        right, down = _scores()

        def validate(value):
            with mock.patch.object(
                evaluator.e14, "_load_cc_cache", return_value=cache
            ), mock.patch.object(
                evaluator.e12, "dense_from_graph", return_value=(right, down)
            ), mock.patch.object(
                evaluator.pose, "run_triangle_potential_dsu", return_value=_result()
            ), mock.patch.object(
                evaluator.e18_core, "build_graph_data", return_value=_graph()
            ):
                evaluator._validate_complete_report(
                    value,
                    contract=contract,
                    contract_digest=digest,
                    e12_report={},
                    scenes=scenes,
                    clean_records=records,
                )

        validate(report)
        corruptions = []
        row = copy.deepcopy(report)
        row["rows"][0]["metrics"]["exact_pose_tiles"] = 3
        corruptions.append(row)
        core = copy.deepcopy(report)
        core["rows"][0]["core"]["selected"]["legal_origin_count"] = 528
        corruptions.append(core)
        summary_bad = copy.deepcopy(report)
        summary_bad["summary"]["mean_rigid_coverage"] = 1.0
        corruptions.append(summary_bad)
        decision_bad = copy.deepcopy(report)
        decision_bad["decision"]["passed"] = not decision_bad["decision"]["passed"]
        corruptions.append(decision_bad)
        stage = copy.deepcopy(report)
        stage["stage"] = "triangle_pose_structure"
        corruptions.append(stage)
        completed = copy.deepcopy(report)
        completed["completed_images"] = completed["completed_images"][:-1]
        corruptions.append(completed)
        for corrupted in corruptions:
            with self.subTest(), self.assertRaises(evaluator.E20ContractError):
                validate(corrupted)


class StagingTests(unittest.TestCase):
    def test_run_gate_passes_only_right_down_and_never_calls_forbidden_routes(self) -> None:
        scenes = [_scene(image) for image in evaluator.e12.CALIBRATION_IDS]
        records = {
            image: {"path": f"E:/cache/{image}.npz", "sha256": SHA}
            for image in evaluator.e12.CALIBRATION_IDS
        }
        cache = SimpleNamespace(
            sha256=SHA,
            cc_candidates=np.zeros((576, 1), dtype=np.int64),
            cc_scores=np.zeros((4, 576, 1), dtype=np.float32),
        )
        right, down = _scores()
        paths = evaluator.E20Paths(
            evaluator.DEFAULT_RAW_CACHE_DIR,
            evaluator.DEFAULT_CALIBRATION_REPORT,
            evaluator.DEFAULT_E12_REPORT,
            evaluator.DEFAULT_E19_REPORT,
            Path(
                "E:/pazzle_work/triangle_pose_e20/"
                "e20_unit_no_write_never_create.json"
            ),
        )
        e12_report = {"scene_provenance_digest": "scene-digest"}
        e19_report = {
            "run_contract_sha256": evaluator.EXPECTED_E19_RUN_CONTRACT_SHA256
        }

        def core_only_scores(*args, **kwargs):
            self.assertEqual(len(args), 2)
            self.assertEqual(kwargs, {})
            self.assertIs(args[0], right)
            self.assertIs(args[1], down)
            return _result()

        with mock.patch.object(
            evaluator, "_verify_e19_cap_kill", return_value=e19_report
        ), mock.patch.object(
            evaluator.e17,
            "_load_verified_structure_inputs",
            return_value=(e12_report, {}, scenes),
        ), mock.patch.object(
            evaluator.e14, "_clean_cache_records", return_value=records
        ), mock.patch.object(
            evaluator.e14, "_load_cc_cache", return_value=cache
        ), mock.patch.object(
            evaluator.e12, "dense_from_graph", return_value=(right, down)
        ), mock.patch.object(
            evaluator.pose,
            "run_triangle_potential_dsu",
            side_effect=core_only_scores,
        ) as core, mock.patch.object(
            evaluator.e18_core, "build_graph_data", return_value=_graph()
        ), mock.patch.object(
            evaluator, "_source_provenance", return_value={"source": SHA}
        ), mock.patch.object(
            evaluator,
            "_runtime_provenance",
            return_value=evaluator.EXPECTED_RUNTIME_PROVENANCE,
        ), mock.patch.object(
            evaluator, "_atomic_write_json"
        ) as writer, mock.patch.object(
            evaluator.e12, "fixed_nlm"
        ) as nlm, mock.patch.object(
            evaluator.e18_core, "solve_absolute_frame"
        ) as absolute_solver, mock.patch.object(
            evaluator.e18_core.e15, "complete_residual"
        ) as residual, mock.patch.object(
            evaluator.e18_eval, "assemble"
        ) as assemble, mock.patch.object(
            evaluator.e18_eval, "sk_ssim"
        ) as ssim:
            output = evaluator.run_gate(paths)

        self.assertEqual(core.call_count, 8)
        nlm.assert_not_called()
        absolute_solver.assert_not_called()
        residual.assert_not_called()
        assemble.assert_not_called()
        ssim.assert_not_called()
        self.assertGreaterEqual(writer.call_count, 10)
        self.assertEqual(output["status"], "complete")
        self.assertEqual(output["completed_images"], list(evaluator.e12.CALIBRATION_IDS))
        self.assertEqual(len(output["rows"]), 8)
        serialized_rows = str(output["rows"]).lower()
        self.assertNotIn("board", serialized_rows)
        self.assertNotIn("ssim", serialized_rows)
        self.assertNotIn("nlm", serialized_rows)


if __name__ == "__main__":
    unittest.main()
