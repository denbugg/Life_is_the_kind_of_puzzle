from __future__ import annotations

import copy
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

import eval_e19_relative_frame_viability as evaluator  # noqa: E402


SHA = "a" * 64


def _synthetic_graph():
    """A tiny valid two-island geometry embedded in the 576-tile contract."""

    component_type = evaluator.e18_core.e15.Component
    components = [
        component_type(0, ((0, 0, 0), (24, 1, 0))),
        component_type(1, ((1, 0, 0), (25, 1, 0))),
    ]
    used = {0, 1, 24, 25}
    for tile in range(evaluator.relative.NUM_TILES):
        if tile in used:
            continue
        component_id = len(components)
        components.append(component_type(component_id, ((tile, 0, 0),)))

    owner = np.full(evaluator.relative.NUM_TILES, -1, dtype=np.int64)
    local_rows = np.zeros(evaluator.relative.NUM_TILES, dtype=np.int64)
    local_cols = np.zeros(evaluator.relative.NUM_TILES, dtype=np.int64)
    for component in components:
        for tile, row, col in component.entries:
            owner[tile] = component.component_id
            local_rows[tile] = row
            local_cols[tile] = col

    claims = (
        evaluator.relative.BridgeClaim(0, 1.25, 0, 1, 0, 1, 0, 1),
        evaluator.relative.BridgeClaim(1, 2.50, 24, 25, 0, 1, 0, 1),
    )
    graph = evaluator.relative.GraphData(
        components=tuple(components),
        owner=owner,
        local_rows=local_rows,
        local_cols=local_cols,
        nontrivial=frozenset({0, 1}),
        claims=claims,
        claims_by_frontier={(0, 3): (claims[0],), (24, 3): (claims[1],)},
        claims_by_component={0: claims, 1: claims},
    )
    right = np.zeros((576, 576), dtype=np.float32)
    down = np.zeros_like(right)
    right[0, 1] = 1.25
    right[24, 25] = 2.50
    return graph, right, down


def _scene(image: int = 10, permutation: np.ndarray | None = None):
    if permutation is None:
        permutation = np.arange(576, dtype=np.int64)
    return SimpleNamespace(
        image_id=image,
        validation_name=f"validation_{image}",
        permutation=np.asarray(permutation, dtype=np.int64),
        tiles_uint8=np.zeros((576, 20, 20, 3), dtype=np.uint8),
    )


def _result() -> evaluator.relative.RelativeBeamResult:
    layout = evaluator.relative.RelativeLayout(
        translations=((0, 0, 0), (1, 0, 1)),
        relative_entries=(
            (0, 0, 0),
            (1, 0, 1),
            (24, 1, 0),
            (25, 1, 1),
        ),
        satisfied_bridge_claims=(0, 1),
        component_contacts=((0, 1),),
        cross_seams=((0, 1, 0, 1), (24, 25, 0, 1)),
        cross_neural_sum=3.75,
        cross_lab_sum=0.0,
        rigid_tiles=4,
        rigid_coverage=4 / 576,
        component_cycle_rank=0,
        component_cycle_rank_ratio=0.0,
        bbox=(0, 1, 0, 1),
        bbox_height=2,
        bbox_width=2,
        legal_origin_bounds=(0, 22, 0, 22),
        legal_origin_count=23 * 23,
    )
    diagnostics = evaluator.relative.RelativeBeamDiagnostics(
        cc192_component_count=574,
        cc192_nontrivial_components=2,
        cc192_nontrivial_tiles=4,
        root_component_id=0,
        root_component_size=2,
        initial_states=1,
        bridge_claims=2,
        rounds=1,
        proposal_evaluations=2,
        cap_hit=False,
        layouts_retained=1,
    )
    return evaluator.relative.RelativeBeamResult(
        layouts=(layout,), diagnostics=diagnostics
    )


def _row(image: int = 10) -> dict[str, object]:
    return evaluator.evaluate_structure(
        _scene(image), _result(), clean_score_cache_sha256=SHA
    )


def _validate_row(row: dict[str, object], image: int = 10) -> None:
    graph, right, down = _synthetic_graph()
    zeros = np.zeros_like(right)
    with mock.patch.object(
        evaluator.e18_core.e15,
        "_lab_pair_matrices",
        return_value=(zeros, zeros),
    ):
        evaluator._validate_success_row(
            row,
            scene=_scene(image),
            cache_sha256=SHA,
            right=right,
            down=down,
            graph=graph,
        )


class FrozenContractTests(unittest.TestCase):
    def test_protocol_decision_cli_and_exclusions_are_literal(self) -> None:
        self.assertEqual(
            evaluator.DECISION_RULE,
            {
                "expansion_cap_hit_scenes_max": 0,
                "one_initial_zero_root_scenes": 8,
                "legal_origin_scenes": 8,
                "mean_rigid_coverage_min": 0.35,
                "worst_rigid_coverage_min": 0.25,
                "mean_accepted_cross_seam_precision_min": 0.85,
                "worst_accepted_cross_seam_precision_min": 0.70,
                "mean_component_cycle_rank_ratio_min": 0.05,
            },
        )
        protocol = evaluator.E19_PROTOCOL
        self.assertEqual(protocol["geometry"]["root_translation"], [0, 0])
        self.assertEqual(protocol["geometry"]["initial_states"], 1)
        self.assertEqual(
            protocol["geometry"]["coordinates"],
            "signed_relative_never_clipped_to_absolute_frame",
        )
        self.assertFalse(protocol["geometry"]["absolute_board"])
        self.assertFalse(protocol["geometry"]["rotation"])
        self.assertFalse(protocol["geometry"]["reflection"])
        self.assertEqual(protocol["components"]["max_edges"], 192)
        self.assertEqual(protocol["bridges"]["top_k"], 8)
        self.assertEqual(protocol["search"]["beam_width"], 256)
        self.assertEqual(
            protocol["search"]["evaluated_translations_per_state"], 64
        )
        self.assertEqual(protocol["search"]["attachment_rounds"], 64)
        self.assertEqual(
            protocol["search"]["relative_layouts_global_per_scene"], 8
        )
        self.assertEqual(
            protocol["search"]["proposal_evaluation_cap_per_scene"], 500000
        )
        self.assertEqual(
            protocol["search"]["cap_reaching"],
            "immediate_complete_KILL_no_truncated_metrics",
        )
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
        required_exclusions = {
            "absolute_board",
            "absolute_origin_inside_search",
            "residual_completion",
            "candidate_solve_metric",
            "placement",
            "neighbour",
            "SSIM",
            "NLM",
            "labels_inside_search",
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
                "e18_report",
                "report",
            },
        )

    def test_source_and_runtime_provenance_are_exact(self) -> None:
        self.assertEqual(
            set(evaluator._source_provenance()),
            {
                "e15_frame_consensus.py",
                "e18_absolute_frame_beam.py",
                "e19_relative_frame_beam.py",
                "eval_clean_score_oracle.py",
                "eval_e14_cc192_discovery.py",
                "eval_e17_cc192_rigid_viability.py",
                "eval_e18_absolute_frame_oracle.py",
                "eval_e19_relative_frame_viability.py",
                "rank96_lab_selector.py",
                "solve_buddies.py",
            },
        )
        self.assertEqual(
            evaluator._runtime_provenance(), evaluator.EXPECTED_RUNTIME_PROVENANCE
        )

    def test_live_e18_cap_kill_is_authenticated_exactly(self) -> None:
        report = evaluator._verify_e18_cap_kill(evaluator.DEFAULT_E18_REPORT)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stage"], "decoder")
        self.assertEqual(report["error"], evaluator.EXPECTED_E18_ERROR)
        self.assertEqual(report["rows"]["candidate"], [])
        self.assertEqual(report["completed_decoder_images"], [])
        self.assertEqual(
            report["run_contract_sha256"],
            evaluator.EXPECTED_E18_RUN_CONTRACT_SHA256,
        )

    def test_e18_authentication_rejects_every_critical_tamper(self) -> None:
        pristine = copy.deepcopy(
            evaluator._load_json(evaluator.DEFAULT_E18_REPORT, label="E18 report")
        )
        real_sha256_file = evaluator.e12.sha256_file

        def authentic_hash(path: Path) -> str:
            if Path(path).resolve() == evaluator.DEFAULT_E18_REPORT.resolve():
                return evaluator.EXPECTED_E18_REPORT_SHA256
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
            ("status", set_nested("status", "complete")),
            ("stage", set_nested("stage", "relative")),
            ("error", set_nested("error", "wrong")),
            ("protocol", set_nested("protocol", {})),
            ("protocol_sha", set_nested("protocol_sha256", "0" * 64)),
            ("contract_sha", set_nested("run_contract_sha256", "0" * 64)),
            (
                "decoder_progress",
                set_nested("completed_decoder_images", [10]),
            ),
            ("nlm_progress", set_nested("completed_nlm_images", [10])),
            ("candidate_row", set_nested("rows", "candidate", [{}])),
            ("rr_rows", set_nested("rows", "RR96", [])),
            ("decisions", set_nested("decisions", {})),
            (
                "e12_provenance",
                set_nested("run_contract", "e12_report", "sha256", "0" * 64),
            ),
            (
                "e17_provenance",
                set_nested("run_contract", "e17_report", "sha256", "0" * 64),
            ),
            ("runtime_bool", set_nested("runtime_seconds", True)),
            ("runtime_nan", set_nested("runtime_seconds", float("nan"))),
        )
        for label, mutate in mutations:
            payload = copy.deepcopy(pristine)
            mutate(payload)
            with self.subTest(label=label), mock.patch.object(
                evaluator.e12,
                "sha256_file",
                side_effect=authentic_hash,
            ), mock.patch.object(evaluator, "_load_json", return_value=payload):
                with self.assertRaises(evaluator.E19ContractError):
                    evaluator._verify_e18_cap_kill(evaluator.DEFAULT_E18_REPORT)

    def test_e18_authentication_rejects_shared_source_byte_drift(self) -> None:
        pristine = copy.deepcopy(
            evaluator._load_json(evaluator.DEFAULT_E18_REPORT, label="E18 report")
        )
        real_sha256_file = evaluator.e12.sha256_file
        drifted = evaluator.E18_SHARED_SOURCE_NAMES[0]

        def drifted_hash(path: Path) -> str:
            resolved = Path(path).resolve()
            if resolved == evaluator.DEFAULT_E18_REPORT.resolve():
                return evaluator.EXPECTED_E18_REPORT_SHA256
            if resolved.name == drifted:
                return "0" * 64
            return real_sha256_file(path)

        with mock.patch.object(
            evaluator.e12, "sha256_file", side_effect=drifted_hash
        ), mock.patch.object(evaluator, "_load_json", return_value=pristine):
            with self.assertRaisesRegex(
                evaluator.E19ContractError, "shared E18-to-E19 source drifted"
            ):
                evaluator._verify_e18_cap_kill(evaluator.DEFAULT_E18_REPORT)

    def test_all_e_drive_and_report_overlap_guards_precede_loading(self) -> None:
        valid = evaluator.E19Paths(
            evaluator.DEFAULT_RAW_CACHE_DIR,
            evaluator.DEFAULT_CALIBRATION_REPORT,
            evaluator.DEFAULT_E12_REPORT,
            evaluator.DEFAULT_E18_REPORT,
            Path("E:/pazzle_work/relative_frame_e19/e19_unit_guard.json"),
        )
        cases = (
            evaluator.E19Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e18_report,
                Path("C:/tmp/e19.json"),
            ),
            evaluator.E19Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e18_report,
                Path("E:/pazzle_work/relative_frame_e19/e19.txt"),
            ),
            evaluator.E19Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e18_report,
                valid.raw_cache_dir / "report.json",
            ),
            evaluator.E19Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e18_report,
                evaluator.DEFAULT_E12_REPORT.parent / "score_cache" / "report.json",
            ),
            evaluator.E19Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e18_report,
                valid.e12_report,
            ),
            evaluator.E19Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                valid.e18_report,
                valid.e18_report,
            ),
            evaluator.E19Paths(
                Path("C:/tmp/raw"),
                valid.calibration_report,
                valid.e12_report,
                valid.e18_report,
                valid.report,
            ),
            evaluator.E19Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                Path("C:/tmp/e12.json"),
                valid.e18_report,
                valid.report,
            ),
            evaluator.E19Paths(
                valid.raw_cache_dir,
                valid.calibration_report,
                valid.e12_report,
                Path("C:/tmp/e18.json"),
                valid.report,
            ),
        )
        for paths in cases:
            with self.subTest(paths=paths), mock.patch.object(
                evaluator, "_verify_e18_cap_kill"
            ) as verify, mock.patch.object(
                evaluator.e17, "_load_verified_structure_inputs"
            ) as loader:
                with self.assertRaises(evaluator.E19ContractError):
                    evaluator.run_gate(paths)
                verify.assert_not_called()
                loader.assert_not_called()


class TruthAndDecisionTests(unittest.TestCase):
    def test_seam_truth_rejects_row_wrap_and_noncanonical_directions(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        self.assertTrue(evaluator._seam_is_true((0, 1, 0, 1), permutation))
        self.assertTrue(evaluator._seam_is_true((0, 24, 1, 0), permutation))
        self.assertFalse(evaluator._seam_is_true((23, 24, 0, 1), permutation))
        for seam in ((1, 0, 0, -1), (24, 0, -1, 0), (0, 1, 1, 1)):
            with self.subTest(seam=seam), self.assertRaisesRegex(
                evaluator.E19ContractError, "canonical"
            ):
                evaluator._seam_is_true(seam, permutation)

    def test_non_self_inverse_permutation_uses_input_tile_to_clean_position(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        permutation[:3] = [1, 2, 0]
        self.assertTrue(evaluator._seam_is_true((0, 1, 0, 1), permutation))
        self.assertFalse(evaluator._seam_is_true((1, 2, 0, 1), permutation))

    def test_zero_precision_duplicates_and_noninteger_seams_fail_closed(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        self.assertEqual(
            evaluator.accepted_cross_seam_precision((), permutation),
            (0, 0, 0.0),
        )
        with self.assertRaisesRegex(evaluator.E19ContractError, "duplicated"):
            evaluator.accepted_cross_seam_precision(
                ((0, 1, 0, 1), (0, 1, 0, 1)), permutation
            )
        for seam in ((0.5, 1, 0, 1), (False, 1, 0, 1)):
            with self.subTest(seam=seam), self.assertRaises(
                evaluator.E19ContractError
            ):
                evaluator.accepted_cross_seam_precision((seam,), permutation)

    def test_decision_is_inclusive_and_every_check_is_required(self) -> None:
        passing = {
            "expansion_cap_hit_scenes": 0,
            "one_initial_zero_root_scenes": 8,
            "legal_origin_scenes": 8,
            "mean_rigid_coverage": 0.35,
            "worst_rigid_coverage": 0.25,
            "mean_accepted_cross_seam_precision": 0.85,
            "worst_accepted_cross_seam_precision": 0.70,
            "mean_component_cycle_rank_ratio": 0.05,
        }
        result = evaluator.decision(passing)
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "go_E20_absolute_origin_residual")
        failures = (
            {**passing, "expansion_cap_hit_scenes": 1},
            {**passing, "one_initial_zero_root_scenes": 7},
            {**passing, "legal_origin_scenes": 7},
            {**passing, "mean_rigid_coverage": 0.349999},
            {**passing, "worst_rigid_coverage": 0.249999},
            {**passing, "mean_accepted_cross_seam_precision": 0.849999},
            {**passing, "worst_accepted_cross_seam_precision": 0.699999},
            {**passing, "mean_component_cycle_rank_ratio": 0.049999},
        )
        for summary in failures:
            with self.subTest(summary=summary):
                rejected = evaluator.decision(summary)
                self.assertFalse(rejected["passed"])
                self.assertEqual(
                    rejected["status"], "kill_dense_top8_single_edge_beam"
                )


class SuccessRowValidationTests(unittest.TestCase):
    def test_success_row_recomputes_geometry_evidence_and_truth(self) -> None:
        row = _row()
        _validate_row(row)
        layout = row["best_layout"]
        self.assertEqual(layout["translations"][0], [0, 0, 0])
        self.assertEqual(layout["bbox"], [0, 1, 0, 1])
        self.assertEqual(layout["legal_origin_bounds"], [0, 22, 0, 22])
        self.assertEqual(layout["legal_origin_count"], 529)
        self.assertEqual(layout["true_accepted_cross_seams"], 2)
        self.assertEqual(layout["accepted_cross_seam_precision"], 1.0)

    def test_every_persisted_algebra_layer_rejects_tampering(self) -> None:
        pristine = _row()
        corruptions: list[tuple[str, dict[str, object]]] = []

        extra = copy.deepcopy(pristine)
        extra["forbidden_board"] = list(range(576))
        corruptions.append(("extra_or_board", extra))

        shifted = copy.deepcopy(pristine)
        shifted["best_layout"]["translations"][0] = [0, 1, 0]
        corruptions.append(("global_shift", shifted))

        entries = copy.deepcopy(pristine)
        entries["best_layout"]["relative_entries"][0][1] = -1
        corruptions.append(("relative_entries", entries))

        bbox = copy.deepcopy(pristine)
        bbox["best_layout"]["bbox_width"] = 3
        corruptions.append(("bbox", bbox))

        origins = copy.deepcopy(pristine)
        origins["best_layout"]["legal_origin_count"] = 528
        corruptions.append(("legal_origins", origins))

        coverage = copy.deepcopy(pristine)
        coverage["best_layout"]["rigid_coverage"] = 0.5
        corruptions.append(("coverage", coverage))

        claims = copy.deepcopy(pristine)
        claims["best_layout"]["satisfied_bridge_claims"] = [0]
        corruptions.append(("claims", claims))

        contacts = copy.deepcopy(pristine)
        contacts["best_layout"]["component_contacts"] = []
        corruptions.append(("contacts", contacts))

        seams = copy.deepcopy(pristine)
        seams["best_layout"]["accepted_cross_seams"] = [[23, 24, 0, 1]]
        corruptions.append(("cross_seams", seams))

        neural = copy.deepcopy(pristine)
        neural["best_layout"]["cross_neural_sum"] = 3.749
        corruptions.append(("neural", neural))

        lab = copy.deepcopy(pristine)
        lab["best_layout"]["cross_lab_sum"] = -1.0
        corruptions.append(("lab", lab))

        truth = copy.deepcopy(pristine)
        truth["best_layout"]["true_accepted_cross_seams"] = 1
        corruptions.append(("truth", truth))

        cycle = copy.deepcopy(pristine)
        cycle["best_layout"]["component_cycle_rank"] = 1
        corruptions.append(("cycle", cycle))

        cap = copy.deepcopy(pristine)
        cap["diagnostics"]["proposal_evaluations"] = evaluator.relative.EXPANSION_CAP
        corruptions.append(("cap_boundary", cap))

        for label, corrupted in corruptions:
            with self.subTest(label=label), self.assertRaises(
                evaluator.E19ContractError
            ):
                _validate_row(corrupted)


class CompleteReportTests(unittest.TestCase):
    def _success_fixture(self):
        graph, right, down = _synthetic_graph()
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
            "protocol": evaluator.E19_PROTOCOL,
            "protocol_sha256": evaluator.e12.canonical_digest(
                evaluator.E19_PROTOCOL
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
        return report, contract, digest, scenes, records, cache, graph, right, down

    def test_success_report_is_fully_recomputed_and_tamper_closed(self) -> None:
        (
            report,
            contract,
            digest,
            scenes,
            records,
            cache,
            graph,
            right,
            down,
        ) = self._success_fixture()
        zeros = np.zeros_like(right)

        def validate(value):
            with mock.patch.object(
                evaluator.e14, "_load_cc_cache", return_value=cache
            ), mock.patch.object(
                evaluator.e12, "dense_from_graph", return_value=(right, down)
            ), mock.patch.object(
                evaluator.relative, "build_graph_data", return_value=graph
            ), mock.patch.object(
                evaluator.e18_core.e15,
                "_lab_pair_matrices",
                return_value=(zeros, zeros),
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
        row["rows"][0]["best_layout"]["legal_origin_count"] = 528
        corruptions.append(row)
        summary = copy.deepcopy(report)
        summary["summary"]["mean_rigid_coverage"] = 1.0
        corruptions.append(summary)
        gate = copy.deepcopy(report)
        gate["decision"]["passed"] = not gate["decision"]["passed"]
        corruptions.append(gate)
        stage = copy.deepcopy(report)
        stage["stage"] = "kill_relative_cap"
        corruptions.append(stage)
        completed = copy.deepcopy(report)
        completed["completed_images"] = completed["completed_images"][:-1]
        corruptions.append(completed)
        forbidden = copy.deepcopy(report)
        forbidden["rows"][0]["best_layout"]["board"] = list(range(576))
        corruptions.append(forbidden)
        for corrupted in corruptions:
            with self.subTest(), self.assertRaises(evaluator.E19ContractError):
                validate(corrupted)

    def test_cap_terminal_discards_all_layout_metrics_and_revalidates(self) -> None:
        scenes = [_scene(image) for image in evaluator.e12.CALIBRATION_IDS]
        records = {
            image: {"path": f"E:/cache/{image}.npz", "sha256": SHA}
            for image in evaluator.e12.CALIBRATION_IDS
        }
        cap_failure = {
            "image": 10,
            "validation_name": "validation_10",
            "clean_score_cache_sha256": SHA,
            "proposal_evaluations": evaluator.relative.EXPANSION_CAP,
            "rounds": 3,
            "initial_states": 1,
            "cap_hit": True,
            "error": (
                "RelativeFrameCapError: relative beam reached the frozen "
                "cumulative proposal cap"
            ),
        }
        contract = {"frozen": True}
        digest = evaluator.e12.canonical_digest(contract)
        report: dict[str, object] = {
            "schema_version": evaluator.SCHEMA_VERSION,
            "schema": evaluator.REPORT_SCHEMA,
            "experiment": evaluator.EXPERIMENT,
            "status": "complete",
            "stage": "kill_relative_cap",
            "protocol": evaluator.E19_PROTOCOL,
            "protocol_sha256": evaluator.e12.canonical_digest(
                evaluator.E19_PROTOCOL
            ),
            "run_contract": contract,
            "run_contract_sha256": digest,
            "rows": [],
            "completed_images": [],
            "cap_failure": cap_failure,
            "decision": evaluator.cap_decision(cap_failure),
            "runtime_seconds": 1.0,
        }

        def validate(value):
            evaluator._validate_complete_report(
                value,
                contract=contract,
                contract_digest=digest,
                e12_report={},
                scenes=scenes,
                clean_records=records,
            )

        corruptions = []
        retained = copy.deepcopy(report)
        retained["rows"] = [_row()]
        corruptions.append(retained)
        completed = copy.deepcopy(report)
        completed["completed_images"] = [10]
        corruptions.append(completed)
        summary = copy.deepcopy(report)
        summary["summary"] = {"mean_rigid_coverage": 1.0}
        corruptions.append(summary)
        below_cap = copy.deepcopy(report)
        below_cap["cap_failure"]["proposal_evaluations"] -= 1
        corruptions.append(below_cap)
        bad_error = copy.deepcopy(report)
        bad_error["cap_failure"]["error"] = "wrong"
        corruptions.append(bad_error)
        late_round = copy.deepcopy(report)
        late_round["cap_failure"]["rounds"] = evaluator.relative.MAX_ATTACHMENTS
        corruptions.append(late_round)
        fake_pass = copy.deepcopy(report)
        fake_pass["decision"]["passed"] = True
        corruptions.append(fake_pass)
        cache = SimpleNamespace(sha256=SHA)
        with mock.patch.object(
            evaluator.e14, "_load_cc_cache", return_value=cache
        ):
            validate(report)
            for corrupted in corruptions:
                with self.subTest(), self.assertRaises(
                    evaluator.E19ContractError
                ):
                    validate(corrupted)


class StagingTests(unittest.TestCase):
    def test_first_cap_is_terminal_and_calls_no_forbidden_route(self) -> None:
        graph, right, down = _synthetic_graph()
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
        paths = evaluator.E19Paths(
            evaluator.DEFAULT_RAW_CACHE_DIR,
            evaluator.DEFAULT_CALIBRATION_REPORT,
            evaluator.DEFAULT_E12_REPORT,
            evaluator.DEFAULT_E18_REPORT,
            Path(
                "E:/pazzle_work/relative_frame_e19/"
                "e19_unit_no_write_never_create.json"
            ),
        )
        e12_report = {"scene_provenance_digest": "scene-digest"}
        e18_report = {
            "run_contract_sha256": evaluator.EXPECTED_E18_RUN_CONTRACT_SHA256
        }
        cap = evaluator.relative.RelativeFrameCapError(
            evaluator.relative.EXPANSION_CAP, 3
        )
        with mock.patch.object(
            evaluator, "_verify_e18_cap_kill", return_value=e18_report
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
            evaluator.relative, "build_graph_data", return_value=graph
        ), mock.patch.object(
            evaluator.relative, "run_relative_frame", side_effect=cap
        ) as solver, mock.patch.object(
            evaluator, "evaluate_structure"
        ) as measured, mock.patch.object(
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

        self.assertEqual(solver.call_count, 1)
        measured.assert_not_called()
        nlm.assert_not_called()
        absolute_solver.assert_not_called()
        residual.assert_not_called()
        assemble.assert_not_called()
        ssim.assert_not_called()
        self.assertGreaterEqual(writer.call_count, 2)
        self.assertEqual(output["status"], "complete")
        self.assertEqual(output["stage"], "kill_relative_cap")
        self.assertFalse(output["decision"]["passed"])
        self.assertEqual(output["rows"], [])
        self.assertEqual(output["completed_images"], [])
        self.assertNotIn("summary", output)
        self.assertEqual(
            output["cap_failure"]["proposal_evaluations"],
            evaluator.relative.EXPANSION_CAP,
        )


if __name__ == "__main__":
    unittest.main()
