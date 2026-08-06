from __future__ import annotations

import ast
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

import eval_e21_posegraph_candidate_ceiling as evaluator  # noqa: E402


SHA = "a" * 64


def _diagnostics(
    *,
    component_count: int,
    nontrivial: int,
    nontrivial_tiles: int,
    claims: int,
    hypotheses: int,
) -> evaluator.pose.CandidatePoolDiagnostics:
    singleton_components = component_count - nontrivial
    singleton_tiles = 576 - nontrivial_tiles
    return evaluator.pose.CandidatePoolDiagnostics(
        component_count=component_count,
        nontrivial_components=nontrivial,
        singleton_components=singleton_components,
        total_tiles=576,
        nontrivial_tiles=nontrivial_tiles,
        singleton_tiles=singleton_tiles,
        emitter_tiles=nontrivial_tiles,
        directional_emitter_rows=4 * nontrivial_tiles,
        positive_top8_before_component_filter=claims,
        same_component_filtered=0,
        claims=claims,
        nontrivial_target_claims=claims,
        singleton_target_claims=0,
        hypotheses=hypotheses,
        component_pairs=int(hypotheses > 0),
        component_pairs_with_alternative_offsets=0,
        unique_physical_seams=hypotheses,
        reciprocal_physical_seams=0,
    )


def _partition(
    grouped_entries: tuple[tuple[tuple[int, int, int], ...], ...],
) -> tuple[
    tuple[evaluator.pose.RigidComponent, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    frozenset[int],
]:
    used = {
        tile
        for entries in grouped_entries
        for tile, _row, _col in entries
    }
    values = [tuple(sorted(entries)) for entries in grouped_entries]
    values.extend(((tile, 0, 0),) for tile in range(576) if tile not in used)
    values.sort(key=lambda entries: (-len(entries), min(row[0] for row in entries), entries))
    components = tuple(
        evaluator.pose.RigidComponent(index, entries)
        for index, entries in enumerate(values)
    )
    owner = np.full(576, -1, dtype=np.int64)
    rows = np.zeros(576, dtype=np.int64)
    cols = np.zeros(576, dtype=np.int64)
    for component in components:
        for tile, row, col in component.entries:
            owner[tile] = component.component_id
            rows[tile] = row
            cols[tile] = col
    nontrivial = frozenset(
        component.component_id for component in components if component.size >= 2
    )
    return components, owner, rows, cols, nontrivial


def _pool(
    *,
    grouped_entries: tuple[tuple[tuple[int, int, int], ...], ...] = (
        ((0, 0, 0), (1, 0, 1)),
        ((24, 0, 0), (25, 0, 1)),
    ),
    relation: tuple[int, int, int, int] | None = (0, 1, 1, 0),
) -> evaluator.pose.CandidatePoolResult:
    components, owner, rows, cols, nontrivial = _partition(grouped_entries)
    claims: tuple[evaluator.pose.CandidateClaim, ...]
    hypotheses: tuple[evaluator.pose.PoseHypothesis, ...]
    if relation is None:
        claims = ()
        hypotheses = ()
    else:
        u, v, dr, dc = relation
        anchor = components[u].tiles[0]
        target = components[v].tiles[0]
        seam_dy, seam_dx = ((1, 0) if dr != 0 else (0, 1))
        claims = (
            evaluator.pose.CandidateClaim(
                0,
                1.25,
                anchor,
                target,
                seam_dy,
                seam_dx,
                u,
                v,
            ),
        )
        hypotheses = (
            evaluator.pose.PoseHypothesis(
                0,
                u,
                v,
                dr,
                dc,
                (((anchor, target, seam_dy, seam_dx), 1.25),),
                (),
            ),
        )
    nontrivial_tiles = sum(components[cid].size for cid in nontrivial)
    diagnostics = _diagnostics(
        component_count=len(components),
        nontrivial=len(nontrivial),
        nontrivial_tiles=nontrivial_tiles,
        claims=len(claims),
        hypotheses=len(hypotheses),
    )
    return evaluator.pose.CandidatePoolResult(
        components=components,
        owner=owner,
        local_rows=rows,
        local_cols=cols,
        nontrivial_component_ids=nontrivial,
        claims=claims,
        hypotheses=hypotheses,
        diagnostics=diagnostics,
    )


def _scene(image: int = 10, permutation: np.ndarray | None = None):
    if permutation is None:
        permutation = np.arange(576, dtype=np.int64)
    return SimpleNamespace(
        image_id=image,
        validation_name=f"validation_{image}",
        cache_path=Path(f"E:/cache/image_{image:04d}.npz"),
        cache_sha256=SHA,
        candidate_ids=np.zeros((576, 1), dtype=np.int64),
        base_scores=np.zeros((4, 576, 1), dtype=np.float32),
        permutation=np.asarray(permutation, dtype=np.int64),
        tiles_uint8=np.zeros((576, 20, 20, 3), dtype=np.uint8),
        target_uint8=np.zeros((480, 480, 3), dtype=np.uint8),
    )


def _scores() -> tuple[np.ndarray, np.ndarray]:
    right = np.zeros((576, 576), dtype=np.float32)
    return right, right.copy()


def _row(image: int = 10, pool: evaluator.pose.CandidatePoolResult | None = None):
    right, down = _scores()
    return evaluator.evaluate_scene(
        _scene(image), pool or _pool(), right=right, down=down
    )


def _cluster(
    *,
    component_ids: tuple[int, ...],
    translations: tuple[tuple[int, int, int], ...],
    tiles: tuple[int, ...],
    relations: int,
    cycle: int,
    minimum_tile: int,
) -> evaluator.OracleCluster:
    entries = tuple((tile, 0, index) for index, tile in enumerate(tiles))
    width = len(entries)
    accepted = tuple((0, index + 1, 0, index + 1) for index in range(relations))
    return evaluator.OracleCluster(
        component_ids=component_ids,
        translations=translations,
        relative_entries=entries,
        accepted_relations=accepted,
        exact_connected_tiles=len(entries),
        exact_connected_coverage=len(entries) / 576,
        accepted_relation_count=relations,
        cycle_rank=cycle,
        minimum_tile=minimum_tile,
        bbox=(0, 0, 0, width - 1),
        bbox_height=1,
        bbox_width=width,
        legal_origin_bounds=(0, 23, 0, 24 - width),
        legal_origin_count=24 * (25 - width),
    )


class FrozenContractTests(unittest.TestCase):
    def test_protocol_gate_cli_and_exclusions_are_literal(self) -> None:
        self.assertEqual(
            evaluator.DECISION_RULE,
            {
                "completed_scenes": 8,
                "max_hypotheses_each": 6000,
                "true_relation_scenes": 8,
                "legal_origin_scenes": 8,
                "mean_exact_connected_coverage_min": 0.30,
                "worst_exact_connected_coverage_min": 0.20,
            },
        )
        protocol = evaluator.E21_PROTOCOL
        self.assertEqual(protocol["components"]["max_edges"], 96)
        self.assertEqual(protocol["components"]["min_margin"], 0.0)
        self.assertTrue(protocol["components"]["partition_includes_singletons"])
        self.assertFalse(protocol["components"]["rotation"])
        self.assertFalse(protocol["components"]["reflection"])
        self.assertEqual(protocol["claims"]["positive_dense_top_k"], 8)
        self.assertTrue(protocol["claims"]["rank_before_component_filter"])
        self.assertFalse(protocol["claims"]["iterative_growth"])
        self.assertFalse(protocol["hypotheses"]["triangle_filter"])
        self.assertFalse(protocol["oracle"]["labels_available_to_core"])
        self.assertFalse(protocol["output"]["absolute_board"])
        self.assertFalse(protocol["output"]["absolute_origin_selection"])
        self.assertEqual(
            protocol["authorization"],
            {
                "e20_report_sha256": evaluator.EXPECTED_E20_REPORT_SHA256,
                "e20_run_contract_sha256": evaluator.EXPECTED_E20_RUN_CONTRACT_SHA256,
                "e20_protocol_sha256": evaluator.EXPECTED_E20_PROTOCOL_SHA256,
                "required_status": "complete",
                "required_stage": "kill_top8_triangle_potential_route",
            },
        )
        required = {
            "clean_score_input",
            "learned_relation_logits",
            "board",
            "residual_completion",
            "placement",
            "neighbour",
            "SSIM",
            "NLM",
            "rotation",
            "reflection",
            "GPU",
            "diffusion",
        }
        self.assertTrue(required.issubset(set(protocol["excluded"])))
        destinations = {
            action.dest for action in evaluator.build_parser()._actions
            if action.dest != "help"
        }
        self.assertEqual(
            destinations,
            {"raw_cache_dir", "calibration_report", "e12_report", "e20_report", "report"},
        )

    def test_source_runtime_and_default_lineage_are_exact(self) -> None:
        self.assertEqual(
            set(evaluator._source_provenance()),
            {
                "e21_posegraph_candidate_oracle.py",
                "eval_buddies_ssim_budget.py",
                "eval_clean_score_oracle.py",
                "eval_e14_cc192_discovery.py",
                "eval_e20_triangle_potential_viability.py",
                "eval_e21_posegraph_candidate_ceiling.py",
                "eval_seeded_qap.py",
                "solve_buddies.py",
            },
        )
        self.assertEqual(
            evaluator._runtime_provenance(), evaluator.EXPECTED_RUNTIME_PROVENANCE
        )
        self.assertEqual(evaluator.DEFAULT_E12_REPORT, evaluator.e14.DEFAULT_E12_REPORT)
        self.assertEqual(
            evaluator.DEFAULT_REPORT,
            Path("E:/pazzle_work/posegraph_e21/cc96_top8_anchor_candidate_ceiling_v1.json"),
        )

    def test_report_and_all_runtime_inputs_are_e_guarded(self) -> None:
        with self.assertRaises(evaluator.E21ContractError):
            evaluator._require_e_drive(Path("C:/tmp/report.json"), label="report")
        self.assertEqual(
            evaluator._require_e_drive(Path("E:/tmp/report.json"), label="report").drive.upper(),
            "E:",
        )

    def test_live_e20_kill_is_authenticated(self) -> None:
        report = evaluator._verify_e20_kill(evaluator.DEFAULT_E20_REPORT)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["stage"], evaluator.EXPECTED_E20_STAGE)
        self.assertFalse(report["decision"]["passed"])

    def test_e20_authorization_rejects_stage_and_numeric_coercion(self) -> None:
        pristine = copy.deepcopy(
            evaluator._load_json(evaluator.DEFAULT_E20_REPORT, label="E20 report")
        )
        real_hash = evaluator.e12.sha256_file

        def authentic_hash(path: Path) -> str:
            if Path(path).resolve() == evaluator.DEFAULT_E20_REPORT.resolve():
                return evaluator.EXPECTED_E20_REPORT_SHA256
            return real_hash(path)

        for mutation in (
            lambda value: value.__setitem__("stage", "triangle_pose_structure"),
            lambda value: value.__setitem__("runtime_seconds", True),
            lambda value: value.__setitem__("schema_version", 1.0),
        ):
            payload = copy.deepcopy(pristine)
            mutation(payload)
            with self.subTest(payload=payload.get("stage")), mock.patch.object(
                evaluator.e12, "sha256_file", side_effect=authentic_hash
            ), mock.patch.object(evaluator, "_load_json", return_value=payload):
                with self.assertRaises(evaluator.E21ContractError):
                    evaluator._verify_e20_kill(evaluator.DEFAULT_E20_REPORT)


class OracleGeometryTests(unittest.TestCase):
    def test_truth_shift_sign_and_exact_relation_are_literal(self) -> None:
        result = _pool()
        shifts = evaluator.component_truth_shifts(
            result, np.arange(576, dtype=np.int64)
        )
        self.assertEqual(shifts[0], (0, 0))
        self.assertEqual(shifts[1], (1, 0))
        true = evaluator.true_pose_hypotheses(result, shifts)
        self.assertEqual(tuple(value.relation for value in true), ((0, 1, 1, 0),))
        wrong = _pool(relation=(0, 1, -1, 0))
        self.assertEqual(
            evaluator.true_pose_hypotheses(
                wrong,
                evaluator.component_truth_shifts(
                    wrong, np.arange(576, dtype=np.int64)
                ),
            ),
            (),
        )

    def test_whole_component_impurity_excludes_component_and_relation(self) -> None:
        result = _pool(
            grouped_entries=(
                ((0, 0, 0), (2, 0, 1)),
                ((24, 0, 0), (25, 0, 1)),
            )
        )
        shifts, true, clusters, selected = evaluator.build_oracle_ceiling(
            result, np.arange(576, dtype=np.int64)
        )
        self.assertIsNone(shifts[0])
        self.assertEqual(true, ())
        self.assertNotIn(0, {cid for cluster in clusters for cid in cluster.component_ids})
        self.assertEqual(selected.exact_connected_tiles, 2)
        self.assertEqual(selected.component_ids, (1,))

    def test_pure_singletons_are_included_without_a_relation(self) -> None:
        result = _pool(grouped_entries=(), relation=None)
        _shifts, true, clusters, selected = evaluator.build_oracle_ceiling(
            result, np.arange(576, dtype=np.int64)
        )
        self.assertEqual(true, ())
        self.assertEqual(len(clusters), 576)
        self.assertEqual(selected.component_ids, (0,))
        self.assertEqual(selected.relative_entries, ((0, 0, 0),))
        self.assertEqual(selected.legal_origin_bounds, (0, 23, 0, 23))
        self.assertEqual(selected.legal_origin_count, 24 * 24)

    def test_full_24_by_24_cluster_has_exactly_one_origin(self) -> None:
        components = (
            evaluator.pose.RigidComponent(
                0,
                tuple(
                    (row * 24 + col, row, col)
                    for row in range(24)
                    for col in range(24)
                ),
            ),
        )
        dsu = evaluator._PotentialDSU(components, (0,))
        cluster = evaluator._make_cluster(dsu, (0,), ())
        self.assertEqual(cluster.bbox_height, 24)
        self.assertEqual(cluster.bbox_width, 24)
        self.assertEqual(cluster.legal_origin_bounds, (0, 0, 0, 0))
        self.assertEqual(cluster.legal_origin_count, 1)

    def test_consistent_cycle_is_evidence_and_inconsistent_cycle_fails(self) -> None:
        components = tuple(
            evaluator.pose.RigidComponent(index, ((index, 0, 0),))
            for index in range(3)
        )
        dsu = evaluator._PotentialDSU(components, (0, 1, 2))
        self.assertTrue(dsu.union(0, 1, 0, 1))
        self.assertTrue(dsu.union(1, 2, 1, 0))
        self.assertFalse(dsu.union(0, 2, 1, 1))
        cluster = evaluator._make_cluster(
            dsu,
            (0, 1, 2),
            ((0, 1, 0, 1), (0, 2, 1, 1), (1, 2, 1, 0)),
        )
        self.assertEqual(cluster.accepted_relation_count, 3)
        self.assertEqual(cluster.cycle_rank, 1)
        with self.assertRaisesRegex(evaluator.E21ContractError, "cycle"):
            dsu.union(0, 2, 1, 0)

    def test_union_rejects_collision_and_span_overflow(self) -> None:
        components = (
            evaluator.pose.RigidComponent(0, ((0, 0, 0),)),
            evaluator.pose.RigidComponent(1, ((1, 0, 0),)),
        )
        collision = evaluator._PotentialDSU(components, (0, 1))
        with self.assertRaisesRegex(evaluator.E21ContractError, "collision"):
            collision.union(0, 1, 0, 0)
        span = evaluator._PotentialDSU(components, (0, 1))
        with self.assertRaisesRegex(evaluator.E21ContractError, "span"):
            span.union(0, 1, 24, 0)

    def test_cluster_is_normalized_and_legal_origins_are_analytic(self) -> None:
        _shifts, _true, _clusters, selected = evaluator.build_oracle_ceiling(
            _pool(), np.arange(576, dtype=np.int64)
        )
        self.assertEqual(selected.bbox, (0, 1, 0, 1))
        self.assertEqual(selected.bbox_height, 2)
        self.assertEqual(selected.bbox_width, 2)
        self.assertEqual(selected.legal_origin_bounds, (0, 22, 0, 22))
        self.assertEqual(selected.legal_origin_count, 23 * 23)
        self.assertEqual(selected.exact_connected_tiles, 4)

    def test_selection_ties_use_relation_cycle_min_tile_then_translations(self) -> None:
        lower_relations = _cluster(
            component_ids=(0, 1),
            translations=((0, 0, 0), (1, 0, 1)),
            tiles=(0, 1, 2, 3),
            relations=1,
            cycle=0,
            minimum_tile=0,
        )
        more_relations = replace(
            lower_relations,
            accepted_relation_count=2,
            cycle_rank=0,
            minimum_tile=10,
            translations=((0, 0, 0), (1, 1, 0)),
        )
        self.assertIs(
            evaluator.select_oracle_cluster((lower_relations, more_relations)),
            more_relations,
        )
        cycle = replace(more_relations, cycle_rank=1, minimum_tile=20)
        self.assertIs(evaluator.select_oracle_cluster((more_relations, cycle)), cycle)
        small_tile = replace(cycle, minimum_tile=3, translations=((0, 1, 0),))
        self.assertIs(evaluator.select_oracle_cluster((cycle, small_tile)), small_tile)
        lexical = replace(small_tile, translations=((0, 0, 1),))
        self.assertIs(evaluator.select_oracle_cluster((small_tile, lexical)), lexical)


class ReportingAndReplayTests(unittest.TestCase):
    def test_legal_claim_target_diagnostics_pass_but_label_payload_keys_fail(self) -> None:
        payload = evaluator._core_payload(_pool())
        self.assertEqual(payload["diagnostics"]["nontrivial_target_claims"], 1)
        self.assertEqual(payload["diagnostics"]["singleton_target_claims"], 0)
        for key in ("target_pixels", "permutation"):
            with self.subTest(key=key), self.assertRaises(evaluator.E21ContractError):
                evaluator._check_forbidden_payload_keys({key: [0]})

    def test_core_is_called_with_right_and_down_only(self) -> None:
        row = _row()
        right, down = _scores()
        calls: list[tuple[object, ...]] = []

        def core(*args):
            calls.append(args)
            return _pool()

        with mock.patch.object(
            evaluator.pose, "run_posegraph_candidate_oracle", side_effect=core
        ):
            evaluator._validate_success_row(
                row, scene=_scene(), right=right, down=down
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)
        self.assertIs(calls[0][0], right)
        self.assertIs(calls[0][1], down)

    def test_row_hashes_and_label_payload_are_recomputed(self) -> None:
        row = _row()
        right, down = _scores()
        evaluator._validate_success_row(
            row,
            scene=_scene(),
            right=right,
            down=down,
            expected_result=_pool(),
        )
        for key, mutate in (
            ("core", lambda value: value["core"].__setitem__("hypothesis_count", 2)),
            ("core_hash", lambda value: value.__setitem__("core_sha256", "0" * 64)),
            ("oracle", lambda value: value["oracle"]["selected"].__setitem__("cycle_rank", 9)),
            ("metrics", lambda value: value["metrics"].__setitem__("true_hypotheses", 0)),
        ):
            tampered = copy.deepcopy(row)
            mutate(tampered)
            with self.subTest(key=key), self.assertRaises(evaluator.E21ContractError):
                evaluator._validate_success_row(
                    tampered,
                    scene=_scene(),
                    right=right,
                    down=down,
                    expected_result=_pool(),
                )

    def test_summary_and_decision_are_all_inclusive(self) -> None:
        rows = []
        for image in range(10, 18):
            row = _row(image)
            row["metrics"]["hypothesis_count"] = 6000
            row["metrics"]["true_hypotheses"] = 1
            row["metrics"]["legal_origin_count"] = 1
            row["metrics"]["selected_exact_connected_tiles"] = 173
            row["metrics"]["selected_exact_connected_coverage"] = 0.30
            rows.append(row)
        summary = evaluator.summarize(rows)
        summary["worst_exact_connected_coverage"] = 0.20
        result = evaluator.decision(summary)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

        mutations = (
            {"completed_scenes": 7},
            {"hypotheses_within_cap_scenes": 7},
            {"max_hypothesis_count": 6001},
            {"true_relation_scenes": 7},
            {"legal_origin_scenes": 7},
            {"mean_exact_connected_coverage": 0.299999},
            {"worst_exact_connected_coverage": 0.199999},
        )
        for mutation in mutations:
            changed = {**summary, **mutation}
            with self.subTest(mutation=mutation):
                self.assertFalse(evaluator.decision(changed)["passed"])

    def test_decision_rejects_bool_fractional_and_nonfinite_types(self) -> None:
        summary = {
            "completed_scenes": 8,
            "hypotheses_within_cap_scenes": 8,
            "max_hypothesis_count": 6000,
            "true_relation_scenes": 8,
            "legal_origin_scenes": 8,
            "mean_exact_connected_coverage": 0.30,
            "worst_exact_connected_coverage": 0.20,
        }
        for field, value in (
            ("completed_scenes", True),
            ("max_hypothesis_count", 6000.0),
            ("true_relation_scenes", 8.5),
            ("mean_exact_connected_coverage", float("nan")),
            ("worst_exact_connected_coverage", True),
        ):
            with self.subTest(field=field), self.assertRaises(evaluator.E21ContractError):
                evaluator.decision({**summary, field: value})

    def test_complete_report_replays_core_and_rejects_tamper(self) -> None:
        pool = _pool()
        right, down = _scores()
        scenes = [_scene(image) for image in range(10, 18)]
        rows = [
            evaluator.evaluate_scene(scene, pool, right=right, down=down)
            for scene in scenes
        ]
        contract = {"frozen": True}
        contract_digest = evaluator.e12.canonical_digest(contract)
        summary = evaluator.summarize(rows)
        result_decision = evaluator.decision(summary)
        report = {
            "schema_version": evaluator.SCHEMA_VERSION,
            "schema": evaluator.REPORT_SCHEMA,
            "experiment": evaluator.EXPERIMENT,
            "status": "complete",
            "stage": result_decision["status"],
            "protocol": evaluator.E21_PROTOCOL,
            "protocol_sha256": evaluator.e12.canonical_digest(evaluator.E21_PROTOCOL),
            "run_contract": contract,
            "run_contract_sha256": contract_digest,
            "rows": rows,
            "completed_images": list(range(10, 18)),
            "summary": summary,
            "decision": result_decision,
            "runtime_seconds": 1.0,
        }
        with mock.patch.object(
            evaluator, "_dense_raw_scene", return_value=(right, down)
        ), mock.patch.object(
            evaluator.pose, "run_posegraph_candidate_oracle", return_value=pool
        ) as replay:
            evaluator._validate_complete_report(
                report,
                contract=contract,
                contract_digest=contract_digest,
                scenes=scenes,
            )
            self.assertEqual(replay.call_count, 8)

            tampered = copy.deepcopy(report)
            tampered["rows"][3]["metrics"]["selected_exact_connected_tiles"] += 1
            with self.assertRaises(evaluator.E21ContractError):
                evaluator._validate_complete_report(
                    tampered,
                    contract=contract,
                    contract_digest=contract_digest,
                    scenes=scenes,
                )

    def test_dense_boundary_requires_exact_float32(self) -> None:
        scene = _scene()
        wrong = np.zeros((576, 576), dtype=np.float64)
        with mock.patch.object(
            evaluator.e12, "dense_from_graph", return_value=(wrong, wrong.copy())
        ), self.assertRaises(evaluator.E21ContractError):
            evaluator._dense_raw_scene(scene)

    def test_forbidden_board_restoration_and_gpu_routes_are_not_imported_or_called(self) -> None:
        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: set[str] = set()
        calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        self.assertTrue({"imgio", "pipeline", "placement_metrics"}.isdisjoint(imports))
        self.assertTrue(
            {
                "assemble",
                "solve_dense",
                "solve_buddies_from_scores",
                "fixed_nlm",
                "nlm_restore",
                "neighbour_accuracy",
                "placement_accuracy",
                "cuda",
            }.isdisjoint(calls)
        )


if __name__ == "__main__":
    unittest.main()
