from __future__ import annotations

import ast
import copy
import gc
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import e22_rcce4_candidate_oracle as core  # noqa: E402
import eval_clean_score_oracle as e12  # noqa: E402
import eval_e22_rcce4_candidate_ceiling as gate  # noqa: E402


E_TEST_ROOT = Path(os.environ.get("E22_TEST_TMP", "E:/pazzle_work/tmp/e22_tests"))
_RESULT_CACHE: core.CandidatePoolResult | None = None


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    candidate_ids = np.zeros((576, 128), dtype=np.int64)
    raw_logits = np.full((4, 576, 128), -np.inf, dtype=np.float32)
    for source in range(576):
        target = source + 1 if source % 2 == 0 else source - 1
        candidate_ids[source, 0] = target
        raw_logits[:, source, 0] = np.asarray(
            (source + 0.1, source + 0.2, source + 0.3, source + 0.4),
            dtype=np.float32,
        )
    return candidate_ids, raw_logits


def _partition(
    groups: tuple[dict[int, tuple[int, int]], ...] = (),
) -> tuple[
    tuple[core.RigidComponent, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    frozenset[int],
]:
    used: set[int] = set()
    entries: list[tuple[tuple[int, int, int], ...]] = []
    for group in groups:
        if used.intersection(group):
            raise AssertionError("test groups overlap")
        used.update(group)
        entries.append(
            tuple(sorted((tile, row, col) for tile, (row, col) in group.items()))
        )
    entries.extend(((tile, 0, 0),) for tile in range(576) if tile not in used)
    entries.sort(key=lambda value: (-len(value), min(item[0] for item in value), value))
    components = tuple(
        core.RigidComponent(component_id=index, entries=value)
        for index, value in enumerate(entries)
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
    for value in (owner, rows, cols):
        value.setflags(write=False)
    return components, owner, rows, cols, nontrivial


def _synthetic_result() -> core.CandidatePoolResult:
    global _RESULT_CACHE
    if _RESULT_CACHE is None:
        candidate_ids, raw_logits = _inputs()
        _RESULT_CACHE = core.run_rcce4_candidate_oracle(candidate_ids, raw_logits)
    return _RESULT_CACHE


def _partition_replay_patch(
    result: core.CandidatePoolResult,
) -> mock._patch:  # type: ignore[name-defined]
    return mock.patch.object(
        gate,
        "_independent_raw_cc96_partition",
        return_value=(
            result.components,
            result.owner,
            result.local_rows,
            result.local_cols,
            result.nontrivial_component_ids,
        ),
    )


def _scene(image: int = 10) -> SimpleNamespace:
    candidate_ids, raw_logits = _inputs()
    return SimpleNamespace(
        image_id=image,
        validation_name=f"synthetic_{image:04d}.png",
        cache_path=Path(f"E:/synthetic/image_{image:04d}.npz"),
        cache_sha256=f"{image:064x}",
        candidate_ids=candidate_ids,
        base_scores=raw_logits,
        permutation=np.arange(576, dtype=np.int64),
        tiles_uint8=np.asarray([image], dtype=np.uint8),
        target_uint8=np.asarray([image + 1], dtype=np.uint8),
    )


def _label_result(
    *,
    include_pair: bool = True,
    include_hypothesis: bool = True,
    exact_physical_seam: bool = True,
) -> SimpleNamespace:
    components, owner, rows, cols, nontrivial = _partition()
    pairs = (
        (core.AffinityPair(0, 0, 1, None, 0),) if include_pair else ()
    )
    claim = core.RCCE4Claim(
        claim_id=0,
        pair_id=0,
        first=0 if exact_physical_seam else 2,
        second=1 if exact_physical_seam else 3,
        dy=0,
        dx=1,
        first_component=0 if exact_physical_seam else 2,
        second_component=1 if exact_physical_seam else 3,
        forward_observation=None,
        reverse_observation=None,
    )
    hypotheses = (
        (
            core.PoseHypothesis(
                hypothesis_id=0,
                relation_id=0,
                u=0,
                v=1,
                dr=0,
                dc=1,
                claim_ids=(0,),
            ),
        )
        if include_hypothesis
        else ()
    )
    return SimpleNamespace(
        components=components,
        owner=owner,
        local_rows=rows,
        local_cols=cols,
        nontrivial_component_ids=nontrivial,
        affinity_pairs=pairs,
        claims=(claim,),
        hypotheses=hypotheses,
    )


def _passing_metrics() -> dict[str, object]:
    return {
        "tile_orientation_degrees": 0,
        "emitter_tiles": 576,
        "directed_valid_memberships": 576,
        "unordered_affinity_pairs": 288,
        "finite_directional_logit_observations": 2304,
        "rcce4_preclaims": 1152,
        "cross_component_claims": 1152,
        "relation_candidates": 900,
        "geometry_valid_hypotheses": 800,
        "bounds_passed": True,
        "eligible_contacts": 100,
        "eligible_pair_hits": 90,
        "eligible_contact_recall": 0.90,
        "unconditional_cross_component_recall": 0.75,
        "postfilter_exact_physical_seam_survivors": 90,
        "postfilter_eligible_true_survival": 1.0,
        "true_hypotheses": 1,
        "selected_exact_connected_tiles": 173,
        "selected_exact_connected_coverage": 0.30,
        "selected_cycle_rank_ratio": 0.05,
        "legal_origin_count": 1,
    }


def _summary_rows() -> list[dict[str, object]]:
    return [
        {"image": image, "metrics": copy.deepcopy(_passing_metrics())}
        for image in e12.CALIBRATION_IDS
    ]


class FrozenEvaluatorContractTests(unittest.TestCase):
    def test_exact_authorization_bounds_gates_and_routing_are_frozen(self) -> None:
        self.assertEqual(
            gate.EXPECTED_E21_REPORT_SHA256,
            "0c43099860c7a16f5e968a8ea6cf637293cd639d9b86e342797ef68c5d53e724",
        )
        self.assertEqual(
            gate.EXPECTED_E21_RUN_CONTRACT_SHA256,
            "1cff1e4ca733a24d69e9b68b410e75ef453f6db712b2709bad6db9f3ed73a992",
        )
        self.assertEqual(
            gate.EXPECTED_E21_PROTOCOL_SHA256,
            "134b1192fcdeb3d63583af938b53b6906930ab725a53df01015836047cd2a04f",
        )
        self.assertEqual(gate.EXPECTED_E21_STAGE, "kill_raw_CC96_anchor_top8_candidate_pool")
        self.assertEqual(gate.MAX_DIRECTED_MEMBERSHIPS, 73_728)
        self.assertEqual(gate.MAX_UNORDERED_PAIRS, 73_728)
        self.assertEqual(gate.MAX_DIRECTIONAL_OBSERVATIONS, 294_912)
        self.assertEqual(gate.MAX_RCCE4_PRECLAIMS, 294_912)
        self.assertEqual(gate.MAX_GEOMETRY_VALID_HYPOTHESES, 294_912)
        self.assertNotIn(200_000, vars(gate).values())
        self.assertEqual(
            gate.DECISION_RULE,
            {
                "completed_scenes": 8,
                "emitters_each": 576,
                "all_bounds_scenes": 8,
                "true_relation_scenes": 8,
                "legal_origin_scenes": 8,
                "positive_eligible_denominator_scenes": 8,
                "exact_postfilter_survival_scenes": 8,
                "mean_eligible_contact_recall_min": 0.90,
                "worst_eligible_contact_recall_min": 0.80,
                "mean_exact_connected_coverage_min": 0.30,
                "worst_exact_connected_coverage_min": 0.20,
                "mean_selected_cycle_rank_ratio_min": 0.05,
                "worst_selected_cycle_rank_ratio_min": 0.01,
            },
        )

    def test_upright_clarification_has_no_orientation_ambiguity(self) -> None:
        clarification = gate.E22_PROTOCOL["upright_orientation_clarification"]
        self.assertEqual(clarification["tile_orientation_degrees"], [0])
        self.assertEqual(
            clarification["forbidden_tile_orientation_degrees"], [90, 180, 270]
        )
        self.assertIs(clarification["reflection"], False)
        self.assertIn("not_tile_rotations", clarification["four_pair_variants"])
        lift = gate.E22_PROTOCOL["pair_lift"]
        self.assertNotIn("orientation_order", lift)
        self.assertEqual(
            lift["upright_adjacency_order"],
            ["a_b_R", "b_a_R", "a_b_D", "b_a_D"],
        )
        self.assertFalse(gate.E22_PROTOCOL["components"]["rotation"])
        self.assertFalse(gate.E22_PROTOCOL["components"]["reflection"])

    def test_core_invocations_have_exactly_two_positional_array_arguments(self) -> None:
        source = Path(gate.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_rcce4_candidate_oracle"
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(len(call.args), 2)
            self.assertEqual(call.keywords, [])
            self.assertEqual(
                tuple(value.id for value in call.args if isinstance(value, ast.Name)),
                ("candidate_ids", "raw_logits"),
            )

    def test_evaluator_has_no_board_rotation_reflection_nlm_ssim_or_gpu_calls(self) -> None:
        source = Path(gate.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls: set[str] = set()
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        self.assertTrue(
            {
                "assemble",
                "rot90",
                "rotate",
                "flip",
                "fliplr",
                "flipud",
                "cuda",
                "solve_dense",
                "fixed_nlm",
                "nlm_restore",
                "structural_similarity",
            }.isdisjoint(calls)
        )
        self.assertTrue({"imgio", "placement_metrics"}.isdisjoint(imports))
        self.assertEqual(gate.DEFAULT_REPORT.drive.upper(), "E:")


class RawAndCoreReplayTests(unittest.TestCase):
    def test_raw_contract_accepts_exact_arrays_and_arbitrary_invalid_ids(self) -> None:
        candidate_ids, raw_logits = _inputs()
        candidate_ids[0, 5] = -999
        candidate_ids[1, 6] = 999
        observed_ids, observed_logits, valid = gate._validate_raw_arrays(
            candidate_ids, raw_logits
        )
        self.assertIs(observed_ids, candidate_ids)
        self.assertIs(observed_logits, raw_logits)
        self.assertEqual(valid.shape, (576, 128))
        self.assertEqual(int(valid.sum()), 576)

    def test_raw_contract_rejects_dtype_shape_contiguity_and_membership_errors(self) -> None:
        candidate_ids, raw_logits = _inputs()
        cases = (
            (candidate_ids.astype(np.int32), raw_logits, "int64"),
            (candidate_ids, raw_logits.astype(np.float64), "float32"),
            (candidate_ids[:, :-1], raw_logits, "576,128"),
            (candidate_ids, raw_logits[:, :, :-1], "4,576,128"),
            (candidate_ids[:, ::-1], raw_logits, "contiguous"),
            (candidate_ids, raw_logits[:, :, ::-1], "contiguous"),
        )
        for bad_ids, bad_logits, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                gate.E22ContractError, message
            ):
                gate._validate_raw_arrays(bad_ids, bad_logits)

        bad_ids = candidate_ids.copy()
        bad_ids[0, 0] = 0
        with self.assertRaisesRegex(gate.E22ContractError, "non-self"):
            gate._validate_raw_arrays(bad_ids, raw_logits)
        bad_ids = candidate_ids.copy()
        bad_ids[0, 1] = bad_ids[0, 0]
        bad_logits = raw_logits.copy()
        bad_logits[:, 0, 1] = 1.0
        with self.assertRaisesRegex(gate.E22ContractError, "unique"):
            gate._validate_raw_arrays(bad_ids, bad_logits)
        bad_ids = candidate_ids.copy()
        bad_ids[0, 0] = 576
        with self.assertRaisesRegex(gate.E22ContractError, "0..575"):
            gate._validate_raw_arrays(bad_ids, raw_logits)

    def test_shared_nan_plusinf_or_nonminusinf_padding_is_rejected(self) -> None:
        candidate_ids, raw_logits = _inputs()
        for value in (np.nan, np.inf):
            bad = raw_logits.copy()
            bad[:, 0, 5] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                gate.E22ContractError, "finite values"
            ):
                gate._validate_raw_arrays(candidate_ids, bad)
        bad = raw_logits.copy()
        bad[0, 0, 5] = 0.0
        with self.assertRaisesRegex(gate.E22ContractError, "finite mask"):
            gate._validate_raw_arrays(candidate_ids, bad)
        bad = raw_logits.copy()
        bad[:, 0, 5] = np.float32(-3.0)
        # It is now a valid common membership and therefore the padding ID is
        # checked; default ID 0 is self for emitter 0.
        with self.assertRaisesRegex(gate.E22ContractError, "non-self"):
            gate._validate_raw_arrays(candidate_ids, bad)

    def test_full_independent_candidate_pool_validation_passes(self) -> None:
        candidate_ids, raw_logits = _inputs()
        ids_before = candidate_ids.copy()
        logits_before = raw_logits.copy()
        gate.validate_candidate_pool(
            _synthetic_result(),
            candidate_ids=candidate_ids,
            raw_logits=raw_logits,
        )
        self.assertTrue(np.array_equal(candidate_ids, ids_before))
        self.assertTrue(np.array_equal(raw_logits, logits_before))

    def test_different_canonical_full_partition_fails_independent_cc96_binding(self) -> None:
        candidate_ids, raw_logits = _inputs()
        result = _synthetic_result()
        components, owner, rows, cols, nontrivial = _partition()
        forged = replace(
            result,
            components=components,
            owner=owner,
            local_rows=rows,
            local_cols=cols,
            nontrivial_component_ids=nontrivial,
        )
        with self.assertRaisesRegex(gate.E22ContractError, "independent frozen dense"):
            gate.validate_candidate_pool(
                forged,
                candidate_ids=candidate_ids,
                raw_logits=raw_logits,
            )

    def test_pair_claim_relation_hypothesis_and_diagnostic_tampering_fail(self) -> None:
        candidate_ids, raw_logits = _inputs()
        result = _synthetic_result()
        with _partition_replay_patch(result):
            bad_pair = replace(result.affinity_pairs[0], a_to_b_slot=7)
            with self.assertRaisesRegex(gate.E22ContractError, "pair OR"):
                gate.validate_candidate_pool(
                    replace(result, affinity_pairs=(bad_pair, *result.affinity_pairs[1:])),
                    candidate_ids=candidate_ids,
                    raw_logits=raw_logits,
                )
            observation = result.claims[0].forward_observation
            self.assertIsNotNone(observation)
            bad_claim = replace(
                result.claims[0],
                forward_observation=replace(observation, logit=observation.logit + 1.0),
            )
            with self.assertRaisesRegex(gate.E22ContractError, "raw-logit metadata"):
                gate.validate_candidate_pool(
                    replace(result, claims=(bad_claim, *result.claims[1:])),
                    candidate_ids=candidate_ids,
                    raw_logits=raw_logits,
                )
            bad_relation = replace(result.relation_candidates[0], dc=999)
            with self.assertRaisesRegex(
                gate.E22ContractError,
                "relation grouping|missing from grouped output",
            ):
                gate.validate_candidate_pool(
                    replace(
                        result,
                        relation_candidates=(bad_relation, *result.relation_candidates[1:]),
                    ),
                    candidate_ids=candidate_ids,
                    raw_logits=raw_logits,
                )
            bad_hypothesis = replace(result.hypotheses[0], claim_ids=(1,))
            with self.assertRaisesRegex(gate.E22ContractError, "hypothesis algebra"):
                gate.validate_candidate_pool(
                    replace(
                        result,
                        hypotheses=(bad_hypothesis, *result.hypotheses[1:]),
                    ),
                    candidate_ids=candidate_ids,
                    raw_logits=raw_logits,
                )
            bad_diagnostics = replace(
                result.diagnostics,
                pre_component_filter_claims=result.diagnostics.pre_component_filter_claims
                - 4,
            )
            with self.assertRaisesRegex(gate.E22ContractError, "diagnostics"):
                gate.validate_candidate_pool(
                    replace(result, diagnostics=bad_diagnostics),
                    candidate_ids=candidate_ids,
                    raw_logits=raw_logits,
                )

    def test_component_partition_requires_canonical_readonly_full_coverage(self) -> None:
        result = _synthetic_result()
        writable = result.owner.copy()
        with self.assertRaisesRegex(gate.E22ContractError, "read-only"):
            gate._validate_components(replace(result, owner=writable))
        swapped = (result.components[1], result.components[0], *result.components[2:])
        with self.assertRaisesRegex(gate.E22ContractError, "component IDs"):
            gate._validate_components(replace(result, components=swapped))


class LabelOnlyContactTests(unittest.TestCase):
    def test_inverse_permutation_enumerates_exactly_1104_upright_rd_seams(self) -> None:
        seams = gate.ground_truth_seams(np.arange(576, dtype=np.int64))
        self.assertEqual(len(seams), 1104)
        self.assertEqual(len(set(seams)), 1104)
        self.assertEqual(seams[:2], ((0, 1, 0, 1), (0, 24, 1, 0)))
        self.assertTrue(all((dy, dx) in ((0, 1), (1, 0)) for _, _, dy, dx in seams))
        self.assertTrue(all(first != second for first, second, _, _ in seams))

        permutation = np.arange(576, dtype=np.int64)
        permutation[1], permutation[5] = permutation[5], permutation[1]
        swapped = gate.ground_truth_seams(permutation)
        self.assertEqual(swapped[0], (0, 5, 0, 1))
        self.assertEqual(swapped[2], (5, 2, 0, 1))

    def test_signed_seam_relation_is_t_v_minus_t_u_in_canonical_order(self) -> None:
        result = _label_result()
        relation = gate.seam_relation(
            (0, 1, 0, 1),
            owner=result.owner,
            local_rows=result.local_rows,
            local_cols=result.local_cols,
        )
        self.assertEqual(relation, (0, 1, 0, 1))
        reverse_component_order = gate.seam_relation(
            (5, 1, 0, 1),
            owner=result.owner,
            local_rows=result.local_rows,
            local_cols=result.local_cols,
        )
        self.assertEqual(reverse_component_order, (1, 5, 0, -1))

    def test_eligible_denominator_is_independent_of_pair_inventory(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        with_pair = _label_result(include_pair=True)
        without_pair = _label_result(include_pair=False, include_hypothesis=False)
        shifts = gate.component_truth_shifts(with_pair, permutation)
        _inventory_a, metrics_a = gate._contact_measurement(
            with_pair, permutation=permutation, shifts=shifts
        )
        _inventory_b, metrics_b = gate._contact_measurement(
            without_pair, permutation=permutation, shifts=shifts
        )
        self.assertEqual(metrics_a["eligible_contacts"], 1104)
        self.assertEqual(
            metrics_a["eligible_contacts"], metrics_b["eligible_contacts"]
        )
        self.assertEqual(metrics_a["eligible_pair_hits"], 1)
        self.assertEqual(metrics_b["eligible_pair_hits"], 0)
        self.assertAlmostEqual(metrics_a["eligible_contact_recall"], 1 / 1104)
        self.assertEqual(metrics_b["eligible_contact_recall"], 0.0)

    def test_pair_or_hit_can_come_from_reverse_membership(self) -> None:
        result = _label_result(include_pair=True)
        self.assertIsNone(result.affinity_pairs[0].a_to_b_slot)
        self.assertEqual(result.affinity_pairs[0].b_to_a_slot, 0)
        permutation = np.arange(576, dtype=np.int64)
        shifts = gate.component_truth_shifts(result, permutation)
        _inventory, metrics = gate._contact_measurement(
            result, permutation=permutation, shifts=shifts
        )
        self.assertEqual(metrics["eligible_pair_hits"], 1)

    def test_survival_requires_exact_physical_seam_not_relation_alone(self) -> None:
        permutation = np.arange(576, dtype=np.int64)
        exact = _label_result(exact_physical_seam=True)
        shifts = gate.component_truth_shifts(exact, permutation)
        _inventory, metrics = gate._contact_measurement(
            exact, permutation=permutation, shifts=shifts
        )
        self.assertEqual(metrics["postfilter_eligible_hits"], 1)
        self.assertEqual(metrics["postfilter_exact_physical_seam_survivors"], 1)
        self.assertEqual(metrics["postfilter_eligible_true_survival"], 1.0)

        relation_only = _label_result(exact_physical_seam=False)
        shifts = gate.component_truth_shifts(relation_only, permutation)
        _inventory, metrics = gate._contact_measurement(
            relation_only, permutation=permutation, shifts=shifts
        )
        self.assertEqual(metrics["postfilter_eligible_hits"], 1)
        self.assertEqual(metrics["postfilter_exact_physical_seam_survivors"], 0)
        self.assertEqual(metrics["postfilter_eligible_true_survival"], 0.0)

    def test_zero_eligible_hits_has_zero_not_vacuous_survival(self) -> None:
        result = _label_result(include_pair=False, include_hypothesis=False)
        permutation = np.arange(576, dtype=np.int64)
        shifts = gate.component_truth_shifts(result, permutation)
        _inventory, metrics = gate._contact_measurement(
            result, permutation=permutation, shifts=shifts
        )
        self.assertEqual(metrics["eligible_contacts"], 1104)
        self.assertEqual(metrics["postfilter_eligible_hits"], 0)
        self.assertEqual(metrics["postfilter_eligible_true_survival"], 0.0)

    def test_unconditional_denominator_includes_impure_cross_component_seams(self) -> None:
        components, owner, rows, cols, nontrivial = _partition(
            ({0: (0, 0), 1: (0, 2)},)
        )
        result = SimpleNamespace(
            components=components,
            owner=owner,
            local_rows=rows,
            local_cols=cols,
            nontrivial_component_ids=nontrivial,
            affinity_pairs=(),
            claims=(),
            hypotheses=(),
        )
        permutation = np.arange(576, dtype=np.int64)
        shifts = gate.component_truth_shifts(result, permutation)
        self.assertIsNone(shifts[0])
        _inventory, metrics = gate._contact_measurement(
            result, permutation=permutation, shifts=shifts
        )
        self.assertGreater(
            metrics["unconditional_cross_component_contacts"],
            metrics["eligible_contacts"],
        )
        self.assertEqual(metrics["unconditional_pair_hits"], 0)
        self.assertEqual(metrics["unconditional_cross_component_recall"], 0.0)


class IndependentPotentialDSUTests(unittest.TestCase):
    def test_keep_u_true_branch_preserves_signed_translation(self) -> None:
        components = _partition()[0]
        dsu = gate._PotentialDSU(components, (0, 1))
        self.assertTrue(dsu.union(0, 1, 3, -4))
        root0, shift0 = dsu.find(0)
        root1, shift1 = dsu.find(1)
        self.assertEqual(root0, root1)
        self.assertEqual(
            (shift1[0] - shift0[0], shift1[1] - shift0[1]), (3, -4)
        )

    def test_keep_u_false_branch_preserves_signed_translation(self) -> None:
        components = _partition()[0]
        dsu = gate._PotentialDSU(components, (0, 1, 2))
        self.assertTrue(dsu.union(1, 2, 0, 1))
        self.assertTrue(dsu.union(0, 1, 1, 0))
        root0, shift0 = dsu.find(0)
        root1, shift1 = dsu.find(1)
        root2, shift2 = dsu.find(2)
        self.assertEqual((root0, root1, root2), (root1, root1, root1))
        self.assertEqual(
            (shift1[0] - shift0[0], shift1[1] - shift0[1]), (1, 0)
        )
        self.assertEqual(
            (shift2[0] - shift1[0], shift2[1] - shift1[1]), (0, 1)
        )

    def test_cycle_rank_ratio_and_legal_origin_formulas_are_exact(self) -> None:
        components = _partition()[0]
        dsu = gate._PotentialDSU(components, (0, 1, 2))
        relations = ((0, 1, 0, 1), (1, 2, 1, 0), (0, 2, 1, 1))
        self.assertTrue(dsu.union(*relations[0]))
        self.assertTrue(dsu.union(*relations[1]))
        self.assertFalse(dsu.union(*relations[2]))
        cluster = gate._make_cluster(dsu, (0, 1, 2), relations)
        self.assertEqual(cluster.accepted_relation_count, 3)
        self.assertEqual(cluster.cycle_rank, 1)
        self.assertEqual(cluster.cycle_rank_ratio, 0.5)
        self.assertEqual(cluster.bbox_height, 2)
        self.assertEqual(cluster.bbox_width, 2)
        self.assertEqual(cluster.legal_origin_bounds, (0, 22, 0, 22))
        self.assertEqual(cluster.legal_origin_count, 23 * 23)

        span = gate._PotentialDSU(components, (0, 1))
        span.union(0, 1, 23, 23)
        full = gate._make_cluster(span, (0, 1), ((0, 1, 23, 23),))
        self.assertEqual(full.bbox_height, 24)
        self.assertEqual(full.bbox_width, 24)
        self.assertEqual(full.legal_origin_bounds, (0, 0, 0, 0))
        self.assertEqual(full.legal_origin_count, 1)

    def test_collision_span_and_contradictory_cycle_fail_closed(self) -> None:
        components = _partition()[0]
        collision = gate._PotentialDSU(components, (0, 1))
        with self.assertRaisesRegex(gate.E22ContractError, "collision"):
            collision.union(0, 1, 0, 0)
        span = gate._PotentialDSU(components, (0, 1))
        with self.assertRaisesRegex(gate.E22ContractError, "span"):
            span.union(0, 1, 24, 0)
        cycle = gate._PotentialDSU(components, (0, 1))
        cycle.union(0, 1, 0, 1)
        with self.assertRaisesRegex(gate.E22ContractError, "contradicts"):
            cycle.union(0, 1, 1, 0)

    def test_all_pure_isolated_components_are_included_and_selected_deterministically(self) -> None:
        components, owner, rows, cols, nontrivial = _partition()
        result = SimpleNamespace(
            components=components,
            owner=owner,
            local_rows=rows,
            local_cols=cols,
            nontrivial_component_ids=nontrivial,
            hypotheses=(),
        )
        shifts, true_hypotheses, clusters, selected = gate.build_oracle_ceiling(
            result, np.arange(576, dtype=np.int64)
        )
        self.assertEqual(len(shifts), 576)
        self.assertEqual(true_hypotheses, ())
        self.assertEqual(len(clusters), 576)
        self.assertEqual(selected.component_ids, (0,))
        self.assertEqual(selected.minimum_tile, 0)
        self.assertEqual(selected.legal_origin_count, 576)


class SummaryAndDecisionTests(unittest.TestCase):
    def test_summary_and_all_inclusive_decision_pass_at_inclusive_boundaries(self) -> None:
        summary = gate.summarize(_summary_rows())
        summary["mean_eligible_contact_recall"] = 0.90
        summary["worst_eligible_contact_recall"] = 0.80
        summary["mean_exact_connected_coverage"] = 0.30
        summary["worst_exact_connected_coverage"] = 0.20
        summary["mean_selected_cycle_rank_ratio"] = 0.05
        summary["worst_selected_cycle_rank_ratio"] = 0.01
        result = gate.decision(summary)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["status"],
            "go_E23_source_group_disjoint_confirmation_same_generator",
        )
        self.assertIn("no_GPU_training_authority", result["scope"])
        self.assertTrue(all(result["checks"].values()))

    def test_each_frozen_gate_is_required_and_thresholds_are_not_rounded(self) -> None:
        passing = gate.summarize(_summary_rows())
        passing.update(
            {
                "mean_eligible_contact_recall": 0.90,
                "worst_eligible_contact_recall": 0.80,
                "mean_exact_connected_coverage": 0.30,
                "worst_exact_connected_coverage": 0.20,
                "mean_selected_cycle_rank_ratio": 0.05,
                "worst_selected_cycle_rank_ratio": 0.01,
            }
        )
        failures = {
            "completed_scenes": 7,
            "upright_orientation_scenes": 7,
            "emitters_exact_scenes": 7,
            "all_bounds_scenes": 7,
            "true_relation_scenes": 7,
            "legal_origin_scenes": 7,
            "positive_eligible_denominator_scenes": 7,
            "exact_postfilter_survival_scenes": 7,
            "mean_eligible_contact_recall": 0.899999,
            "worst_eligible_contact_recall": 0.799999,
            "mean_exact_connected_coverage": 0.299999,
            "worst_exact_connected_coverage": 0.199999,
            "mean_selected_cycle_rank_ratio": 0.049999,
            "worst_selected_cycle_rank_ratio": 0.009999,
        }
        for key, value in failures.items():
            changed = {**passing, key: value}
            with self.subTest(key=key):
                observed = gate.decision(changed)
                self.assertFalse(observed["passed"])
                self.assertEqual(
                    observed["status"],
                    "kill_existing_affinity_full_union_generator",
                )

    def test_summary_binds_orientation_survival_and_all_claim_algebra_bounds(self) -> None:
        cases = {
            "orientation": ("tile_orientation_degrees", 90),
            "emitters": ("emitter_tiles", 575),
            "survival": ("postfilter_eligible_true_survival", 0.999999),
            "preclaim equality": ("rcce4_preclaims", 1148),
            "retained above pre": ("cross_component_claims", 1156),
            "retained groups of four": ("cross_component_claims", 1151),
            "relations above claims": ("relation_candidates", 1153),
            "hypotheses above relations": ("geometry_valid_hypotheses", 901),
            "negative hypotheses": ("geometry_valid_hypotheses", -1),
            "observation equality": ("finite_directional_logit_observations", 2300),
        }
        for label, (key, value) in cases.items():
            rows = _summary_rows()
            rows[0]["metrics"][key] = value  # type: ignore[index]
            with self.subTest(label=label):
                summary = gate.summarize(rows)
                if key == "tile_orientation_degrees":
                    self.assertEqual(summary["upright_orientation_scenes"], 7)
                elif key == "emitter_tiles":
                    self.assertEqual(summary["emitters_exact_scenes"], 7)
                elif key == "postfilter_eligible_true_survival":
                    self.assertEqual(summary["exact_postfilter_survival_scenes"], 7)
                else:
                    self.assertEqual(summary["all_bounds_scenes"], 7)

    def test_boolean_and_nonfinite_metric_coercions_fail_closed(self) -> None:
        passing = gate.summarize(_summary_rows())
        for key, value in (
            ("completed_scenes", True),
            ("mean_eligible_contact_recall", True),
            ("mean_exact_connected_coverage", float("nan")),
            ("mean_selected_cycle_rank_ratio", float("inf")),
        ):
            with self.subTest(key=key), self.assertRaises(gate.E22ContractError):
                gate.decision({**passing, key: value})

    def test_summary_rejects_duplicate_or_incomplete_image_ids(self) -> None:
        rows = _summary_rows()
        rows[-1]["image"] = 16
        with self.assertRaisesRegex(gate.E22ContractError, "image IDs"):
            gate.summarize(rows)
        with self.assertRaisesRegex(gate.E22ContractError, "exactly eight"):
            gate.summarize(_summary_rows()[:-1])


class RowAndReportReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        E_TEST_ROOT.mkdir(parents=True, exist_ok=True)
        if E_TEST_ROOT.resolve().drive.upper() != "E:":
            raise AssertionError("E22 tests must use E: temporary storage")

    def test_scene_row_is_compact_hashed_and_replays_exactly(self) -> None:
        scene = _scene(10)
        result = _synthetic_result()
        with _partition_replay_patch(result):
            row = gate.evaluate_scene(
                scene,
                result,
                candidate_ids=scene.candidate_ids,
                raw_logits=scene.base_scores,
            )
        self.assertEqual(row["orientation"], "upright_0_degrees_no_rotation_no_reflection")
        self.assertEqual(row["oracle"]["contact_inventory"]["ground_truth_upright_rd_seam_count"], 1104)
        self.assertEqual(row["metrics"]["postfilter_eligible_true_survival"], 1.0)
        self.assertNotIn("affinity_pairs", row["core"])
        self.assertNotIn("claims", row["core"])
        self.assertNotIn("relation_candidates", row["core"])
        self.assertNotIn("hypotheses", row["core"])
        self.assertLess(len(json.dumps(row, sort_keys=True)), 100_000)
        with _partition_replay_patch(result):
            gate._validate_success_row(
                row,
                scene=scene,
                candidate_ids=scene.candidate_ids,
                raw_logits=scene.base_scores,
                expected_result=result,
            )

        tampered = copy.deepcopy(row)
        tampered["oracle"]["selected"]["cycle_rank"] = 99
        with _partition_replay_patch(result), self.assertRaisesRegex(
            gate.E22ContractError, "row replay"
        ):
            gate._validate_success_row(
                tampered,
                scene=scene,
                candidate_ids=scene.candidate_ids,
                raw_logits=scene.base_scores,
                expected_result=result,
            )

    def test_row_replay_calls_core_with_exact_two_original_arrays(self) -> None:
        scene = _scene(10)
        result = _synthetic_result()
        with _partition_replay_patch(result):
            row = gate.evaluate_scene(
                scene,
                result,
                candidate_ids=scene.candidate_ids,
                raw_logits=scene.base_scores,
            )
        with mock.patch.object(
            gate.rcce, "run_rcce4_candidate_oracle", return_value=result
        ) as runner, _partition_replay_patch(result):
            gate._validate_success_row(
                row,
                scene=scene,
                candidate_ids=scene.candidate_ids,
                raw_logits=scene.base_scores,
            )
        runner.assert_called_once()
        self.assertIs(runner.call_args.args[0], scene.candidate_ids)
        self.assertIs(runner.call_args.args[1], scene.base_scores)
        self.assertEqual(runner.call_args.kwargs, {})

    def test_row_replay_rejects_a_core_that_mutates_either_raw_input(self) -> None:
        result = _synthetic_result()
        for mutated in ("candidate_ids", "raw_logits"):
            scene = _scene(10)
            with _partition_replay_patch(result):
                row = gate.evaluate_scene(
                    scene,
                    result,
                    candidate_ids=scene.candidate_ids,
                    raw_logits=scene.base_scores,
                )

            def mutating_core(candidate_ids: np.ndarray, raw_logits: np.ndarray):
                if mutated == "candidate_ids":
                    candidate_ids[0, 7] += 1
                else:
                    raw_logits[0, 0, 0] += np.float32(1.0)
                return result

            with self.subTest(mutated=mutated), mock.patch.object(
                gate.rcce,
                "run_rcce4_candidate_oracle",
                side_effect=mutating_core,
            ), self.assertRaisesRegex(gate.E22ContractError, "mutated"):
                gate._validate_success_row(
                    row,
                    scene=scene,
                    candidate_ids=scene.candidate_ids,
                    raw_logits=scene.base_scores,
                )

    def test_complete_report_replays_every_row_and_rejects_tamper(self) -> None:
        result = _synthetic_result()
        scenes = [_scene(image) for image in e12.CALIBRATION_IDS]
        with _partition_replay_patch(result):
            rows = [
                gate.evaluate_scene(
                    scene,
                    result,
                    candidate_ids=scene.candidate_ids,
                    raw_logits=scene.base_scores,
                )
                for scene in scenes
            ]
        contract = {"synthetic": True}
        contract_digest = e12.canonical_digest(contract)
        summary = gate.summarize(rows)
        report = {
            "schema_version": gate.SCHEMA_VERSION,
            "schema": gate.REPORT_SCHEMA,
            "experiment": gate.EXPERIMENT,
            "status": "complete",
            "stage": gate.decision(summary)["status"],
            "protocol": gate.E22_PROTOCOL,
            "protocol_sha256": e12.canonical_digest(gate.E22_PROTOCOL),
            "run_contract": contract,
            "run_contract_sha256": contract_digest,
            "rows": rows,
            "completed_images": list(e12.CALIBRATION_IDS),
            "summary": summary,
            "decision": gate.decision(summary),
            "runtime_seconds": 1.0,
        }
        with mock.patch.object(
            gate.rcce, "run_rcce4_candidate_oracle", return_value=result
        ) as runner, _partition_replay_patch(result):
            gate._validate_complete_report(
                report,
                contract=contract,
                contract_digest=contract_digest,
                scenes=scenes,
            )
        self.assertEqual(runner.call_count, 8)
        for call, scene in zip(runner.call_args_list, scenes):
            self.assertIs(call.args[0], scene.candidate_ids)
            self.assertIs(call.args[1], scene.base_scores)
            self.assertEqual(call.kwargs, {})

        tampered = copy.deepcopy(report)
        tampered["rows"][0]["metrics"]["eligible_pair_hits"] += 1
        with mock.patch.object(
            gate.rcce, "run_rcce4_candidate_oracle", return_value=result
        ), _partition_replay_patch(result), self.assertRaisesRegex(
            gate.E22ContractError, "row replay"
        ):
            gate._validate_complete_report(
                tampered,
                contract=contract,
                contract_digest=contract_digest,
                scenes=scenes,
            )

    def test_e_only_atomic_write_has_no_partial_file_and_rejects_c(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as temporary:
            root = Path(temporary)
            path = root / "atomic.json"
            gate._atomic_write_json(path, {"ok": True, "finite": 1.25})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"finite": 1.25, "ok": True})
            self.assertEqual(list(root.glob(".*.tmp")), [])
        with self.assertRaisesRegex(gate.E22ContractError, "must stay on E"):
            gate._atomic_write_json(ROOT / "forbidden_e22.json", {"no": True})

    def test_run_gate_restarts_matching_in_progress_and_replays_complete_report(self) -> None:
        result = _synthetic_result()
        scenes = [_scene(image) for image in e12.CALIBRATION_IDS]

        class WeakPool(core.CandidatePoolResult):
            pass

        previous_pool: weakref.ReferenceType[core.CandidatePoolResult] | None = None

        def fresh_pool(_candidate_ids: np.ndarray, _raw_logits: np.ndarray):
            nonlocal previous_pool
            if previous_pool is not None:
                gc.collect()
                self.assertIsNone(
                    previous_pool(),
                    "previous near-cap pool survived into the next core RHS",
                )
            observed = WeakPool(
                result.components,
                result.owner,
                result.local_rows,
                result.local_cols,
                result.nontrivial_component_ids,
                result.affinity_pairs,
                result.claims,
                result.relation_candidates,
                result.hypotheses,
                result.geometry_rejections,
                result.diagnostics,
            )
            previous_pool = weakref.ref(observed)
            return observed

        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as temporary:
            root = Path(temporary)
            paths = gate.E22Paths(
                raw_cache_dir=root / "raw",
                calibration_report=root / "calibration.json",
                e12_report=root / "e12.json",
                e21_report=root / "e21.json",
                report=root / "report.json",
            )
            e21_report = {
                "run_contract_sha256": gate.EXPECTED_E21_RUN_CONTRACT_SHA256,
                "stage": gate.EXPECTED_E21_STAGE,
            }
            e12_report = {"scene_provenance_digest": "d" * 64}
            patches = (
                mock.patch.object(gate, "_verify_e21_kill", return_value=e21_report),
                mock.patch.object(
                    gate,
                    "_load_verified_raw_inputs",
                    return_value=(e12_report, {}, scenes),
                ),
                mock.patch.object(
                    gate, "_source_provenance", return_value={"synthetic.py": "a" * 64}
                ),
                mock.patch.object(
                    gate,
                    "_runtime_provenance",
                    return_value=dict(gate.EXPECTED_RUNTIME_PROVENANCE),
                ),
                mock.patch.object(
                    gate.rcce,
                    "run_rcce4_candidate_oracle",
                    side_effect=fresh_pool,
                ),
                _partition_replay_patch(result),
            )
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4] as runner,
                patches[5],
            ):
                first = gate.run_gate(paths)
                self.assertEqual(first["status"], "complete")
                self.assertEqual(runner.call_count, 8)

                incomplete = json.loads(paths.report.read_text(encoding="utf-8"))
                incomplete["status"] = "in_progress"
                incomplete["stage"] = "upright_rcce4_candidate_availability_ceiling"
                incomplete.pop("summary")
                incomplete.pop("runtime_seconds")
                gate._atomic_write_json(paths.report, incomplete)
                second = gate.run_gate(paths)
                self.assertEqual(second["status"], "complete")
                self.assertEqual(runner.call_count, 16)

                replayed = gate.run_gate(paths)
                self.assertEqual(replayed, second)
                self.assertEqual(runner.call_count, 24)
                gc.collect()
                self.assertIsNone(previous_pool())
                self.assertEqual(list(root.glob(".*.tmp")), [])
                self.assertEqual(list(root.glob("**/.*.tmp")), [])


class StreamingDigestTests(unittest.TestCase):
    def test_near_cap_digest_is_streaming_and_has_bounded_auxiliary_memory(self) -> None:
        produced = 0

        def records():
            nonlocal produced
            for claim_id in range(gate.MAX_RCCE4_PRECLAIMS):
                produced += 1
                yield (claim_id, claim_id % 576, "upright")

        tracemalloc.start()
        try:
            digest = gate._stream_digest(records())
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(produced, 294_912)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertLess(peak, 8 * 1024 * 1024)

    def test_core_payload_never_bulk_jsonifies_large_pool_tuples(self) -> None:
        source = inspect.getsource(gate._core_payload)
        for name in (
            "result.affinity_pairs",
            "result.claims",
            "result.relation_candidates",
            "result.hypotheses",
            "result.geometry_rejections",
        ):
            self.assertNotIn(f"_jsonable({name})", source)
            self.assertIn(f"_stream_digest({name})", source)
        validation_source = inspect.getsource(gate.validate_candidate_pool)
        self.assertNotIn("expected_claims: list", validation_source)
        self.assertNotIn("expected_valid: list", validation_source)
        self.assertNotIn("seen_seams", validation_source)

    def test_digest_is_pythonhashseed_independent(self) -> None:
        code = (
            "import sys;"
            f"sys.path.insert(0,{str(SRC)!r});"
            "import eval_e22_rcce4_candidate_ceiling as g;"
            "print(g._stream_digest([{'z':{'gamma','alpha','beta'},'a':1},(3,2,1)]))"
        )
        outputs = []
        for seed in ("1", "987654"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONPYCACHEPREFIX"] = str(E_TEST_ROOT / "pycache")
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-B", "-c", code],
                    cwd=ROOT,
                    env=environment,
                    text=True,
                ).strip()
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertRegex(outputs[0], r"^[0-9a-f]{64}$")


class LineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        E_TEST_ROOT.mkdir(parents=True, exist_ok=True)

    def test_exact_e21_kill_bytes_authenticate_and_tamper_fails_sha_first(self) -> None:
        report = gate._verify_e21_kill(gate.DEFAULT_E21_REPORT)
        self.assertEqual(report["stage"], gate.EXPECTED_E21_STAGE)
        self.assertFalse(report["decision"]["passed"])
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as temporary:
            tampered = Path(temporary) / "e21_tampered.json"
            tampered.write_bytes(gate.DEFAULT_E21_REPORT.read_bytes() + b"\n")
            with self.assertRaisesRegex(gate.E22ContractError, "SHA256 mismatch"):
                gate._verify_e21_kill(tampered)

    def test_source_provenance_is_sha256_complete_and_contains_current_core(self) -> None:
        provenance = gate._source_provenance()
        self.assertIn("e22_rcce4_candidate_oracle.py", provenance)
        self.assertIn("eval_e22_rcce4_candidate_ceiling.py", provenance)
        for name, digest in provenance.items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(
                digest,
                hashlib.sha256((SRC / name).read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
