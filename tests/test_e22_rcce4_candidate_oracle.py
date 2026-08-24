from __future__ import annotations

import ast
import inspect
import sys
import unittest
import weakref
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import e22_rcce4_candidate_oracle as e22  # noqa: E402


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    candidates = np.zeros((576, 128), dtype=np.int64)
    logits = np.full((4, 576, 128), -np.inf, dtype=np.float32)
    # Every row is nonempty; each even/odd pair has both directed memberships.
    for source in range(576):
        target = source + 1 if source % 2 == 0 else source - 1
        candidates[source, 0] = target
        logits[:, source, 0] = np.asarray(
            (source + 0.1, source + 0.2, source + 0.3, source + 0.4),
            dtype=np.float32,
        )
    return candidates, logits


def _add_membership(
    candidates: np.ndarray,
    logits: np.ndarray,
    *,
    source: int,
    target: int,
    slot: int,
    values: tuple[float, float, float, float],
) -> None:
    candidates[source, slot] = target
    logits[:, source, slot] = np.asarray(values, dtype=np.float32)


def _partition(
    groups: tuple[dict[int, tuple[int, int]], ...] = (),
) -> tuple[
    tuple[e22.RigidComponent, ...],
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
        e22.RigidComponent(component_id=index, entries=value)
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


def _claim(
    claim_id: int,
    first: int,
    second: int,
    dy: int,
    dx: int,
    first_component: int,
    second_component: int,
) -> e22.RCCE4Claim:
    return e22.RCCE4Claim(
        claim_id=claim_id,
        pair_id=claim_id,
        first=first,
        second=second,
        dy=dy,
        dx=dx,
        first_component=first_component,
        second_component=second_component,
        forward_observation=None,
        reverse_observation=None,
    )


class FrozenContractTests(unittest.TestCase):
    def test_constants_signature_fields_and_theoretical_bounds_are_literal(self) -> None:
        self.assertEqual(e22.GRID, 24)
        self.assertEqual(e22.NUM_TILES, 576)
        self.assertEqual(e22.CANDIDATE_WIDTH, 128)
        self.assertEqual(e22.NUM_DIRECTIONS, 4)
        self.assertEqual((e22.UP, e22.DOWN, e22.LEFT, e22.RIGHT), (0, 1, 2, 3))
        self.assertEqual(e22.DIRECTION_NAMES, ("U", "D", "L", "R"))
        self.assertEqual(e22.COMPONENT_MAX_EDGES, 96)
        self.assertEqual(e22.MIN_MARGIN, 0.0)
        self.assertEqual(e22.MAX_DIRECTED_MEMBERSHIPS, 73_728)
        self.assertEqual(e22.MAX_UNORDERED_AFFINITY_PAIRS, 73_728)
        self.assertEqual(e22.MAX_LOGIT_OBSERVATIONS, 294_912)
        self.assertEqual(e22.MAX_RCCE4_CLAIMS, 294_912)
        self.assertEqual(e22.MAX_GEOMETRY_HYPOTHESES, 294_912)
        self.assertNotIn(200_000, vars(e22).values())
        self.assertEqual(
            tuple(inspect.signature(e22.run_rcce4_candidate_oracle).parameters),
            ("candidate_ids", "raw_logits"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(e22.CandidatePoolResult)),
            (
                "components",
                "owner",
                "local_rows",
                "local_cols",
                "nontrivial_component_ids",
                "affinity_pairs",
                "claims",
                "relation_candidates",
                "hypotheses",
                "geometry_rejections",
                "diagnostics",
            ),
        )
        self.assertEqual(
            e22.RCCE4_CLAIM_ORDER,
            (
                ("a", "b", 0, 1),
                ("b", "a", 0, 1),
                ("a", "b", 1, 0),
                ("b", "a", 1, 0),
            ),
        )

    def test_no_pixel_board_rotation_reflection_or_gpu_route(self) -> None:
        source = Path(e22.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
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
            }.isdisjoint(calls)
        )
        forbidden_inputs = {
            "tiles",
            "pixels",
            "permutation",
            "truth",
            "target",
            "board",
            "labels",
            "rotation",
        }
        self.assertTrue(
            forbidden_inputs.isdisjoint(
                inspect.signature(e22.run_rcce4_candidate_oracle).parameters
            )
        )

    def test_strict_input_contract_fails_closed(self) -> None:
        candidates, logits = _inputs()
        cases: list[tuple[str, np.ndarray, np.ndarray, str]] = []
        cases.append(("candidate dtype", candidates.astype(np.int32), logits, "int64"))
        cases.append(("logit dtype", candidates, logits.astype(np.float64), "float32"))
        cases.append(("candidate shape", candidates[:, :-1], logits, "576,128"))
        cases.append(("logit shape", candidates, logits[:, :, :-1], "4,576,128"))
        cases.append(("candidate contiguous", candidates[:, ::-1], logits, "contiguous"))
        cases.append(("logit contiguous", candidates, logits[:, :, ::-1], "contiguous"))
        for label, bad_candidates, bad_logits, message in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                e22.RCCE4OracleError, message
            ):
                e22.run_rcce4_candidate_oracle(bad_candidates, bad_logits)

        bad = logits.copy()
        bad[e22.UP, 0, 0] = -np.inf
        with self.assertRaisesRegex(e22.RCCE4OracleError, "mask"):
            e22._validate_inputs(candidates, bad)
        bad = logits.copy()
        bad[:, 0, 0] = -np.inf
        with self.assertRaisesRegex(e22.RCCE4OracleError, "every affinity row"):
            e22._validate_inputs(candidates, bad)
        bad = logits.copy()
        bad[0, 0, 5] = np.nan
        with self.assertRaisesRegex(e22.RCCE4OracleError, "finite values"):
            e22._validate_inputs(candidates, bad)
        bad_candidates = candidates.copy()
        bad_candidates[0, 0] = 0
        with self.assertRaisesRegex(e22.RCCE4OracleError, "self"):
            e22._validate_inputs(bad_candidates, logits)
        bad_candidates = candidates.copy()
        bad_logits = logits.copy()
        bad_candidates[0, 1] = bad_candidates[0, 0]
        bad_logits[:, 0, 1] = 0.0
        with self.assertRaisesRegex(e22.RCCE4OracleError, "duplicate"):
            e22._validate_inputs(bad_candidates, bad_logits)
        bad_candidates = candidates.copy()
        bad_candidates[0, 0] = 576
        with self.assertRaisesRegex(e22.RCCE4OracleError, "outside"):
            e22._validate_inputs(bad_candidates, logits)

    def test_invalid_padding_ids_are_arbitrary_sanitized_and_nonmutating(self) -> None:
        candidates, logits = _inputs()
        candidates[0, 10] = -1
        candidates[1, 11] = 999
        original_candidates = candidates.copy()
        original_logits = logits.copy()
        values = e22._validate_inputs(candidates, logits)
        self.assertEqual(int(values.candidate_ids[0, 10]), 0)
        self.assertEqual(int(values.candidate_ids[1, 11]), 0)
        self.assertTrue(np.array_equal(candidates, original_candidates))
        self.assertTrue(np.array_equal(logits, original_logits))
        right, down = e22.derive_dense_scores(candidates, logits)
        self.assertEqual(right.shape, (576, 576))
        self.assertEqual(down.shape, (576, 576))
        self.assertTrue(np.array_equal(candidates, original_candidates))
        self.assertTrue(np.array_equal(logits, original_logits))

    def test_exact_cpu_torch_dense_path_matches_frozen_dense_rd(self) -> None:
        candidates, logits = _inputs()
        right, down = e22.derive_dense_scores(candidates, logits)
        with torch.inference_mode():
            expected_right, expected_down = e22.dense_rd(
                torch.from_numpy(candidates).long(), torch.from_numpy(logits).float()
            )
        self.assertTrue(np.array_equal(right, expected_right.numpy()))
        self.assertTrue(np.array_equal(down, expected_down.numpy()))
        self.assertEqual(right.dtype, np.dtype(np.float32))
        self.assertTrue(right.flags.c_contiguous)
        self.assertTrue(candidates.flags.writeable)
        self.assertTrue(logits.flags.writeable)

    def test_full_run_uses_fail_closed_dense_output_validation(self) -> None:
        candidates, logits = _inputs()
        invalid = torch.zeros((575, 576), dtype=torch.float32)
        with mock.patch.object(e22, "dense_rd", return_value=(invalid, invalid)):
            with mock.patch.object(e22.e21, "build_components") as builder:
                with self.assertRaisesRegex(e22.RCCE4OracleError, "derived dense"):
                    e22.run_rcce4_candidate_oracle(candidates, logits)
        builder.assert_not_called()


class PairAndClaimTests(unittest.TestCase):
    def test_unordered_or_pair_is_unique_sorted_and_records_membership(self) -> None:
        candidates, logits = _inputs()
        _add_membership(
            candidates,
            logits,
            source=0,
            target=2,
            slot=1,
            values=(10.0, 20.0, 30.0, 40.0),
        )
        values = e22._validate_inputs(candidates, logits)
        pairs = e22._build_affinity_pairs(values)
        pair = next(value for value in pairs if value.identity == (0, 2))
        self.assertEqual(pair.a_to_b_slot, 1)
        self.assertIsNone(pair.b_to_a_slot)
        self.assertEqual(pair.membership_count, 1)
        self.assertFalse(pair.reciprocal)
        self.assertEqual(
            tuple(value.identity for value in pairs),
            tuple(sorted(value.identity for value in pairs)),
        )
        self.assertEqual(tuple(value.pair_id for value in pairs), tuple(range(len(pairs))))
        self.assertEqual(
            sum(value.membership_count for value in pairs), values.directed_memberships
        )

    def test_exact_four_upright_claims_and_literal_logit_metadata(self) -> None:
        candidates, logits = _inputs()
        _add_membership(
            candidates,
            logits,
            source=0,
            target=2,
            slot=1,
            values=(10.0, 20.0, 30.0, 40.0),
        )
        _add_membership(
            candidates,
            logits,
            source=2,
            target=0,
            slot=1,
            values=(11.0, 21.0, 31.0, 41.0),
        )
        values = e22._validate_inputs(candidates, logits)
        pairs = e22._build_affinity_pairs(values)
        pair = next(value for value in pairs if value.identity == (0, 2))
        _components, owner, _rows, _cols, _nontrivial = _partition()
        claims, _same, _cross = e22._build_claims(values, pairs, owner)
        selected = [claim for claim in claims if claim.pair_id == pair.pair_id]
        self.assertEqual(
            tuple(claim.physical_seam for claim in selected),
            ((0, 2, 0, 1), (2, 0, 0, 1), (0, 2, 1, 0), (2, 0, 1, 0)),
        )
        self.assertEqual(tuple(claim.adjacency[2] for claim in selected), ("R", "R", "D", "D"))
        self.assertEqual(
            tuple(
                (
                    claim.forward_observation.direction,
                    claim.forward_observation.logit,
                    claim.reverse_observation.direction,
                    claim.reverse_observation.logit,
                )
                for claim in selected
            ),
            (
                (e22.RIGHT, 40.0, e22.LEFT, 31.0),
                (e22.RIGHT, 41.0, e22.LEFT, 30.0),
                (e22.DOWN, 20.0, e22.UP, 11.0),
                (e22.DOWN, 21.0, e22.UP, 10.0),
            ),
        )

    def test_missing_reverse_membership_is_explicit_and_scores_do_not_admit(self) -> None:
        candidates, logits = _inputs()
        _add_membership(
            candidates,
            logits,
            source=0,
            target=2,
            slot=1,
            values=(-1000.0, -999.0, -998.0, -997.0),
        )
        values = e22._validate_inputs(candidates, logits)
        pairs = e22._build_affinity_pairs(values)
        pair = next(value for value in pairs if value.identity == (0, 2))
        _components, owner, _rows, _cols, _nontrivial = _partition()
        claims, _same, _cross = e22._build_claims(values, pairs, owner)
        selected = [claim for claim in claims if claim.pair_id == pair.pair_id]
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            tuple(
                (
                    claim.forward_observation is None,
                    claim.reverse_observation is None,
                )
                for claim in selected
            ),
            ((False, True), (True, False), (False, True), (True, False)),
        )
        self.assertEqual(sum(len(claim.observations) for claim in selected), 4)

    def test_same_component_pair_removes_all_four_claims_together(self) -> None:
        candidates, logits = _inputs()
        _add_membership(
            candidates,
            logits,
            source=0,
            target=2,
            slot=1,
            values=(1.0, 2.0, 3.0, 4.0),
        )
        values = e22._validate_inputs(candidates, logits)
        pairs = e22._build_affinity_pairs(values)
        pair = next(value for value in pairs if value.identity == (0, 2))
        _components, owner, _rows, _cols, _nontrivial = _partition(
            ({0: (0, 0), 2: (0, 1)},)
        )
        claims, same, cross = e22._build_claims(values, pairs, owner)
        self.assertFalse(any(claim.pair_id == pair.pair_id for claim in claims))
        self.assertGreaterEqual(same, 1)
        self.assertEqual(len(claims), 4 * cross)


class RelationAndGeometryTests(unittest.TestCase):
    def test_signed_relations_are_canonical_and_offsets_are_not_collapsed(self) -> None:
        rows = np.zeros(576, dtype=np.int64)
        cols = np.zeros(576, dtype=np.int64)
        cols[1] = 1
        claims = (
            _claim(0, 0, 2, 0, 1, 0, 1),
            _claim(1, 1, 2, 0, 1, 0, 1),
            _claim(2, 2, 0, 1, 0, 1, 0),
        )
        relations = e22._build_relation_candidates(claims, rows, cols)
        self.assertEqual(
            tuple(value.relation for value in relations),
            ((0, 1, -1, 0), (0, 1, 0, 1), (0, 1, 0, 2)),
        )
        self.assertEqual(
            tuple(value.relation_id for value in relations), tuple(range(3))
        )

    def test_geometry_rejects_adjacency_collision_and_span_separately(self) -> None:
        owner = np.zeros(576, dtype=np.int64)
        rows = np.zeros(576, dtype=np.int64)
        cols = np.zeros(576, dtype=np.int64)
        owner[2] = 1

        components = (
            e22.RigidComponent(0, ((0, 0, 0),)),
            e22.RigidComponent(1, ((2, 0, 0),)),
        )
        claim = _claim(0, 0, 2, 0, 1, 0, 1)
        wrong = e22.RelationCandidate(0, 0, 1, 0, 2, (0,))
        hypotheses, rejected = e22.filter_relation_geometry(
            (wrong,), (claim,), components, owner, rows, cols
        )
        self.assertEqual(hypotheses, ())
        self.assertEqual(rejected, (e22.GeometryRejection(0, "adjacency"),))

        components = (
            e22.RigidComponent(0, ((0, 0, 0), (1, 0, 1))),
            e22.RigidComponent(1, ((2, 0, 0),)),
        )
        owner[1] = 0
        collision = e22.RelationCandidate(0, 0, 1, 0, 1, (0,))
        hypotheses, rejected = e22.filter_relation_geometry(
            (collision,), (claim,), components, owner, rows, cols
        )
        self.assertEqual(hypotheses, ())
        self.assertEqual(rejected, (e22.GeometryRejection(0, "collision"),))

        components = (
            e22.RigidComponent(0, ((0, 0, 0), (1, 0, 23))),
            e22.RigidComponent(1, ((2, 0, 0),)),
        )
        cols[1] = 23
        span_claim = _claim(0, 1, 2, 0, 1, 0, 1)
        span = e22.RelationCandidate(0, 0, 1, 0, 24, (0,))
        hypotheses, rejected = e22.filter_relation_geometry(
            (span,), (span_claim,), components, owner, rows, cols
        )
        self.assertEqual(hypotheses, ())
        self.assertEqual(rejected, (e22.GeometryRejection(0, "span"),))

    def test_incidental_component_contacts_are_allowed_without_added_evidence(self) -> None:
        components = (
            e22.RigidComponent(0, ((0, 0, 0), (1, 1, 0))),
            e22.RigidComponent(1, ((2, 0, 0), (3, 1, 0))),
        )
        owner = np.zeros(576, dtype=np.int64)
        owner[[2, 3]] = 1
        rows = np.zeros(576, dtype=np.int64)
        rows[[1, 3]] = 1
        cols = np.zeros(576, dtype=np.int64)
        claim = _claim(0, 0, 2, 0, 1, 0, 1)
        relation = e22.RelationCandidate(0, 0, 1, 0, 1, (0,))
        hypotheses, rejected = e22.filter_relation_geometry(
            (relation,), (claim,), components, owner, rows, cols
        )
        self.assertEqual(rejected, ())
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0].claim_ids, (0,))
        self.assertEqual(hypotheses[0].relation, (0, 1, 0, 1))


class FullCoreTests(unittest.TestCase):
    def test_full_core_is_deterministic_bounded_and_diagnostic_exact(self) -> None:
        candidates, logits = _inputs()
        original_candidates = candidates.copy()
        original_logits = logits.copy()
        partition = _partition()
        with mock.patch.object(
            e22.e21, "build_components", return_value=partition
        ) as builder:
            first = e22.run_rcce4_candidate_oracle(candidates, logits)
            second = e22.run_rcce4_candidate_oracle(candidates, logits)
        self.assertEqual(builder.call_count, 2)
        for call in builder.call_args_list:
            right, down = call.args
            self.assertEqual(right.shape, (576, 576))
            self.assertEqual(right.dtype, np.dtype(np.float32))
            self.assertTrue(np.isfinite(right).all())
            self.assertTrue(np.isfinite(down).all())
        self.assertEqual(first.affinity_pairs, second.affinity_pairs)
        self.assertEqual(first.claims, second.claims)
        self.assertEqual(first.relation_candidates, second.relation_candidates)
        self.assertEqual(first.hypotheses, second.hypotheses)
        self.assertEqual(first.geometry_rejections, ())
        self.assertEqual(first.diagnostics, second.diagnostics)
        self.assertTrue(candidates.flags.writeable)
        self.assertTrue(logits.flags.writeable)
        self.assertTrue(np.array_equal(candidates, original_candidates))
        self.assertTrue(np.array_equal(logits, original_logits))
        for value in (first.owner, first.local_rows, first.local_cols):
            self.assertFalse(value.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            first.affinity_pairs[0].pair_id = 99  # type: ignore[misc]

        observation = first.claims[0].forward_observation
        self.assertIsNotNone(observation)
        rejection = e22.GeometryRejection(0, "collision")
        slotted_values = (
            first.affinity_pairs[0],
            observation,
            first.claims[0],
            first.relation_candidates[0],
            first.hypotheses[0],
            rejection,
            first.diagnostics,
            first,
        )
        for value in slotted_values:
            with self.subTest(slotted_type=type(value).__name__):
                self.assertTrue(hasattr(type(value), "__slots__"))
                self.assertFalse(hasattr(value, "__dict__"))
        for value in slotted_values[:-1]:
            self.assertEqual(replace(value), value)
        replaced_result = replace(first)
        for field in fields(e22.CandidatePoolResult):
            self.assertIs(getattr(replaced_result, field.name), getattr(first, field.name))
        self.assertIs(weakref.ref(first)(), first)
        for value in slotted_values[:-1]:
            self.assertEqual(asdict(value), asdict(replace(value)))
        result_payload = asdict(first)
        self.assertEqual(
            tuple(result_payload),
            tuple(field.name for field in fields(e22.CandidatePoolResult)),
        )
        self.assertTrue(np.array_equal(result_payload["owner"], first.owner))

        self.assertEqual(len(first.affinity_pairs), 288)
        self.assertEqual(len(first.claims), 1_152)
        self.assertEqual(len(first.relation_candidates), 1_152)
        self.assertEqual(len(first.hypotheses), 1_152)
        diagnostics = first.diagnostics
        self.assertEqual(diagnostics.component_count, 576)
        self.assertEqual(diagnostics.nontrivial_components, 0)
        self.assertEqual(diagnostics.singleton_components, 576)
        self.assertEqual(diagnostics.emitter_tiles, 576)
        self.assertEqual(diagnostics.directed_valid_memberships, 576)
        self.assertEqual(diagnostics.input_logit_observations, 2_304)
        self.assertEqual(diagnostics.unordered_affinity_pairs, 288)
        self.assertEqual(diagnostics.one_way_affinity_pairs, 0)
        self.assertEqual(diagnostics.reciprocal_affinity_pairs, 288)
        self.assertEqual(diagnostics.pre_component_filter_claims, 1_152)
        self.assertEqual(diagnostics.same_component_pairs, 0)
        self.assertEqual(diagnostics.cross_component_pairs, 288)
        self.assertEqual(diagnostics.claims, 1_152)
        self.assertEqual(diagnostics.claim_logit_observations, 2_304)
        self.assertEqual(diagnostics.geometry_valid_hypotheses, 1_152)
        self.assertEqual(diagnostics.geometry_rejected_relations, 0)
        self.assertEqual(diagnostics.component_pairs, 288)
        self.assertEqual(diagnostics.component_pairs_with_alternative_offsets, 288)


if __name__ == "__main__":
    unittest.main()
