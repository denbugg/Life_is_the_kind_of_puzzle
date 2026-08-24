from __future__ import annotations

import ast
import inspect
import sys
import unittest
import weakref
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import e22_rcce4_candidate_oracle as e22  # noqa: E402
import e23_i21_residual_candidate_oracle as e23  # noqa: E402


def _base_inputs() -> tuple[np.ndarray, np.ndarray]:
    candidates = np.zeros((576, 128), dtype=np.int64)
    logits = np.full((4, 576, 128), -np.inf, dtype=np.float32)
    for source in range(576):
        target = source + 1 if source % 2 == 0 else source - 1
        candidates[source, 0] = target
        logits[:, source, 0] = np.asarray(
            (source + 0.1, source + 0.2, source + 0.3, source + 0.4),
            dtype=np.float32,
        )
    return candidates, logits


def _spatial(fill: float = 0.0) -> np.ndarray:
    return np.full((4, 576, 576), fill, dtype=np.float32)


def _base_pair(
    pair_id: int, a: int, b: int, *, reciprocal: bool = False
) -> e22.AffinityPair:
    return e22.AffinityPair(
        pair_id=pair_id,
        a=a,
        b=b,
        a_to_b_slot=0,
        b_to_a_slot=0 if reciprocal else None,
    )


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
    def test_constants_signature_fields_and_caps_are_literal(self) -> None:
        self.assertEqual(e23.GRID, 24)
        self.assertEqual(e23.NUM_TILES, 576)
        self.assertEqual(e23.CANDIDATE_WIDTH, 128)
        self.assertEqual(e23.NUM_DIRECTIONS, 4)
        self.assertEqual((e23.UP, e23.DOWN, e23.LEFT, e23.RIGHT), (0, 1, 2, 3))
        self.assertEqual(e23.SPATIAL_K, 64)
        self.assertEqual(e23.SPATIAL_LOGIT_VALUES, 1_327_104)
        self.assertEqual(e23.SPATIAL_SELECTIONS, 147_456)
        self.assertEqual(e23.MAX_DIRECTED_MEMBERSHIPS, 73_728)
        self.assertEqual(e23.MAX_BASE_AFFINITY_PAIRS, 73_728)
        self.assertEqual(e23.MAX_ALL_UNORDERED_PAIRS, 165_600)
        self.assertEqual(e23.MAX_NEW_RCCE4_CLAIMS, 589_824)
        self.assertEqual(e23.MAX_COMBINED_RCCE4_CLAIMS, 662_400)
        self.assertEqual(e23.MAX_CROSS_COMPONENT_CLAIMS, 662_400)
        self.assertEqual(e23.MAX_RELATION_CANDIDATES, 662_400)
        self.assertEqual(e23.MAX_GEOMETRY_HYPOTHESES, 662_400)
        self.assertEqual(
            tuple(inspect.signature(e23.run_i21_residual_candidate_oracle).parameters),
            ("candidate_ids", "raw_logits", "spatial_logits"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(e23.CandidatePoolResult)),
            (
                "components",
                "owner",
                "local_rows",
                "local_cols",
                "nontrivial_component_ids",
                "affinity_pairs",
                "base_affinity_pairs",
                "spatial_selected_ids",
                "spatial_pairs",
                "claims",
                "relation_candidates",
                "hypotheses",
                "rejections",
                "diagnostics",
            ),
        )
        self.assertEqual(
            e23.RCCE4_CLAIM_ORDER,
            (
                ("a", "b", 0, 1),
                ("b", "a", 0, 1),
                ("a", "b", 1, 0),
                ("b", "a", 1, 0),
            ),
        )

    def test_no_labels_pixels_model_gpu_rotation_or_reflection_route(self) -> None:
        source = Path(e23.__file__).read_text(encoding="utf-8")
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
                "encode_tiles",
                "directional_edge_scores",
                "solve_dense",
            }.isdisjoint(calls)
        )
        forbidden_inputs = {
            "tiles",
            "pixels",
            "permutation",
            "truth",
            "target",
            "labels",
            "board",
            "rotation",
        }
        self.assertTrue(
            forbidden_inputs.isdisjoint(
                inspect.signature(e23.run_i21_residual_candidate_oracle).parameters
            )
        )


class InputValidationTests(unittest.TestCase):
    def test_spatial_contract_is_exact_finite_float32_contiguous(self) -> None:
        spatial = _spatial()
        cases: list[tuple[str, object, str]] = [
            ("type", [[0.0]], "numpy"),
            ("shape", spatial[:, :, :-1], "4,576,576"),
            ("dtype", spatial.astype(np.float64), "float32"),
            ("contiguous", spatial[:, :, ::-1], "contiguous"),
        ]
        for label, value, message in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                e23.I21ResidualOracleError, message
            ):
                e23._validate_spatial_logits(value)  # type: ignore[arg-type]
        for bad_value in (np.nan, np.inf, -np.inf):
            bad = spatial.copy()
            bad[0, 0, 0] = bad_value
            with self.subTest(bad_value=bad_value), self.assertRaisesRegex(
                e23.I21ResidualOracleError, "finite"
            ):
                e23._validate_spatial_logits(bad)

    def test_candidate_and_raw_contract_remains_exact_e22_contract(self) -> None:
        candidates, logits = _base_inputs()
        spatial = _spatial()
        with self.assertRaisesRegex(e23.I21ResidualOracleError, "E22 base replay"):
            e23.run_i21_residual_candidate_oracle(
                candidates.astype(np.int32), logits, spatial
            )
        with self.assertRaisesRegex(e23.I21ResidualOracleError, "E22 base replay"):
            e23.run_i21_residual_candidate_oracle(
                candidates, logits.astype(np.float64), spatial
            )


class ResidualSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_pairs = (
            _base_pair(0, 0, 1),
            _base_pair(1, 0, 2),
            _base_pair(2, 10, 20),
        )
        spatial = _spatial()
        spatial[e23.RIGHT, 5, 500] = 10.0
        spatial[e23.RIGHT, 5, 501] = 10.0
        cls.selected, cls.counts = e23._select_spatial_residuals(
            spatial, cls.base_pairs
        )

    def test_exact_k64_ties_score_order_and_exclusions(self) -> None:
        selected = self.selected
        self.assertEqual(selected.shape, (4, 576, 64))
        self.assertEqual(selected.dtype, np.dtype(np.int64))
        self.assertTrue(selected.flags.c_contiguous)
        self.assertFalse(selected.flags.writeable)
        self.assertEqual(selected.size, 147_456)
        self.assertTrue(np.array_equal(selected[e23.UP, 0], np.arange(3, 67)))
        self.assertEqual(tuple(selected[e23.RIGHT, 5, :2]), (500, 501))
        for direction in range(4):
            for source in range(576):
                row = selected[direction, source]
                self.assertEqual(np.unique(row).size, 64)
                self.assertNotIn(source, row)
        for direction in range(4):
            self.assertNotIn(1, selected[direction, 0])
            self.assertNotIn(2, selected[direction, 0])
            self.assertNotIn(0, selected[direction, 1])
            self.assertNotIn(20, selected[direction, 10])
            self.assertNotIn(10, selected[direction, 20])
        self.assertEqual(sum(self.counts.values()), 147_456)

    def test_pair_or_is_disjoint_lexicographic_and_counts_all_nominations(self) -> None:
        pairs = e23._build_spatial_pairs(self.counts, self.base_pairs)
        self.assertTrue(pairs)
        self.assertEqual(
            tuple(pair.pair_id for pair in pairs),
            tuple(range(len(self.base_pairs), len(self.base_pairs) + len(pairs))),
        )
        identities = tuple(pair.identity for pair in pairs)
        self.assertEqual(identities, tuple(sorted(identities)))
        self.assertFalse(set(identities).intersection(pair.identity for pair in self.base_pairs))
        self.assertEqual(sum(pair.nomination_count for pair in pairs), 147_456)
        self.assertTrue(all(1 <= pair.nomination_count <= 8 for pair in pairs))

    def test_selection_is_deterministic_for_identical_float32_logits(self) -> None:
        spatial = _spatial()
        spatial[e23.RIGHT, 5, 500] = 10.0
        spatial[e23.RIGHT, 5, 501] = 10.0
        selected, counts = e23._select_spatial_residuals(spatial, self.base_pairs)
        self.assertTrue(np.array_equal(selected, self.selected))
        self.assertEqual(counts, self.counts)

    def test_less_than_64_residual_targets_fails_instead_of_truncating(self) -> None:
        # Anchor zero has only 63 non-self, non-base targets left.
        base_pairs = tuple(
            _base_pair(index, 0, target)
            for index, target in enumerate(range(1, 513))
        )
        with self.assertRaisesRegex(e23.I21ResidualOracleError, "fewer than 64"):
            e23._select_spatial_residuals(_spatial(), base_pairs)

    def test_pair_cap_fails_instead_of_clipping(self) -> None:
        with mock.patch.object(
            e23,
            "MAX_ALL_UNORDERED_PAIRS",
            len(self.base_pairs) + len(self.counts) - 1,
        ):
            with self.assertRaisesRegex(e23.I21ResidualOracleError, "bound"):
                e23._build_spatial_pairs(self.counts, self.base_pairs)


class RCCE4AndGeometryTests(unittest.TestCase):
    def test_new_pair_gets_exact_four_upright_claims_without_side_metadata(self) -> None:
        owner = np.arange(576, dtype=np.int64)
        pair = e23.SpatialPair(10, 2, 7, 3)
        claims, same, cross = e23._build_spatial_claims(
            (pair,), owner, first_claim_id=20
        )
        self.assertEqual((same, cross), (0, 1))
        self.assertEqual(tuple(claim.claim_id for claim in claims), (20, 21, 22, 23))
        self.assertEqual(
            tuple(claim.physical_seam for claim in claims),
            ((2, 7, 0, 1), (7, 2, 0, 1), (2, 7, 1, 0), (7, 2, 1, 0)),
        )
        self.assertEqual(tuple(claim.pair_id for claim in claims), (10, 10, 10, 10))
        self.assertTrue(
            all(
                claim.forward_observation is None
                and claim.reverse_observation is None
                and claim.observations == ()
                and (claim.dy, claim.dx) in ((0, 1), (1, 0))
                for claim in claims
            )
        )

    def test_same_component_removes_all_four_and_bound_fails_not_truncates(self) -> None:
        owner = np.arange(576, dtype=np.int64)
        owner[7] = owner[2]
        pair = e23.SpatialPair(10, 2, 7, 1)
        claims, same, cross = e23._build_spatial_claims(
            (pair,), owner, first_claim_id=0
        )
        self.assertEqual(claims, ())
        self.assertEqual((same, cross), (1, 0))
        owner[7] = 7
        with mock.patch.object(e23, "MAX_NEW_RCCE4_CLAIMS", 0):
            with self.assertRaisesRegex(e23.I21ResidualOracleError, "bound"):
                e23._build_spatial_claims((pair,), owner, first_claim_id=0)

    def test_unchanged_geometry_rejects_adjacency_collision_and_span(self) -> None:
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
        hypotheses, rejected = e23.filter_relation_geometry(
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
        hypotheses, rejected = e23.filter_relation_geometry(
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
        hypotheses, rejected = e23.filter_relation_geometry(
            (span,), (span_claim,), components, owner, rows, cols
        )
        self.assertEqual(hypotheses, ())
        self.assertEqual(rejected, (e22.GeometryRejection(0, "span"),))

    def test_geometry_cap_fails_instead_of_truncating(self) -> None:
        relation = e22.RelationCandidate(0, 0, 1, 0, 1, (0,))
        claim = _claim(0, 0, 1, 0, 1, 0, 1)
        components = (
            e22.RigidComponent(0, ((0, 0, 0),)),
            e22.RigidComponent(1, ((1, 0, 0),)),
        )
        owner = np.arange(576, dtype=np.int64)
        coordinates = np.zeros(576, dtype=np.int64)
        with mock.patch.object(e23, "MAX_RELATION_CANDIDATES", 0):
            with self.assertRaisesRegex(e23.I21ResidualOracleError, "bound"):
                e23.filter_relation_geometry(
                    (relation,), (claim,), components, owner, coordinates, coordinates
                )


class FullCoreSmokeTests(unittest.TestCase):
    def test_full_core_preserves_exact_e22_prefix_and_is_bounded_immutable(self) -> None:
        candidates, raw_logits = _base_inputs()
        original_candidates = candidates.copy()
        original_raw = raw_logits.copy()
        spatial = _spatial()
        original_spatial = spatial.copy()
        partition = _partition()
        with mock.patch.object(e22.e21, "build_components", return_value=partition):
            base = e22.run_rcce4_candidate_oracle(candidates, raw_logits)
        with mock.patch.object(
            e23.e22, "run_rcce4_candidate_oracle", return_value=base
        ) as replay:
            result = e23.run_i21_residual_candidate_oracle(
                candidates, raw_logits, spatial
            )
        replay.assert_called_once_with(candidates, raw_logits)

        self.assertEqual(
            result.affinity_pairs[: len(base.affinity_pairs)], base.affinity_pairs
        )
        self.assertEqual(result.base_affinity_pairs, base.affinity_pairs)
        self.assertEqual(result.claims[: len(base.claims)], base.claims)
        self.assertEqual(
            tuple(pair.identity for pair in result.spatial_pairs),
            tuple(sorted(pair.identity for pair in result.spatial_pairs)),
        )
        self.assertFalse(
            {pair.identity for pair in result.base_affinity_pairs}.intersection(
                pair.identity for pair in result.spatial_pairs
            )
        )
        self.assertEqual(
            tuple(pair.pair_id for pair in result.affinity_pairs),
            tuple(range(len(result.affinity_pairs))),
        )
        self.assertEqual(
            tuple(claim.claim_id for claim in result.claims),
            tuple(range(len(result.claims))),
        )
        self.assertTrue(
            all(
                claim.forward_observation is None
                and claim.reverse_observation is None
                for claim in result.claims[len(base.claims) :]
            )
        )
        self.assertEqual(result.geometry_rejections, result.rejections)
        self.assertEqual(result.rejections, ())
        self.assertEqual(len(result.hypotheses), len(result.relation_candidates))
        self.assertIs(result.components, base.components)
        self.assertIs(result.owner, base.owner)
        self.assertIs(result.local_rows, base.local_rows)
        self.assertIs(result.local_cols, base.local_cols)
        for value in (
            result.owner,
            result.local_rows,
            result.local_cols,
            result.spatial_selected_ids,
        ):
            self.assertFalse(value.flags.writeable)
        self.assertTrue(np.array_equal(candidates, original_candidates))
        self.assertTrue(np.array_equal(raw_logits, original_raw))
        self.assertTrue(np.array_equal(spatial, original_spatial))

        diagnostics = result.diagnostics
        self.assertEqual(diagnostics.emitter_tiles, 576)
        self.assertEqual(diagnostics.spatial_logit_values, 1_327_104)
        self.assertEqual(diagnostics.spatial_selections, 147_456)
        self.assertEqual(diagnostics.spatial_pair_nominations, 147_456)
        self.assertEqual(
            diagnostics.unordered_affinity_pairs,
            diagnostics.base_affinity_pairs + diagnostics.spatial_pairs,
        )
        self.assertEqual(
            diagnostics.pre_component_filter_claims,
            4 * diagnostics.unordered_affinity_pairs,
        )
        self.assertEqual(diagnostics.claims, 4 * diagnostics.cross_component_pairs)
        self.assertLessEqual(
            diagnostics.pre_component_filter_claims,
            e23.MAX_COMBINED_RCCE4_CLAIMS,
        )
        self.assertLessEqual(diagnostics.claims, e23.MAX_CROSS_COMPONENT_CLAIMS)
        self.assertLessEqual(
            diagnostics.relation_candidates, e23.MAX_RELATION_CANDIDATES
        )
        self.assertLessEqual(
            diagnostics.geometry_valid_hypotheses,
            e23.MAX_GEOMETRY_HYPOTHESES,
        )

        slotted = (
            result.spatial_pairs[0],
            result.diagnostics,
            result,
        )
        for value in slotted:
            self.assertTrue(hasattr(type(value), "__slots__"))
            self.assertFalse(hasattr(value, "__dict__"))
        self.assertEqual(replace(result.spatial_pairs[0]), result.spatial_pairs[0])
        self.assertEqual(replace(result.diagnostics), result.diagnostics)
        self.assertIs(weakref.ref(result)(), result)
        with self.assertRaises(FrozenInstanceError):
            result.spatial_pairs[0].pair_id = 0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
