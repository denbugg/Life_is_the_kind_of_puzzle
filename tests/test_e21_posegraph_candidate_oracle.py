from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import e21_posegraph_candidate_oracle as e21  # noqa: E402


def _zeros() -> tuple[np.ndarray, np.ndarray]:
    shape = (e21.NUM_TILES, e21.NUM_TILES)
    return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)


def _partition(
    groups: tuple[dict[int, tuple[int, int]], ...],
) -> tuple[
    tuple[e21.RigidComponent, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    frozenset[int],
]:
    used: set[int] = set()
    entries: list[tuple[e21.ComponentEntry, ...]] = []
    for group in groups:
        if used & set(group):
            raise AssertionError("test partition groups overlap")
        used.update(group)
        value = tuple(
            sorted((tile, row, col) for tile, (row, col) in group.items())
        )
        entries.append(value)
    entries.extend(
        ((tile, 0, 0),) for tile in range(e21.NUM_TILES) if tile not in used
    )
    entries.sort(key=lambda value: (-len(value), min(item[0] for item in value), value))
    components = tuple(
        e21.RigidComponent(component_id=index, entries=value)
        for index, value in enumerate(entries)
    )
    owner = np.full(e21.NUM_TILES, -1, dtype=np.int64)
    rows = np.zeros(e21.NUM_TILES, dtype=np.int64)
    cols = np.zeros(e21.NUM_TILES, dtype=np.int64)
    for component in components:
        for tile, row, col in component.entries:
            owner[tile] = component.component_id
            rows[tile] = row
            cols[tile] = col
    nontrivial = frozenset(
        component.component_id for component in components if component.size >= 2
    )
    return components, owner, rows, cols, nontrivial


def _claim(
    claim_id: int,
    score: float,
    anchor: int,
    target: int,
    dy: int,
    dx: int,
    owner: np.ndarray,
) -> e21.CandidateClaim:
    return e21.CandidateClaim(
        claim_id=claim_id,
        score=score,
        anchor=anchor,
        target=target,
        dy=dy,
        dx=dx,
        anchor_component=int(owner[anchor]),
        target_component=int(owner[target]),
    )


class FrozenContractTests(unittest.TestCase):
    def test_constants_signature_and_result_fields_are_literal(self) -> None:
        self.assertEqual(e21.GRID, 24)
        self.assertEqual(e21.NUM_TILES, 576)
        self.assertEqual(e21.COMPONENT_MAX_EDGES, 96)
        self.assertEqual(e21.MIN_MARGIN, 0.0)
        self.assertEqual(e21.CANDIDATE_TOP_K, 8)
        self.assertEqual(e21.DELTAS, ((-1, 0), (1, 0), (0, -1), (0, 1)))
        self.assertEqual(
            tuple(inspect.signature(e21.run_posegraph_candidate_oracle).parameters),
            ("right", "down"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(e21.CandidatePoolResult)),
            (
                "components",
                "owner",
                "local_rows",
                "local_cols",
                "nontrivial_component_ids",
                "claims",
                "hypotheses",
                "diagnostics",
            ),
        )
        forbidden_inputs = {
            "permutation",
            "truth",
            "target",
            "board",
            "labels",
            "pixels",
            "tiles",
            "rotation",
        }
        self.assertTrue(
            forbidden_inputs.isdisjoint(
                inspect.signature(e21.run_posegraph_candidate_oracle).parameters
            )
        )

    def test_dense_contract_is_exact_and_does_not_freeze_caller(self) -> None:
        right, _down = _zeros()
        frozen = e21._dense(right, label="right")
        self.assertEqual(frozen.dtype, np.dtype(np.float32))
        self.assertTrue(frozen.flags.c_contiguous)
        self.assertFalse(frozen.flags.writeable)
        self.assertTrue(right.flags.writeable)
        self.assertIsNot(frozen, right)

        with self.assertRaisesRegex(e21.CandidateOracleError, "576x576"):
            e21._dense(right[:-1], label="right")
        with self.assertRaisesRegex(e21.CandidateOracleError, "float32"):
            e21._dense(right.astype(np.float64), label="right")
        right[0, 1] = -1.0
        with self.assertRaisesRegex(e21.CandidateOracleError, "nonnegative"):
            e21._dense(right, label="right")
        right[0, 1] = np.nan
        with self.assertRaisesRegex(e21.CandidateOracleError, "finite"):
            e21._dense(right, label="right")
        right[0, 1] = 0.0
        right[0, 0] = 1.0
        with self.assertRaisesRegex(e21.CandidateOracleError, "diagonal"):
            e21._dense(right, label="right")


class CorrectedComponentTests(unittest.TestCase):
    def test_cc96_literal_full_partition_order_and_read_only_geometry(self) -> None:
        right, down = _zeros()
        raw = [
            {9: (4, 4), 10: (4, 5)},
            {5: (8, 9), 6: (9, 9), 7: (9, 10)},
        ]
        with mock.patch.object(
            e21, "build_buddies_components", return_value=raw
        ) as builder:
            components, owner, rows, cols, nontrivial = e21.build_components(
                right, down
            )
        builder.assert_called_once()
        self.assertEqual(
            builder.call_args.kwargs, {"max_edges": 96, "min_margin": 0.0}
        )
        self.assertEqual(components[0].tiles, (5, 6, 7))
        self.assertEqual(
            components[0].positions(), {5: (0, 0), 6: (1, 0), 7: (1, 1)}
        )
        self.assertEqual(components[1].tiles, (9, 10))
        self.assertEqual(components[2].tiles, (0,))
        self.assertEqual(nontrivial, frozenset({0, 1}))
        self.assertEqual(
            sorted(tile for component in components for tile in component.tiles),
            list(range(e21.NUM_TILES)),
        )
        self.assertEqual(int(owner[5]), 0)
        self.assertEqual((int(rows[7]), int(cols[7])), (1, 1))
        for value in (owner, rows, cols):
            self.assertFalse(value.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            components[0].component_id = 99  # type: ignore[misc]

    def test_component_normalization_fails_closed_on_bad_builder_geometry(self) -> None:
        right, down = _zeros()
        cases = (
            ([{0: (0, 0), 1: (0, 1)}, {1: (2, 2), 2: (2, 3)}], "overlap"),
            ([{0: (0, 0), 1: (0, 0)}], "collision"),
            ([{0: (0, 0), 1: (24, 0)}], "24x24"),
        )
        for raw, message in cases:
            with self.subTest(message=message), mock.patch.object(
                e21, "build_buddies_components", return_value=raw
            ), self.assertRaisesRegex(e21.CandidateOracleError, message):
                e21.build_components(right, down)


class CandidateClaimTests(unittest.TestCase):
    def test_positive_top8_is_selected_before_cross_component_filter(self) -> None:
        right, down = _zeros()
        components, owner, _rows, _cols, nontrivial = _partition(
            ({0: (0, 0), 1: (0, 1)},)
        )
        right[0, 1] = 1.0
        for target in range(2, 10):
            right[0, target] = 0.9

        claims, diagnostics = e21._build_candidate_claims_with_diagnostics(
            right, down, components, owner, nontrivial
        )
        selected = [
            claim.target
            for claim in claims
            if claim.anchor == 0 and (claim.dy, claim.dx) == (0, 1)
        ]
        self.assertEqual(selected, list(range(2, 9)))
        self.assertNotIn(9, selected)
        self.assertEqual(diagnostics.positive_top8_before_component_filter, 9)
        self.assertEqual(diagnostics.same_component_filtered, 2)
        self.assertEqual(len(claims), 7)
        self.assertEqual(tuple(claim.claim_id for claim in claims), tuple(range(7)))

    def test_singleton_target_is_allowed_but_singleton_emitter_is_forbidden(self) -> None:
        right, down = _zeros()
        components, owner, _rows, _cols, nontrivial = _partition(
            ({0: (0, 0), 1: (0, 1)},)
        )
        right[0, 2] = 0.9
        right[2, 3] = 0.8
        claims, diagnostics = e21._build_candidate_claims_with_diagnostics(
            right, down, components, owner, nontrivial
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].identity, (0, 2, 0, 1))
        self.assertEqual(claims[0].target_component, int(owner[2]))
        self.assertNotIn(2, {claim.anchor for claim in claims})
        self.assertEqual(diagnostics.emitter_tiles, 2)
        self.assertEqual(diagnostics.singleton_target_claims, 1)
        self.assertEqual(diagnostics.nontrivial_target_claims, 0)

    def test_component_owner_must_exactly_match_full_partition(self) -> None:
        right, down = _zeros()
        components, owner, _rows, _cols, nontrivial = _partition(
            ({0: (0, 0), 1: (0, 1)}, {2: (0, 0), 3: (0, 1)})
        )
        bad_owner = owner.copy()
        bad_owner[0] = int(owner[2])
        with self.assertRaisesRegex(e21.CandidateOracleError, "does not match"):
            e21.build_candidate_claims(
                right, down, components, bad_owner, nontrivial
            )


class PoseHypothesisTests(unittest.TestCase):
    def test_canonical_signed_relation_when_emitter_component_is_higher(self) -> None:
        _components, owner, rows, cols, _nontrivial = _partition(
            (
                {0: (0, 0), 1: (0, 1)},
                {2: (1, 1), 3: (0, 0)},
            )
        )
        claims = (_claim(0, 0.7, 2, 0, 1, 0, owner),)
        hypotheses = e21.build_pose_hypotheses(claims, owner, rows, cols)
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0].relation, (0, 1, -2, -1))
        self.assertEqual(hypotheses[0].physical_seams, ((2, 0, 1, 0),))

    def test_physical_seam_is_deduplicated_and_reverse_is_metadata(self) -> None:
        _components, owner, rows, cols, _nontrivial = _partition(
            (
                {0: (0, 0), 1: (0, 1)},
                {2: (0, 0), 3: (0, 1)},
            )
        )
        claims = (
            _claim(0, 0.7, 0, 2, 0, 1, owner),
            _claim(1, 0.9, 2, 0, 0, -1, owner),
            _claim(2, 0.8, 0, 2, 0, 1, owner),
        )
        hypotheses = e21.build_pose_hypotheses(claims, owner, rows, cols)
        self.assertEqual(len(hypotheses), 1)
        hypothesis = hypotheses[0]
        self.assertEqual(hypothesis.relation, (0, 1, 0, 1))
        self.assertEqual(hypothesis.seam_scores, (((0, 2, 0, 1), 0.9),))
        self.assertEqual(hypothesis.reciprocal_seams, ((0, 2, 0, 1),))
        self.assertEqual(hypothesis.unique_physical_seams, 1)
        self.assertEqual(hypothesis.reciprocal_physical_seams, 1)
        self.assertAlmostEqual(hypothesis.direct_neural_sum, 0.9)
        self.assertAlmostEqual(hypothesis.direct_max_score, 0.9)

    def test_alternative_offsets_for_one_component_pair_are_not_collapsed(self) -> None:
        _components, owner, rows, cols, _nontrivial = _partition(
            (
                {0: (0, 0), 1: (0, 1)},
                {2: (0, 0), 3: (0, 1)},
            )
        )
        claims = (
            _claim(0, 0.8, 0, 2, 0, 1, owner),
            _claim(1, 0.7, 1, 2, 0, 1, owner),
        )
        hypotheses = e21.build_pose_hypotheses(claims, owner, rows, cols)
        self.assertEqual(
            tuple(hypothesis.relation for hypothesis in hypotheses),
            ((0, 1, 0, 1), (0, 1, 0, 2)),
        )
        self.assertEqual(
            tuple(hypothesis.hypothesis_id for hypothesis in hypotheses), (0, 1)
        )
        self.assertEqual(
            tuple(hypothesis.unique_physical_seams for hypothesis in hypotheses),
            (1, 1),
        )

    def test_claim_ids_and_directions_fail_closed(self) -> None:
        _components, owner, rows, cols, _nontrivial = _partition(
            ({0: (0, 0), 1: (0, 1)}, {2: (0, 0), 3: (0, 1)})
        )
        with self.assertRaisesRegex(e21.CandidateOracleError, "IDs"):
            e21.build_pose_hypotheses(
                (_claim(4, 0.8, 0, 2, 0, 1, owner),), owner, rows, cols
            )
        bad_direction = _claim(0, 0.8, 0, 2, 1, 1, owner)
        with self.assertRaisesRegex(e21.CandidateOracleError, "cardinal"):
            e21.build_pose_hypotheses(
                (bad_direction,), owner, rows, cols
            )


class FullCoreTests(unittest.TestCase):
    def test_full_core_is_deterministic_read_only_and_diagnostic_complete(self) -> None:
        right, down = _zeros()
        right[0, 2] = 0.8
        raw = [
            {2: (5, 7), 3: (5, 8)},
            {0: (9, 2), 1: (9, 3)},
        ]
        with mock.patch.object(
            e21, "build_buddies_components", return_value=raw
        ) as builder:
            first = e21.run_posegraph_candidate_oracle(right, down)
            second = e21.run_posegraph_candidate_oracle(right, down)
        self.assertEqual(builder.call_count, 2)
        self.assertEqual(first.components, second.components)
        self.assertEqual(first.claims, second.claims)
        self.assertEqual(first.hypotheses, second.hypotheses)
        self.assertEqual(first.diagnostics, second.diagnostics)
        self.assertTrue(np.array_equal(first.owner, second.owner))
        self.assertTrue(right.flags.writeable)
        for value in (first.owner, first.local_rows, first.local_cols):
            self.assertFalse(value.flags.writeable)
        with self.assertRaises(ValueError):
            first.owner[0] = 99

        self.assertEqual(tuple(claim.claim_id for claim in first.claims), (0, 1))
        self.assertEqual(len(first.hypotheses), 1)
        self.assertEqual(first.hypotheses[0].relation, (0, 1, 0, 1))
        self.assertEqual(first.hypotheses[0].reciprocal_physical_seams, 1)
        self.assertEqual(
            first.diagnostics,
            e21.CandidatePoolDiagnostics(
                component_count=574,
                nontrivial_components=2,
                singleton_components=572,
                total_tiles=576,
                nontrivial_tiles=4,
                singleton_tiles=572,
                emitter_tiles=4,
                directional_emitter_rows=16,
                positive_top8_before_component_filter=2,
                same_component_filtered=0,
                claims=2,
                nontrivial_target_claims=2,
                singleton_target_claims=0,
                hypotheses=1,
                component_pairs=1,
                component_pairs_with_alternative_offsets=0,
                unique_physical_seams=1,
                reciprocal_physical_seams=1,
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            first.diagnostics.claims = 0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
