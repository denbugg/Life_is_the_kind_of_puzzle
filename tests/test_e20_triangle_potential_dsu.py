from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import e18_absolute_frame_beam as e18  # noqa: E402
import e20_triangle_potential_dsu as e20  # noqa: E402


def _component(
    component_id: int, values: dict[int, tuple[int, int]]
) -> e18.e15.Component:
    return e18.e15.Component(
        component_id=component_id,
        entries=tuple(
            sorted((tile, row, col) for tile, (row, col) in values.items())
        ),
    )


def _claim(
    claim_id: int,
    score: float,
    anchor: int,
    target: int,
    dy: int,
    dx: int,
    anchor_component: int,
    target_component: int,
) -> e18.BridgeClaim:
    return e18.BridgeClaim(
        claim_id,
        score,
        anchor,
        target,
        dy,
        dx,
        anchor_component,
        target_component,
    )


def _graph(
    components: tuple[e18.e15.Component, ...],
    claims: tuple[e18.BridgeClaim, ...] = (),
) -> e18.GraphData:
    owner = np.full(576, -1, dtype=np.int64)
    rows = np.zeros(576, dtype=np.int64)
    cols = np.zeros(576, dtype=np.int64)
    for component in components:
        for tile, row, col in component.entries:
            owner[tile] = component.component_id
            rows[tile] = row
            cols[tile] = col
    by_frontier: dict[tuple[int, int], list[e18.BridgeClaim]] = {}
    by_component: dict[int, list[e18.BridgeClaim]] = {
        component.component_id: [] for component in components
    }
    for claim in claims:
        direction = e18.DELTAS.index((claim.dy, claim.dx))
        by_frontier.setdefault((claim.anchor, direction), []).append(claim)
        by_component[claim.anchor_component].append(claim)
        by_component[claim.target_component].append(claim)
    for value in (owner, rows, cols):
        value.setflags(write=False)
    return e18.GraphData(
        components=components,
        owner=owner,
        local_rows=rows,
        local_cols=cols,
        nontrivial=frozenset(component.component_id for component in components),
        claims=claims,
        claims_by_frontier={key: tuple(value) for key, value in by_frontier.items()},
        claims_by_component={key: tuple(value) for key, value in by_component.items()},
    )


def _hypothesis(
    hypothesis_id: int,
    u: int,
    v: int,
    dr: int,
    dc: int,
    scores: tuple[float, ...] = (1.0,),
    reciprocal: int = 0,
) -> e20.PoseHypothesis:
    base = hypothesis_id * 12
    seams = tuple(
        ((base + 2 * index, base + 2 * index + 1, 0, 1), float(score))
        for index, score in enumerate(scores)
    )
    return e20.PoseHypothesis(
        hypothesis_id=hypothesis_id,
        u=u,
        v=v,
        dr=dr,
        dc=dc,
        seam_scores=seams,
        reciprocal_seams=tuple(seam for seam, _score in seams[:reciprocal]),
    )


class FrozenContractTests(unittest.TestCase):
    def test_constants_and_public_run_api_are_literal(self) -> None:
        self.assertEqual(e20.GRID, 24)
        self.assertEqual(e20.NUM_TILES, 576)
        self.assertEqual(e20.COMPONENT_MAX_EDGES, 192)
        self.assertEqual(e20.MIN_MARGIN, 0.0)
        self.assertEqual(e20.CANDIDATE_TOP_K, 8)
        self.assertEqual(e20.INCIDENT_TOP_K, 8)
        self.assertEqual(
            set(inspect.signature(e20.run_triangle_potential_dsu).parameters),
            {"right", "down"},
        )

    def test_run_delegates_to_exact_e18_graph_and_structure_solver(self) -> None:
        graph = _graph((_component(0, {0: (0, 0), 1: (0, 1)}),))
        sentinel = object()
        right = np.zeros((576, 576), dtype=np.float32)
        down = right.copy()
        with mock.patch.object(
            e20.e18, "build_graph_data", return_value=graph
        ) as builder, mock.patch.object(
            e20, "solve_pose_forest", return_value=sentinel
        ) as solver:
            self.assertIs(e20.run_triangle_potential_dsu(right, down), sentinel)
        builder.assert_called_once_with(right, down)
        solver.assert_called_once_with(graph)

    def test_result_is_sparse_and_has_no_board_or_label_input(self) -> None:
        graph = _graph((_component(0, {0: (0, 0), 1: (1, 0)}),))
        result = e20.solve_pose_forest(graph)
        self.assertFalse(hasattr(result, "board"))
        self.assertFalse(hasattr(result.selected, "board"))
        self.assertEqual(result.selected.relative_entries, ((0, 0, 0), (1, 1, 0)))
        self.assertEqual(result.selected.legal_origin_count, 23 * 24)
        self.assertEqual(result.diagnostics.eligible_processed, 0)


class HypothesisConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
        )

    def test_physical_seams_are_deduped_and_reverse_is_only_reciprocity(self) -> None:
        claims = (
            _claim(0, 0.7, 0, 2, 0, 1, 0, 1),
            _claim(1, 0.9, 2, 0, 0, -1, 1, 0),
        )
        hypothesis = e20.build_pose_hypotheses(_graph(self.components, claims))[0]
        self.assertEqual(hypothesis.relation, (0, 1, 0, 1))
        self.assertEqual(hypothesis.unique_physical_seams, 1)
        self.assertEqual(hypothesis.reciprocal_physical_seams, 1)
        self.assertAlmostEqual(hypothesis.direct_neural_sum, 0.9)
        self.assertAlmostEqual(hypothesis.direct_max_score, 0.9)
        self.assertEqual(hypothesis.independent_paths, 1)
        self.assertFalse(hypothesis.eligible)

    def test_distinct_physical_seams_make_one_exact_offset_eligible(self) -> None:
        claims = (
            _claim(0, 0.9, 0, 2, 0, 1, 0, 1),
            _claim(1, 0.8, 1, 3, 0, 1, 0, 1),
        )
        hypothesis = e20.build_pose_hypotheses(_graph(self.components, claims))[0]
        self.assertEqual(hypothesis.unique_physical_seams, 2)
        self.assertEqual(hypothesis.reciprocal_physical_seams, 0)
        self.assertAlmostEqual(hypothesis.direct_neural_sum, 1.7)
        self.assertEqual(hypothesis.independent_paths, 2)
        self.assertTrue(hypothesis.eligible)

    def test_multiple_exact_offsets_for_one_pair_are_never_collapsed(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (0, 2)}),
            _component(1, {2: (0, 0), 3: (0, 2)}),
        )
        claims = (
            _claim(0, 1.0, 0, 2, 0, 1, 0, 1),
            _claim(1, 0.8, 1, 2, 0, 1, 0, 1),
        )
        hypotheses = e20.build_pose_hypotheses(_graph(components, claims))
        self.assertEqual(
            tuple(hypothesis.relation for hypothesis in hypotheses),
            ((0, 1, 0, 1), (0, 1, 0, 3)),
        )

    def test_invalid_owner_and_nonpositive_claim_fail_closed(self) -> None:
        bad_owner = _claim(0, 1.0, 0, 2, 0, 1, 1, 0)
        with self.assertRaisesRegex(e20.TrianglePoseError, "owner"):
            e20.build_pose_hypotheses(_graph(self.components, (bad_owner,)))
        nonpositive = _claim(0, 0.0, 0, 2, 0, 1, 0, 1)
        with self.assertRaisesRegex(e20.TrianglePoseError, "positive"):
            e20.build_pose_hypotheses(_graph(self.components, (nonpositive,)))

    def test_duplicate_same_orientation_observation_uses_max_once(self) -> None:
        claims = (
            _claim(0, 0.4, 0, 2, 0, 1, 0, 1),
            _claim(1, 0.8, 0, 2, 0, 1, 0, 1),
        )
        hypothesis = e20.build_pose_hypotheses(_graph(self.components, claims))[0]
        self.assertEqual(hypothesis.unique_physical_seams, 1)
        self.assertEqual(hypothesis.reciprocal_physical_seams, 0)
        self.assertAlmostEqual(hypothesis.direct_neural_sum, 0.8)


class TriangleSupportTests(unittest.TestCase):
    def test_exact_signed_composition_adds_one_distinct_intermediary(self) -> None:
        hypotheses = (
            _hypothesis(0, 0, 1, 0, 1),
            _hypothesis(1, 0, 2, 1, 1),
            _hypothesis(2, 1, 2, 1, 0),
        )
        supported = e20.add_triangle_support(hypotheses)
        outer = supported[1]
        self.assertEqual(outer.triangle_intermediates, (1,))
        self.assertEqual(outer.independent_paths, 2)
        self.assertTrue(outer.eligible)
        witness = outer.triangle_witnesses[0]
        self.assertEqual(
            (witness.first_hypothesis, witness.second_hypothesis), (0, 2)
        )
        self.assertAlmostEqual(witness.bottleneck_sum, 1.0)

    def test_only_the_exact_outer_offset_is_targeted(self) -> None:
        hypotheses = (
            _hypothesis(0, 0, 1, 0, 1),
            _hypothesis(1, 0, 2, 1, 1),
            _hypothesis(2, 0, 2, 4, 4),
            _hypothesis(3, 1, 2, 1, 0),
        )
        supported = e20.add_triangle_support(hypotheses)
        self.assertEqual(supported[1].triangle_intermediates, (1,))
        self.assertEqual(supported[2].triangle_intermediates, ())

    def test_incident_top8_keeps_hypotheses_not_unique_component_pairs(self) -> None:
        hypotheses = tuple(
            _hypothesis(index, 0, 1, 0, index, (float(20 - index),))
            for index in range(9)
        )
        top = e20.incident_top_hypotheses(hypotheses)
        self.assertEqual(len(top[0]), 8)
        self.assertEqual(len(top[1]), 8)
        self.assertEqual(
            tuple(value.hypothesis_id for value in top[0]), tuple(range(8))
        )

    def test_incident_direct_rank_is_literal_before_other_and_offset(self) -> None:
        values = (
            _hypothesis(0, 0, 7, 0, 0, (2.0, 0.1)),
            _hypothesis(1, 0, 1, 0, 0, (100.0,), reciprocal=1),
            _hypothesis(2, 0, 2, 0, 0, (200.0, 100.0)),
            _hypothesis(3, 0, 3, 0, 0, (4.0, 3.0), reciprocal=1),
        )
        top = e20.incident_top_hypotheses(values)[0]
        self.assertEqual(tuple(value.hypothesis_id for value in top), (3, 2, 0, 1))

    def test_same_intermediary_retains_best_bottleneck_then_leg_ids(self) -> None:
        hypotheses = (
            _hypothesis(0, 0, 1, 0, 1, (5.0,)),
            _hypothesis(1, 0, 1, 1, 0, (4.0, 4.0)),
            _hypothesis(2, 0, 2, 0, 2, (1.0,)),
            _hypothesis(3, 1, 2, 0, 1, (6.0,)),
            _hypothesis(4, 1, 2, -1, 2, (4.0, 4.0)),
        )
        outer = e20.add_triangle_support(hypotheses)[2]
        self.assertEqual(len(outer.triangle_witnesses), 1)
        witness = outer.triangle_witnesses[0]
        self.assertEqual(
            (witness.first_hypothesis, witness.second_hypothesis), (1, 4)
        )
        self.assertAlmostEqual(witness.bottleneck_sum, 8.0)
        self.assertAlmostEqual(witness.bottleneck_max, 4.0)
        self.assertTrue(witness.strong)

    def test_strong_requires_two_physical_seams_on_each_retained_leg(self) -> None:
        hypotheses = (
            _hypothesis(0, 0, 1, 0, 1, (9.0,), reciprocal=1),
            _hypothesis(1, 0, 2, 0, 2),
            _hypothesis(2, 1, 2, 0, 1, (8.0,), reciprocal=1),
        )
        witness = e20.add_triangle_support(hypotheses)[1].triangle_witnesses[0]
        self.assertFalse(witness.strong)

    def test_top8_bound_can_exclude_a_ninth_triangle_leg(self) -> None:
        values = [
            _hypothesis(index, 0, index + 1, 0, index + 1, (20.0 - index,))
            for index in range(9)
        ]
        # At intermediary 0, the low-score 0--9 leg is displaced by 0--10.
        # It therefore cannot combine with 0--10 to witness outer edge 9--10.
        values.extend(
            (
                _hypothesis(9, 0, 10, 0, 10, (30.0,)),
                _hypothesis(10, 9, 10, 0, 1, (10.0,)),
            )
        )
        supported = e20.add_triangle_support(tuple(values))
        target = next(value for value in supported if value.hypothesis_id == 10)
        self.assertNotIn(0, target.triangle_intermediates)


class PotentialForestTests(unittest.TestCase):
    def test_two_direct_supported_merges_propagate_signed_potentials(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
            _component(2, {4: (0, 0), 5: (1, 0)}),
        )
        claims = (
            _claim(0, 1.0, 0, 2, 0, 1, 0, 1),
            _claim(1, 0.9, 1, 3, 0, 1, 0, 1),
            _claim(2, 0.8, 2, 4, 0, 1, 1, 2),
            _claim(3, 0.7, 3, 5, 0, 1, 1, 2),
        )
        result = e20.solve_pose_forest(_graph(components, claims))
        selected = result.selected
        self.assertEqual(selected.component_ids, (0, 1, 2))
        self.assertEqual(
            selected.translations, ((0, 0, 0), (1, 0, 1), (2, 0, 2))
        )
        self.assertEqual(selected.bbox, (0, 1, 0, 2))
        self.assertEqual(selected.legal_origin_bounds, (0, 22, 0, 21))
        self.assertEqual(selected.legal_origin_count, 23 * 22)
        self.assertEqual(len(selected.tree_hypothesis_ids), 2)
        self.assertEqual(selected.cycle_hypothesis_ids, ())
        self.assertEqual(result.diagnostics.tree_merges, 2)

    def test_negative_signed_merge_is_normalized_only_after_forest(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
        )
        claims = (
            _claim(0, 1.0, 2, 0, 0, 1, 1, 0),
            _claim(1, 0.9, 3, 1, 0, 1, 1, 0),
        )
        selected = e20.solve_pose_forest(_graph(components, claims)).selected
        self.assertEqual(selected.translations, ((0, 0, 1), (1, 0, 0)))
        self.assertEqual(selected.bbox, (0, 1, 0, 1))
        self.assertEqual(selected.legal_origin_count, 23 * 23)

    def test_exact_triangle_relation_is_accepted_as_cycle_evidence(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (0, 1)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
            _component(2, {4: (0, 0), 5: (1, 0)}),
        )
        claims = (
            _claim(0, 0.9, 0, 2, 1, 0, 0, 1),
            _claim(1, 0.8, 1, 4, 1, 0, 0, 2),
            _claim(2, 1.0, 2, 4, 0, 1, 1, 2),
            _claim(3, 0.95, 3, 5, 0, 1, 1, 2),
        )
        result = e20.solve_pose_forest(_graph(components, claims))
        selected = result.selected
        self.assertEqual(
            selected.translations, ((0, 0, 0), (1, 1, 0), (2, 1, 1))
        )
        self.assertEqual(len(selected.tree_hypothesis_ids), 2)
        self.assertEqual(len(selected.cycle_hypothesis_ids), 1)
        self.assertEqual(selected.component_cycle_rank, 1)
        self.assertEqual(selected.component_cycle_rank_ratio, 0.5)
        self.assertEqual(len(selected.component_contacts), 3)
        self.assertEqual(len(selected.accepted_cross_seams), 4)
        self.assertEqual(result.diagnostics.cycle_acceptances, 1)

    def test_reciprocal_only_weak_relation_cannot_merge_or_add_evidence(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
        )
        claims = (
            _claim(0, 1.0, 0, 2, 0, 1, 0, 1),
            _claim(1, 1.0, 2, 0, 0, -1, 1, 0),
        )
        result = e20.solve_pose_forest(_graph(components, claims))
        self.assertEqual(result.diagnostics.eligible_hypotheses, 0)
        self.assertEqual(result.diagnostics.weak_hypotheses, 1)
        self.assertEqual(result.diagnostics.cluster_count, 2)
        self.assertEqual(result.selected.accepted_hypothesis_ids, ())
        self.assertEqual(result.selected.accepted_cross_seams, ())
        self.assertEqual(result.selected.cross_neural_sum, 0.0)

    def test_inconsistent_second_offset_is_a_pose_conflict(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (1, 0), 2: (0, 2), 3: (1, 2)}),
            _component(1, {4: (0, 0), 5: (1, 0), 6: (0, 2), 7: (1, 2)}),
        )
        claims = (
            _claim(0, 2.0, 0, 4, 0, 1, 0, 1),
            _claim(1, 2.0, 1, 5, 0, 1, 0, 1),
            _claim(2, 1.0, 6, 2, 0, 1, 1, 0),
            _claim(3, 1.0, 7, 3, 0, 1, 1, 0),
        )
        result = e20.solve_pose_forest(_graph(components, claims))
        self.assertEqual(result.diagnostics.tree_merges, 1)
        self.assertEqual(result.diagnostics.pose_conflicts, 1)
        self.assertEqual(len(result.selected.accepted_hypothesis_ids), 1)

    def test_collision_rejects_an_otherwise_supported_contact(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (0, 2), 2: (0, 4)}),
            _component(1, {3: (0, 0), 4: (0, 2), 5: (0, 3)}),
        )
        claims = (
            _claim(0, 1.0, 0, 3, 0, 1, 0, 1),
            _claim(1, 0.9, 1, 4, 0, 1, 0, 1),
        )
        result = e20.solve_pose_forest(_graph(components, claims))
        self.assertEqual(result.diagnostics.collision_rejections, 1)
        self.assertEqual(result.diagnostics.tree_merges, 0)
        self.assertEqual(result.diagnostics.cluster_count, 2)

    def test_bbox_width_24_is_accepted_and_25_is_rejected(self) -> None:
        accepted_components = (
            _component(0, {0: (0, 0), 1: (0, 22), 2: (1, 22)}),
            _component(1, {3: (0, 0), 4: (1, 0)}),
        )
        accepted_claims = (
            _claim(0, 1.0, 1, 3, 0, 1, 0, 1),
            _claim(1, 1.0, 2, 4, 0, 1, 0, 1),
        )
        accepted = e20.solve_pose_forest(
            _graph(accepted_components, accepted_claims)
        )
        self.assertEqual(accepted.diagnostics.tree_merges, 1)
        self.assertEqual(accepted.selected.bbox_width, 24)
        self.assertEqual(accepted.selected.legal_origin_count, 23)

        rejected_components = (
            _component(0, {0: (0, 0), 1: (0, 23), 2: (1, 23)}),
            _component(1, {3: (0, 0), 4: (1, 0)}),
        )
        rejected_claims = (
            _claim(0, 1.0, 1, 3, 0, 1, 0, 1),
            _claim(1, 1.0, 2, 4, 0, 1, 0, 1),
        )
        rejected = e20.solve_pose_forest(
            _graph(rejected_components, rejected_claims)
        )
        self.assertEqual(rejected.diagnostics.span_rejections, 1)
        self.assertEqual(rejected.diagnostics.tree_merges, 0)

    def test_injected_noncontact_equation_fails_geometry(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
        )
        graph = _graph(components)
        dsu = e20._PotentialDSU(graph)
        invalid = e20.PoseHypothesis(
            hypothesis_id=0,
            u=0,
            v=1,
            dr=0,
            dc=2,
            seam_scores=(((0, 2, 0, 1), 1.0), ((1, 3, 0, 1), 0.9)),
            reciprocal_seams=(),
        )
        self.assertEqual(dsu.accept(invalid), "contact")

    def test_weak_hypothesis_cannot_be_passed_to_dsu(self) -> None:
        graph = _graph(
            (
                _component(0, {0: (0, 0), 1: (1, 0)}),
                _component(1, {2: (0, 0), 3: (1, 0)}),
            )
        )
        with self.assertRaisesRegex(e20.TrianglePoseError, "weak"):
            e20._PotentialDSU(graph).accept(_hypothesis(0, 0, 1, 0, 1))

    def test_cluster_selection_prefers_tiles_then_cycles_then_seams(self) -> None:
        base = e20.PoseCluster(
            component_ids=(0,),
            translations=((0, 0, 0),),
            relative_entries=((0, 0, 0), (1, 0, 1)),
            bbox=(0, 0, 0, 1),
            bbox_height=1,
            bbox_width=2,
            legal_origin_bounds=(0, 23, 0, 22),
            legal_origin_count=24 * 23,
            tree_hypothesis_ids=(),
            cycle_hypothesis_ids=(),
            accepted_hypothesis_ids=(),
            accepted_relations=(),
            component_contacts=(),
            accepted_cross_seams=(),
            rigid_tiles=2,
            rigid_coverage=2 / 576,
            component_cycle_rank=0,
            component_cycle_rank_ratio=0.0,
            cross_neural_sum=0.0,
            minimum_tile=0,
        )
        more_tiles = replace(base, rigid_tiles=3)
        cyclic = replace(base, component_cycle_rank=1)
        more_seams = replace(base, accepted_cross_seams=((0, 1, 0, 1),))
        self.assertLess(
            e20._cluster_selection_key(more_tiles),
            e20._cluster_selection_key(cyclic),
        )
        self.assertLess(
            e20._cluster_selection_key(cyclic),
            e20._cluster_selection_key(more_seams),
        )

    def test_global_translation_gauge_does_not_change_normalized_cluster(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
        )
        graph = _graph(components)
        hypotheses = {
            0: e20.PoseHypothesis(
                0,
                0,
                1,
                0,
                1,
                (((0, 2, 0, 1), 1.0), ((1, 3, 0, 1), 0.9)),
                (),
            )
        }
        first = e20._PotentialDSU(graph)
        self.assertEqual(first.accept(hypotheses[0]), "tree")
        first_root = first.find(0)[0]
        cluster_first = e20._cluster(first_root, first, hypotheses)

        shifted = e20._PotentialDSU(graph)
        self.assertEqual(shifted.accept(hypotheses[0]), "tree")
        shifted_root = shifted.find(0)[0]
        shift = (7, -11)
        shifted.entries[shifted_root] = {
            e20._add(coordinate, shift): tile
            for coordinate, tile in shifted.entries[shifted_root].items()
        }
        shifted.translations[shifted_root] = {
            component: e20._add(value, shift)
            for component, value in shifted.translations[shifted_root].items()
        }
        cluster_shifted = e20._cluster(shifted_root, shifted, hypotheses)
        self.assertEqual(cluster_first.translations, cluster_shifted.translations)
        self.assertEqual(cluster_first.relative_entries, cluster_shifted.relative_entries)
        self.assertEqual(cluster_first.legal_origin_count, cluster_shifted.legal_origin_count)


if __name__ == "__main__":
    unittest.main()
