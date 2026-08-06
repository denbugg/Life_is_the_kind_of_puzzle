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
import e19_relative_frame_beam as e19  # noqa: E402


def _component(
    component_id: int, values: dict[int, tuple[int, int]]
) -> e18.e15.Component:
    return e18.e15.Component(
        component_id=component_id,
        entries=tuple(
            sorted((tile, row, col) for tile, (row, col) in values.items())
        ),
    )


def _zeros() -> tuple[np.ndarray, np.ndarray]:
    value = np.zeros((576, 576), dtype=np.float32)
    return value.copy(), value.copy()


def _tiles() -> np.ndarray:
    return np.zeros((576, 20, 20, 3), dtype=np.uint8)


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
        direction = e19.DELTAS.index((claim.dy, claim.dx))
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
        nontrivial=frozenset(
            component.component_id for component in components if component.size >= 2
        ),
        claims=claims,
        claims_by_frontier={key: tuple(value) for key, value in by_frontier.items()},
        claims_by_component={key: tuple(value) for key, value in by_component.items()},
    )


def _absolute_root_state(
    graph: e18.GraphData, origin: tuple[int, int]
) -> e18.PartialState:
    board = np.full((24, 24), -1, dtype=np.int64)
    root = graph.components[0]
    for tile, row, col in root.entries:
        board[row + origin[0], col + origin[1]] = tile
    board.setflags(write=False)
    return e18.PartialState(
        translations=((0, origin[0], origin[1]),),
        board=board,
        satisfied_bridge_claims=frozenset(),
        component_contacts=frozenset(),
        cross_seams=frozenset(),
        cross_neural_sum=0.0,
        cross_lab_sum=0.0,
        rigid_tiles=root.size,
        root_origin=origin,
    )


class FrozenContractTests(unittest.TestCase):
    def test_constants_and_public_api_are_literal(self) -> None:
        self.assertEqual(e19.COMPONENT_MAX_EDGES, 192)
        self.assertEqual(e19.MIN_MARGIN, 0.0)
        self.assertEqual(e19.CANDIDATE_TOP_K, 8)
        self.assertEqual(e19.BEAM_WIDTH, 256)
        self.assertEqual(e19.PROPOSALS_PER_STATE, 64)
        self.assertEqual(e19.MAX_ATTACHMENTS, 64)
        self.assertEqual(e19.RELATIVE_LAYOUTS, 8)
        self.assertEqual(e19.EXPANSION_CAP, 500_000)
        self.assertEqual(
            set(inspect.signature(e19.run_relative_frame).parameters),
            {"right", "down", "tiles"},
        )

    def test_initial_state_is_one_zero_root_without_absolute_origin(self) -> None:
        graph = _graph((_component(0, {0: (0, 0), 1: (1, 0)}),))
        state = e19.initial_relative_state(graph)
        self.assertEqual(state.translations, ((0, 0, 0),))
        self.assertEqual(state.bbox, (0, 1, 0, 0))
        self.assertFalse(hasattr(state, "legal_origin_bounds"))
        self.assertFalse(hasattr(state, "legal_origin_count"))
        self.assertFalse(hasattr(state, "board"))

    def test_run_api_returns_structure_only_best_first(self) -> None:
        graph = _graph((_component(0, {0: (0, 0), 1: (1, 0)}),))
        state = e19.initial_relative_state(graph)
        with mock.patch.object(e19, "build_graph_data", return_value=graph), mock.patch.object(
            e19, "relative_path_beam", return_value=((state,), 7, 2, False)
        ):
            result = e19.run_relative_frame(*_zeros(), _tiles())
        self.assertEqual(len(result.layouts), 1)
        self.assertEqual(result.layouts[0].translations, ((0, 0, 0),))
        self.assertEqual(result.layouts[0].legal_origin_count, 23 * 24)
        self.assertFalse(hasattr(result.layouts[0], "board"))
        self.assertFalse(hasattr(result, "board"))
        self.assertEqual(result.diagnostics.initial_states, 1)
        self.assertEqual(result.diagnostics.proposal_evaluations, 7)
        self.assertEqual(result.diagnostics.rounds, 2)
        self.assertFalse(result.diagnostics.cap_hit)


class QuotientGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
        )
        self.claim = e18.BridgeClaim(0, 0.9, 0, 2, 0, 1, 0, 1)
        self.graph = _graph(self.components, (self.claim,))

    def test_two_shifted_e18_geometries_collapse_to_one_relative_state(self) -> None:
        right, down = _zeros()
        right[0, 2] = 0.9
        relative_root = e19.initial_relative_state(self.graph)
        relative_proposals = e19.induced_relative_translations(
            relative_root, self.graph
        )
        self.assertEqual(relative_proposals[0][:4], (1, 0, 1, (0,)))

        normalized_absolute = []
        absolute_children = []
        for origin in ((5, 5), (10, 11)):
            absolute = _absolute_root_state(self.graph, origin)
            proposal = e18.induced_translations(absolute, self.graph)[0]
            normalized_absolute.append(
                (proposal[0], proposal[1] - origin[0], proposal[2] - origin[1])
            )
            child = e18.place_induced_component(
                absolute,
                self.graph,
                proposal[0],
                proposal[1],
                proposal[2],
                right,
                down,
                *_zeros(),
            )
            self.assertIsNotNone(child)
            absolute_children.append(child)
        self.assertEqual(normalized_absolute, [(1, 0, 1), (1, 0, 1)])

        relative_child = e19.place_relative_component(
            relative_root,
            self.graph,
            1,
            0,
            1,
            right,
            down,
            *_zeros(),
        )
        self.assertIsNotNone(relative_child)
        assert relative_child is not None
        for origin, absolute_child in zip(((5, 5), (10, 11)), absolute_children):
            normalized = tuple(
                (component, row - origin[0], col - origin[1])
                for component, row, col in absolute_child.translations
            )
            self.assertEqual(normalized, relative_child.translations)
            self.assertEqual(absolute_child.cross_seams, relative_child.cross_seams)
            self.assertEqual(
                absolute_child.satisfied_bridge_claims,
                relative_child.satisfied_bridge_claims,
            )

    def test_negative_signed_coordinates_are_not_absolute_clipped(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
        )
        claim = e18.BridgeClaim(0, 0.9, 0, 2, 0, -1, 0, 1)
        graph = _graph(components, (claim,))
        state = e19.initial_relative_state(graph)
        proposals = e19.induced_relative_translations(state, graph)
        self.assertEqual(proposals[0][:4], (1, 0, -1, (0,)))
        right, down = _zeros()
        right[2, 0] = 0.9
        child = e19.place_relative_component(
            state, graph, 1, 0, -1, right, down, *_zeros()
        )
        self.assertIsNotNone(child)
        assert child is not None
        self.assertEqual(child.bbox, (0, 1, -1, 0))
        self.assertFalse(hasattr(child, "legal_origin_count"))
        layout = e19._layout(child)
        self.assertEqual(layout.legal_origin_bounds, (0, 22, 1, 23))
        self.assertEqual(layout.legal_origin_count, 23 * 23)

    def test_bbox_24_is_accepted_but_25_and_collision_are_rejected(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (0, 1)}),
            _component(1, {2: (0, 0), 3: (0, 21)}),
            _component(2, {4: (0, 0), 5: (0, 22)}),
        )
        graph = _graph(components)
        state = e19.initial_relative_state(graph)
        right, down = _zeros()
        right[1, 2] = 1.0
        right[1, 4] = 1.0
        accepted = e19.place_relative_component(
            state, graph, 1, 0, 2, right, down, *_zeros()
        )
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertEqual(accepted.bbox, (0, 0, 0, 23))
        self.assertEqual(e19._layout(accepted).legal_origin_count, 24)
        self.assertIsNone(
            e19.place_relative_component(
                state, graph, 2, 0, 2, right, down, *_zeros()
            )
        )
        self.assertIsNone(
            e19.place_relative_component(
                state, graph, 1, 0, 0, right, down, *_zeros()
            )
        )

    def test_post_rank_origin_count_is_translation_invariant(self) -> None:
        self.assertEqual(e19._legal_origin_count((0, 23, 0, 23)), 1)
        self.assertEqual(e19._legal_origin_count((-12, 11, 7, 30)), 1)
        self.assertEqual(e19._legal_origin_count((0, 1, 0, 0)), 23 * 24)
        self.assertEqual(e19._legal_origin_count((-9, -8, 13, 13)), 23 * 24)

    def test_injected_globally_shifted_root_fails_closed(self) -> None:
        state = e19.initial_relative_state(self.graph)
        shifted = replace(
            state,
            translations=((0, 4, -3),),
            relative_entries=tuple(
                (tile, row + 4, col - 3)
                for tile, row, col in state.relative_entries
            ),
            bbox=(4, 5, -3, -3),
        )
        with self.assertRaisesRegex(e19.RelativeFrameError, "root"):
            e19._select_states((shifted,), 1)

    def test_pregeometry_top64_rank_is_identical_in_relative_coordinates(self) -> None:
        components = [_component(0, {0: (0, 0), 1: (0, 1)})]
        claims: list[e18.BridgeClaim] = []
        claim_id = 0
        for component_id in range(1, 66):
            first = 2 * component_id
            second = first + 1
            components.append(
                _component(component_id, {first: (0, 0), second: (0, 1)})
            )
            claims.append(
                e18.BridgeClaim(
                    claim_id, 1.0, 0, first, -1, 0, 0, component_id
                )
            )
            claim_id += 1
            if component_id == 65:
                claims.append(
                    e18.BridgeClaim(
                        claim_id, 0.01, 1, second, -1, 0, 0, component_id
                    )
                )
                claim_id += 1
        graph = _graph(tuple(components), tuple(claims))
        proposals = e19.induced_relative_translations(
            e19.initial_relative_state(graph), graph
        )
        self.assertEqual(len(proposals), 64)
        self.assertEqual(proposals[0][0], 65)
        self.assertEqual([value[0] for value in proposals[1:]], list(range(1, 64)))
        self.assertNotIn(64, [value[0] for value in proposals])

    def test_cycle_rank_remains_first_state_rank_field(self) -> None:
        base = e19.initial_relative_state(self.graph)
        cyclic = replace(
            base,
            translations=((0, 0, 0), (1, 0, 1), (2, 1, 0)),
            component_contacts=frozenset({(0, 1), (0, 2), (1, 2)}),
            rigid_tiles=6,
            cross_neural_sum=0.1,
        )
        stronger_tree = replace(
            base,
            translations=((0, 0, 0), (1, 0, 1), (2, 1, 1)),
            satisfied_bridge_claims=frozenset({0, 1, 2, 3}),
            component_contacts=frozenset({(0, 1), (1, 2)}),
            rigid_tiles=6,
            cross_neural_sum=10.0,
        )
        self.assertIs(e19._select_states((stronger_tree, cyclic), 1)[0], cyclic)


class SearchTests(unittest.TestCase):
    def test_duplicate_proposal_key_is_skipped_and_not_counted_twice(self) -> None:
        graph = _graph(
            (
                _component(0, {0: (0, 0), 1: (1, 0)}),
                _component(1, {2: (0, 0), 3: (1, 0)}),
            )
        )
        state = e19.initial_relative_state(graph)
        proposal = (1, 0, 1, (0,), 1.0, 1.0)
        with mock.patch.object(
            e19, "initial_relative_state", return_value=state
        ), mock.patch.object(
            e19,
            "induced_relative_translations",
            return_value=(proposal, proposal),
        ), mock.patch.object(
            e19, "place_relative_component", return_value=None
        ) as placer, mock.patch.object(
            e19.e18.e15, "_lab_pair_matrices", return_value=_zeros()
        ):
            retained, evaluations, rounds, cap_hit = e19.relative_path_beam(
                *_zeros(), _tiles(), graph
            )
        self.assertEqual(retained, (state,))
        self.assertEqual(evaluations, 1)
        self.assertEqual(rounds, 0)
        self.assertFalse(cap_hit)
        self.assertEqual(placer.call_count, 1)

    def test_real_two_round_relative_beam_selects_cycle(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (0, 1)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
            _component(2, {4: (0, 0), 5: (0, 1)}),
        )
        claims = (
            e18.BridgeClaim(0, 0.1, 1, 2, 0, 1, 0, 1),
            e18.BridgeClaim(1, 1.0, 0, 2, 1, 0, 0, 1),
            e18.BridgeClaim(2, 0.05, 3, 5, 0, -1, 1, 2),
        )
        graph = _graph(components, claims)
        right, down = _zeros()
        right[1, 2] = 0.1
        down[0, 2] = 1.0
        down[0, 4] = 0.05
        down[1, 5] = 0.05
        right[5, 3] = 0.05
        with mock.patch.object(
            e19.e18.e15, "_lab_pair_matrices", return_value=_zeros()
        ), mock.patch.object(e19, "MAX_ATTACHMENTS", 2), mock.patch.object(
            e19, "BEAM_WIDTH", 2
        ), mock.patch.object(e19, "RELATIVE_LAYOUTS", 1):
            retained, evaluations, rounds, cap_hit = e19.relative_path_beam(
                right, down, _tiles(), graph
            )
        self.assertFalse(cap_hit)
        self.assertEqual(rounds, 2)
        self.assertGreaterEqual(evaluations, 4)
        best = retained[0]
        self.assertEqual(
            best.translations, ((0, 0, 0), (1, 0, 2), (2, 1, 0))
        )
        self.assertEqual(best.component_cycle_rank, 1)
        self.assertEqual(
            best.component_contacts, frozenset({(0, 1), (0, 2), (1, 2)})
        )

    def test_cap_is_global_distinct_and_counted_before_geometry(self) -> None:
        graph = _graph(
            (
                _component(0, {0: (0, 0), 1: (1, 0)}),
                _component(1, {2: (0, 0), 3: (1, 0)}),
            )
        )
        state = e19.initial_relative_state(graph)
        proposals = (
            (1, 0, 1, (0,), 1.0, 1.0),
            (1, 0, 2, (1,), 0.9, 0.9),
        )
        with mock.patch.object(
            e19, "initial_relative_state", return_value=state
        ), mock.patch.object(
            e19, "induced_relative_translations", return_value=proposals
        ), mock.patch.object(
            e19, "place_relative_component", return_value=state
        ) as placer, mock.patch.object(
            e19.e18.e15, "_lab_pair_matrices", return_value=_zeros()
        ), mock.patch.object(e19, "EXPANSION_CAP", 2):
            _retained, evaluations, _rounds, cap_hit = e19.relative_path_beam(
                *_zeros(), _tiles(), graph
            )
        self.assertEqual(evaluations, 2)
        self.assertTrue(cap_hit)
        self.assertEqual(placer.call_count, 1)

    def test_run_relative_frame_raises_typed_hard_cap_failure(self) -> None:
        graph = _graph((_component(0, {0: (0, 0), 1: (1, 0)}),))
        state = e19.initial_relative_state(graph)
        with mock.patch.object(e19, "build_graph_data", return_value=graph), mock.patch.object(
            e19,
            "relative_path_beam",
            return_value=((state,), 500_000, 17, True),
        ), mock.patch.object(e19, "_layout") as layout:
            with self.assertRaises(e19.RelativeFrameCapError) as caught:
                e19.run_relative_frame(*_zeros(), _tiles())
        self.assertEqual(caught.exception.proposal_evaluations, 500_000)
        self.assertEqual(caught.exception.rounds, 17)
        self.assertEqual(caught.exception.initial_states, 1)
        self.assertTrue(caught.exception.cap_hit)
        layout.assert_not_called()


if __name__ == "__main__":
    unittest.main()
