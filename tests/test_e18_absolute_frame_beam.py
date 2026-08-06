from __future__ import annotations

import inspect
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

import e18_absolute_frame_beam as e18  # noqa: E402


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
    shape = (576, 576)
    return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)


def _tiles() -> np.ndarray:
    return np.zeros((576, 20, 20, 3), dtype=np.uint8)


def _valid_completion(partial: np.ndarray) -> np.ndarray:
    flat = np.asarray(partial, dtype=np.int64).reshape(-1)
    board = flat.copy()
    locked_tiles = set(map(int, flat[flat >= 0].tolist()))
    remaining = iter(tile for tile in range(576) if tile not in locked_tiles)
    for index in np.flatnonzero(board < 0).tolist():
        board[index] = next(remaining)
    return board


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
        nontrivial=frozenset(
            component.component_id for component in components if component.size >= 2
        ),
        claims=claims,
        claims_by_frontier={key: tuple(value) for key, value in by_frontier.items()},
        claims_by_component={key: tuple(value) for key, value in by_component.items()},
    )


def _root_state(
    graph: e18.GraphData, shift: tuple[int, int] = (5, 5)
) -> e18.PartialState:
    root = graph.components[0]
    board = np.full((24, 24), -1, dtype=np.int64)
    for tile, row, col in root.entries:
        board[row + shift[0], col + shift[1]] = tile
    board.setflags(write=False)
    return e18.PartialState(
        translations=((0, shift[0], shift[1]),),
        board=board,
        satisfied_bridge_claims=frozenset(),
        component_contacts=frozenset(),
        cross_seams=frozenset(),
        cross_neural_sum=0.0,
        cross_lab_sum=0.0,
        rigid_tiles=root.size,
        root_origin=shift,
    )


class FrozenContractTests(unittest.TestCase):
    def test_constants_and_solver_signature_are_literal(self) -> None:
        self.assertEqual(e18.COMPONENT_MAX_EDGES, 192)
        self.assertEqual(e18.MIN_MARGIN, 0.0)
        self.assertEqual(e18.CANDIDATE_TOP_K, 8)
        self.assertEqual(e18.BEAM_WIDTH, 256)
        self.assertEqual(e18.PROPOSALS_PER_STATE, 64)
        self.assertEqual(e18.MAX_ATTACHMENTS, 64)
        self.assertEqual(e18.ABSOLUTE_LAYOUTS, 8)
        self.assertEqual(e18.EXPANSION_CAP, 500_000)
        self.assertEqual(e18.HUNGARIAN_ROUNDS, 2)
        self.assertEqual(e18.IDENTITY_BONUS, 0.0)
        self.assertEqual(e18.REPAIR_PASSES, 0)
        self.assertEqual(
            set(inspect.signature(e18.solve_absolute_frame).parameters),
            {"right", "down", "tiles"},
        )

    def test_dense_and_tile_contracts_are_fail_closed(self) -> None:
        right, down = _zeros()
        right[0, 1] = -1.0
        with self.assertRaisesRegex(e18.AbsoluteFrameError, "nonnegative"):
            e18.solve_absolute_frame(right, down, _tiles())
        with self.assertRaisesRegex(e18.AbsoluteFrameError, "upright"):
            e18._tiles(np.zeros((576, 20, 20, 1), dtype=np.uint8))


class ComponentAndBridgeTests(unittest.TestCase):
    def test_components_use_exact_cc192_and_stable_largest_root(self) -> None:
        right, down = _zeros()
        raw = (
            {9: (4, 4), 10: (4, 5)},
            {5: (8, 9), 6: (9, 9), 7: (9, 10)},
        )
        with mock.patch.object(
            e18, "build_buddies_components", return_value=list(raw)
        ) as builder:
            components, owner, rows, cols = e18.build_components(right, down)
        builder.assert_called_once()
        self.assertEqual(
            builder.call_args.kwargs, {"max_edges": 192, "min_margin": 0.0}
        )
        self.assertEqual(components[0].tiles, (5, 6, 7))
        self.assertEqual(components[0].positions(), {5: (0, 0), 6: (1, 0), 7: (1, 1)})
        self.assertEqual(int(owner[5]), 0)
        self.assertEqual((int(rows[7]), int(cols[7])), (1, 1))
        flattened = sorted(tile for component in components for tile in component.tiles)
        self.assertEqual(flattened, list(range(576)))

    def test_dense_top8_is_exact_before_cross_component_filtering(self) -> None:
        right, down = _zeros()
        components = (
            _component(0, {0: (0, 0), 1: (0, 1)}),
            _component(1, {10: (0, 0), 11: (0, 1)}),
        )
        owner = np.full(576, 2, dtype=np.int64)
        owner[[0, 1]] = 0
        owner[[10, 11, 12, 13, 14, 15, 16, 17, 18]] = 1
        # Same-component tile 1 consumes rank one. Cross targets 10..16 occupy
        # ranks two through eight; 17 and 18 must not leak in after filtering.
        right[0, 1] = 1.0
        for offset, target in enumerate(range(10, 19)):
            right[0, target] = 0.99 - 0.01 * offset
        claims, _frontier, _by_component = e18.build_bridge_claims(
            right, down, components, owner
        )
        selected = [
            claim.target
            for claim in claims
            if claim.anchor == 0 and (claim.dy, claim.dx) == (0, 1)
        ]
        self.assertEqual(selected, list(range(10, 17)))
        self.assertNotIn(17, selected)
        self.assertEqual(len(selected), len(set(selected)))

    def test_dense_top8_is_cardinal_tie_stable_positive_and_unique(self) -> None:
        right, down = _zeros()
        components = (
            _component(0, {0: (0, 0), 1: (0, 1)}),
            _component(1, {tile: (0, tile - 10) for tile in range(10, 19)}),
        )
        owner = np.full(576, 2, dtype=np.int64)
        owner[[0, 1]] = 0
        owner[10:19] = 1
        for matrix, anchor_first in (
            (down, False),
            (down, True),
            (right, False),
            (right, True),
        ):
            if anchor_first:
                matrix[0, 1] = 1.0
                matrix[0, 10] = 0.9
                matrix[0, 11] = 0.9
                for offset, target in enumerate(range(12, 19)):
                    matrix[0, target] = 0.8 - 0.05 * offset
            else:
                matrix[1, 0] = 1.0
                matrix[10, 0] = 0.9
                matrix[11, 0] = 0.9
                for offset, target in enumerate(range(12, 19)):
                    matrix[target, 0] = 0.8 - 0.05 * offset
        claims, _frontier, _by_component = e18.build_bridge_claims(
            right, down, components, owner
        )
        for delta in e18.DELTAS:
            selected = [
                claim.target
                for claim in claims
                if claim.anchor == 0 and (claim.dy, claim.dx) == delta
            ]
            self.assertEqual(selected, list(range(10, 17)))
            self.assertEqual(len(selected), len(set(selected)))
            self.assertNotIn(17, selected)

    def test_cardinal_contact_values_are_directionally_exact(self) -> None:
        right, down = _zeros()
        right[1, 2] = 0.1
        right[2, 1] = 0.2
        down[1, 2] = 0.3
        down[2, 1] = 0.4
        self.assertAlmostEqual(
            e18.e15._contact_value(1, 2, 0, 1, right, down), 0.1
        )
        self.assertAlmostEqual(
            e18.e15._contact_value(1, 2, 0, -1, right, down), 0.2
        )
        self.assertAlmostEqual(
            e18.e15._contact_value(1, 2, 1, 0, right, down), 0.3
        )
        self.assertAlmostEqual(
            e18.e15._contact_value(1, 2, -1, 0, right, down), 0.4
        )


class AbsoluteGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
            _component(2, {4: (0, 0), 5: (1, 0)}),
        )
        self.claim = e18.BridgeClaim(0, 0.9, 0, 2, 0, 1, 0, 1)
        self.graph = _graph(self.components, (self.claim,))

    def test_all_legal_root_origins_are_enumerated_without_canonicalisation(self) -> None:
        states = e18.initial_absolute_states(self.graph)
        self.assertEqual(len(states), 23 * 24)
        origins = {state.root_origin for state in states}
        self.assertIn((0, 0), origins)
        self.assertIn((22, 23), origins)
        last = next(state for state in states if state.root_origin == (22, 23))
        self.assertEqual(last.translations, ((0, 22, 23),))
        self.assertEqual(int(last.board[23, 23]), 1)

    def test_single_bridge_induces_one_absolute_translation(self) -> None:
        state = _root_state(self.graph)
        proposals = e18.induced_translations(state, self.graph)
        self.assertEqual(len(proposals), 1)
        component, row, col, claims, _score_sum, _maximum = proposals[0]
        self.assertEqual((component, row, col), (1, 5, 6))
        self.assertEqual(claims, (0,))

    def test_multiple_claims_dedupe_to_one_component_translation(self) -> None:
        components = (
            _component(0, {0: (0, 0), 1: (0, 1)}),
            _component(1, {2: (0, 0), 3: (0, 1)}),
        )
        claims = (
            e18.BridgeClaim(0, 0.9, 0, 2, -1, 0, 0, 1),
            e18.BridgeClaim(1, 0.8, 1, 3, -1, 0, 0, 1),
        )
        graph = _graph(components, claims)
        proposals = e18.induced_translations(_root_state(graph), graph)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0][:4], (1, 4, 5, (0, 1)))
        self.assertAlmostEqual(proposals[0][4], 1.7)
        self.assertAlmostEqual(proposals[0][5], 0.9)

    def test_pregeometry_rank_is_frozen_before_top64_boundary(self) -> None:
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
        proposals = e18.induced_translations(_root_state(graph), graph)
        self.assertEqual(len(proposals), 64)
        self.assertEqual(proposals[0][0], 65)
        selected = [value[0] for value in proposals]
        self.assertEqual(selected[1:], list(range(1, 64)))
        self.assertNotIn(64, selected)

    def test_pregeometry_sum_then_max_order_is_literal(self) -> None:
        components = [_component(0, {0: (0, 0), 1: (0, 1)})]
        score_pairs = {
            1: (0.1, 0.1),
            2: (0.6, 0.6),
            3: (0.7, 0.4),
            4: (0.8, 0.2),
            5: (0.6, 0.4),
        }
        claims: list[e18.BridgeClaim] = []
        claim_id = 0
        for component_id, scores in score_pairs.items():
            first = 2 * component_id
            second = first + 1
            components.append(
                _component(component_id, {first: (0, 0), second: (0, 1)})
            )
            for anchor, target, score in (
                (0, first, scores[0]),
                (1, second, scores[1]),
            ):
                claims.append(
                    e18.BridgeClaim(
                        claim_id, score, anchor, target, -1, 0, 0, component_id
                    )
                )
                claim_id += 1
        components.append(_component(6, {12: (0, 0), 13: (0, 1)}))
        claims.append(e18.BridgeClaim(claim_id, 10.0, 0, 12, -1, 0, 0, 6))
        graph = _graph(tuple(components), tuple(claims))
        proposals = e18.induced_translations(_root_state(graph), graph)
        self.assertEqual([value[0] for value in proposals], [2, 3, 4, 5, 1, 6])

    def test_placement_preserves_whole_component_and_collects_all_contacts(self) -> None:
        right, down = _zeros()
        right[0, 2] = 0.9
        right[1, 3] = 0.8
        state = _root_state(self.graph)
        child = e18.place_induced_component(
            state,
            self.graph,
            1,
            5,
            6,
            right,
            down,
            *_zeros(),
        )
        self.assertIsNotNone(child)
        assert child is not None
        self.assertEqual(int(child.board[5, 6]), 2)
        self.assertEqual(int(child.board[6, 6]), 3)
        self.assertEqual(len(child.cross_seams), 2)
        self.assertEqual(child.component_contacts, frozenset({(0, 1)}))
        self.assertEqual(child.satisfied_bridge_claims, frozenset({0}))
        self.assertEqual(child.rigid_tiles, 4)
        self.assertEqual(child.component_cycle_rank, 0)

    def test_collision_overflow_and_repeated_component_fail_closed(self) -> None:
        right, down = _zeros()
        right[0, 2] = 1.0
        state = _root_state(self.graph)
        self.assertIsNone(
            e18.place_induced_component(
                state, self.graph, 1, 5, 5, right, down, *_zeros()
            )
        )
        self.assertIsNone(
            e18.place_induced_component(
                state, self.graph, 1, 23, 23, right, down, *_zeros()
            )
        )
        with self.assertRaisesRegex(e18.AbsoluteFrameError, "unplaced"):
            e18.place_induced_component(
                state, self.graph, 0, 0, 0, right, down, *_zeros()
            )

    def test_real_two_round_beam_keeps_weak_path_then_selects_cycle(self) -> None:
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
        root = _root_state(graph)
        right, down = _zeros()
        right[1, 2] = 0.1
        down[0, 2] = 1.0
        down[0, 4] = 0.05
        down[1, 5] = 0.05
        right[5, 3] = 0.05
        with mock.patch.object(
            e18, "initial_absolute_states", return_value=(root,)
        ), mock.patch.object(
            e18.e15, "_lab_pair_matrices", return_value=_zeros()
        ), mock.patch.object(e18, "MAX_ATTACHMENTS", 2), mock.patch.object(
            e18, "BEAM_WIDTH", 2
        ), mock.patch.object(e18, "ABSOLUTE_LAYOUTS", 1):
            retained, evaluations, rounds, cap_hit, _origins = e18.absolute_path_beam(
                right, down, _tiles(), graph
            )
        self.assertFalse(cap_hit)
        self.assertEqual(rounds, 2)
        self.assertGreaterEqual(evaluations, 4)
        best = retained[0]
        self.assertEqual(best.translations, ((0, 5, 5), (1, 5, 7), (2, 6, 5)))
        self.assertEqual(best.component_cycle_rank, 1)
        self.assertEqual(
            best.component_contacts, frozenset({(0, 1), (0, 2), (1, 2)})
        )
        self.assertEqual(int(best.board[5, 5]), 0)
        self.assertEqual(int(best.board[6, 7]), 3)
        self.assertEqual(int(best.board[6, 5]), 4)

    def test_exact_score_ties_keep_spatially_diverse_origins(self) -> None:
        states = []
        for row in range(4):
            for col in range(4):
                state = _root_state(self.graph, (row, col))
                states.append(state)
        retained = e18._diverse_tied_states(states, 8)
        origins = {state.root_origin for state in retained}
        self.assertEqual(len(origins), 8)
        self.assertGreater(len({row for row, _col in origins}), 1)
        self.assertGreater(len({col for _row, col in origins}), 1)

    def test_identical_translation_with_different_evidence_fails_closed(self) -> None:
        state = _root_state(self.graph)
        drifted = e18.PartialState(
            **{**state.__dict__, "cross_neural_sum": 1.0}
        )
        with self.assertRaisesRegex(e18.AbsoluteFrameError, "different evidence"):
            e18._diverse_tied_states((state, drifted), 1)


class SearchAndCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.components = (
            _component(0, {0: (0, 0), 1: (1, 0)}),
            _component(1, {2: (0, 0), 3: (1, 0)}),
        )
        self.graph = _graph(self.components)

    def test_every_root_origin_reaches_first_proposal_layer(self) -> None:
        states = tuple(_root_state(self.graph, (0, col)) for col in range(3))
        observed: list[tuple[int, int]] = []

        def no_proposals(state: e18.PartialState, _graph: e18.GraphData):
            observed.append(state.root_origin)
            return ()

        with mock.patch.object(e18, "initial_absolute_states", return_value=states), mock.patch.object(
            e18, "induced_translations", side_effect=no_proposals
        ), mock.patch.object(
            e18.e15, "_lab_pair_matrices", return_value=_zeros()
        ), mock.patch.object(e18, "MAX_ATTACHMENTS", 1):
            retained, evaluations, rounds, cap_hit, origins = e18.absolute_path_beam(
                *_zeros(), _tiles(), self.graph
            )
        self.assertEqual(observed, [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(origins, 3)
        self.assertEqual(evaluations, 0)
        self.assertEqual(rounds, 0)
        self.assertFalse(cap_hit)
        self.assertEqual(len(retained), 3)

    def test_expansion_cap_is_one_global_pre_geometry_counter(self) -> None:
        states = tuple(_root_state(self.graph, (0, col)) for col in range(2))
        proposal = ((1, 0, 1, (0,), 1.0, 1.0),)
        with mock.patch.object(e18, "initial_absolute_states", return_value=states), mock.patch.object(
            e18, "induced_translations", return_value=proposal
        ), mock.patch.object(
            e18, "place_induced_component", return_value=states[0]
        ) as placer, mock.patch.object(
            e18.e15, "_lab_pair_matrices", return_value=_zeros()
        ), mock.patch.object(e18, "EXPANSION_CAP", 2):
            _retained, evaluations, _rounds, cap_hit, _origins = e18.absolute_path_beam(
                *_zeros(), _tiles(), self.graph
            )
        self.assertEqual(evaluations, 2)
        self.assertTrue(cap_hit)
        self.assertEqual(placer.call_count, 1)

    def test_expansion_cap_remains_global_across_attachment_rounds(self) -> None:
        state = _root_state(self.graph)
        proposal = ((1, 0, 1, (0,), 1.0, 1.0),)
        with mock.patch.object(
            e18, "initial_absolute_states", return_value=(state,)
        ), mock.patch.object(
            e18, "induced_translations", return_value=proposal
        ), mock.patch.object(
            e18, "place_induced_component", return_value=state
        ) as placer, mock.patch.object(
            e18.e15, "_lab_pair_matrices", return_value=_zeros()
        ), mock.patch.object(e18, "EXPANSION_CAP", 2):
            _retained, evaluations, rounds, cap_hit, _origins = e18.absolute_path_beam(
                *_zeros(), _tiles(), self.graph
            )
        self.assertEqual(evaluations, 2)
        self.assertEqual(rounds, 1)
        self.assertTrue(cap_hit)
        self.assertEqual(placer.call_count, 1)

    def test_dead_end_keeps_rigid_state_instead_of_exploding_it(self) -> None:
        state = _root_state(self.graph)
        with mock.patch.object(e18, "initial_absolute_states", return_value=(state,)), mock.patch.object(
            e18, "induced_translations", return_value=()
        ), mock.patch.object(
            e18.e15, "_lab_pair_matrices", return_value=_zeros()
        ):
            retained, *_rest = e18.absolute_path_beam(
                *_zeros(), _tiles(), self.graph
            )
        self.assertEqual(retained[0].translations, state.translations)
        self.assertTrue(np.array_equal(retained[0].board, state.board))

    def test_solver_completion_locks_rigid_core_and_returns_strict_board(self) -> None:
        state = _root_state(self.graph, (0, 0))
        completed = _valid_completion(state.board)
        residual = SimpleNamespace(
            wave_commits=3, wave_rounds=2, hungarian_rounds=2
        )
        with mock.patch.object(e18, "build_graph_data", return_value=self.graph), mock.patch.object(
            e18,
            "absolute_path_beam",
            return_value=((state,), 12, 1, False, 23 * 24),
        ), mock.patch.object(
            e18.e15, "complete_residual", return_value=(completed, residual)
        ) as completion, mock.patch.object(
            e18.e15, "terminal_neural_objective", return_value=1.0
        ), mock.patch.object(
            e18, "lab_depth1_board_score", return_value=-1.0
        ):
            result = e18.solve_absolute_frame(*_zeros(), _tiles())
        partial = completion.call_args.args[0]
        self.assertEqual(int(partial[0, 0]), 0)
        self.assertEqual(int(partial[1, 0]), 1)
        self.assertTrue(np.array_equal(result.board, completed))
        self.assertFalse(result.board.flags.writeable)
        self.assertEqual(result.diagnostics.hungarian_rounds, 2)
        self.assertEqual(result.diagnostics.rigid_tiles_placed, 2)
        self.assertEqual(result.diagnostics.proposal_evaluations, 12)

    def test_solver_rejects_cap_hit_before_residual_completion(self) -> None:
        state = _root_state(self.graph)
        with mock.patch.object(
            e18, "build_graph_data", return_value=self.graph
        ), mock.patch.object(
            e18,
            "absolute_path_beam",
            return_value=((state,), 500_000, 3, True, 1),
        ), mock.patch.object(e18.e15, "complete_residual") as completion:
            with self.assertRaisesRegex(e18.AbsoluteFrameError, "proposal cap"):
                e18.solve_absolute_frame(*_zeros(), _tiles())
        completion.assert_not_called()

    def test_solver_rejects_completion_that_moves_locked_core(self) -> None:
        state = _root_state(self.graph, (0, 0))
        residual = SimpleNamespace(
            wave_commits=0, wave_rounds=0, hungarian_rounds=2
        )
        with mock.patch.object(
            e18, "build_graph_data", return_value=self.graph
        ), mock.patch.object(
            e18,
            "absolute_path_beam",
            return_value=((state,), 1, 1, False, 1),
        ), mock.patch.object(
            e18.e15,
            "complete_residual",
            return_value=(np.arange(576, dtype=np.int64), residual),
        ):
            with self.assertRaisesRegex(e18.AbsoluteFrameError, "locked rigid core"):
                e18.solve_absolute_frame(*_zeros(), _tiles())

    def test_solver_rejects_completion_without_two_hungarian_rounds(self) -> None:
        state = _root_state(self.graph, (0, 0))
        residual = SimpleNamespace(
            wave_commits=574, wave_rounds=1, hungarian_rounds=0
        )
        with mock.patch.object(
            e18, "build_graph_data", return_value=self.graph
        ), mock.patch.object(
            e18,
            "absolute_path_beam",
            return_value=((state,), 1, 1, False, 1),
        ), mock.patch.object(
            e18.e15,
            "complete_residual",
            return_value=(_valid_completion(state.board), residual),
        ):
            with self.assertRaisesRegex(e18.AbsoluteFrameError, "two Hungarian"):
                e18.solve_absolute_frame(*_zeros(), _tiles())

    def test_cross_lab_precedes_terminal_objectives_in_final_rank(self) -> None:
        state = _root_state(self.graph)
        stronger_partial = e18.PartialState(
            **{**state.__dict__, "cross_lab_sum": 1.0}
        )
        board = _valid_completion(state.board)
        first = e18.ResidualCandidate(
            state=stronger_partial,
            board=board,
            wave_commits=0,
            wave_rounds=0,
            hungarian_rounds=2,
            terminal_neural_objective=-100.0,
            terminal_lab_tie_score=-100.0,
        )
        second = e18.ResidualCandidate(
            state=state,
            board=board,
            wave_commits=0,
            wave_rounds=0,
            hungarian_rounds=2,
            terminal_neural_objective=100.0,
            terminal_lab_tie_score=100.0,
        )
        self.assertGreater(e18._final_rank(first), e18._final_rank(second))


if __name__ == "__main__":
    unittest.main()
