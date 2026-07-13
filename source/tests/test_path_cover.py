from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from puzzle_assembly.path_cover import (
    DirectedCandidate,
    PathCoverInfeasibleError,
    exhaustive_path_cover_reference,
    extract_topk_directed_candidates,
    extract_union_directed_candidates,
    path_cover_edges,
    solve_candidate_path_cover,
    solve_path_cover,
    validate_exact_path_cover,
)


ORTOOLS_AVAILABLE = importlib.util.find_spec("ortools") is not None


def _matrix(
    node_count: int, edges: dict[tuple[int, int], float]
) -> np.ndarray:
    values = np.full((node_count, node_count), np.inf, dtype=np.float64)
    np.fill_diagonal(values, np.nan)
    for (source, destination), cost in edges.items():
        values[source, destination] = cost
    return values


def _rank2_false_hub_matrix() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    # Truth is second-ranked in every nonfinal row.  The cheaper false edges
    # reuse destination 2 and cannot jointly form two length-three paths.
    truth = ((0, 1, 2), (3, 4, 5))
    true_edges = {(0, 1), (1, 2), (3, 4), (4, 5)}
    false_edges = {(0, 2), (1, 3), (3, 2), (4, 2)}
    edges = {edge: 1.0 for edge in true_edges}
    edges.update({edge: 0.0 for edge in false_edges})
    return _matrix(6, edges), truth


def _false_cycle_matrix() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    truth = ((0, 1, 2), (3, 4, 5))
    true_edges = {(0, 1), (1, 2), (3, 4), (4, 5)}
    false_cycles = {(0, 3), (3, 0), (1, 4), (4, 1)}
    edges = {edge: 1.0 for edge in true_edges}
    edges.update({edge: 0.0 for edge in false_cycles})
    return _matrix(6, edges), truth


def test_candidate_extraction_is_finite_topk_and_tie_stable() -> None:
    costs = np.asarray(
        [
            [np.nan, 2.0, 1.0, 1.0],
            [np.inf, np.nan, -1.0, 3.0],
            [5.0, np.inf, np.nan, 4.0],
            [np.inf, np.inf, np.inf, np.nan],
        ],
        dtype=np.float64,
    )
    first = extract_topk_directed_candidates(costs, top_k=2)
    second = extract_topk_directed_candidates(costs.copy(), top_k=2)
    assert first == second
    assert [candidate.edge for candidate in first] == [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 0),
    ]
    assert all(np.isfinite(candidate.cost) for candidate in first)
    assert [candidate.outgoing_rank for candidate in first] == [0, 1, 0, 1, 0, 1]


def test_union_extraction_adds_incoming_and_expensive_rescue_edges() -> None:
    costs = np.asarray(
        [
            [np.inf, 0.0, 2.0, 9.0],
            [4.0, np.inf, 0.0, 8.0],
            [3.0, -1.0, np.inf, -2.0],
            [0.0, 5.0, 7.0, np.inf],
        ],
        dtype=np.float64,
    )
    candidates = extract_union_directed_candidates(
        costs,
        outgoing_top_k=1,
        incoming_top_k=1,
        rescue_edges=((0, 3), (0, 1)),
    )
    by_edge = {candidate.edge: candidate for candidate in candidates}
    assert {(0, 1), (1, 2), (2, 3), (3, 0)} <= set(by_edge)
    # 2->1 is incoming-top1 for destination 1 but not outgoing-top1 for node 2.
    assert (2, 1) in by_edge
    assert (0, 3) in by_edge
    regular_max = max(
        candidate.cost for edge, candidate in by_edge.items() if edge != (0, 3)
    )
    assert by_edge[(0, 3)].cost > regular_max
    # Existing edge keeps its observed score rather than rescue penalty.
    assert by_edge[(0, 1)].cost == 0.0


def test_union_extraction_is_input_deterministic() -> None:
    costs = np.ones((5, 5), dtype=np.float64)
    np.fill_diagonal(costs, np.inf)
    first = extract_union_directed_candidates(
        costs,
        outgoing_top_k=2,
        incoming_top_k=2,
        rescue_edges=((4, 3), (3, 4)),
    )
    second = extract_union_directed_candidates(
        costs.copy(),
        outgoing_top_k=2,
        incoming_top_k=2,
        rescue_edges=reversed(((4, 3), (3, 4))),
    )
    assert first == second


@pytest.mark.parametrize(
    ("matrix", "top_k", "error"),
    [
        (np.zeros((2, 3)), 1, ValueError),
        (np.zeros((1, 1)), 1, ValueError),
        (np.zeros((3, 3)), 0, ValueError),
        (np.zeros((3, 3)), 3, ValueError),
        (np.zeros((3, 3)), True, TypeError),
        (np.asarray([[object(), object()], [object(), object()]]), 1, TypeError),
    ],
)
def test_candidate_extraction_rejects_bad_inputs(
    matrix: np.ndarray, top_k: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        extract_topk_directed_candidates(matrix, top_k=top_k)


def test_validator_canonicalizes_path_labels_and_edges() -> None:
    paths = validate_exact_path_cover(
        ((3, 4, 5), (0, 1, 2)),
        node_count=6,
        path_count=2,
        path_length=3,
        allowed_edges={(0, 1), (1, 2), (3, 4), (4, 5)},
    )
    assert paths == ((0, 1, 2), (3, 4, 5))
    assert path_cover_edges(paths) == ((0, 1), (1, 2), (3, 4), (4, 5))


@pytest.mark.parametrize(
    ("paths", "error"),
    [
        (((0, 1, 2),), ValueError),
        (((0, 1), (2, 3, 4)), ValueError),
        (((0, 1, 2), (2, 4, 5)), ValueError),
        (((0, 1, 2), (3, 4, 6)), ValueError),
        (((0, 1, 2), (3, 4, 4.5)), TypeError),
        (((0, 1, 2), (3, 4, True)), TypeError),
    ],
)
def test_validator_rejects_invalid_covers(
    paths: tuple[tuple[object, ...], ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        validate_exact_path_cover(
            paths,
            node_count=6,
            path_count=2,
            path_length=3,
        )


def test_validator_rejects_unavailable_edge() -> None:
    with pytest.raises(ValueError, match="unavailable edge"):
        validate_exact_path_cover(
            ((0, 1, 2), (3, 4, 5)),
            node_count=6,
            path_count=2,
            path_length=3,
            allowed_edges={(0, 1), (1, 2), (3, 4)},
        )


def test_exhaustive_reference_recovers_rank2_truth_behind_false_hubs() -> None:
    costs, truth = _rank2_false_hub_matrix()
    candidates = extract_topk_directed_candidates(costs, top_k=2)
    assert all(
        next(
            candidate.outgoing_rank
            for candidate in candidates
            if candidate.edge == edge
        )
        == 1
        for edge in {(0, 1), (1, 2), (3, 4), (4, 5)}
    )
    result = exhaustive_path_cover_reference(
        candidates,
        node_count=6,
        path_count=2,
        path_length=3,
    )
    assert result.paths == truth
    assert result.accepted_candidate is True
    assert result.used_reference_fallback is False
    assert result.diagnostics["feasible_cover_count"] == 1
    assert result.diagnostics["objective_cost"] == pytest.approx(4.0)


def test_exhaustive_reference_rejects_cheaper_false_cycles() -> None:
    costs, truth = _false_cycle_matrix()
    candidates = extract_topk_directed_candidates(costs, top_k=2)
    result = exhaustive_path_cover_reference(
        candidates,
        node_count=6,
        path_count=2,
        path_length=3,
    )
    assert result.paths == truth
    selected = set(path_cover_edges(result.paths))
    assert not selected.intersection({(0, 3), (3, 0), (1, 4), (4, 1)})


def test_exhaustive_reference_ties_and_input_order_are_deterministic() -> None:
    # Every permutation is feasible and has equal cost.  The reference's
    # canonical lexicographic tie rule must ignore candidate input order.
    costs = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(costs, np.inf)
    candidates = extract_topk_directed_candidates(costs, top_k=3)
    first = exhaustive_path_cover_reference(
        candidates,
        node_count=4,
        path_count=2,
        path_length=2,
    )
    second = exhaustive_path_cover_reference(
        reversed(candidates),
        node_count=4,
        path_count=2,
        path_length=2,
    )
    assert first.paths == second.paths == ((0, 1), (2, 3))


def test_exhaustive_reference_reports_infeasible_graph() -> None:
    candidates = (
        DirectedCandidate(0, 1, 0.0, 0),
        DirectedCandidate(2, 3, 0.0, 0),
    )
    with pytest.raises(PathCoverInfeasibleError):
        exhaustive_path_cover_reference(
            candidates,
            node_count=4,
            path_count=1,
            path_length=4,
        )


@pytest.mark.parametrize(
    "candidate",
    [
        DirectedCandidate(0, 0, 0.0, 0),
        DirectedCandidate(-1, 1, 0.0, 0),
        DirectedCandidate(0, 4, 0.0, 0),
        DirectedCandidate(0, 1, np.inf, 0),
        DirectedCandidate(0, 1, 0.0, -1),
    ],
)
def test_candidate_validator_rejects_invalid_candidates(
    candidate: DirectedCandidate,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        exhaustive_path_cover_reference(
            (candidate,),
            node_count=4,
            path_count=2,
            path_length=2,
        )


def test_candidate_validator_rejects_duplicate_edges() -> None:
    duplicate = (
        DirectedCandidate(0, 1, 0.0, 0),
        DirectedCandidate(0, 1, 1.0, 1),
    )
    with pytest.raises(ValueError, match="duplicate"):
        exhaustive_path_cover_reference(
            duplicate,
            node_count=4,
            path_count=2,
            path_length=2,
        )


def test_shape_contract_rejects_nonfactorization() -> None:
    with pytest.raises(ValueError, match="must equal"):
        validate_exact_path_cover(
            ((0, 1), (2, 3)),
            node_count=5,
            path_count=2,
            path_length=2,
        )


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="optional ortools is not installed")
@pytest.mark.parametrize("fixture", [_rank2_false_hub_matrix, _false_cycle_matrix])
def test_cp_sat_matches_exhaustive_adversarial_fixtures(fixture) -> None:
    costs, truth = fixture()
    candidates = extract_topk_directed_candidates(costs, top_k=2)
    expected = exhaustive_path_cover_reference(
        candidates,
        node_count=6,
        path_count=2,
        path_length=3,
    )
    result = solve_candidate_path_cover(
        reversed(candidates),
        node_count=6,
        path_count=2,
        path_length=3,
        time_limit_seconds=10.0,
    )
    assert result.paths == expected.paths == truth
    assert result.diagnostics["optimal"] is True
    assert result.diagnostics["depth_counts"] == [2, 2, 2]
    assert result.diagnostics["selected_edge_count"] == 4


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="optional ortools is not installed")
def test_cp_sat_ties_are_repeatable_and_candidate_order_invariant() -> None:
    costs = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(costs, np.inf)
    candidates = extract_topk_directed_candidates(costs, top_k=3)
    first = solve_candidate_path_cover(
        candidates,
        node_count=4,
        path_count=2,
        path_length=2,
        time_limit_seconds=10.0,
    )
    second = solve_candidate_path_cover(
        reversed(candidates),
        node_count=4,
        path_count=2,
        path_length=2,
        time_limit_seconds=10.0,
    )
    assert first.paths == second.paths
    assert first.diagnostics["primary_integer_objective"] == 0


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="optional ortools is not installed")
def test_solve_path_cover_wrapper_adds_topk_diagnostic() -> None:
    costs, truth = _rank2_false_hub_matrix()
    result = solve_path_cover(
        costs,
        path_count=2,
        path_length=3,
        outgoing_top_k=2,
        incoming_top_k=2,
        time_limit_seconds=10.0,
    )
    assert result.paths == truth
    assert result.diagnostics["outgoing_top_k"] == 2
    assert result.diagnostics["incoming_top_k"] == 2


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="optional ortools is not installed")
def test_cp_sat_reference_is_hint_and_fail_closed_fallback() -> None:
    costs = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(costs, np.inf)
    reference = ((0, 1), (2, 3))
    result = solve_path_cover(
        costs,
        path_count=2,
        path_length=2,
        outgoing_top_k=3,
        incoming_top_k=3,
        rescue_edges=path_cover_edges(reference),
        reference_paths=reference,
        time_limit_seconds=10.0,
    )
    assert result.paths == reference
    assert result.accepted_candidate is False
    assert result.used_reference_fallback is True
    assert result.fallback_reason == "no_strict_raw_cost_improvement"
    assert result.diagnostics["reference_objective_cost"] == pytest.approx(2.0)


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="optional ortools is not installed")
def test_cp_sat_accepts_strict_cost_improvement_over_reference() -> None:
    reference = ((0, 1, 2), (3, 4, 5))
    better = ((0, 3, 2), (1, 4, 5))
    edges = {
        (0, 1): 4.0,
        (1, 2): 4.0,
        (3, 4): 4.0,
        (4, 5): 4.0,
        (0, 3): 0.0,
        (3, 2): 0.0,
        (1, 4): 0.0,
    }
    costs = _matrix(6, edges)
    result = solve_path_cover(
        costs,
        path_count=2,
        path_length=3,
        outgoing_top_k=2,
        incoming_top_k=2,
        rescue_edges=path_cover_edges(reference),
        reference_paths=reference,
        time_limit_seconds=10.0,
    )
    assert result.paths == better
    assert result.accepted_candidate is True
    assert result.used_reference_fallback is False
    assert result.diagnostics["candidate_objective_cost"] < result.diagnostics[
        "reference_objective_cost"
    ]


def test_reference_feasibility_mode_falls_back_before_solver_at_zero_bound() -> None:
    # This exercises the sound lower-bound short circuit even without optional
    # OR-Tools: non-negative normalized integer costs cannot beat zero.
    reference = ((0, 1), (2, 3))
    candidates = (
        DirectedCandidate(0, 1, 1.0, 0),
        DirectedCandidate(2, 3, 1.0, 0),
        DirectedCandidate(0, 2, 1.0, 1),
        DirectedCandidate(1, 3, 1.0, 0),
    )
    result = solve_candidate_path_cover(
        candidates,
        node_count=4,
        path_count=2,
        path_length=2,
        reference_paths=reference,
        reference_improvement_feasibility=True,
        time_limit_seconds=5.0,
    )
    assert result.paths == reference
    assert result.used_reference_fallback is True
    assert result.fallback_reason == "no_strict_integer_improvement_possible"
    assert result.diagnostics["reference_primary_integer_objective"] == 0
    assert result.diagnostics["deterministic_time"] == 0.0
    assert result.diagnostics["time_limit_kind"] == "cp_sat_deterministic_time"


def test_reference_feasibility_mode_requires_reference_and_strictness() -> None:
    candidate = (DirectedCandidate(0, 1, 0.0, 0),)
    with pytest.raises(ValueError, match="requires reference_paths"):
        solve_candidate_path_cover(
            candidate,
            node_count=2,
            path_count=1,
            path_length=2,
            reference_improvement_feasibility=True,
        )
    with pytest.raises(ValueError, match="requires strict"):
        solve_candidate_path_cover(
            candidate,
            node_count=2,
            path_count=1,
            path_length=2,
            reference_paths=((0, 1),),
            require_strict_reference_improvement=False,
            reference_improvement_feasibility=True,
        )


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="optional ortools is not installed")
def test_reference_feasibility_mode_accepts_deterministic_improving_cover() -> None:
    reference = ((0, 1, 2), (3, 4, 5))
    better = ((0, 3, 2), (1, 4, 5))
    edges = {
        (0, 1): 4.0,
        (1, 2): 4.0,
        (3, 4): 4.0,
        (4, 5): 4.0,
        (0, 3): 0.0,
        (3, 2): 0.0,
        (1, 4): 0.0,
    }
    candidates = extract_union_directed_candidates(
        _matrix(6, edges),
        outgoing_top_k=2,
        incoming_top_k=2,
        rescue_edges=path_cover_edges(reference),
    )
    first = solve_candidate_path_cover(
        candidates,
        node_count=6,
        path_count=2,
        path_length=3,
        reference_paths=reference,
        reference_improvement_feasibility=True,
        time_limit_seconds=10.0,
    )
    second = solve_candidate_path_cover(
        reversed(candidates),
        node_count=6,
        path_count=2,
        path_length=3,
        reference_paths=reference,
        reference_improvement_feasibility=True,
        time_limit_seconds=10.0,
    )
    assert first.paths == second.paths == better
    assert first.accepted_candidate is True
    assert first.diagnostics["candidate_primary_integer_objective"] < first.diagnostics[
        "reference_primary_integer_objective"
    ]
    assert first.diagnostics["candidate_objective_cost"] < first.diagnostics[
        "reference_objective_cost"
    ]
    assert first.diagnostics["best_objective_bound"] is None
    assert first.diagnostics["deterministic_time"] >= 0.0
    assert first.diagnostics["time_limit_kind"] == "cp_sat_deterministic_time"


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="optional ortools is not installed")
def test_reference_feasibility_mode_falls_back_when_no_better_cover_exists() -> None:
    costs, truth = _rank2_false_hub_matrix()
    candidates = extract_topk_directed_candidates(costs, top_k=2)
    result = solve_candidate_path_cover(
        candidates,
        node_count=6,
        path_count=2,
        path_length=3,
        reference_paths=truth,
        reference_improvement_feasibility=True,
        time_limit_seconds=10.0,
    )
    assert result.paths == truth
    assert result.accepted_candidate is False
    assert result.used_reference_fallback is True
    assert result.fallback_reason == "solver_status_INFEASIBLE"
    assert result.diagnostics["candidate_primary_integer_objective"] is None
    assert result.diagnostics["reference_primary_integer_objective"] > 0
