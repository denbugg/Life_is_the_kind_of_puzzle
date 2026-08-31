from __future__ import annotations

import numpy as np

from aiijc_puzzle.component_relation_confidence import (
    FEATURE_NAMES,
    QueryConfidenceFeatures,
    aggregate_confidence_observations,
    build_query_confidence_features,
    calibrated_component_edge_priorities,
    fit_confidence_calibrator,
    relation_forest_score_substitution,
)
from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationCandidate,
    RelationContact,
)
from aiijc_puzzle.component_shift_head import ComponentDescriptor
from aiijc_puzzle.socket_decoder import hard_partial_axis_matching


def _candidate(
    target: int,
    *,
    baseline: float,
    features: tuple[float, ...],
) -> ComponentRelationCandidate:
    return ComponentRelationCandidate(
        source_component=0,
        target_component=target,
        direction="right",
        target_row_offset=0,
        target_column_offset=1,
        contacts=(RelationContact(0, target, features),),
        proposal_count=2,
        baseline_score=baseline,
    )


def test_target_blind_query_features_are_candidate_order_invariant() -> None:
    grid = 2
    components = tuple(
        ComponentDescriptor((tile,), (0,), (0,), float(tile) / 10)
        for tile in range(grid * grid)
    )
    first = _candidate(
        1,
        baseline=0.7,
        features=(1.0, 0.5, 0.4, 0.3, 1.0, 0.5, 0.2, -0.1),
    )
    second = _candidate(
        2,
        baseline=0.2,
        features=(0.1, 0.0, -0.1, 0.2, 0.5, 1.0, -0.2, 0.1),
    )
    expected = build_query_confidence_features(
        np.asarray([1.4, 0.3]),
        (first, second),
        components,
        board_id="board",
        grid=grid,
    )
    observed = build_query_confidence_features(
        np.asarray([0.3, 1.4]),
        (second, first),
        components,
        board_id="board",
        grid=grid,
    )
    assert len(expected) == len(observed) == 1
    np.testing.assert_allclose(expected[0].values, observed[0].values)
    assert len(expected[0].values) == len(FEATURE_NAMES) == 67


def test_tiny_logistic_calibrator_is_portable_and_learns_signal() -> None:
    rows: list[QueryConfidenceFeatures] = []
    labels: list[bool] = []
    for index in range(80):
        positive = index % 2 == 0
        values = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        values[0] = 2.0 if positive else -2.0
        values[1] = (index % 7) / 10
        rows.append(
            QueryConfidenceFeatures(
                board_id=f"board-{index // 4}",
                source_component=index % 4,
                direction="right",
                learned_top_candidate=0,
                raw_top_candidate=0,
                learned_margin=float(values[1]),
                raw_margin=0.0,
                values=tuple(values),
            )
        )
        labels.append(positive)
    calibrator = fit_confidence_calibrator(rows, labels)
    probabilities = calibrator.predict_probabilities([row.values for row in rows])
    assert calibrator.parameter_count == 68
    assert probabilities[np.asarray(labels)].mean() > 0.9
    assert probabilities[~np.asarray(labels)].mean() < 0.1
    restored = type(calibrator).from_dict(calibrator.as_dict())
    np.testing.assert_allclose(
        probabilities,
        restored.predict_probabilities([row.values for row in rows]),
    )


def test_cross_query_metrics_compare_calibrated_learned_top1_to_raw() -> None:
    rows = []
    for board in ("a", "b"):
        for component in range(4):
            learned_correct = component in {0, 1}
            raw_correct = component in {2, 3}
            rows.append(
                {
                    "board_id": board,
                    "source_component": component,
                    "direction": "right",
                    "calibrated_confidence": 0.9 - 0.1 * component,
                    "learned_margin": 0.2,
                    "raw_margin": 0.9 if raw_correct else 0.1,
                    "learned_top1_correct": learned_correct,
                    "raw_top1_correct": raw_correct,
                }
            )
    metrics = aggregate_confidence_observations(rows, caps=(2, 4))
    assert metrics["calibrated"]["high_confidence"]["top2"]["precision"] == 1.0
    assert (
        metrics["raw_socket_component_baseline"]["high_confidence"]["top2"][
            "precision"
        ]
        == 1.0
    )
    assert metrics["calibrated"]["high_confidence"]["top4"]["correct_per_board"] == 2


def test_decoder_priority_can_only_boost_supported_hard_edges() -> None:
    grid = 2
    count = grid * grid

    def assignment(first: tuple[int, int], second: tuple[int, int]) -> np.ndarray:
        value = np.full((count + 1, count + 1), -10.0, dtype=np.float64)
        value[:count, count] = 0.0
        value[count, :count] = 0.0
        value[count, count] = -1e4
        value[first] = 12.0
        value[second] = 8.0
        return value

    right = assignment((0, 1), (2, 3))
    down = assignment((0, 2), (1, 3))
    candidate = _candidate(
        1,
        baseline=0.7,
        features=(1.0, 0.5, 0.4, 0.3, 1.0, 0.5, 0.2, -0.1),
    )
    row = QueryConfidenceFeatures(
        board_id="board",
        source_component=0,
        direction="right",
        learned_top_candidate=0,
        raw_top_candidate=0,
        learned_margin=1.0,
        raw_margin=1.0,
        values=tuple(np.zeros(len(FEATURE_NAMES))),
    )
    priorities, diagnostics = calibrated_component_edge_priorities(
        right,
        down,
        (row,),
        np.asarray([0.9]),
        (candidate,),
        grid=grid,
        top_cap=1,
    )
    matching = hard_partial_axis_matching(right, grid=grid, axis="right")
    base = next(edge.confidence for edge in matching.edges if (edge.source, edge.target) == (0, 1))
    assert priorities["right"][0, 1] > base
    assert diagnostics["boosted_hard_edges"] == 1
    assert priorities["right"][1, 0] == 0.0

    substituted, forest = relation_forest_score_substitution(
        right,
        down,
        (row,),
        np.asarray([0.9]),
        (candidate,),
        grid=grid,
        top_cap=1,
        component_edge_budget_per_axis=1,
    )
    assert forest["accepted_relations"] == 1
    assert forest["accepted_contacts"] == 1
    assert substituted["right"][0, 1] >= right[0, 1]
    np.testing.assert_array_equal(substituted["right"][:count, count], right[:count, count])
    np.testing.assert_array_equal(substituted["down"], down)
