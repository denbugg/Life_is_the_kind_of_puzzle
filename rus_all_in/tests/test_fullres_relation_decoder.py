from __future__ import annotations

import numpy as np

from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationCandidate,
    RelationContact,
)
from aiijc_puzzle.fullres_relation_decoder import build_fusion_forest_inputs


def _candidate(source: int, target: int, baseline: float) -> ComponentRelationCandidate:
    return ComponentRelationCandidate(
        source_component=source,
        target_component=target,
        direction="right",
        target_row_offset=0,
        target_column_offset=1,
        contacts=(
            RelationContact(
                source_tile=source,
                target_tile=target,
                features=(0.0,) * 8,
            ),
        ),
        proposal_count=1,
        baseline_score=baseline,
    )


def test_forest_adapter_can_select_a_restored_only_relation() -> None:
    raw = _candidate(0, 1, 2.0)
    restored_only = _candidate(0, 2, 1.0)
    candidates = (raw, restored_only)
    result = build_fusion_forest_inputs(
        candidates,
        np.asarray([0.5, 3.0]),
        np.asarray([-2.0, 2.0]),
        raw_candidate_keys=frozenset({raw.relation_key}),
        board_id="capacity",
    )
    assert len(result.rows) == 1
    assert result.rows[0].learned_top_candidate == 1
    assert result.diagnostics["restored_only_query_winners"] == 1
    assert result.probabilities[0] > 0.8


def test_forest_adapter_is_deterministic_under_equal_scores() -> None:
    candidates = (_candidate(0, 2, 1.0), _candidate(0, 1, 1.0))
    first = build_fusion_forest_inputs(
        candidates,
        np.zeros(2),
        np.zeros(2),
        raw_candidate_keys=frozenset(),
        board_id="tie",
    )
    second = build_fusion_forest_inputs(
        candidates,
        np.zeros(2),
        np.zeros(2),
        raw_candidate_keys=frozenset(),
        board_id="tie",
    )
    assert first.rows == second.rows
    assert first.diagnostics == second.diagnostics
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    assert first.rows[0].learned_top_candidate == 1
