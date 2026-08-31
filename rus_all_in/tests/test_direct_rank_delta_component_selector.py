from __future__ import annotations

import pytest

from aiijc_puzzle.direct_rank_delta_component_selector import (
    ComponentConsistencyEvidence,
    select_direct_rank_delta_component_arm,
)


def _evidence(consistent: int, largest: int, *, count: int = 576) -> ComponentConsistencyEvidence:
    return ComponentConsistencyEvidence(
        consistent_redundant_constraints=consistent,
        largest_component=largest,
        tile_count=count,
    )


def test_more_consistent_cycles_select_rank_delta_even_with_smaller_component() -> None:
    decision = select_direct_rank_delta_component_arm(
        _evidence(7, 100),
        _evidence(8, 20),
    )
    assert decision.selected_arm == "rank_delta_transfer"
    assert decision.reason == "more_consistent_redundant_constraints"


def test_larger_component_breaks_consistent_count_tie() -> None:
    decision = select_direct_rank_delta_component_arm(
        _evidence(8, 100),
        _evidence(8, 101),
    )
    assert decision.selected_arm == "rank_delta_transfer"
    assert decision.reason == "consistent_tie_larger_component"


@pytest.mark.parametrize(
    ("baseline", "treatment"),
    [
        (_evidence(8, 100), _evidence(8, 100)),
        (_evidence(8, 100), _evidence(7, 576)),
        (_evidence(8, 100), _evidence(8, 99)),
    ],
)
def test_nonpositive_evidence_reverts_to_union(
    baseline: ComponentConsistencyEvidence,
    treatment: ComponentConsistencyEvidence,
) -> None:
    decision = select_direct_rank_delta_component_arm(baseline, treatment)
    assert decision.selected_arm == "union_v2"
    assert decision.reason == "union_conservative_fallback"


def test_report_freezes_rule_and_contains_no_layout_or_target() -> None:
    decision = select_direct_rank_delta_component_arm(
        _evidence(8, 100),
        _evidence(8, 101),
    )
    report = decision.report()
    assert report["schema"] == "aiijc-direct-rank-delta-component-selector-v1"
    assert report["selected_arm"] == "rank_delta_transfer"
    assert "target" not in str(report).lower()
    assert "layout" not in str(report).lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"consistent_redundant_constraints": -1, "largest_component": 1, "tile_count": 4},
        {"consistent_redundant_constraints": 0, "largest_component": 0, "tile_count": 4},
        {"consistent_redundant_constraints": 0, "largest_component": 5, "tile_count": 4},
        {"consistent_redundant_constraints": True, "largest_component": 1, "tile_count": 4},
    ],
)
def test_invalid_evidence_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ComponentConsistencyEvidence(**kwargs)


def test_arms_must_describe_same_board_size() -> None:
    with pytest.raises(ValueError, match="tile_count"):
        select_direct_rank_delta_component_arm(
            _evidence(1, 2, count=4),
            _evidence(2, 3, count=9),
        )
