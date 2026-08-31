from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_drunet_unique_supply import (
    _mutual_argmax_edges,
    accept_unique_drunet_proposals,
    compose_drunet_unique_fusion,
    unique_drunet_proposals,
)
from aiijc_puzzle.taska_edge_calibrator import solve_prioritized_raw_tail_global
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import SOLVER_CONFIG
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    SELECTIVE_ARM,
    compose_selective_fullres_fusion,
)


def edge(source: int, target: int, axis: str = "right") -> RawTailEdge:
    return RawTailEdge(source, target, axis)


def test_mutual_nominator_requires_row_and_column_top_one() -> None:
    scores = np.asarray(
        [
            [-4.0, 3.0, 1.0, 0.0],
            [0.0, -4.0, 5.0, 4.0],
            [0.0, 2.0, -4.0, 1.0],
            [0.0, 1.0, 2.0, -4.0],
        ]
    )
    assert _mutual_argmax_edges(scores, axis="right") == (
        edge(0, 1),
        edge(1, 2),
    )


def test_unique_filter_drops_all_three_parent_supplies_in_frozen_order() -> None:
    nominated = (
        edge(0, 1),
        edge(2, 3),
        edge(1, 3, "down"),
        edge(0, 2, "down"),
    )
    result = unique_drunet_proposals(
        nominated_edges=nominated,
        current_edges=(nominated[0],),
        selective_edges=(nominated[1],),
        fullres_edges=(nominated[2],),
    )
    assert result.unique_edges == (nominated[3],)
    assert result.overlap_current_count == 1
    assert result.overlap_selective_count == 1
    assert result.overlap_fullres_count == 1


def test_focal_zero_gate_is_inclusive_and_alignment_preserved() -> None:
    proposals = (edge(0, 1), edge(1, 2), edge(2, 3))
    accepted, logits = accept_unique_drunet_proposals(
        proposals, np.asarray([-0.1, 0.0, 1.2], dtype=np.float32)
    )
    assert accepted == proposals[1:]
    np.testing.assert_array_equal(logits, np.asarray([0.0, 1.2], dtype=np.float32))


def _tiny_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    tuple[RawTailEdge, ...],
    np.ndarray,
    tuple[RawTailEdge, ...],
    np.ndarray,
]:
    right = np.asarray(
        [
            [9.0, 0.1, 3.0, 2.0],
            [3.0, 9.0, 0.2, 2.0],
            [2.0, 3.0, 9.0, 0.1],
            [0.2, 2.0, 3.0, 9.0],
        ]
    )
    down = right.T.copy()
    four = {
        "raw": np.asarray([0, 1, 2, 3]),
        "logistic": np.asarray([1, 0, 3, 2]),
        "focal_top5": np.asarray([2, 3, 0, 1]),
        "nonlinear": np.asarray([3, 2, 1, 0]),
    }
    current = (edge(0, 1), edge(2, 3))
    current_logits = np.asarray([1.0, 1.0], dtype=np.float32)
    selective = (edge(0, 2, "down"),)
    selective_logits = np.asarray([1.0], dtype=np.float32)
    return right, down, four, current, current_logits, selective, selective_logits


def _frozen_selective_control() -> np.ndarray:
    right, down, four, current, current_logits, selective, selective_logits = _tiny_inputs()
    union = current + selective
    logits = np.concatenate((current_logits, selective_logits))
    solved = solve_prioritized_raw_tail_global(
        right, down, union, logits, grid=2, config=SOLVER_CONFIG
    )
    selection = select_lowest_taska_seam_cost_layout(
        {**four, SELECTIVE_ARM: solved.layout}, right, down, grid=2
    )
    return polish_taska_tail_with_focal_gate(
        selection.layout,
        right,
        down,
        union if selection.choice == SELECTIVE_ARM else current,
        logits if selection.choice == SELECTIVE_ARM else current_logits,
        grid=2,
    ).layout


def _frozen_fullres_control() -> np.ndarray:
    right, down, four, current, current_logits, selective, selective_logits = _tiny_inputs()
    return compose_selective_fullres_fusion(
        cost_right=right,
        cost_down=down,
        four_layouts=four,
        frozen_selective_control=_frozen_selective_control(),
        current_edges=current,
        current_logits=current_logits,
        selective_new_edges=selective,
        selective_new_logits=selective_logits,
        fullres_accepted_edges=(edge(1, 3, "down"),),
        fullres_accepted_logits=[2.0],
        grid=2,
    ).candidate_layout


def test_composition_replays_parent_and_keeps_six_arm_roster() -> None:
    right, down, four, current, current_logits, selective, selective_logits = _tiny_inputs()
    result = compose_drunet_unique_fusion(
        cost_right=right,
        cost_down=down,
        four_layouts=four,
        frozen_selective_control=_frozen_selective_control(),
        frozen_fullres_fusion_control=_frozen_fullres_control(),
        current_edges=current,
        current_logits=current_logits,
        selective_new_edges=selective,
        selective_new_logits=selective_logits,
        fullres_accepted_edges=(edge(1, 3, "down"),),
        fullres_accepted_logits=[2.0],
        drunet_accepted_edges=(edge(3, 1),),
        drunet_accepted_logits=[1.5],
        grid=2,
    )
    np.testing.assert_array_equal(result.control_layout, _frozen_fullres_control())
    assert tuple(name for name, _ in result.costs)[-1] == COMBINED_ARM
    np.testing.assert_array_equal(np.sort(result.candidate_layout), np.arange(4))


def test_composition_rejects_parent_overlap() -> None:
    right, down, four, current, current_logits, selective, selective_logits = _tiny_inputs()
    with pytest.raises(ValueError, match="unique to all parent"):
        compose_drunet_unique_fusion(
            cost_right=right,
            cost_down=down,
            four_layouts=four,
            frozen_selective_control=_frozen_selective_control(),
            frozen_fullres_fusion_control=_frozen_fullres_control(),
            current_edges=current,
            current_logits=current_logits,
            selective_new_edges=selective,
            selective_new_logits=selective_logits,
            fullres_accepted_edges=(edge(1, 3, "down"),),
            fullres_accepted_logits=[2.0],
            drunet_accepted_edges=(current[0],),
            drunet_accepted_logits=[1.5],
            grid=2,
        )
