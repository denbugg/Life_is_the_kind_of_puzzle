from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_edge_calibrator import solve_prioritized_raw_tail_global
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES, SOLVER_CONFIG
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    FUSION_ARM_NAMES,
    SELECTIVE_ARM,
    compose_selective_fullres_fusion,
    fuse_unique_fullres_supply,
)


def edge(source: int, target: int, axis: str = "right") -> RawTailEdge:
    return RawTailEdge(source, target, axis)


def test_unique_fullres_filter_preserves_edge_logit_alignment() -> None:
    current = (edge(0, 1),)
    selective = (edge(2, 3, "down"),)
    fullres = (
        edge(0, 1),
        edge(4, 5),
        edge(2, 3, "down"),
        edge(6, 7, "down"),
    )
    supply = fuse_unique_fullres_supply(
        current_edges=current,
        current_logits=[0.1],
        selective_new_edges=selective,
        selective_new_logits=[0.2],
        fullres_accepted_edges=fullres,
        fullres_accepted_logits=[1.0, 2.0, 3.0, 4.0],
    )
    assert supply.unique_fullres_edges == (fullres[1], fullres[3])
    np.testing.assert_array_equal(supply.unique_fullres_logits, [2.0, 4.0])
    assert supply.combined_union_edges == current + selective + (
        fullres[1],
        fullres[3],
    )
    assert supply.fullres_overlap_current_count == 1
    assert supply.fullres_overlap_selective_count == 1


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


def _tiny_frozen_control() -> np.ndarray:
    right, down, four, current, current_logits, selective, selective_logits = _tiny_inputs()
    union = current + selective
    logits = np.concatenate((current_logits, selective_logits))
    solved = solve_prioritized_raw_tail_global(
        right,
        down,
        union,
        logits,
        grid=2,
        config=SOLVER_CONFIG,
    )
    selection = select_lowest_taska_seam_cost_layout(
        {**four, SELECTIVE_ARM: solved.layout}, right, down, grid=2
    )
    uses_union = selection.choice == SELECTIVE_ARM
    return polish_taska_tail_with_focal_gate(
        selection.layout,
        right,
        down,
        union if uses_union else current,
        logits if uses_union else current_logits,
        grid=2,
    ).layout


def test_composition_replays_control_and_emits_strict_fixed_roster() -> None:
    right, down, four, current, current_logits, selective, selective_logits = _tiny_inputs()
    result = compose_selective_fullres_fusion(
        cost_right=right,
        cost_down=down,
        four_layouts=four,
        frozen_selective_control=_tiny_frozen_control(),
        current_edges=current,
        current_logits=current_logits,
        selective_new_edges=selective,
        selective_new_logits=selective_logits,
        fullres_accepted_edges=(selective[0], edge(1, 3, "down")),
        fullres_accepted_logits=[2.0, 3.0],
        grid=2,
    )
    assert tuple(name for name, _ in result.costs) == FUSION_ARM_NAMES
    assert COMBINED_ARM in dict(result.costs)
    np.testing.assert_array_equal(result.control_layout, _tiny_frozen_control())
    np.testing.assert_array_equal(np.sort(result.candidate_layout), np.arange(4))


def test_composition_fails_closed_on_control_replay_mismatch() -> None:
    right, down, four, current, current_logits, selective, selective_logits = _tiny_inputs()
    with pytest.raises(RuntimeError, match="control replay mismatch"):
        compose_selective_fullres_fusion(
            cost_right=right,
            cost_down=down,
            four_layouts=four,
            frozen_selective_control=np.roll(_tiny_frozen_control(), 1),
            current_edges=current,
            current_logits=current_logits,
            selective_new_edges=selective,
            selective_new_logits=selective_logits,
            fullres_accepted_edges=(edge(1, 3, "down"),),
            fullres_accepted_logits=[3.0],
            grid=2,
        )


def test_fixed_production_arm_order_is_unchanged() -> None:
    assert (*ARM_NAMES, SELECTIVE_ARM, COMBINED_ARM) == FUSION_ARM_NAMES
