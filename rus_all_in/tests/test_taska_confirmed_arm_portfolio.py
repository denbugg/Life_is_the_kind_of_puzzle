from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_confirmed_arm_portfolio import (
    CONFIRMED_ARM_NAMES,
    FULLRES_ARM,
    compose_confirmed_arm_portfolio,
)
from aiijc_puzzle.taska_focal_gated_protected_tail import polish_taska_tail_with_focal_gate
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    FUSION_ARM_NAMES,
    SELECTIVE_ARM,
)


def edge(source: int, target: int, axis: str = "right") -> RawTailEdge:
    return RawTailEdge(source, target, axis)


def _inputs() -> dict[str, object]:
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
    current = (edge(0, 1),)
    selective = current + (edge(2, 3),)
    combined = selective + (edge(0, 2, "down"),)
    fullres = current + (edge(1, 3, "down"),)
    selective_layout = np.asarray([0, 2, 1, 3])
    combined_layout = np.asarray([0, 1, 3, 2])
    fullres_layout = np.asarray([3, 1, 2, 0])
    control_layouts = {
        **four,
        SELECTIVE_ARM: selective_layout,
        COMBINED_ARM: combined_layout,
    }
    selection = select_lowest_taska_seam_cost_layout(
        control_layouts, right, down, grid=2
    )
    supplies = {
        **{name: current for name in four},
        SELECTIVE_ARM: selective,
        COMBINED_ARM: combined,
    }
    frozen_control = polish_taska_tail_with_focal_gate(
        selection.layout,
        right,
        down,
        supplies[selection.choice],
        np.ones(len(supplies[selection.choice]), dtype=np.float32),
        grid=2,
    ).layout
    return {
        "cost_right": right,
        "cost_down": down,
        "four_layouts": four,
        "selective_union_layout": selective_layout,
        "combined_union_layout": combined_layout,
        "fullres_union_layout": fullres_layout,
        "frozen_fusion_control": frozen_control,
        "current_edges": current,
        "current_logits": np.ones(len(current)),
        "selective_union_edges": selective,
        "selective_union_logits": np.ones(len(selective)),
        "combined_union_edges": combined,
        "combined_union_logits": np.ones(len(combined)),
        "fullres_union_edges": fullres,
        "fullres_union_logits": np.ones(len(fullres)),
        "grid": 2,
    }


def test_fixed_seven_arm_portfolio_replays_six_arm_control() -> None:
    result = compose_confirmed_arm_portfolio(**_inputs())
    assert tuple(name for name, _ in result.control_costs) == FUSION_ARM_NAMES
    assert tuple(name for name, _ in result.costs) == CONFIRMED_ARM_NAMES
    assert FULLRES_ARM in dict(result.costs)
    np.testing.assert_array_equal(result.control_layout, result.mechanical_control_layout)
    np.testing.assert_array_equal(np.sort(result.candidate_layout), np.arange(4))


def test_fixed_portfolio_fails_closed_on_control_mismatch() -> None:
    values = _inputs()
    values["frozen_fusion_control"] = np.roll(values["frozen_fusion_control"], 1)
    with pytest.raises(RuntimeError, match="six-arm fusion control replay mismatch"):
        compose_confirmed_arm_portfolio(**values)

