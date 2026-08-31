from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_component_relation_anchor import (
    translate_component_with_local_fill,
)
from aiijc_puzzle.taska_cross_arm_component_anchor import (
    anchor_one_component_from_cross_arm_agreement,
)
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES


def _arms(control: np.ndarray, translated: np.ndarray, support: int) -> dict[str, np.ndarray]:
    return {
        name: translated if index < support else control
        for index, name in enumerate(FUSION_ARM_NAMES)
    }


def test_two_distinct_arms_anchor_one_complete_component() -> None:
    control = np.arange(16, dtype=np.int32)
    translated = translate_component_with_local_fill(
        control, (0, 1), 1, 0, grid=4
    )
    result = anchor_one_component_from_cross_arm_agreement(
        control,
        _arms(control, translated, support=2),
        (RawTailEdge(0, 1, "right"),),
        np.asarray([1.0]),
        grid=4,
    )
    assert np.array_equal(result.layout, translated)
    assert result.diagnostics.changed
    assert result.diagnostics.selected_component_size == 2
    assert result.diagnostics.selected_distinct_arm_support == 2
    assert len(np.unique(result.layout)) == 16


def test_one_arm_support_falls_back_bit_for_bit() -> None:
    control = np.arange(16, dtype=np.int32)
    translated = translate_component_with_local_fill(
        control, (0, 1), 1, 0, grid=4
    )
    result = anchor_one_component_from_cross_arm_agreement(
        control,
        _arms(control, translated, support=1),
        (RawTailEdge(0, 1, "right"),),
        np.asarray([1.0]),
        grid=4,
    )
    assert np.array_equal(result.layout, control)
    assert not result.diagnostics.changed


def test_partial_component_agreement_does_not_vote() -> None:
    control = np.arange(16, dtype=np.int32)
    partial = control.copy()
    partial[[0, 4]] = partial[[4, 0]]
    result = anchor_one_component_from_cross_arm_agreement(
        control,
        _arms(control, partial, support=2),
        (RawTailEdge(0, 1, "right"),),
        np.asarray([1.0]),
        grid=4,
    )
    assert np.array_equal(result.layout, control)


def test_roster_and_fixed_support_are_enforced() -> None:
    control = np.arange(16, dtype=np.int32)
    with pytest.raises(ValueError, match="six-arm roster"):
        anchor_one_component_from_cross_arm_agreement(
            control,
            {"raw": control},
            (),
            np.empty(0),
            grid=4,
        )
    with pytest.raises(ValueError, match="two distinct"):
        anchor_one_component_from_cross_arm_agreement(
            control,
            {name: control for name in FUSION_ARM_NAMES},
            (),
            np.empty(0),
            grid=4,
            minimum_distinct_arm_support=3,
        )
