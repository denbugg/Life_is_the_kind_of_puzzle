from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_consensus_component_arm import (
    CONSENSUS_ARM_NAMES,
    build_taska_consensus_bonds,
    solve_taska_consensus_component_arm,
    taska_layout_bonds,
)


def _layouts(*values: np.ndarray) -> dict[str, np.ndarray]:
    return dict(zip(CONSENSUS_ARM_NAMES, values, strict=True))


def test_layout_bonds_have_fixed_right_then_down_order() -> None:
    layout = np.arange(9, dtype=np.int32)
    assert taska_layout_bonds(layout, grid=3) == (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(1, 2, "right"),
        RawTailEdge(3, 4, "right"),
        RawTailEdge(4, 5, "right"),
        RawTailEdge(6, 7, "right"),
        RawTailEdge(7, 8, "right"),
        RawTailEdge(0, 3, "down"),
        RawTailEdge(1, 4, "down"),
        RawTailEdge(2, 5, "down"),
        RawTailEdge(3, 6, "down"),
        RawTailEdge(4, 7, "down"),
        RawTailEdge(5, 8, "down"),
    )


def test_consensus_filters_support_one_and_uses_fixed_lexicographic_order() -> None:
    base = np.arange(9, dtype=np.int32)
    alternate = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 6], dtype=np.int32)
    right = np.zeros((9, 9), dtype=np.float64)
    down = np.zeros((9, 9), dtype=np.float64)
    # Among support-four bonds, lower raw cost must precede higher raw cost.
    right[0, 1] = -3.0
    right[1, 2] = -2.0
    bonds = build_taska_consensus_bonds(
        _layouts(base, base, base, alternate),
        right,
        down,
        grid=3,
    )

    assert bonds[0].edge == RawTailEdge(0, 1, "right")
    assert bonds[0].support == 4
    assert bonds[1].edge == RawTailEdge(1, 2, "right")
    assert all(bond.support >= 2 for bond in bonds)
    assert RawTailEdge(8, 6, "right") not in {bond.edge for bond in bonds}
    assert [bond.support for bond in bonds] == sorted(
        (bond.support for bond in bonds), reverse=True
    )


def test_identical_arms_recover_layout_and_tail_is_strict_read_only() -> None:
    layout = np.asarray([8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
    generator = np.random.default_rng(193)
    right = generator.normal(size=(9, 9))
    down = generator.normal(size=(9, 9))
    result = solve_taska_consensus_component_arm(
        _layouts(layout, layout, layout, layout),
        right,
        down,
        grid=3,
    )

    assert np.array_equal(result.component.layout, layout)
    assert np.array_equal(result.layout, layout)
    assert np.array_equal(np.sort(result.layout), np.arange(9))
    assert not result.layout.flags.writeable
    assert result.diagnostics.consensus_edge_count == 12
    assert result.diagnostics.support_counts == ((4, 12),)


def test_arm_roster_and_layout_permutations_are_fail_closed() -> None:
    layout = np.arange(4, dtype=np.int32)
    costs = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="exactly"):
        build_taska_consensus_bonds({"raw": layout}, costs, costs, grid=2)
    malformed = _layouts(layout, layout, layout, np.asarray([0, 1, 1, 3]))
    with pytest.raises(ValueError, match="every original tile"):
        build_taska_consensus_bonds(malformed, costs, costs, grid=2)
