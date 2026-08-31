from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.taska_six_arm_consensus_selector import (
    directed_adjacencies,
    select_adjacency_consensus_layout,
)


def test_directed_adjacencies_has_exact_pair_denominator() -> None:
    layout = np.arange(9, dtype=np.int32)
    edges = directed_adjacencies(layout, grid=3)
    assert len(edges) == 12
    assert ("right", 0, 1) in edges
    assert ("down", 0, 3) in edges


def test_fallback_wins_an_exact_consensus_tie() -> None:
    first = np.arange(9, dtype=np.int32)
    second = np.roll(first, 1)
    result = select_adjacency_consensus_layout(
        {"first": first, "second": second},
        second,
        grid=3,
    )
    assert result.choice == "second"
    np.testing.assert_array_equal(result.layout, second)


def test_selector_rejects_non_permutation() -> None:
    invalid = np.zeros(9, dtype=np.int32)
    with pytest.raises(ValueError, match="strict"):
        select_adjacency_consensus_layout(
            {"invalid": invalid},
            np.arange(9, dtype=np.int32),
            grid=3,
        )
