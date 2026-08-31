from __future__ import annotations

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_fullres_translation_consensus import (
    CONSENSUS_MINIMUM,
    translation_consensus_evidence,
)


def _edge(source: int, target: int, axis: str = "right") -> RawTailEdge:
    return RawTailEdge(source, target, axis)  # type: ignore[arg-type]


def test_repeated_component_translation_is_promoted_strictly_above_old_max() -> None:
    costs = np.zeros((9, 9), dtype=np.float64)
    current = (_edge(0, 1), _edge(3, 4))
    unique = (
        _edge(0, 3, "down"),
        _edge(1, 4, "down"),
        _edge(0, 6, "down"),
    )
    original_unique_priorities = np.asarray([1.0, 3.0, 8.0])

    evidence = translation_consensus_evidence(
        cost_right=costs,
        cost_down=costs,
        current_edges=current,
        current_priorities=[10.0, 9.0],
        selective_new_edges=(),
        selective_new_priorities=[],
        unique_fullres_edges=unique,
        unique_fullres_priorities=original_unique_priorities,
        grid=3,
    )

    np.testing.assert_array_equal(evidence.support, [2, 2, 0])
    np.testing.assert_array_equal(evidence.mask, [True, True, False])
    assert evidence.adjusted_unique_priorities.dtype == np.float64
    assert np.all(evidence.adjusted_unique_priorities[:2] > 10.0)
    assert evidence.adjusted_unique_priorities[1] > evidence.adjusted_unique_priorities[0]
    assert evidence.adjusted_unique_priorities[2] == original_unique_priorities[2]
    assert evidence.diagnostics()["consensus_minimum"] == CONSENSUS_MINIMUM


def test_no_repeated_translation_preserves_every_unique_priority() -> None:
    costs = np.zeros((9, 9), dtype=np.float64)
    original = np.asarray([2.0, 4.0])

    evidence = translation_consensus_evidence(
        cost_right=costs,
        cost_down=costs,
        current_edges=(_edge(0, 1), _edge(3, 4)),
        current_priorities=[10.0, 9.0],
        selective_new_edges=(),
        selective_new_priorities=[],
        unique_fullres_edges=(_edge(0, 3, "down"), _edge(0, 6, "down")),
        unique_fullres_priorities=original,
        grid=3,
    )

    assert not np.any(evidence.mask)
    np.testing.assert_array_equal(evidence.support, [1, 0])
    np.testing.assert_array_equal(evidence.adjusted_unique_priorities, original)
