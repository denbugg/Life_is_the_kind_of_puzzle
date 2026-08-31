from __future__ import annotations

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_context_bridge_relocator import relocate_one_weak_bridge_subtree


def test_high_confidence_core_leaves_no_eligible_weak_subtree() -> None:
    layout = np.arange(576, dtype=np.int32)
    result = relocate_one_weak_bridge_subtree(
        layout,
        np.zeros((576, 576), dtype=np.float64),
        np.zeros((576, 576), dtype=np.float64),
        (RawTailEdge(0, 1, "right"), RawTailEdge(1, 2, "right")),
        np.asarray([0.9, 0.2]),
    )
    assert np.array_equal(result.layout, layout)
    assert not result.diagnostics.changed
    assert result.diagnostics.weak_bridge_count == 1
    assert result.diagnostics.eligible_weak_subtree_count == 0


def test_empty_bridge_corpus_is_a_strict_noop() -> None:
    layout = np.arange(576, dtype=np.int32)
    result = relocate_one_weak_bridge_subtree(
        layout,
        np.zeros((576, 576), dtype=np.float64),
        np.zeros((576, 576), dtype=np.float64),
        (),
        np.empty(0),
    )
    assert np.array_equal(result.layout, layout)
    assert result.diagnostics.high_confidence_core_tile_count == 0
