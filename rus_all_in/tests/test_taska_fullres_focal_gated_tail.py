from __future__ import annotations

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_fullres_focal_gated_tail import (
    FOCAL_PROTECTION_MAX_SWAPS,
    polish_fullres_winner_with_focal_gate,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(87)
    return (
        np.arange(16, dtype=np.int32),
        generator.normal(size=(16, 16)),
        generator.normal(size=(16, 16)),
    )


def test_old_winner_matches_current_only_focal_gate() -> None:
    layout, right, down = _inputs()
    current = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(2, 3, "right"),
        RawTailEdge(4, 8, "down"),
    )
    new = (RawTailEdge(5, 9, "down"),)
    current_logits = np.asarray([1.0, -1.0, 0.0])
    result = polish_fullres_winner_with_focal_gate(
        layout,
        right,
        down,
        current,
        current_logits,
        new,
        np.asarray([2.0]),
        winner_is_fullres=False,
        grid=4,
    )
    literal = polish_taska_tail_with_focal_gate(
        layout, right, down, current, current_logits, grid=4
    )
    assert np.array_equal(result.layout, literal.layout)
    assert result.diagnostics.protection_input == "current_only"
    assert result.diagnostics.focal_gate.harvested_edge_count == len(current)


def test_fullres_winner_matches_union_focal_gate() -> None:
    layout, right, down = _inputs()
    current = (RawTailEdge(0, 1, "right"), RawTailEdge(4, 8, "down"))
    new = (RawTailEdge(2, 3, "right"), RawTailEdge(5, 9, "down"))
    current_logits = np.asarray([1.0, -1.0])
    new_logits = np.asarray([0.0, 2.0])
    result = polish_fullres_winner_with_focal_gate(
        layout,
        right,
        down,
        current,
        current_logits,
        new,
        new_logits,
        winner_is_fullres=True,
        grid=4,
    )
    literal = polish_taska_tail_with_focal_gate(
        layout,
        right,
        down,
        current + new,
        np.concatenate((current_logits, new_logits)),
        grid=4,
    )
    assert np.array_equal(result.layout, literal.layout)
    assert np.array_equal(np.sort(result.layout), np.arange(16))
    assert result.diagnostics.protection_input == "current_plus_accepted_new"
    assert result.diagnostics.focal_gate.harvested_edge_count == 4


def test_composition_exposes_no_new_threshold_or_budget() -> None:
    assert FOCAL_PROTECTION_LOGIT_THRESHOLD == 0.0
    assert FOCAL_PROTECTION_MAX_SWAPS == 96
