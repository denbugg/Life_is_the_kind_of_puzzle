from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
    FOCAL_PROTECTION_MAX_SWAPS,
    FOCAL_PROTECTION_MINIMUM_GAIN,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail


def _is_realised(layout: np.ndarray, edge: RawTailEdge, *, grid: int) -> bool:
    position = np.empty(len(layout), dtype=np.int32)
    position[layout] = np.arange(len(layout), dtype=np.int32)
    source = int(position[edge.source])
    target = int(position[edge.target])
    if edge.axis == "right":
        return target == source + 1 and target // grid == source // grid
    return target == source + grid


def test_fixed_gate_matches_literal_filtered_control_and_preserves_kept_bonds() -> None:
    generator = np.random.default_rng(83)
    layout = np.arange(16, dtype=np.int32)
    right = generator.normal(size=(16, 16))
    down = generator.normal(size=(16, 16))
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(2, 3, "right"),
        RawTailEdge(4, 8, "down"),
    )
    logits = np.asarray([0.0, -0.01, 1.25], dtype=np.float64)

    result = polish_taska_tail_with_focal_gate(
        layout, right, down, edges, logits, grid=4
    )
    literal = polish_unprotected_taska_tail(
        layout,
        right,
        down,
        (edges[0], edges[2]),
        grid=4,
        max_swaps=96,
        minimum_gain=1e-9,
    )

    assert np.array_equal(result.layout, literal.layout)
    assert np.array_equal(np.sort(result.layout), np.arange(16))
    assert not result.layout.flags.writeable
    assert result.diagnostics.harvested_edge_count == 3
    assert result.diagnostics.focal_kept_edge_count == 2
    assert result.diagnostics.focal_dropped_edge_count == 1
    assert result.diagnostics.focal_logit_threshold == 0.0
    assert result.diagnostics.tail == literal.diagnostics
    assert result.diagnostics.tail.final_total_cost <= (
        result.diagnostics.tail.initial_total_cost
    )
    assert _is_realised(result.layout, edges[0], grid=4)
    assert _is_realised(result.layout, edges[2], grid=4)


def test_fixed_rule_has_no_threshold_or_budget_surface() -> None:
    assert FOCAL_PROTECTION_LOGIT_THRESHOLD == 0.0
    assert FOCAL_PROTECTION_MAX_SWAPS == 96
    assert FOCAL_PROTECTION_MINIMUM_GAIN == 1e-9


@pytest.mark.parametrize(
    ("logits", "match"),
    [
        (np.zeros(0), "shape"),
        (np.asarray([np.nan]), "finite"),
    ],
)
def test_malformed_logits_are_rejected(logits: np.ndarray, match: str) -> None:
    layout = np.arange(4, dtype=np.int32)
    costs = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match=match):
        polish_taska_tail_with_focal_gate(
            layout,
            costs,
            costs,
            (RawTailEdge(0, 1, "right"),),
            logits,
            grid=2,
        )


def test_even_dropped_invalid_edge_is_rejected() -> None:
    layout = np.arange(4, dtype=np.int32)
    costs = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="out-of-range"):
        polish_taska_tail_with_focal_gate(
            layout,
            costs,
            costs,
            (RawTailEdge(0, 9, "right"),),
            np.asarray([-10.0]),
            grid=2,
        )
