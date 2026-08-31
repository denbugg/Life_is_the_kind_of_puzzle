from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.socket_decoder import SocketEdge
from aiijc_puzzle.taska_twin_unique_supply import (
    accept_twin_unique_edges,
    filter_twin_nominated_unique_edges,
)


def _socket(source: int, target: int, axis: str, confidence: float) -> SocketEdge:
    delta = (0, 1) if axis == "right" else (1, 0)
    return SocketEdge(source, target, delta[0], delta[1], confidence, axis)


def test_twin_only_filter_applies_budget_before_all_exclusions() -> None:
    learned = {
        "right": (
            _socket(0, 1, "right", 4.0),
            _socket(1, 2, "right", 3.0),
            _socket(2, 3, "right", 2.0),
        ),
        "down": (
            _socket(4, 5, "down", 4.0),
            _socket(5, 6, "down", 3.0),
            _socket(6, 7, "down", 2.0),
        ),
    }
    union = tuple(
        RawTailEdge(edge.source, edge.target, axis)
        for axis, values in learned.items()
        for edge in values
    )
    result = filter_twin_nominated_unique_edges(
        learned_edges_by_axis=learned,
        immutable_union_edges=union,
        twin_top_edges=(
            RawTailEdge(0, 1, "right"),
            RawTailEdge(1, 2, "right"),
            RawTailEdge(5, 6, "down"),
        ),
        excluded_edges=(
            RawTailEdge(0, 1, "right"),
            RawTailEdge(1, 2, "right"),
        ),
        budget_per_axis=2,
    )
    assert result == (RawTailEdge(5, 6, "down"),)


def test_twin_only_filter_rejects_projection_outside_immutable_union() -> None:
    with pytest.raises(RuntimeError, match="escaped"):
        filter_twin_nominated_unique_edges(
            learned_edges_by_axis={
                "right": (_socket(0, 1, "right", 1.0),),
                "down": (),
            },
            immutable_union_edges=(),
            twin_top_edges=(RawTailEdge(0, 1, "right"),),
            excluded_edges=(),
            budget_per_axis=1,
        )


def test_focal_acceptance_is_exactly_nonnegative_and_order_preserving() -> None:
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(2, 3, "down"),
        RawTailEdge(4, 5, "right"),
    )
    accepted, logits = accept_twin_unique_edges(
        edges,
        np.asarray([-1e-6, 0.0, 2.0], dtype=np.float32),
    )
    assert accepted == edges[1:]
    np.testing.assert_array_equal(logits, np.asarray([0.0, 2.0], dtype=np.float32))
