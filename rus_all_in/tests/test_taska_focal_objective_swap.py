import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_objective_swap import (
    propose_one_focal_objective_swap,
)


def test_positive_objective_creates_visible_edge() -> None:
    layout = np.arange(16, dtype=np.int32)
    # Moving tile 1 from position 1 to position 6 creates the requested
    # vertical relation 1 -> 10 without an adjacent-position swap.
    edge = RawTailEdge(1, 10, "down")
    result = propose_one_focal_objective_swap(
        layout,
        (edge,),
        np.asarray([2.0]),
        objective="positive_softplus",
        grid=4,
    )
    assert result.changed
    assert result.objective_gain > 0.0
    positions = np.empty(16, dtype=np.int32)
    positions[result.layout] = np.arange(16)
    assert positions[10] == positions[1] + 4
    assert np.array_equal(np.sort(result.layout), np.arange(16))


def test_realised_positive_edge_is_protected() -> None:
    layout = np.arange(16, dtype=np.int32)
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(2, 7, "down"),
    )
    result = propose_one_focal_objective_swap(
        layout,
        edges,
        np.asarray([1.0, 10.0]),
        objective="signed_logit",
        grid=4,
    )
    positions = np.empty(16, dtype=np.int32)
    positions[result.layout] = np.arange(16)
    assert positions[1] == positions[0] + 1
    assert result.layout[0] == 0
    assert result.layout[1] == 1


def test_invalid_logits_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        propose_one_focal_objective_swap(
            np.arange(16),
            (RawTailEdge(0, 1, "right"),),
            np.asarray([np.nan]),
            objective="signed_logit",
            grid=4,
        )
