from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.taska_dense_contact_consensus import (
    build_dense_contact_consensus,
)


def _case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = 4
    count = grid * grid
    layout = np.arange(count, dtype=np.int32)
    component = np.arange(count, dtype=np.int32) + 2
    component[[0, 1]] = 0
    component[[8, 9, 12, 13]] = 1
    # Relabel remaining singleton components contiguously.
    labels = {value: index for index, value in enumerate(np.unique(component))}
    component = np.asarray([labels[value] for value in component], dtype=np.int32)
    relative = np.zeros((count, 2), dtype=np.int16)
    relative[0] = (0, 0)
    relative[1] = (0, 1)
    relative[8] = (0, 0)
    relative[9] = (0, 1)
    relative[12] = (1, 0)
    relative[13] = (1, 1)
    indices = np.arange(count * count, dtype=np.float64).reshape(count, count)
    right = 1000.0 + indices
    down = 2000.0 + indices
    # Two reciprocal contacts with the same O_B-O_A=(0,0), one per axis.
    right[0, 9] = -100.0
    down[1, 13] = -100.0
    return layout, component, relative, right, down


def test_fixed_consensus_retains_reciprocal_two_axis_translation() -> None:
    layout, component, relative, right, down = _case()
    board = build_dense_contact_consensus(
        layout=layout,
        component_of_tile=component,
        component_relative_coordinates=relative,
        cost_right=right,
        cost_down=down,
        grid=4,
    )
    emitted = set(
        zip(
            board.edge_source.tolist(),
            board.edge_target.tolist(),
            board.edge_axis.tolist(),
            strict=True,
        )
    )
    assert (0, 9, 0) in emitted
    assert (1, 13, 1) in emitted
    matching = np.flatnonzero(
        (board.group_component_low == component[0])
        & (board.group_component_high == component[9])
        & np.all(board.group_relative_translation == (0, 0), axis=1)
    )
    assert len(matching) == 1
    assert board.group_right_support[matching[0]] >= 1
    assert board.group_down_support[matching[0]] >= 1


def test_nonreciprocal_contact_is_not_counted() -> None:
    layout, component, relative, right, down = _case()
    # Give target 9 eight even better incoming predecessors, so 0->9 remains
    # outgoing top-8 but is no longer reciprocal.
    for source in (2, 3, 4, 5, 6, 7, 10, 11, 14, 15):
        right[source, 9] = -200.0 - source
    board = build_dense_contact_consensus(
        layout=layout,
        component_of_tile=component,
        component_relative_coordinates=relative,
        cost_right=right,
        cost_down=down,
        grid=4,
    )
    emitted = set(
        zip(
            board.edge_source.tolist(),
            board.edge_target.tolist(),
            board.edge_axis.tolist(),
            strict=True,
        )
    )
    assert (0, 9, 0) not in emitted


def test_fixed_hyperparameters_are_enforced() -> None:
    layout, component, relative, right, down = _case()
    with pytest.raises(ValueError, match="dense_topk=8"):
        build_dense_contact_consensus(
            layout=layout,
            component_of_tile=component,
            component_relative_coordinates=relative,
            cost_right=right,
            cost_down=down,
            grid=4,
            dense_topk=7,
        )
