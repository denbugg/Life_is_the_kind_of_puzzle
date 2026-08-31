from __future__ import annotations

import numpy as np

from aiijc_puzzle.component_anchor_diagnostic import (
    diagnose_component_translation,
    rebuild_decoder_components,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments


def _perfect_assignment(layout: np.ndarray, *, grid: int, axis: str) -> np.ndarray:
    count = grid * grid
    board = np.asarray(layout).reshape(grid, grid)
    value = np.full((count + 1, count + 1), -20.0, dtype=np.float64)
    value[count, count] = -1e4
    if axis == "right":
        for row in range(grid):
            value[count, board[row, 0]] = 0.0
            value[board[row, -1], count] = 0.0
            for column in range(grid - 1):
                value[board[row, column], board[row, column + 1]] = 0.0
    else:
        for column in range(grid):
            value[count, board[0, column]] = 0.0
            value[board[-1, column], count] = 0.0
            for row in range(grid - 1):
                value[board[row, column], board[row + 1, column]] = 0.0
    return value


def test_rebuild_matches_decoder_component_sizes() -> None:
    grid = 4
    count = grid * grid
    generator = np.random.default_rng(91)
    right = generator.normal(size=(count + 1, count + 1))
    down = generator.normal(size=(count + 1, count + 1))
    right[count, count] = down[count, count] = -1e4
    config = SocketDecoderConfig(
        component_edge_budget_per_axis=6,
        swap_edge_budget_per_axis=6,
        max_swap_steps=0,
    )
    decoded = decode_socket_assignments(right, down, grid=grid, config=config)
    rebuilt = rebuild_decoder_components(
        right,
        down,
        grid=grid,
        edge_budget_per_axis=6,
    )
    sizes = tuple(sorted((len(component) for component in rebuilt.components), reverse=True))
    assert sizes == decoded.diagnostics.component_sizes
    assert rebuilt.status_counts["added"] == decoded.diagnostics.added_constraints


def test_oracle_component_is_internally_exact_and_anchor_errors_are_exact() -> None:
    grid = 4
    reference = np.random.default_rng(19).permutation(grid * grid)
    right = _perfect_assignment(reference, grid=grid, axis="right")
    down = _perfect_assignment(reference, grid=grid, axis="down")
    rebuilt = rebuild_decoder_components(
        right,
        down,
        grid=grid,
        edge_budget_per_axis=grid * (grid - 1),
    )
    assert len(rebuilt.components) == 1
    diagnostic = diagnose_component_translation(
        rebuilt.components[0],
        reference,
        {"decoder_border": reference, "decoder_texture_centre": reference},
        grid=grid,
        component_id=0,
    )
    assert diagnostic.internally_exact
    assert diagnostic.translation_purity == 1.0
    assert diagnostic.pairwise_relative_accuracy == 1.0
    assert diagnostic.anchors["decoder_border"].exact
    assert diagnostic.anchors["decoder_texture_centre"].exact
    assert diagnostic.anchors["geometric_centre"].exact


def test_purity_and_pairwise_accuracy_separate_two_translation_clusters() -> None:
    grid = 3
    reference = np.arange(grid * grid)
    # Tiles 0/1 agree with one horizontal translation, while tile 8 is a
    # disconnected true location assigned the relative coordinate (0, 2).
    component = {0: (0, 0), 1: (0, 1), 8: (0, 2)}
    diagnostic = diagnose_component_translation(
        component,
        reference,
        {"decoder_border": reference},
        grid=grid,
        component_id=7,
    )
    assert diagnostic.true_shift_support == 2
    assert np.isclose(diagnostic.translation_purity, 2 / 3)
    assert np.isclose(diagnostic.pairwise_relative_accuracy, 1 / 3)
    assert not diagnostic.internally_exact


def test_trusted_evidence_can_measure_a_component_without_ambiguous_tiles() -> None:
    grid = 3
    reference = np.arange(grid * grid)
    component = {0: (0, 0), 1: (0, 1), 8: (0, 2)}
    diagnostic = diagnose_component_translation(
        component,
        reference,
        {"decoder_border": reference},
        grid=grid,
        component_id=7,
        evidence_tiles=np.asarray([0, 1]),
    )
    assert diagnostic.size == 3
    assert diagnostic.evidence_size == 2
    assert diagnostic.internally_exact
    assert diagnostic.translation_purity == 1.0
