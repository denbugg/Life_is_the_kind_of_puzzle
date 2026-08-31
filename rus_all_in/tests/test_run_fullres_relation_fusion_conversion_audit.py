from __future__ import annotations

import numpy as np

from scripts.run_fullres_relation_fusion_conversion_audit import (
    GRID,
    TILE_COUNT,
    component_stats,
    edge_stats,
    oracle_best_cyclic,
)


def test_edge_truth_uses_upright_grid_displacements() -> None:
    tile_to_position = np.arange(TILE_COUNT, dtype=np.int32)
    result = edge_stats(
        {
            ("right", 0, 1),
            ("right", GRID - 1, GRID),
            ("down", 0, GRID),
        },
        tile_to_position,
    )
    assert result["edge_count"] == 3
    assert result["correct_edges"] == 2


def test_oracle_cyclic_recovers_a_known_whole_board_roll() -> None:
    reference = np.arange(TILE_COUNT, dtype=np.int32)
    shifted = np.roll(
        reference.reshape(GRID, GRID),
        shift=(3, 5),
        axis=(0, 1),
    ).reshape(-1)
    result = oracle_best_cyclic(shifted, reference)
    assert result["row_roll"] == GRID - 3
    assert result["column_roll"] == GRID - 5
    assert result["metrics"]["correct_tile_count"] == TILE_COUNT
    assert result["target_assisted_not_deployable"]


def test_component_audit_recognises_exact_geometry_and_anchor() -> None:
    component = {
        tile: divmod(tile, GRID)
        for tile in range(TILE_COUNT)
    }
    identity = np.arange(TILE_COUNT, dtype=np.int32)
    result = component_stats((component,), identity, identity)
    assert result["component_count"] == 1
    assert result["tile_weighted_truth_translation_purity"] == 1.0
    assert result["pairwise_relative_accuracy"] == 1.0
    assert result["truth_mode_anchor_correct_support"] == TILE_COUNT
