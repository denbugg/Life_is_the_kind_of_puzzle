from __future__ import annotations

import numpy as np
import pytest

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.components import (
    ProposedEdge,
    project_rigid_components_around_reference,
    reference_manhattan_placement_costs,
)
from puzzle_assembly.geometry import TILE_COUNT


def _flat_compatibility() -> CompatibilityMatrices:
    right = np.ones((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    down = np.ones((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices(name="flat", right=right, down=down)


def _pair_partition() -> list[dict[int, tuple[int, int]]]:
    return [
        {0: (0, 0), 1: (1, 0)},
        *[{tile: (0, 0)} for tile in range(2, TILE_COUNT)],
    ]


def test_reference_manhattan_costs_are_normalized_and_exact_on_reference() -> None:
    reference = np.arange(TILE_COUNT, dtype=np.int32)
    costs = reference_manhattan_placement_costs(reference)
    assert costs.shape == (TILE_COUNT, TILE_COUNT)
    assert costs.dtype == np.float32
    np.testing.assert_array_equal(costs[np.arange(TILE_COUNT), reference], 0.0)
    assert float(costs[0, TILE_COUNT - 1]) == pytest.approx(1.0)


def test_rigid_projection_preserves_accepted_pair_without_post_refinement() -> None:
    reference = np.arange(TILE_COUNT, dtype=np.int32)
    accepted = [
        ProposedEdge(
            first=0,
            second=1,
            dx=1,
            dy=0,
            cost=0.0,
            margin=1.0,
            reciprocal=True,
            in_loop=True,
        )
    ]
    result = project_rigid_components_around_reference(
        _pair_partition(),
        accepted,
        _flat_compatibility(),
        reference,
        selected_proposals=1,
        reference_weight=0.5,
        beam_width=2,
        beam_components=1,
        translations_per_state=2,
    )
    np.testing.assert_array_equal(result.position_to_slot, reference)
    assert result.accepted_edges == 1
    assert result.retained_accepted_edges == 1
    assert result.retained_accepted_edge_fraction == 1.0
    assert result.placed_component_tiles == 2
    assert result.unresolved_tiles_before_assignment == TILE_COUNT - 2


def test_rigid_projection_is_deterministic() -> None:
    reference = np.arange(TILE_COUNT, dtype=np.int32)
    accepted = [
        ProposedEdge(0, 1, 1, 0, 0.0, 1.0, True, True),
    ]
    kwargs = dict(
        components=_pair_partition(),
        accepted_edges=accepted,
        placement_compatibility=_flat_compatibility(),
        reference_position_to_slot=reference,
        selected_proposals=1,
        reference_weight=0.5,
        beam_width=2,
        beam_components=1,
        translations_per_state=2,
    )
    first = project_rigid_components_around_reference(**kwargs)
    second = project_rigid_components_around_reference(**kwargs)
    np.testing.assert_array_equal(first.position_to_slot, second.position_to_slot)


def test_rigid_projection_rejects_non_partition() -> None:
    reference = np.arange(TILE_COUNT, dtype=np.int32)
    with pytest.raises(ValueError, match="partition"):
        project_rigid_components_around_reference(
            [{0: (0, 0)}],
            [],
            _flat_compatibility(),
            reference,
            selected_proposals=0,
            beam_width=2,
            beam_components=1,
        )
