from __future__ import annotations

import numpy as np

from puzzle_assembly.anneal_refine import (
    _finite_matrix,
    _incremental_delta,
    anneal_refine,
    build_protected_edges,
    layout_energy,
)
from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.geometry import GRID, TILE_COUNT
from puzzle_assembly.solvers import placement_unary


def _random_compatibility(seed: int = 9) -> CompatibilityMatrices:
    rng = np.random.default_rng(seed)
    right = rng.uniform(0.0, 1.0, size=(TILE_COUNT, TILE_COUNT)).astype(np.float32)
    down = rng.uniform(0.0, 1.0, size=(TILE_COUNT, TILE_COUNT)).astype(np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices("random", right, down)


def _identity_favouring_compatibility() -> CompatibilityMatrices:
    right = np.ones((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    down = np.ones((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    grid = np.arange(TILE_COUNT, dtype=np.int32).reshape(GRID, GRID)
    right[grid[:, :-1], grid[:, 1:]] = 0.0
    down[grid[:-1, :], grid[1:, :]] = 0.0
    return CompatibilityMatrices("identity", right, down)


def test_incremental_swap_delta_matches_full_objective() -> None:
    compatibility = _random_compatibility()
    current = np.arange(TILE_COUNT, dtype=np.int32)
    candidate = current.copy()
    candidate[7], candidate[511] = candidate[511], candidate[7]
    changed = np.asarray([7, 511], dtype=np.int32)
    unary = placement_unary(compatibility).astype(np.float64)
    delta, affected = _incremental_delta(
        current,
        candidate,
        changed,
        augmented_right=_finite_matrix(compatibility.right, name="right"),
        augmented_down=_finite_matrix(compatibility.down, name="down"),
        unary=unary,
        boundary_weight=0.05,
    )
    full_delta = layout_energy(candidate, compatibility, boundary_weight=0.05)
    full_delta -= layout_energy(current, compatibility, boundary_weight=0.05)
    assert affected <= 8
    assert abs(delta - full_delta) < 1e-9


def test_protected_edges_are_input_only_seed_adjacencies() -> None:
    compatibility = _identity_favouring_compatibility()
    identity = np.arange(TILE_COUNT, dtype=np.int32)
    protected = build_protected_edges(
        compatibility,
        protected_layout=identity,
        confidence_quantile=0.5,
        max_edges=1200,
    )
    assert protected.count_right == GRID * (GRID - 1)
    assert protected.count_down == GRID * (GRID - 1)
    assert protected.total_confidence == protected.count
    identity_energy = layout_energy(
        identity,
        compatibility,
        boundary_weight=0.0,
        protected=protected,
        protection_weight=0.3,
    )
    swapped = identity.copy()
    swapped[0], swapped[-1] = swapped[-1], swapped[0]
    unprotected_swapped = layout_energy(swapped, compatibility, boundary_weight=0.0)
    protected_swapped = layout_energy(
        swapped,
        compatibility,
        boundary_weight=0.0,
        protected=protected,
        protection_weight=0.3,
    )
    assert identity_energy == 0.0
    assert protected_swapped > unprotected_swapped


def test_annealing_is_deterministic_and_never_loses_augmented_best() -> None:
    compatibility = _random_compatibility(seed=17)
    initial = np.arange(TILE_COUNT, dtype=np.int32)
    kwargs = dict(
        seed=20260711,
        seed_compatibility=compatibility,
        protected_layout=initial,
        evaluations_per_restart=120,
        restarts=1,
        protection_strength=0.1,
        weak_pool_size=24,
        weak_refresh=16,
        calibration_samples=8,
        polish_moves=2,
        polish_weak_cells=12,
        audit_interval=32,
    )
    first = anneal_refine(initial, compatibility, **kwargs)
    second = anneal_refine(initial, compatibility, **kwargs)
    assert np.array_equal(first.position_to_slot, second.position_to_slot)
    assert np.array_equal(np.sort(first.position_to_slot), initial)
    assert first.augmented_energy_after <= first.augmented_energy_before + 1e-9
    assert first.proposed_by_move == second.proposed_by_move
    assert first.accepted_by_move == second.accepted_by_move
