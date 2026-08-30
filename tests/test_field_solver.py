import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from field_diffusion import BagEncoder, GRID, N, render
from field_solver import (constrained_layout, descriptor_cost, hungarian_layout,
                          components_from_score_tail, seam_energy, select_layout,
                          solve_field)


def _unique_fragments(seed=0):
    """Fragments whose exact 4x4 descriptors are unique and recoverable."""
    rng = np.random.default_rng(seed)
    blocks = rng.uniform(5, 250, (N, 4, 4, 3)).astype(np.float32)
    return np.repeat(np.repeat(blocks, 5, axis=1), 5, axis=2)


def test_field_hungarian_recovers_rendered_permutation():
    frags = _unique_fragments()
    order = np.random.default_rng(1).permutation(N)
    field = render(frags, order)
    got, value = solve_field(field, frags)
    assert np.array_equal(got, order)
    assert value == pytest.approx(0.0, abs=1e-4)


def test_bag_encoder_is_permutation_equivariant():
    torch.manual_seed(0)
    model = BagEncoder(d=16, layers=1, heads=4, view=2).eval()
    tiles = torch.rand(1, 12, 20, 20, 3) * 255.0
    perm = torch.randperm(tiles.shape[1])
    with torch.no_grad():
        direct = model(tiles)
        shuffled = model(tiles[:, perm])
    assert torch.allclose(shuffled, direct[:, perm], atol=2e-6, rtol=2e-6)


def test_zscore_is_invariant_to_fragment_affine():
    frags = _unique_fragments(2)
    order = np.random.default_rng(3).permutation(N)
    field = render(frags, order)
    changed = frags * 0.72 + 31.0
    cost = descriptor_cost(field, changed, "zscore")
    got = hungarian_layout(cost)
    assert np.array_equal(got, order)


def test_constrained_layout_preserves_relative_island():
    frags = _unique_fragments(4)
    order = np.random.default_rng(5).permutation(N)
    field = render(frags, order)
    # Deliberately make the field favour swapping the two tiles.  The rigid
    # component must still preserve their known left-to-right relation.
    cell = 8 * GRID + 9
    left, right = int(order[cell]), int(order[cell + 1])
    cost = descriptor_cost(field, frags)
    cost[cell, left], cost[cell, right] = cost[cell, right], -1e6
    cost[cell + 1, right], cost[cell + 1, left] = cost[cell + 1, left], -1e6
    got, _ = constrained_layout(cost, [{left: (0, 0), right: (0, 1)}],
                                beam=16, offsets=N)
    pos = np.empty(N, np.int64)
    pos[got] = np.arange(N)
    assert pos[right] == pos[left] + 1
    assert pos[left] // GRID == pos[right] // GRID
    assert len(np.unique(got)) == N


def test_component_validation_rejects_duplicate_tiles():
    cost = np.zeros((N, N), np.float64)
    with pytest.raises(ValueError, match="more than one component"):
        constrained_layout(cost, [{1: (0, 0), 2: (0, 1)},
                                  {2: (0, 0), 3: (1, 0)}])


def test_selector_uses_seam_only_as_finite_hypothesis_tiebreak():
    identity = np.arange(N, dtype=np.int64)
    swapped = identity.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    right = np.ones((N, N), np.float64)
    down = np.ones((N, N), np.float64)
    a = identity.reshape(GRID, GRID)
    right[a[:, :-1], a[:, 1:]] = 0.0
    down[a[:-1, :], a[1:, :]] = 0.0
    assert seam_energy(identity, right, down) < seam_energy(swapped, right, down)
    pick = select_layout([swapped, identity], [1.0, 1.0], right, down,
                         seam_weight=1.0)
    assert pick == 1


def test_score_tail_builds_rigid_island_from_best_edges():
    right = np.full((N, N), 10.0, np.float64)
    down = np.full((N, N), 10.0, np.float64)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    right[10, 11] = -5.0
    down[11, 35] = -4.0
    comps = components_from_score_tail(right, down, keep=2)
    assert len(comps) == 1
    c = comps[0]
    assert c[11][1] - c[10][1] == 1
    assert c[11][0] == c[10][0]
    assert c[35][0] - c[11][0] == 1
    assert c[35][1] == c[11][1]
