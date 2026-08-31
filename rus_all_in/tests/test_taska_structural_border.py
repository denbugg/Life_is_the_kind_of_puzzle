from __future__ import annotations

import numpy as np
import pytest
import torch

from aiijc_puzzle.taska_structural_border import (
    SIDES,
    structural_border_scores,
    structural_border_unary,
)


class _EquivariantMatcher:
    def __init__(self, *, scale: float, bias: float) -> None:
        self.scale = scale
        self.bias = bias

    def right_down_logits(
        self,
        tiles_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = tiles_tensor[:, 0, 0, 0]
        source = value[:, None]
        target = value[None, :]
        right = self.scale * (0.7 * source - (target - source - 1.0).square()) + self.bias
        down = self.scale * (-0.4 * source - (target - source - 2.0).square()) - self.bias
        return right, down


class _StaticMatcher:
    def __init__(self, right: torch.Tensor, down: torch.Tensor) -> None:
        self.right = right
        self.down = down

    def right_down_logits(
        self,
        tiles_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.right.to(tiles_tensor.device), self.down.to(tiles_tensor.device)


def _tiles(grid: int) -> np.ndarray:
    count = grid * grid
    tiles = np.zeros((count, 20, 20, 3), dtype=np.float32)
    tiles[:, :, :, 0] = np.arange(count, dtype=np.float32)[:, None, None]
    return tiles


def _historical_slack(
    logits: torch.Tensor,
    *,
    slack: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(logits)
    augmented = torch.zeros(count + 1, count + 1)
    augmented[:count, :count] = logits
    augmented.fill_diagonal_(-1.0e4)
    row_mass = torch.ones(count + 1)
    column_mass = torch.ones(count + 1)
    row_mass[count] = column_mass[count] = slack
    for _ in range(iterations):
        augmented = (
            augmented
            - torch.logsumexp(augmented, 1, keepdim=True)
            + row_mass.log()[:, None]
        )
        augmented = (
            augmented
            - torch.logsumexp(augmented, 0, keepdim=True)
            + column_mass.log()[None, :]
        )
    return augmented[:count, count].exp().numpy(), augmented[count, :count].exp().numpy()


def test_side_mapping_matches_historical_reference() -> None:
    right = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 7.0
    down = torch.flip(right, dims=(0,)) * 0.6 - 0.3
    scores = structural_border_scores(
        [_StaticMatcher(right, down)],
        _tiles(2),
        grid=2,
        slack=3.0,
        sinkhorn_iterations=4,
    )
    right_out, right_in = _historical_slack(right, slack=3.0, iterations=4)
    down_out, down_in = _historical_slack(down, slack=3.0, iterations=4)

    assert np.allclose(scores["right"], right_out)
    assert np.allclose(scores["left"], right_in)
    assert np.allclose(scores["bottom"], down_out)
    assert np.allclose(scores["top"], down_in)


def test_scores_average_models_and_are_finite() -> None:
    tiles = _tiles(3)
    first = _EquivariantMatcher(scale=0.3, bias=-0.2)
    second = _EquivariantMatcher(scale=0.7, bias=0.4)

    first_scores = structural_border_scores([first], tiles, grid=3, sinkhorn_iterations=5)
    second_scores = structural_border_scores([second], tiles, grid=3, sinkhorn_iterations=5)
    averaged = structural_border_scores(
        [first, second],
        tiles,
        grid=3,
        sinkhorn_iterations=5,
    )

    assert tuple(averaged) == SIDES
    for side in SIDES:
        assert averaged[side].shape == (9,)
        assert averaged[side].dtype == np.float64
        assert np.isfinite(averaged[side]).all()
        assert np.allclose(
            averaged[side],
            0.5 * (first_scores[side] + second_scores[side]),
            atol=1e-7,
        )


def test_tile_axis_is_permutation_equivariant() -> None:
    tiles = _tiles(3)
    models = [
        _EquivariantMatcher(scale=0.3, bias=-0.2),
        _EquivariantMatcher(scale=0.7, bias=0.4),
    ]
    permutation = np.asarray([4, 1, 7, 0, 8, 3, 5, 2, 6])

    scores = structural_border_scores(models, tiles, grid=3, sinkhorn_iterations=7)
    permuted_scores = structural_border_scores(
        models,
        tiles[permutation],
        grid=3,
        sinkhorn_iterations=7,
    )
    unary = structural_border_unary(models, tiles, grid=3, sinkhorn_iterations=7)
    permuted_unary = structural_border_unary(
        models,
        tiles[permutation],
        grid=3,
        sinkhorn_iterations=7,
    )

    for side in SIDES:
        assert np.allclose(permuted_scores[side], scores[side][permutation], atol=2e-6)
    assert np.allclose(permuted_unary, unary[permutation], atol=2e-6)


def test_unary_is_border_only_with_historical_shape() -> None:
    unary = structural_border_unary(
        [_EquivariantMatcher(scale=0.5, bias=0.0)],
        _tiles(3),
        grid=3,
        sinkhorn_iterations=5,
    )

    assert unary.shape == (9, 3, 3)
    assert unary.dtype == np.float64
    assert np.isfinite(unary).all()
    assert np.array_equal(unary[:, 1, 1], np.zeros(9))
    assert np.any(unary[:, 0, :] != 0.0)
    assert np.any(unary[:, -1, :] != 0.0)
    assert np.any(unary[:, :, 0] != 0.0)
    assert np.any(unary[:, :, -1] != 0.0)


def test_production_grid_shape_and_finiteness() -> None:
    unary = structural_border_unary(
        [_EquivariantMatcher(scale=0.01, bias=0.0)],
        _tiles(24),
        grid=24,
        sinkhorn_iterations=1,
    )

    assert unary.shape == (576, 24, 24)
    assert np.isfinite(unary).all()
    assert np.array_equal(unary[:, 1:-1, 1:-1], np.zeros((576, 22, 22)))


def test_invalid_inputs_fail_closed() -> None:
    tiles = _tiles(2)
    model = _EquivariantMatcher(scale=1.0, bias=0.0)
    with pytest.raises(ValueError, match="at least one"):
        structural_border_scores([], tiles, grid=2)
    with pytest.raises(ValueError, match="tiles must have shape"):
        structural_border_scores([model], tiles[:3], grid=2)
    with pytest.raises(ValueError, match="slack"):
        structural_border_scores([model], tiles, grid=2, slack=0.0)
    damaged = tiles.copy()
    damaged[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        structural_border_scores([model], damaged, grid=2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_path_matches_cpu_shape_and_is_finite() -> None:
    unary = structural_border_unary(
        [_EquivariantMatcher(scale=0.5, bias=0.0)],
        _tiles(2),
        device="mps",
        grid=2,
        sinkhorn_iterations=3,
    )

    assert unary.shape == (4, 2, 2)
    assert np.isfinite(unary).all()
