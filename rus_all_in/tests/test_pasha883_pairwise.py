from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aiijc_puzzle.pasha883_pairwise import (
    Pasha883PairwiseNet,
    load_pasha883_pairwise,
    pasha883_directional_retrieval_metrics,
)


def test_archived_network_contract_and_strict_checkpoint_load(tmp_path: Path) -> None:
    model = Pasha883PairwiseNet()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_953_025
    assert model(torch.rand(2, 3, 20, 40)).shape == (2,)
    with pytest.raises(ValueError, match="20 x 40"):
        model(torch.rand(2, 3, 20, 20))

    checkpoint = tmp_path / "pair.pt"
    torch.save({"model": model.state_dict(), "step": 6500, "val": 0.4765625}, checkpoint)
    loaded = load_pasha883_pairwise(checkpoint, device=torch.device("cpu"))
    assert loaded.step == 6500
    assert loaded.sampled_validation_accuracy_at_32_mislabeled_acc_at_48 == pytest.approx(
        0.4765625
    )
    assert loaded.model.training is False


def test_all_candidate_directional_metrics_recover_perfect_neighbours() -> None:
    grid = 3
    count = grid * grid
    layout = np.array([7, 2, 5, 1, 0, 6, 3, 8, 4])
    right = np.full((count, count), -2.0)
    down = np.full((count, count), -2.0)
    for position, tile in enumerate(layout):
        if position % grid != grid - 1:
            right[tile, layout[position + 1]] = 3.0
        if position < count - grid:
            down[tile, layout[position + grid]] = 3.0
    metrics = pasha883_directional_retrieval_metrics(
        right,
        down,
        layout,
        grid=grid,
        ks=(1, 2, 5),
    )
    assert metrics["right_query_count"] == 6
    assert metrics["down_query_count"] == 6
    assert metrics["pooled_query_count"] == 12
    assert metrics["pooled_r1"] == 1.0
    assert metrics["pooled_r5"] == 1.0
    assert metrics["pooled_median_rank"] == 1.0


def test_directional_metric_candidate_pool_includes_non_neighbours() -> None:
    grid = 2
    layout = np.arange(4)
    right = np.zeros((4, 4))
    down = np.zeros((4, 4))
    # Every true neighbour has exactly two strictly better non-self candidates.
    right[0, 2:] = 2.0
    right[0, 1] = 1.0
    right[2, [0, 1]] = 2.0
    right[2, 3] = 1.0
    down[0, [1, 3]] = 2.0
    down[0, 2] = 1.0
    down[1, [0, 2]] = 2.0
    down[1, 3] = 1.0
    metrics = pasha883_directional_retrieval_metrics(
        right,
        down,
        layout,
        grid=grid,
        ks=(1, 2, 3),
    )
    assert metrics["pooled_r1"] == 0.0
    assert metrics["pooled_r2"] == 0.0
    assert metrics["pooled_r3"] == 1.0
