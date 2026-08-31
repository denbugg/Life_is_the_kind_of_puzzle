from __future__ import annotations

import numpy as np
import pytest
import torch

from aiijc_puzzle.border_pointer_repair import (
    BaselineRepairConfig,
    baseline_guided_pointer_repair,
)
from aiijc_puzzle.border_pointer_sorter import BorderPointerSorter


def _model() -> BorderPointerSorter:
    return BorderPointerSorter(
        feature_width=8,
        feature_blocks=1,
        dimension=16,
        heads=4,
        board_layers=1,
        pointer_layers=2,
        max_grid=4,
    ).eval()


def test_high_margin_repair_preserves_baseline_and_freezes_traces() -> None:
    torch.manual_seed(7)
    model = _model()
    tiles = torch.rand(1, 16, 3, 20, 20)
    baseline = np.random.default_rng(7).permutation(16).astype(np.int32)
    scores = np.random.default_rng(8).normal(size=(16, 16))
    result = baseline_guided_pointer_repair(
        model,
        tiles,
        baseline,
        scores,
        scores.T.copy(),
        grid=4,
        config=BaselineRepairConfig(
            logit_margin=1e6,
            budgets=(1, 2),
            socket_support_topk=3,
        ),
    )
    assert np.array_equal(result.layouts[1], baseline)
    assert np.array_equal(result.layouts[2], baseline)
    assert not result.proposals
    assert result.trace.prefix_topk.shape == (16, 5)
    assert result.trace.no_prefix_topk.shape == (16, 5)


def test_low_margin_repair_always_returns_strict_permutations() -> None:
    torch.manual_seed(9)
    model = _model()
    tiles = torch.rand(1, 16, 3, 20, 20)
    baseline = np.random.default_rng(9).permutation(16).astype(np.int32)
    scores = np.random.default_rng(10).normal(size=(16, 16))
    result = baseline_guided_pointer_repair(
        model,
        tiles,
        baseline,
        scores,
        scores.T.copy(),
        grid=4,
        config=BaselineRepairConfig(
            logit_margin=0.0,
            budgets=(1, 2),
            socket_support_topk=15,
        ),
    )
    for layout in result.layouts.values():
        assert np.array_equal(np.sort(layout), np.arange(16))


def test_repair_config_rejects_invalid_budget_contract() -> None:
    with pytest.raises(ValueError, match="budgets"):
        BaselineRepairConfig(budgets=(2, 1)).validate(count=16)
